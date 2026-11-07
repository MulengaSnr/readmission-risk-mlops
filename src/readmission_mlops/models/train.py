"""Train model v2, track it in MLflow and register it behind promotion gates.

Run from the project root (after ``make data``):

    uv run python -m readmission_mlops.models.train

Each run:

1. builds one pipeline: stateless feature engineering -> fitted preprocessing
   (without ``payer_code``) -> XGBoost with project 1's tuned settings;
2. scores it with 5-fold patient-grouped cross-validation (same folds as
   project 1);
3. refits on the full training set and scores the test set. This is the
   **second** use of the test set (v1.0 used it first), so it checks v2
   against v1.0 and is not used to choose between options;
4. logs parameters, metrics, per-fold results, data fingerprints and code
   versions to MLflow, and registers the model;
5. points the ``champion`` alias at the new version only if it passes the
   promotion gates in ``configs/default.yaml``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.features.build import add_features, build_preprocessor
from readmission_risk.models.baselines import (
    cross_validate_models,
    grouped_folds,
    summarise,
)
from readmission_risk.models.metrics import evaluate
from readmission_risk.models.tuning import make_candidates
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from readmission_mlops.config import PROJECT_ROOT, load_config
from readmission_mlops.data.prepare import data_fingerprint
from readmission_mlops.features.schema import serving_columns
from readmission_mlops.models.serialization import TRUSTED_TYPES

logger = logging.getLogger(__name__)


def build_model(config: dict[str, Any]) -> Pipeline:
    """The unfitted v2 pipeline: features -> preprocessing -> XGBoost.

    Args:
        config: Parsed configuration (``model`` and ``seed`` sections).

    Returns:
        A pipeline whose input is a frame of :func:`serving_columns` fields.
    """
    exclude = tuple(config["model"]["exclude_features"])
    boosted = clone(make_candidates(config["seed"])["xgboost"][0])
    boosted.set_params(
        **{f"model__{k}": v for k, v in config["model"]["xgboost_params"].items()}
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("prep", build_preprocessor(exclude=exclude)),
            ("model", boosted.named_steps["model"]),
        ]
    )


def check_gates(
    cv_pr_auc: float, test_pr_auc: float, promotion: dict[str, float]
) -> list[str]:
    """Reasons the model must not be promoted; an empty list means it passes.

    Args:
        cv_pr_auc: Mean cross-validated PR-AUC.
        test_pr_auc: PR-AUC on the test set.
        promotion: The ``promotion`` section of the configuration.

    Returns:
        Human-readable failure reasons.
    """
    failures = []
    if cv_pr_auc < promotion["min_cv_pr_auc"]:
        failures.append(
            f"CV PR-AUC {cv_pr_auc:.4f} < minimum {promotion['min_cv_pr_auc']}"
        )
    floor = promotion["reference_test_pr_auc"] - promotion["max_test_pr_auc_drop"]
    if test_pr_auc < floor:
        failures.append(
            f"test PR-AUC {test_pr_auc:.4f} < floor {floor:.4f} (v1.0 reference "
            f"minus allowed drop {promotion['max_test_pr_auc_drop']})"
        )
    return failures


def resolve_tracking_uri(uri: str, root: Path) -> str:
    """Make a relative ``sqlite:///`` URI absolute under ``root``."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix) :]).is_absolute():
        return prefix + str(root / uri[len(prefix) :])
    return uri


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _experiment_id(name: str, artifact_root: Path) -> str:
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_root.as_uri())


def train(config: dict[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Run the full training, tracking and registration flow.

    Args:
        config: Parsed configuration.
        root: Project root for data, tracking database and artefacts.

    Returns:
        Summary with run ID, model version, metrics and promotion outcome.
    """
    processed = root / config["paths"]["processed"]
    train_df = pd.read_parquet(processed / "train.parquet")
    test_df = pd.read_parquet(processed / "test.parquet")
    columns = serving_columns(tuple(config["model"]["exclude_features"]))
    x_train, y_train = train_df[columns], train_df[TARGET].to_numpy()
    x_test, y_test = test_df[columns], test_df[TARGET].to_numpy()
    capacities = config["evaluation"]["capacities"]

    model = build_model(config)
    folds = grouped_folds(
        y_train, train_df[GROUP_COL], config["cv"]["n_splits"], config["seed"]
    )
    cv = cross_validate_models(x_train, y_train, folds, {"v2": model}, capacities)
    cv_summary = summarise(cv).loc["v2"]
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate(y_test, prob, capacities)

    name = config["model"]["name"]
    failures = check_gates(
        cv_summary[("pr_auc", "mean")], test_metrics["pr_auc"], config["promotion"]
    )
    mlflow.set_tracking_uri(
        resolve_tracking_uri(config["mlflow"]["tracking_uri"], root)
    )
    experiment = _experiment_id(config["mlflow"]["experiment"], root / "mlruns")
    with mlflow.start_run(
        experiment_id=experiment, run_name="v2-xgboost-no-payer"
    ) as run:
        mlflow.log_params(
            {
                **{f"xgb_{k}": v for k, v in config["model"]["xgboost_params"].items()},
                "excluded_features": ",".join(config["model"]["exclude_features"]),
                "n_input_fields": len(columns),
                "cv_folds": config["cv"]["n_splits"],
                "seed": config["seed"],
            }
        )
        mlflow.set_tags(
            {
                "model_version_label": config["model"]["version_label"],
                "train_fingerprint": data_fingerprint(train_df),
                "test_fingerprint": data_fingerprint(test_df),
                "test_set_use": "second (v1.0 was evaluated on it first)",
                "readmission_risk_version": importlib.metadata.version(
                    "readmission-risk"
                ),
                "git_commit": _git_commit(root),
                "promotion_failures": "; ".join(failures) or "none",
            }
        )
        for (metric, stat), value in cv_summary.items():
            mlflow.log_metric(f"cv_{metric}_{stat}", float(value))
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_text(cv.to_csv(index=False), "cv_folds.csv")
        mlflow.log_dict({"input_fields": columns}, "input_schema.json")
        example = x_train.head(5)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict_proba(example)[:, 1]),
            input_example=example,
            registered_model_name=name,
            skops_trusted_types=TRUSTED_TYPES,
        )

    client = MlflowClient()
    version = str(info.registered_model_version)
    promoted = not failures
    client.set_model_version_tag(name, version, "promoted", str(promoted).lower())
    if promoted:
        client.set_registered_model_alias(
            name, config["mlflow"]["champion_alias"], version
        )
    return {
        "run_id": run.info.run_id,
        "model_name": name,
        "model_version": version,
        "promoted": promoted,
        "promotion_failures": failures,
        "cv_pr_auc_mean": float(cv_summary[("pr_auc", "mean")]),
        "cv_pr_auc_std": float(cv_summary[("pr_auc", "std")]),
        "test": test_metrics,
    }


def _jsonable(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    """Command-line entry point."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = train(load_config())
    print(json.dumps(result, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
