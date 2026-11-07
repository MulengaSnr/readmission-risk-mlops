"""Export the champion model from the MLflow registry for the service image.

Run from the project root (after ``make train``):

    uv run python -m readmission_mlops.models.export

Writes ``artifacts/model/model.skops`` (safe format, see ``serialization``)
and ``metadata.json``. The Docker
image (step 4) copies this folder, so the running service never needs to
reach the registry (design decision D3), and the image is the release.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from readmission_mlops.config import PROJECT_ROOT, load_config
from readmission_mlops.models import serialization
from readmission_mlops.models.train import resolve_tracking_uri

logger = logging.getLogger(__name__)


def export_champion(config: dict[str, Any], root: Path = PROJECT_ROOT) -> Path:
    """Load the model behind the champion alias and write it with metadata.

    Args:
        config: Parsed configuration.
        root: Project root.

    Returns:
        The export folder.

    Raises:
        mlflow.exceptions.MlflowException: If no version has the alias yet.
    """
    mlflow.set_tracking_uri(
        resolve_tracking_uri(config["mlflow"]["tracking_uri"], root)
    )
    name, alias = config["model"]["name"], config["mlflow"]["champion_alias"]
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    run = client.get_run(version.run_id)
    # Download the raw artefact and load it with *our* allow-list. MLflow's own
    # loader would read the trusted types from the model's MLmodel file, which
    # a tampered model could edit to approve its own dangerous types.
    local = Path(mlflow.artifacts.download_artifacts(f"models:/{name}@{alias}"))
    model = serialization.load(local / "model.skops")
    schema = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/input_schema.json")
    serving = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/serving_config.json")
    reference = mlflow.artifacts.load_dict(
        f"runs:/{version.run_id}/monitoring_reference.json"
    )

    dest = root / config["paths"]["model_export"]
    dest.mkdir(parents=True, exist_ok=True)
    serialization.save(model, dest / "model.skops")
    metadata = {
        "model_name": name,
        "model_version": version.version,
        "version_label": run.data.tags.get("model_version_label"),
        "run_id": version.run_id,
        "input_fields": schema["input_fields"],
        **serving,
        "excluded_features": run.data.params.get("excluded_features", "").split(","),
        "train_fingerprint": run.data.tags.get("train_fingerprint"),
        "git_commit": run.data.tags.get("git_commit"),
        "readmission_risk_version": run.data.tags.get("readmission_risk_version"),
        "metrics": {
            k: round(v, 5)
            for k, v in sorted(run.data.metrics.items())
            if k.startswith(("cv_pr_auc", "cv_roc_auc", "test_"))
        },
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (dest / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    return dest


def main() -> None:
    """Command-line entry point."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dest = export_champion(load_config())
    logger.info("Exported champion model to %s", dest)


if __name__ == "__main__":
    main()
