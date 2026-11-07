"""A tiny model trained on synthetic encounters, for tests and PR images.

CI has no data. Pull-request builds use this model to test the service and
container mechanics; it is never used for results. Release images on `main`
contain the real model (see `.github/workflows/ci.yml`).

    uv run python -m readmission_mlops.synthetic artifacts/model
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from readmission_mlops.config import load_config
from readmission_mlops.features.schema import serving_columns
from readmission_mlops.models import serialization
from readmission_mlops.models.train import OPEN_CATEGORIES, build_model

AGES = ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]
DRUGS = ["No", "Steady", "Up", "Down"]
CATEGORIES = {
    "admission_type": ["Emergency", "Elective", "Urgent", None],
    "admission_source": ["Emergency Room", "Physician Referral", None],
    "discharge_disposition": ["Discharged to home", "Discharged/transferred to SNF"],
    "medical_specialty": ["InternalMedicine", "Cardiology", None],
}
FIXED = {
    "age": AGES,
    "gender": ["Female", "Male"],
    "A1Cresult": ["Not measured", "Norm", ">7", ">8"],
    "max_glu_serum": ["Not measured", "Norm", ">200"],
    "change": ["No", "Ch"],
    "diabetesMed": ["Yes", "No"],
}


def synthetic_encounters(n: int, seed: int = 0) -> pd.DataFrame:
    """Random, schema-valid encounters with the model's input columns."""
    rng = np.random.default_rng(seed)
    data = {}
    for c in serving_columns(("payer_code",)):
        if c in CATEGORIES:
            data[c] = rng.choice(np.array(CATEGORIES[c], dtype=object), n)
        elif c in FIXED:
            data[c] = rng.choice(FIXED[c], n)
        elif c.startswith("diag_"):
            data[c] = rng.choice(["428", "250.02", "486", "V57", "715"], n)
        elif c == "time_in_hospital":
            data[c] = rng.integers(1, 15, n)
        elif c.startswith(("number_", "num_")):
            data[c] = rng.poisson(2 if c.startswith("number_") else 20, n)
        else:  # medications
            data[c] = rng.choice(DRUGS, n, p=[0.7, 0.2, 0.05, 0.05])
    df = pd.DataFrame(data)
    df["number_diagnoses"] = df["number_diagnoses"].clip(lower=1)
    return df


def build_test_model(dest: Path, n: int = 600, n_estimators: int = 15) -> Path:
    """Fit a small v2 pipeline on synthetic data and export it like the real one.

    Risk rises with prior inpatient stays, so tests can check the direction.

    Args:
        dest: Folder to write ``model.skops`` and ``metadata.json`` into.
        n: Number of synthetic encounters.
        n_estimators: Trees in the XGBoost model.

    Returns:
        ``dest``.
    """
    config = copy.deepcopy(load_config())
    config["model"]["xgboost_params"]["n_estimators"] = n_estimators
    x = synthetic_encounters(n)
    logit = 0.4 * x["number_inpatient"] - 2.5
    y = (np.random.default_rng(1).random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = build_model(config).fit(x, y)
    prob = model.predict_proba(x)[:, 1]

    serialization.save(model, dest / "model.skops")
    metadata = {
        "model_name": "readmission-risk",
        "model_version": 0,
        "version_label": "synthetic-test",
        "run_id": "synthetic",
        "input_fields": list(x.columns),
        "excluded_features": list(config["model"]["exclude_features"]),
        "train_fingerprint": "synthetic",
        "git_commit": "synthetic",
        "readmission_risk_version": "0.1.0",
        "band_thresholds": {
            "top_10pct": float(np.quantile(prob, 0.9)),
            "top_20pct": float(np.quantile(prob, 0.8)),
        },
        "training_readmission_rate": float(y.mean()),
        "known_categories": {
            c: sorted(v for v in CATEGORIES[c] if v is not None)
            for c in OPEN_CATEGORIES
        },
        "metrics": {},
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def main() -> None:
    """Command-line entry point: write the synthetic model to a folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", type=Path)
    build_test_model(parser.parse_args().dest)


if __name__ == "__main__":
    main()
