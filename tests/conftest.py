"""Shared fixtures: a tiny model trained on synthetic encounters.

CI has no data, so API tests use a model fitted on random but schema-valid
records. It is only for testing the service mechanics, never for results.
"""

import copy
import json

import numpy as np
import pandas as pd
import pytest

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


def synthetic_encounters(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = serving_columns(("payer_code",))
    data = {}
    for c in cols:
        if c in CATEGORIES:
            data[c] = rng.choice(np.array(CATEGORIES[c], dtype=object), n)
        elif c == "age":
            data[c] = rng.choice(AGES, n)
        elif c == "gender":
            data[c] = rng.choice(["Female", "Male"], n)
        elif c.startswith("diag_"):
            data[c] = rng.choice(["428", "250.02", "486", "V57", "715"], n)
        elif c == "A1Cresult":
            data[c] = rng.choice(["Not measured", "Norm", ">7", ">8"], n)
        elif c == "max_glu_serum":
            data[c] = rng.choice(["Not measured", "Norm", ">200"], n)
        elif c == "change":
            data[c] = rng.choice(["No", "Ch"], n)
        elif c == "diabetesMed":
            data[c] = rng.choice(["Yes", "No"], n)
        elif c == "time_in_hospital":
            data[c] = rng.integers(1, 15, n)
        elif c.startswith(("number_", "num_")):
            data[c] = rng.poisson(2 if c.startswith("number_") else 20, n)
        else:  # medications
            data[c] = rng.choice(DRUGS, n, p=[0.7, 0.2, 0.05, 0.05])
    df = pd.DataFrame(data, columns=cols)
    df["number_diagnoses"] = df["number_diagnoses"].clip(lower=1)
    return df


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    config = copy.deepcopy(load_config())
    config["model"]["xgboost_params"]["n_estimators"] = 15
    x = synthetic_encounters(600)
    logit = 0.4 * x["number_inpatient"] - 2.5
    y = (np.random.default_rng(1).random(len(x)) < 1 / (1 + np.exp(-logit))).astype(int)
    model = build_model(config).fit(x, y)
    prob = model.predict_proba(x)[:, 1]

    folder = tmp_path_factory.mktemp("model")
    serialization.save(model, folder / "model.skops")
    metadata = {
        "model_name": "readmission-risk",
        "model_version": 7,
        "version_label": "test",
        "run_id": "test-run",
        "input_fields": list(x.columns),
        "excluded_features": ["payer_code"],
        "train_fingerprint": "0000000000000000",
        "git_commit": "test",
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
        "metrics": {"test_pr_auc": 0.5},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return folder


@pytest.fixture
def example_encounter():
    return {
        "age": "[70-80)",
        "gender": "Female",
        "time_in_hospital": 5,
        "number_inpatient": 2,
        "number_emergency": 0,
        "number_outpatient": 1,
        "num_lab_procedures": 44,
        "num_procedures": 1,
        "num_medications": 16,
        "number_diagnoses": 9,
        "admission_type": "Emergency",
        "admission_source": "Emergency Room",
        "discharge_disposition": "Discharged/transferred to SNF",
        "medical_specialty": "InternalMedicine",
        "diag_1": "428",
        "diag_2": "250.02",
        "diag_3": "401",
        "diabetesMed": "Yes",
        "insulin": "Up",
    }
