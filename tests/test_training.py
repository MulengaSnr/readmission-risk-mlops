import copy
import json

import numpy as np
import pandas as pd
import pytest
import skops.io
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from readmission_mlops.config import PROJECT_ROOT, load_config
from readmission_mlops.data.prepare import data_fingerprint
from readmission_mlops.features.schema import ENGINEERED, serving_columns
from readmission_mlops.models import serialization
from readmission_mlops.models.train import check_gates, resolve_tracking_uri

# --- data fingerprint ----------------------------------------------------


def test_fingerprint_ignores_row_order_but_not_outcomes():
    df = pd.DataFrame({"encounter_id": [3, 1, 2], "readmitted_30d": [0, 1, 0]})
    shuffled = df.sample(frac=1, random_state=0)
    assert data_fingerprint(df) == data_fingerprint(shuffled)
    flipped = df.assign(readmitted_30d=[1, 1, 0])
    assert data_fingerprint(df) != data_fingerprint(flipped)
    assert len(data_fingerprint(df)) == 16


# --- input contract ------------------------------------------------------


def test_serving_columns_are_raw_and_exclude_payer_and_race():
    cols = serving_columns(("payer_code",))
    assert "payer_code" not in cols and "race" not in cols
    assert not ENGINEERED & set(cols)
    assert {"age", "diag_1", "diag_2", "diag_3", "insulin"} <= set(cols)
    assert len(cols) == len(set(cols))
    assert "payer_code" in serving_columns()


# --- promotion gates -----------------------------------------------------

PROMOTION = {"min_cv_pr_auc": 0.20, "reference_cv_pr_auc": 0.2335,
             "max_cv_pr_auc_drop": 0.01}  # fmt: skip


@pytest.mark.parametrize(
    ("cv", "n_failures"),
    [(0.2298, 0), (0.2236, 0), (0.2234, 1), (0.19, 2)],
)
def test_check_gates(cv, n_failures):
    assert len(check_gates(cv, PROMOTION)) == n_failures


def test_promotion_settings_do_not_refer_to_the_test_set():
    assert not [key for key in load_config()["promotion"] if "test" in key]


def test_resolve_tracking_uri(tmp_path):
    assert resolve_tracking_uri("sqlite:///mlflow.db", tmp_path) == (
        f"sqlite:///{tmp_path}/mlflow.db"
    )
    assert (
        resolve_tracking_uri("sqlite:////abs/x.db", tmp_path) == "sqlite:////abs/x.db"
    )
    assert resolve_tracking_uri("http://host:5000", tmp_path) == "http://host:5000"


# --- safe serialisation --------------------------------------------------


def untrusted_function(x):
    return x


def test_untrusted_types_are_refused(tmp_path):
    path = tmp_path / "m.skops"
    skops.io.dump(Pipeline([("f", FunctionTransformer(untrusted_function))]), path)
    with pytest.raises(skops.io.exceptions.UntrustedTypesFoundException):
        serialization.load(path)


# --- end to end on real data (skipped in CI: no data) ---------------------

DATA = PROJECT_ROOT / "data" / "processed"


@pytest.mark.integration
@pytest.mark.skipif(not (DATA / "train.parquet").exists(), reason="data not prepared")
def test_train_register_promote_export(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    from readmission_mlops.models.export import export_champion
    from readmission_mlops.models.train import train

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for part, n in (("train", 3000), ("test", 1000)):
        pd.read_parquet(DATA / f"{part}.parquet").sample(n, random_state=0).to_parquet(
            processed / f"{part}.parquet", index=False
        )
    config = copy.deepcopy(load_config())
    config["cv"]["n_splits"] = 2
    config["model"]["xgboost_params"]["n_estimators"] = 20
    config["promotion"].update(min_cv_pr_auc=0.0, reference_cv_pr_auc=0.0)

    result = train(config, root=tmp_path)
    assert result["promoted"] and result["model_version"] == "1"

    dest = export_champion(config, root=tmp_path)
    meta = json.loads((dest / "metadata.json").read_text())
    assert meta["version_label"] == "v2"
    assert meta["excluded_features"] == ["payer_code"]
    bands = meta["band_thresholds"]
    assert 0 < bands["top_20pct"] <= bands["top_10pct"] < 1
    assert "Emergency" in meta["known_categories"]["admission_type"]
    reference = json.loads((dest / "reference.json").read_text())
    assert {"score", "features", "missing", "performance"} <= set(reference)
    model = serialization.load(dest / "model.skops")
    test = pd.read_parquet(processed / "test.parquet")
    prob = model.predict_proba(test[meta["input_fields"]])[:, 1]
    assert np.all((prob > 0) & (prob < 1))

    # A failing gate registers the model but does not move the champion alias.
    config["promotion"]["min_cv_pr_auc"] = 0.99
    second = train(config, root=tmp_path)
    assert not second["promoted"] and second["model_version"] == "2"
    meta_again = json.loads(
        (export_champion(config, root=tmp_path) / "metadata.json").read_text()
    )
    assert str(meta_again["model_version"]) == "1"


# --- archive reuse -------------------------------------------------------


def test_ensure_archive_reuses_only_a_verified_file(tmp_path, monkeypatch):
    import hashlib

    from readmission_mlops.data import prepare

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"good")
    good = hashlib.sha256(b"good").hexdigest()
    calls = []
    monkeypatch.setattr(
        prepare, "fetch", lambda url, dest, sha: calls.append(url) or dest
    )

    assert prepare.ensure_archive("http://x", archive, good) == archive
    assert calls == []  # verified: no download

    archive.write_bytes(b"tampered")
    prepare.ensure_archive("http://x", archive, good)
    assert calls == ["http://x"]  # checksum failed: downloaded again


def test_audit_reports_every_group() -> None:
    """The fairness audit runs on a fitted model and covers each group."""
    from readmission_mlops.models.audit import audit

    class Model:
        """Scores encounters by their first feature."""

        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            p = x["score"].to_numpy()
            return np.column_stack([1 - p, p])

    n = 200
    df = pd.DataFrame(
        {
            "score": np.linspace(0.01, 0.99, n),
            "readmitted_30d": [int(i % 4 == 0) for i in range(n)],
            "patient_nbr": range(n),
            "gender": ["Male", "Female"] * (n // 2),
            "age": ["[80-90)", "[60-70)"] * (n // 2),
            "race": ["Caucasian"] * n,
        }
    )
    table = audit(Model(), df, ["score"], n_boot=5, seed=1)
    assert {"80+", "60-79", "Male", "Female", "Caucasian"} <= set(table["group"])
    assert (table["metric"] == "recall").any()
