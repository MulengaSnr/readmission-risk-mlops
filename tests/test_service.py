import json

import pytest
from fastapi.testclient import TestClient

from readmission_mlops.service.app import create_app


@pytest.fixture
def client(model_dir, tmp_path):
    app = create_app(model_dir=model_dir, log_path=tmp_path / "predictions.jsonl")
    with TestClient(app) as c:
        c.log_path = tmp_path / "predictions.jsonl"
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "0"}
    info = client.get("/model-info").json()
    assert info["model_version"] == "0"
    assert info["excluded_features"] == ["payer_code"]
    assert set(info["band_thresholds"]) == {"top_10pct", "top_20pct"}


def test_predict_returns_probability_band_and_three_reasons(client, example_encounter):
    r = client.post("/predict", json=example_encounter)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < body["probability"] < 1
    assert body["risk_band"] in {"highest 10%", "next 10%", "below top 20%"}
    assert len(body["reasons"]) == 3
    assert all(x["effect"] in {"raises risk", "lowers risk"} for x in body["reasons"])
    assert body["warnings"] == []
    assert body["model_version"] == "0"


def test_prediction_is_logged_without_identifiers(client, example_encounter):
    r = client.post("/predict", json=example_encounter)
    line = json.loads(client.log_path.read_text().splitlines()[-1])
    assert line["request_id"] == r.json()["request_id"]
    assert line["probability"] == r.json()["probability"]
    assert line["inputs"]["number_inpatient"] == 2
    assert "patient_nbr" not in line["inputs"] and "encounter_id" not in line["inputs"]


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"gender": "Femail"}, "gender"),
        ({"time_in_hospital": 30}, "time_in_hospital"),
        ({"number_inpatient": -1}, "number_inpatient"),
        ({"diag_1": "not-a-code"}, "diag_1"),
        ({"patient_nbr": 123}, "patient_nbr"),  # identifiers are rejected
    ],
)
def test_invalid_input_is_rejected(client, example_encounter, change, fragment):
    r = client.post("/predict", json={**example_encounter, **change})
    assert r.status_code == 422
    assert fragment in r.text


def test_missing_required_field_is_rejected(client, example_encounter):
    payload = {k: v for k, v in example_encounter.items() if k != "age"}
    assert client.post("/predict", json=payload).status_code == 422


def test_unseen_open_category_is_scored_with_warning(client, example_encounter):
    payload = {**example_encounter, "admission_source": "Brand new route"}
    body = client.post("/predict", json=payload).json()
    assert len(body["warnings"]) == 1 and "admission_source" in body["warnings"][0]


def test_hyphenated_drug_fields_and_missing_values(client, example_encounter):
    payload = {**example_encounter, "glyburide-metformin": "Steady", "diag_3": None,
               "medical_specialty": ""}  # fmt: skip
    assert client.post("/predict", json=payload).status_code == 200


def test_batch_matches_single_predictions(client, example_encounter):
    other = {**example_encounter, "number_inpatient": 8}
    batch = client.post(
        "/predict/batch", json={"encounters": [example_encounter, other]}
    )
    assert batch.status_code == 200
    probs = [p["probability"] for p in batch.json()["predictions"]]
    single = client.post("/predict", json=example_encounter).json()["probability"]
    assert probs[0] == single
    assert probs[1] > probs[0]  # synthetic model: more prior stays, more risk


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"encounters": []}).status_code == 422


def test_service_without_model_reports_unavailable(tmp_path, example_encounter):
    app = create_app(model_dir=tmp_path / "missing", log_path=tmp_path / "log.jsonl")
    with TestClient(app) as c:
        health = c.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "model not loaded"
        assert c.post("/predict", json=example_encounter).status_code == 503
