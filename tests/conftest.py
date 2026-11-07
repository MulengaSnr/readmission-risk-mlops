"""Shared fixtures: the synthetic test model (see readmission_mlops.synthetic)."""

import pytest

from readmission_mlops.synthetic import build_test_model


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return build_test_model(tmp_path_factory.mktemp("model"))


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
