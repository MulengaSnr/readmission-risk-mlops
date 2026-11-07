"""Request and response schemas for the prediction API.

Validation rules come from the training data (README "input contract"):

- closed categories (sex, age band, test results, medication status) must be
  one of the known values, so a typo is rejected rather than silently scored;
- open categories (admission type/source, discharge destination, specialty)
  accept unseen values, which the model pools as rare, and the response
  warns about them;
- diagnosis codes must be valid ICD-9-CM codes (project 1's validator);
- length of stay must be 1-14 days, the model's training population;
- unknown fields are rejected, so identifiers can never reach the log.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from readmission_risk.features.icd9 import icd9_chapter

AgeBand = Literal[
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]  # fmt: skip
DrugStatus = Literal["No", "Steady", "Up", "Down"]


class Encounter(BaseModel):
    """One discharge to score. Field names match the source dataset."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
                "A1Cresult": "Not measured",
                "max_glu_serum": "Not measured",
                "change": "Ch",
                "diabetesMed": "Yes",
                "insulin": "Up",
                "metformin": "No",
            }
        },
    )

    # Demographics (race is deliberately not an input).
    age: AgeBand
    gender: Literal["Female", "Male"]

    # This encounter.
    time_in_hospital: int = Field(
        ge=1, le=14, description="Days; model trained on 1-14"
    )
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=20)
    num_medications: int = Field(ge=0, le=150)
    number_diagnoses: int = Field(ge=1, le=30)

    # Utilisation in the previous year.
    number_inpatient: int = Field(ge=0, le=100)
    number_emergency: int = Field(ge=0, le=100)
    number_outpatient: int = Field(ge=0, le=100)

    # Open categories: unseen values are allowed (with a warning).
    admission_type: str | None = None
    admission_source: str | None = None
    discharge_disposition: str | None = None
    medical_specialty: str | None = None

    # ICD-9-CM diagnosis codes.
    diag_1: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None

    # Results and medication.
    A1Cresult: Literal["Not measured", "Norm", ">7", ">8"] = "Not measured"  # noqa: N815
    max_glu_serum: Literal["Not measured", "Norm", ">200", ">300"] = "Not measured"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["Yes", "No"]  # noqa: N815
    insulin: DrugStatus = "No"
    metformin: DrugStatus = "No"
    glipizide: DrugStatus = "No"
    glyburide: DrugStatus = "No"
    pioglitazone: DrugStatus = "No"
    rosiglitazone: DrugStatus = "No"
    glimepiride: DrugStatus = "No"
    repaglinide: DrugStatus = "No"
    nateglinide: DrugStatus = "No"
    chlorpropamide: DrugStatus = "No"
    acetohexamide: DrugStatus = "No"
    tolbutamide: DrugStatus = "No"
    acarbose: DrugStatus = "No"
    miglitol: DrugStatus = "No"
    troglitazone: DrugStatus = "No"
    tolazamide: DrugStatus = "No"
    glyburide_metformin: DrugStatus = Field("No", alias="glyburide-metformin")
    glipizide_metformin: DrugStatus = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: DrugStatus = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: DrugStatus = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: DrugStatus = Field("No", alias="metformin-pioglitazone")

    @field_validator("diag_1", "diag_2", "diag_3")
    @classmethod
    def _valid_icd9(cls, code: str | None) -> str | None:
        if code is None or not code.strip():
            return None
        icd9_chapter(code)  # raises ValueError for anything that is not ICD-9
        return code.strip().upper()

    @field_validator(
        "admission_type",
        "admission_source",
        "discharge_disposition",
        "medical_specialty",
    )
    @classmethod
    def _blank_is_missing(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


class Reason(BaseModel):
    """One factor behind a prediction (exact TreeSHAP, in log-odds)."""

    feature: str
    label: str
    value: str
    effect: Literal["raises risk", "lowers risk"]
    shap_log_odds: float


class Prediction(BaseModel):
    """Risk for one encounter."""

    probability: float = Field(
        description="Predicted probability of 30-day readmission"
    )
    risk_band: Literal["highest 10%", "next 10%", "below top 20%"]
    reasons: list[Reason]
    warnings: list[str]


class PredictionResponse(Prediction):
    """Single prediction with tracing information."""

    request_id: str
    model_version: str


class BatchRequest(BaseModel):
    """Several encounters scored in one call."""

    model_config = ConfigDict(extra="forbid")
    encounters: list[Encounter] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    """Predictions in the same order as the request."""

    request_id: str
    model_version: str
    predictions: list[Prediction]


class Health(BaseModel):
    """Liveness and readiness."""

    status: Literal["ok", "model not loaded"]
    model_version: str | None


class ModelInfo(BaseModel):
    """What is serving: version, provenance and offline metrics."""

    model_name: str
    model_version: str
    version_label: str | None
    run_id: str
    train_fingerprint: str | None
    git_commit: str | None
    readmission_risk_version: str | None
    excluded_features: list[str]
    band_thresholds: dict[str, float]
    training_readmission_rate: float
    metrics: dict[str, float]
