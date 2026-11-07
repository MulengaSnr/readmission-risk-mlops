"""Load the exported model once and turn encounters into explained predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from readmission_risk.models.explain import FEATURE_LABELS, shap_by_feature
from sklearn.pipeline import Pipeline

from readmission_mlops.models import serialization

N_REASONS = 3
BANDS = (("top_10pct", "highest 10%"), ("top_20pct", "next 10%"))
BELOW = "below top 20%"


@dataclass(frozen=True)
class LoadedModel:
    """The champion pipeline plus the metadata exported with it."""

    pipeline: Pipeline
    metadata: dict[str, Any]

    @classmethod
    def from_dir(cls, folder: Path) -> LoadedModel:
        """Load ``model.skops`` (with our type allow-list) and ``metadata.json``."""
        metadata = json.loads((folder / "metadata.json").read_text())
        return cls(serialization.load(folder / "model.skops"), metadata)

    @property
    def version(self) -> str:
        """Registry version number as a string."""
        return str(self.metadata["model_version"])

    @property
    def input_fields(self) -> list[str]:
        """Raw fields the pipeline expects, in order."""
        return list(self.metadata["input_fields"])

    def frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Records -> a frame with exactly the model's input columns.

        Missing values are ``None`` in JSON; they become NaN so the pipeline
        imputes them exactly as in training.
        """
        df = pd.DataFrame.from_records(records, columns=self.input_fields)
        return df.astype(object).where(df.notna(), float("nan"))

    def band(self, probability: float) -> str:
        """Risk band from the out-of-fold thresholds exported with the model."""
        thresholds = self.metadata["band_thresholds"]
        for key, label in BANDS:
            if probability >= thresholds[key]:
                return label
        return BELOW

    def warnings(self, record: dict[str, Any]) -> list[str]:
        """Notes for values the model has not seen in training."""
        notes = []
        for field, known in self.metadata["known_categories"].items():
            value = record.get(field)
            if value is not None and value not in known:
                notes.append(
                    f"{field}={value!r} was not seen in training; "
                    "the model treats it as a rare category"
                )
        return notes

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Probability, band, top reasons and warnings for each record.

        Args:
            records: Encounters as dicts keyed by the dataset's field names.

        Returns:
            One dict per record, in order.
        """
        x = self.frame(records)
        probabilities = self.pipeline.predict_proba(x)[:, 1]
        engineered = self.pipeline.named_steps["features"].transform(x)
        values, _ = shap_by_feature(Pipeline(self.pipeline.steps[1:]), engineered)
        results = []
        for i, record in enumerate(records):
            p = float(probabilities[i])
            results.append(
                {
                    "probability": round(p, 4),
                    "risk_band": self.band(p),
                    "reasons": self._reasons(
                        values.iloc[i], engineered.iloc[i], record
                    ),
                    "warnings": self.warnings(record),
                }
            )
        return results

    @staticmethod
    def _reasons(
        shap_row: pd.Series, engineered_row: pd.Series, record: dict[str, Any]
    ) -> list[dict[str, Any]]:
        top = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)
        reasons = []
        for feature, contribution in top.head(N_REASONS).items():
            reasons.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "value": _display_value(feature, engineered_row, record),
                    "effect": "raises risk" if contribution > 0 else "lowers risk",
                    "shap_log_odds": round(float(contribution), 4),
                }
            )
        return reasons


def _display_value(feature: str, engineered_row: pd.Series, record: dict) -> str:
    """A readable value for an explanation, using the raw input where clearer."""
    if feature == "age_ordinal":
        return str(record.get("age"))
    if feature.endswith("_group"):
        raw = record.get(feature.removesuffix("_group"))
        return f"{engineered_row[feature]} ({raw})" if raw else "Missing"
    value = engineered_row.get(feature, record.get(feature))
    return "Missing" if value is None or value != value else str(value)
