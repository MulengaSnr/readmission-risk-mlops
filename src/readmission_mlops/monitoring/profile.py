"""Reference profiles and drift statistics shared by training and monitoring.

At training time, :func:`build_reference` records what "normal" looks like:
the distribution of out-of-fold scores, of key inputs and of missing values,
and recall at the fixed top-20% cut-off, overall and by age group. The
monitoring job compares each month of logged predictions with it.

Population Stability Index (PSI) compares two distributions over the same
bins: ``sum((cur - ref) * ln(cur / ref))``. Common reading: < 0.1 stable,
0.1-0.25 moderate shift, > 0.25 significant shift.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from readmission_risk.models.fairness import AGE_BANDS

MISSING = "<missing>"
EPSILON = 1e-4  # floor for empty bins, so PSI stays finite

# Continuous inputs: decile bins from the reference data.
NUMERIC_FEATURES = ("time_in_hospital", "num_lab_procedures", "num_medications",
                    "number_diagnoses")  # fmt: skip
# Small counts with many zeros: one level per count, the tail pooled at the cap.
COUNT_FEATURES = {"number_inpatient": 5, "number_emergency": 3, "number_outpatient": 3}
CATEGORICAL_FEATURES = ("age", "gender", "admission_type", "admission_source",
                        "discharge_disposition", "A1Cresult")  # fmt: skip


def numeric_profile(values: pd.Series | np.ndarray, n_bins: int = 10) -> dict:
    """Quantile bin edges and the share of values in each bin."""
    v = pd.Series(values, dtype=float).dropna().to_numpy()
    inner = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = [-math.inf, *inner.tolist(), math.inf]
    return {"type": "numeric", "edges": edges, "props": _binned(v, edges).tolist()}


def _binned(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts = np.histogram(values, bins=np.asarray(edges, dtype=float))[0]
    return counts / max(counts.sum(), 1)


def categorical_profile(values: pd.Series, cap: int | None = None) -> dict:
    """Share of each level, with missing values as their own level."""
    s = _levels(values, cap)
    props = s.value_counts(normalize=True).sort_index()
    return {"type": "categorical", "cap": cap, "props": props.to_dict()}


def _levels(values: pd.Series, cap: int | None) -> pd.Series:
    s = pd.Series(values)
    if cap is not None:
        s = s.astype(float).clip(upper=cap)
        return s.map(lambda v: MISSING if pd.isna(v) else
                     (f"{int(v)}+" if v >= cap else str(int(v))))  # fmt: skip
    return s.astype(object).where(s.notna(), MISSING).astype(str)


def psi(
    reference: list[float] | np.ndarray, current: list[float] | np.ndarray
) -> float:
    """Population Stability Index between two aligned sets of proportions."""
    ref = np.clip(np.asarray(reference, dtype=float), EPSILON, None)
    cur = np.clip(np.asarray(current, dtype=float), EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift(profile: dict, values: pd.Series | np.ndarray) -> float:
    """PSI of ``values`` against a stored profile (numeric or categorical)."""
    if profile["type"] == "numeric":
        v = pd.Series(values, dtype=float).dropna().to_numpy()
        return psi(profile["props"], _binned(v, profile["edges"]))
    current = _levels(pd.Series(values), profile.get("cap")).value_counts(
        normalize=True
    )
    levels = sorted(set(profile["props"]) | set(current.index))
    return psi(
        [profile["props"].get(level, 0.0) for level in levels],
        [current.get(level, 0.0) for level in levels],
    )


def age_group(age: pd.Series) -> pd.Series:
    """Map age bands to the audit groups used in project 1 (<40 ... 80+)."""
    return age.astype(str).map(AGE_BANDS)


def recall_at(prob: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Share of readmissions with a score at or above the cut-off."""
    positives = np.asarray(y) == 1
    if not positives.any():
        return float("nan")
    return float((np.asarray(prob)[positives] >= threshold).mean())


def performance(
    prob: np.ndarray, y: np.ndarray, age: pd.Series, threshold: float
) -> dict[str, float]:
    """Recall at the fixed cut-off, overall and for ages 80+ and 60-79."""
    groups = age_group(age).to_numpy()
    prob, y = np.asarray(prob), np.asarray(y)
    r80 = recall_at(prob[groups == "80+"], y[groups == "80+"], threshold)
    r60 = recall_at(prob[groups == "60-79"], y[groups == "60-79"], threshold)
    return {
        "recall": recall_at(prob, y, threshold),
        "recall_80plus": r80,
        "recall_60_79": r60,
        "gap_80_vs_60_79": r60 - r80,
    }


def build_reference(
    inputs: pd.DataFrame, oof: np.ndarray, y: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Everything monitoring compares against, from training data only.

    Args:
        inputs: Training encounters with the model's input columns.
        oof: Out-of-fold predicted probabilities for those encounters.
        y: Training outcomes.
        threshold: The fixed top-20% score cut-off used in production.

    Returns:
        A JSON-serialisable reference profile.
    """
    features = {c: numeric_profile(inputs[c]) for c in NUMERIC_FEATURES}
    features |= {
        c: categorical_profile(inputs[c], cap) for c, cap in COUNT_FEATURES.items()
    }
    features |= {c: categorical_profile(inputs[c]) for c in CATEGORICAL_FEATURES}
    return {
        "score": numeric_profile(oof),
        "features": features,
        "missing": {c: float(inputs[c].isna().mean()) for c in inputs.columns},
        "performance": {
            "threshold": threshold,
            **performance(oof, y, inputs["age"], threshold),
        },
        "n": len(inputs),
    }
