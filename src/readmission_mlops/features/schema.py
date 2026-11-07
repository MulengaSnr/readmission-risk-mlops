"""The input contract: which raw encounter fields the model needs.

The registered model is a single pipeline (stateless feature engineering ->
fitted preprocessing -> XGBoost), so it takes a cleaned encounter record, not
engineered features. This module lists exactly the raw fields required, which
the API schema (step 3) and the monitoring jobs (step 5) also use.
"""

from __future__ import annotations

from readmission_risk.features.build import DIAG_COLUMNS, DRUG_COLUMNS, feature_columns

# Columns created by readmission_risk.features.build.add_features.
ENGINEERED = frozenset(
    {
        "age_ordinal",
        "prior_visits_total",
        "n_diabetes_drugs",
        "n_drug_changes",
        *(f"{c}_group" for c in DIAG_COLUMNS),
    }
)
# Raw fields add_features reads to build those columns.
ENGINEERING_INPUTS = ("age", *DIAG_COLUMNS, *DRUG_COLUMNS)


def serving_columns(exclude: tuple[str, ...] = ()) -> list[str]:
    """Raw encounter fields the deployed model needs, in a stable order.

    Args:
        exclude: Features left out of the model (e.g. ``("payer_code",)``).

    Returns:
        Model inputs that are not engineered, followed by the raw fields that
        feature engineering reads, without duplicates.
    """
    direct = [c for c in feature_columns(exclude=exclude) if c not in ENGINEERED]
    ordered = direct + [c for c in ENGINEERING_INPUTS if c not in direct]
    return [c for c in ordered if c not in exclude]
