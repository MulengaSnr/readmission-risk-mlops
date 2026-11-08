"""Fairness audit of the exported champion on the test set.

Run from the project root (after ``make export``):

    uv run python -m readmission_mlops.models.audit

Applies project 1's audit (``readmission_risk.models.fairness``) to model v2,
so its results can be compared directly with the v1.0 audit: flag the top 20%
of test encounters with one cut-off for everyone, then measure each group,
with 95% intervals from a patient-level bootstrap.

This is a further use of the test set. It reports on a model that is already
chosen and no modelling decision depends on it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.models.fairness import audit_groups, fairness_table

from readmission_mlops.config import PROJECT_ROOT, load_config
from readmission_mlops.service.model_store import LoadedModel

logger = logging.getLogger(__name__)


def audit(
    pipeline: Any,
    test_df: pd.DataFrame,
    input_fields: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Per-group audit metrics for a fitted model.

    Args:
        pipeline: Fitted model with ``predict_proba``.
        test_df: Test encounters, with outcome, patient ID and audit columns.
        input_fields: Columns the model expects.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        The table from project 1's ``fairness_table``.
    """
    prob = pipeline.predict_proba(test_df[input_fields])[:, 1]
    return fairness_table(
        test_df[TARGET].to_numpy(),
        prob,
        audit_groups(test_df),
        test_df[GROUP_COL],
        n_boot,
        seed,
    )


def main(root: Path = PROJECT_ROOT) -> None:
    """Audit the exported model and write ``reports/audit/fairness_test.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config()
    model = LoadedModel.from_dir(root / config["paths"]["model_export"])
    test_df = pd.read_parquet(root / config["paths"]["processed"] / "test.parquet")
    table = audit(
        model.pipeline,
        test_df,
        model.input_fields,
        config["evaluation"]["n_bootstrap"],
        config["seed"],
    )
    out = root / config["paths"]["reports"] / "audit" / "fairness_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.5f")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
