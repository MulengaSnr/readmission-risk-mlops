"""Download, clean and split the data with project 1's own functions.

Run from the project root:

    uv run python -m readmission_mlops.data.prepare

Reusing ``readmission_risk`` (pinned to v1.0) instead of copying its code
means the cohort, target and patient split are defined in one place. With the
same seed and test size, the split is identical to project 1's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from readmission_risk.data.clean import (
    GROUP_COL,
    TARGET,
    clean,
    parse_id_mappings,
    read_raw,
    validate,
)
from readmission_risk.data.download import extract_members, fetch
from readmission_risk.data.split import patient_split

from readmission_mlops.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short, order-independent hash of which encounters and outcomes a set holds.

    Two frames get the same fingerprint exactly when they contain the same
    encounter IDs with the same outcomes, whatever the row order or file
    format. Logged with every training run, so each model can be traced to
    its data.

    Args:
        df: Encounters with ``encounter_id`` and ``TARGET`` columns.

    Returns:
        The first 16 hex characters of a SHA-256 digest.
    """
    pairs = df[["encounter_id", TARGET]].sort_values("encounter_id")
    payload = pairs.to_csv(index=False, header=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def prepare(config: dict[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Run download -> clean -> validate -> split and write parquet files.

    Args:
        config: Parsed ``configs/default.yaml``.
        root: Project root that relative paths in ``config`` resolve against.

    Returns:
        A summary with row counts, patients, readmission rates and fingerprints.
    """
    raw = root / config["paths"]["raw"]
    processed = root / config["paths"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)
    ds = config["dataset"]

    archive = fetch(ds["url"], raw / ds["archive_name"], ds["sha256"])
    extract_members(archive, ds["members"], raw)
    cleaned, _ = clean(
        read_raw(raw / "diabetic_data.csv"), parse_id_mappings(raw / "IDS_mapping.csv")
    )
    validate(cleaned)
    train, test = patient_split(cleaned, config["split"]["test_size"], config["seed"])

    summary: dict[str, Any] = {}
    for name, part in (("train", train), ("test", test)):
        part.to_parquet(processed / f"{name}.parquet", index=False)
        summary[name] = {
            "encounters": len(part),
            "patients": int(part[GROUP_COL].nunique()),
            "readmission_rate": round(float(part[TARGET].mean()), 5),
            "fingerprint": data_fingerprint(part),
        }
    (processed / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Command-line entry point."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    summary = prepare(load_config())
    logger.info("Prepared data: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
