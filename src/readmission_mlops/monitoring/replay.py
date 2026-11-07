"""Simulate six months of production traffic through the prediction API.

Run from the project root (after ``make export``):

    uv run python -m readmission_mlops.monitoring.replay

Months 1-4 split the held-out test set into four random parts (normal
traffic). Month 5 is a **population shift**: the test set resampled towards
older patients with more prior inpatient stays. Month 6 is a **pipeline bug**:
normal patients, but ``discharge_disposition`` arrives empty for everyone.

Each month is sent to ``/predict/batch``, in-process by default (the same
FastAPI app, validation and logging code) or to a live server with ``--url``.
Outcomes are written to a separate file keyed by request ID and position, as
a hospital system would supply them 30 days later; the API never sees them.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from readmission_risk.data.clean import TARGET

from readmission_mlops.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

CHUNK = 1000  # the API's batch limit
SHIFT_SIZE = 5000

Sender = Callable[[list[dict[str, Any]]], dict[str, Any]]


def build_months(test: pd.DataFrame, seed: int = 42) -> list[dict[str, Any]]:
    """The six simulated months, each with a label, description and encounters."""
    rng = np.random.default_rng(seed)
    shuffled = test.sample(frac=1, random_state=seed).reset_index(drop=True)
    months = [
        {
            "month": i + 1,
            "scenario": "normal",
            "data": shuffled.iloc[rows].reset_index(drop=True),
        }
        for i, rows in enumerate(np.array_split(np.arange(len(shuffled)), 4))
    ]
    old = test["age"].astype(str).isin(["[80-90)", "[90-100)"])
    weights = np.where(old, 4.0, 1.0) * np.where(
        test["number_inpatient"] >= 2, 3.0, 1.0
    )
    shifted = test.sample(SHIFT_SIZE, replace=True, weights=weights, random_state=seed)
    months.append(
        {
            "month": 5,
            "scenario": "population shift (older, more prior stays)",
            "data": shifted.reset_index(drop=True),
        }
    )
    bug = test.sample(SHIFT_SIZE, replace=True, random_state=int(rng.integers(1e9)))
    bug = bug.assign(discharge_disposition=None)
    months.append(
        {
            "month": 6,
            "scenario": "pipeline bug (discharge destination missing)",
            "data": bug.reset_index(drop=True),
        }
    )
    return months


def to_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Encounters -> JSON-ready dicts with ``None`` for missing values."""
    return json.loads(df[columns].to_json(orient="records"))


def http_sender(url: str) -> Sender:
    """Send batches to a live server's ``/predict/batch``."""

    def send(records: list[dict[str, Any]]) -> dict[str, Any]:
        req = urllib.request.Request(
            url.rstrip("/") + "/predict/batch",
            data=json.dumps({"encounters": records}).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    return send


def replay(
    months: list[dict[str, Any]], columns: list[str], send: Sender
) -> tuple[list, list]:
    """Score every month in API-sized chunks.

    Args:
        months: Output of :func:`build_months`.
        columns: The model's input fields.
        send: Function posting one batch and returning the parsed response.

    Returns:
        ``(manifest, outcomes)``: which request IDs belong to which month, and
        one outcome row per scored encounter.
    """
    manifest, outcomes = [], []
    for month in months:
        data = month["data"]
        ids = []
        for start in range(0, len(data), CHUNK):
            chunk = data.iloc[start : start + CHUNK]
            response = send(to_records(chunk, columns))
            ids.append(response["request_id"])
            outcomes += [
                {"request_id": response["request_id"], "item": i, TARGET: int(y)}
                for i, y in enumerate(chunk[TARGET])
            ]
        manifest.append(
            {
                "month": month["month"],
                "scenario": month["scenario"],
                "encounters": len(data),
                "request_ids": ids,
            }
        )
        logger.info(
            "Month %d (%s): %d encounters", month["month"], month["scenario"], len(data)
        )
    return manifest, outcomes


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="live server base URL; default: in-process")
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/model"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=PROJECT_ROOT / "logs/replay_predictions.jsonl",
        help="prediction log written by the in-process app",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports/monitoring")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = load_config()
    test = pd.read_parquet(PROJECT_ROOT / config["paths"]["processed"] / "test.parquet")
    columns = json.loads((args.model_dir / "metadata.json").read_text())["input_fields"]
    months = build_months(test, config["seed"])

    if args.url:
        manifest, outcomes = replay(months, columns, http_sender(args.url))
    else:
        from fastapi.testclient import TestClient

        from readmission_mlops.service.app import create_app

        args.log.unlink(missing_ok=True)  # one clean log per replay
        app = create_app(model_dir=args.model_dir, log_path=args.log)
        with TestClient(app) as client:

            def send(records: list[dict[str, Any]]) -> dict[str, Any]:
                r = client.post("/predict/batch", json={"encounters": records})
                r.raise_for_status()
                return r.json()

            manifest, outcomes = replay(months, columns, send)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.out / "outcomes.jsonl").open("w") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in outcomes)
    logger.info("Wrote manifest and %d outcomes to %s", len(outcomes), args.out)


if __name__ == "__main__":
    main()
