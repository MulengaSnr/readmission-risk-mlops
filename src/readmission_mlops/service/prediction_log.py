"""Append-only log of predictions, the input for monitoring (step 5).

Each line is one JSON object: time, request ID, model version, the raw
inputs and the prediction. The API schema forbids extra fields, so patient or
encounter identifiers cannot be logged. Outcomes arrive 30 days later and are
joined by request ID in the monitoring job.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PredictionLog:
    """Thread-safe JSON-lines writer. ``path=None`` disables logging."""

    def __init__(self, path: Path | None) -> None:
        """Create the parent folder if logging is enabled."""
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        request_id: str,
        model_version: str,
        records: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> None:
        """Append one line per scored encounter."""
        if self.path is None:
            return
        now = datetime.now(UTC).isoformat(timespec="seconds")
        lines = [
            json.dumps(
                {
                    "timestamp": now,
                    "request_id": request_id,
                    "item": i,
                    "model_version": model_version,
                    "inputs": record,
                    "probability": pred["probability"],
                    "risk_band": pred["risk_band"],
                },
                separators=(",", ":"),
            )
            for i, (record, pred) in enumerate(zip(records, predictions, strict=True))
        ]
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
