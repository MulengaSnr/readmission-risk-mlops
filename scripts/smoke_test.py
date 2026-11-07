"""Smoke-test a running container: health, model info, prediction, validation.

    python scripts/smoke_test.py http://127.0.0.1:8000 [--expect-label v2]

Uses only the standard library, so it runs anywhere CI has Python.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ENCOUNTER = {
    "age": "[80-90)", "gender": "Male", "time_in_hospital": 8,
    "number_inpatient": 4, "number_emergency": 1, "number_outpatient": 0,
    "num_lab_procedures": 60, "num_procedures": 0, "num_medications": 22,
    "number_diagnoses": 9, "admission_type": "Emergency",
    "admission_source": "Emergency Room",
    "discharge_disposition": "Discharged/transferred to SNF",
    "medical_specialty": "InternalMedicine", "diag_1": "428",
    "diag_2": "250.02", "diag_3": "584", "diabetesMed": "Yes", "insulin": "Up",
}  # fmt: skip


def call(url: str, payload: dict | None = None) -> tuple[int, dict]:
    """GET (or POST JSON) and return status code and parsed body."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def wait_healthy(base: str, timeout: float = 60) -> None:
    """Poll /health until it returns 200 or the timeout passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if call(f"{base}/health")[0] == 200:
                return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(1)
    raise SystemExit(f"FAIL: {base}/health not healthy after {timeout}s")


def main() -> None:
    """Run the checks; exit non-zero on the first failure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--expect-label", help="required model version_label")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    wait_healthy(base)
    checks = []
    status, info = call(f"{base}/model-info")
    checks.append(("model-info returns 200", status == 200))
    if args.expect_label:
        checks.append((f"model label is {args.expect_label}",
                       info.get("version_label") == args.expect_label))  # fmt: skip
    status, pred = call(f"{base}/predict", ENCOUNTER)
    checks.append(("predict returns 200", status == 200))
    checks.append(("probability in (0, 1)", 0 < pred.get("probability", -1) < 1))
    checks.append(("three reasons", len(pred.get("reasons", [])) == 3))
    status, _ = call(f"{base}/predict", {**ENCOUNTER, "patient_nbr": 1})
    checks.append(("identifier field rejected (422)", status == 422))
    status, _ = call(f"{base}/predict", {**ENCOUNTER, "gender": "Femail"})
    checks.append(("invalid category rejected (422)", status == 422))

    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    label, version = info.get("version_label"), info.get("model_version")
    print(f"model {label} v{version}: probability {pred.get('probability')}, "
          f"band {pred.get('risk_band')}")  # fmt: skip
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
