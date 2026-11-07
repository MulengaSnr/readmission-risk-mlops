import json

import numpy as np
import pandas as pd
import pytest

from readmission_mlops.config import load_config
from readmission_mlops.monitoring.profile import (
    MISSING,
    build_reference,
    categorical_profile,
    drift,
    numeric_profile,
    performance,
    psi,
)
from readmission_mlops.monitoring.replay import build_months, replay
from readmission_mlops.monitoring.report import build_report, grade, to_markdown
from readmission_mlops.synthetic import synthetic_encounters

# --- PSI and profiles ----------------------------------------------------


def test_psi_is_zero_for_identical_and_large_for_shifted():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))  # empty bins floored


def test_numeric_profile_and_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    profile = numeric_profile(ref)
    assert len(profile["props"]) == len(profile["edges"]) - 1 == 10
    assert sum(profile["props"]) == pytest.approx(1.0)
    assert drift(profile, rng.normal(size=5000)) < 0.02
    assert drift(profile, rng.normal(loc=1.0, size=5000)) > 0.25


def test_categorical_profile_handles_missing_unseen_and_caps():
    profile = categorical_profile(pd.Series(["a", "a", "b", None]))
    assert profile["props"] == {MISSING: 0.25, "a": 0.5, "b": 0.25}
    assert drift(profile, pd.Series(["a", "a", "b", None])) == pytest.approx(0.0)
    assert drift(profile, pd.Series(["c", "c", "c", "c"])) > 1  # unseen level
    counts = categorical_profile(pd.Series([0, 1, 5, 9]), cap=5)
    assert set(counts["props"]) == {"0", "1", "5+"}


def test_performance_at_fixed_cutoff():
    prob = np.array([0.9, 0.1, 0.9, 0.1, 0.5])
    y = np.array([1, 1, 1, 1, 0])
    age = pd.Series(["[80-90)", "[80-90)", "[60-70)", "[70-80)", "[60-70)"])
    perf = performance(prob, y, age, threshold=0.5)
    assert perf["recall"] == 0.5
    assert perf["recall_80plus"] == 0.5 and perf["recall_60_79"] == 0.5
    assert perf["gap_80_vs_60_79"] == 0.0


@pytest.mark.parametrize(
    ("value", "status"), [(0.05, "OK"), (0.1, "WARN"), (0.3, "ALERT")]
)
def test_grade(value, status):
    assert grade(value, warn=0.1, alert=0.25) == status


# --- report --------------------------------------------------------------


def scored_months(shift_month=None, n=2000, months=4):
    rng = np.random.default_rng(0)
    frames = []
    for m in range(1, months + 1):
        df = synthetic_encounters(n, seed=m)
        if m == shift_month:
            df["discharge_disposition"] = None
        df["probability"] = rng.uniform(0.02, 0.4, n)
        df["readmitted_30d"] = (rng.random(n) < df["probability"]).astype(int)
        df["month"] = m
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_report_waits_for_full_window_and_flags_pipeline_bug():
    train = scored_months(months=1)
    ref = build_reference(
        train,
        train["probability"].to_numpy(),
        train["readmitted_30d"].to_numpy(),
        threshold=0.25,
    )
    scored = scored_months(shift_month=4)
    manifest = [{"month": m, "scenario": "s"} for m in range(1, 5)]
    report = build_report(scored, manifest, ref, load_config())
    by_check = lambda r: {c["check"]: c for c in r["checks"]}  # noqa: E731
    month1 = by_check(report[0])
    assert month1["Calibration, worst decile (pts)"]["status"] == "N/A"
    assert (
        "window filling: 1 of 3" in month1["Calibration, worst decile (pts)"]["detail"]
    )
    assert report[2]["checks"][3]["status"] != "N/A"  # window full from month 3
    month4 = by_check(report[3])
    missing = month4["Missing values, largest increase (pts)"]
    assert missing["status"] == "ALERT" and missing["detail"] == "discharge_disposition"
    assert report[3]["status"] == "ALERT"
    markdown = to_markdown(report, 3)
    assert "n/a" in markdown and "Month 4, **ALERT**" in markdown


# --- replay --------------------------------------------------------------


def fake_test_set(n=3000):
    df = synthetic_encounters(n)
    df["readmitted_30d"] = np.random.default_rng(0).integers(0, 2, n)
    return df


def test_build_months_scenarios():
    test = fake_test_set()
    months = build_months(test, seed=1)
    assert [m["month"] for m in months] == [1, 2, 3, 4, 5, 6]
    assert sum(len(m["data"]) for m in months[:4]) == len(test)  # a partition
    old = lambda d: d["age"].isin(["[80-90)", "[90-100)"]).mean()  # noqa: E731
    assert old(months[4]["data"]) > old(test) + 0.1
    assert months[5]["data"]["discharge_disposition"].isna().all()


def test_replay_writes_manifest_and_outcomes():
    test = fake_test_set(2500)
    months = build_months(test, seed=1)[:1]
    columns = [c for c in test.columns if c != "readmitted_30d"]
    calls = []

    def send(records):
        calls.append(len(records))
        return {"request_id": f"r{len(calls)}"}

    manifest, outcomes = replay(months, columns, send)
    assert calls == [625]  # one chunk (<= 1000)
    assert manifest[0]["request_ids"] == ["r1"]
    assert len(outcomes) == 625 and set(outcomes[0]) == {
        "request_id",
        "item",
        "readmitted_30d",
    }
    json.dumps(manifest)  # serialisable


def test_load_scored_joins_log_outcomes_and_months(tmp_path):
    from readmission_mlops.monitoring.report import load_scored

    log = tmp_path / "log.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": rid,
                    "item": i,
                    "probability": p,
                    "inputs": {"age": "[60-70)"},
                }
            )  # fmt: skip
            for rid, i, p in [
                ("a", 0, 0.1),
                ("a", 1, 0.2),
                ("b", 0, 0.3),
                ("x", 0, 0.9),
            ]
        )
    )
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        "\n".join(
            json.dumps({"request_id": rid, "item": i, "readmitted_30d": y})
            for rid, i, y in [("a", 0, 0), ("a", 1, 1), ("b", 0, 1)]
        )
    )
    manifest = [{"month": 1, "request_ids": ["a"]}, {"month": 2, "request_ids": ["b"]}]
    scored = load_scored(log, outcomes, manifest)
    assert len(scored) == 3  # request "x" is not part of the replay
    assert scored.set_index(["request_id", "item"])["readmitted_30d"].to_dict() == {
        ("a", 0): 0, ("a", 1): 1, ("b", 0): 1,
    }  # fmt: skip
    assert scored.groupby("month").size().to_dict() == {1: 2, 2: 1}
