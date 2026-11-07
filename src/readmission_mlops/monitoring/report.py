"""Monthly monitoring report: drift, data quality, calibration, recall, fairness.

Run from the project root (after ``make replay``):

    uv run python -m readmission_mlops.monitoring.report

Inputs: the prediction log, the outcomes file and the replay manifest. Each
check is compared with the reference profile exported with the model and
graded OK / WARN / ALERT using the thresholds in ``configs/default.yaml``
(fixed in ``docs/DESIGN.md`` before any data was seen):

- **Every month:** score drift and input drift (PSI), missing values.
- **Rolling window of the latest months** (default 3): calibration by
  decile, recall at the fixed top-20% cut-off and the recall gap between
  ages 80+ and 60-79. One month of outcomes is too few to estimate these
  reliably.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from readmission_risk.data.clean import TARGET
from readmission_risk.visualisation import plots

from readmission_mlops.config import PROJECT_ROOT, load_config
from readmission_mlops.monitoring.profile import drift, performance

logger = logging.getLogger(__name__)

# "N/A": not assessed yet (the rolling window is still filling).
STATUSES = ("N/A", "OK", "WARN", "ALERT")


def grade(value: float, warn: float, alert: float) -> str:
    """Status for a metric where larger values are worse."""
    if value != value:  # NaN: not enough data
        return "OK"
    return "ALERT" if value >= alert else "WARN" if value >= warn else "OK"


def load_scored(log: Path, outcomes: Path, manifest: list[dict]) -> pd.DataFrame:
    """Join logged predictions with outcomes and month labels.

    Returns:
        One row per scored encounter: month, probability, outcome and inputs.
    """
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    preds = pd.DataFrame(
        [
            {
                "request_id": r["request_id"],
                "item": r["item"],
                "probability": r["probability"],
                **r["inputs"],
            }
            for r in rows
        ]
    )
    labels = pd.read_json(outcomes, lines=True)
    month_of = {rid: m["month"] for m in manifest for rid in m["request_ids"]}
    scored = preds.merge(labels, on=["request_id", "item"], how="left", validate="1:1")
    scored["month"] = scored["request_id"].map(month_of)
    return scored.dropna(subset=["month"]).astype({"month": int})


def max_calibration_gap(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Largest |observed - mean predicted| across deciles of predicted risk."""
    deciles = pd.qcut(prob, bins, labels=False, duplicates="drop")
    table = pd.DataFrame({"p": prob, "y": y, "d": deciles}).groupby("d").mean()
    return float((table["y"] - table["p"]).abs().max())


def monthly_checks(
    month: pd.DataFrame, reference: dict, thresholds: dict
) -> list[dict[str, Any]]:
    """Drift and data-quality checks for one month."""
    t = thresholds
    score_psi = drift(reference["score"], month["probability"])
    feature_psi = {c: drift(p, month[c]) for c, p in reference["features"].items()}
    worst_feature = max(feature_psi, key=feature_psi.get)
    missing_up = {
        c: 100 * (month[c].isna().mean() - ref)
        for c, ref in reference["missing"].items()
    }
    worst_missing = max(missing_up, key=missing_up.get)
    return [
        {
            "check": "Score drift (PSI)",
            "value": score_psi,
            "status": grade(score_psi, **t["psi"]),
            "detail": "",
        },
        {
            "check": "Input drift, worst feature (PSI)",
            "value": feature_psi[worst_feature],
            "status": grade(feature_psi[worst_feature], **t["psi"]),
            "detail": worst_feature,
        },
        {
            "check": "Missing values, largest increase (pts)",
            "value": missing_up[worst_missing],
            "status": grade(missing_up[worst_missing], **t["missing_increase_pts"]),
            "detail": worst_missing,
        },
    ]


def window_checks(
    window: pd.DataFrame, reference: dict, thresholds: dict
) -> list[dict[str, Any]]:
    """Calibration, recall and fairness checks on a window with outcomes."""
    t = thresholds
    ref = reference["performance"]
    prob, y = window["probability"].to_numpy(), window[TARGET].to_numpy()
    perf = performance(prob, y, window["age"], ref["threshold"])
    calibration = 100 * max_calibration_gap(prob, y)
    recall_drop = 100 * (ref["recall"] - perf["recall"])
    gap = 100 * perf["gap_80_vs_60_79"]
    return [
        {
            "check": "Calibration, worst decile (pts)",
            "value": calibration,
            "status": grade(calibration, **t["calibration_pts"]),
            "detail": "",
        },
        {
            "check": "Recall at 20% cut-off, drop vs training (pts)",
            "value": recall_drop,
            "status": grade(recall_drop, **t["recall_drop_pts"]),
            "detail": f"recall {perf['recall']:.1%} vs {ref['recall']:.1%}",
        },
        {
            "check": "Recall gap, ages 80+ vs 60-79 (pts)",
            "value": gap,
            "status": grade(gap, **t["gap_80_pts"]),
            "detail": f"{perf['recall_80plus']:.1%} vs {perf['recall_60_79']:.1%}",
        },
    ]


def build_report(
    scored: pd.DataFrame, manifest: list[dict], reference: dict, config: dict
) -> list[dict[str, Any]]:
    """All checks for every month, with the worst status per month."""
    thresholds = config["monitoring"]
    window_size = thresholds["performance_window_months"]
    report = []
    for entry in manifest:
        m = entry["month"]
        month = scored[scored["month"] == m]
        window = scored[scored["month"].between(m - window_size + 1, m)]
        checks = monthly_checks(month, reference, thresholds)
        filled = window["month"].nunique()
        if filled < window_size:
            # Too few outcomes: grading now raised false warnings on normal
            # traffic in the first replay, so these checks wait for a full window.
            checks += [
                {**c, "value": float("nan"), "status": "N/A",
                 "detail": f"window filling: {filled} of {window_size} months"}
                for c in window_checks(window, reference, thresholds)
            ]  # fmt: skip
        else:
            checks += window_checks(window, reference, thresholds)
        worst = max((c["status"] for c in checks), key=STATUSES.index)
        report.append(
            {
                "month": m,
                "scenario": entry["scenario"],
                "encounters": len(month),
                "window_encounters": len(window),
                "status": worst,
                "checks": checks,
            }
        )
    return report


def _cell(check: dict[str, Any]) -> str:
    if check["status"] == "N/A":
        return "n/a"
    value = f"{check['value']:.2f}"
    return value if check["status"] == "OK" else f"{value} {check['status']}"


def to_markdown(report: list[dict[str, Any]], window_size: int) -> str:
    """Human-readable report: one summary table and the alerts explained."""
    checks = [c["check"] for c in report[0]["checks"]]
    lines = [
        "# Monitoring report",
        "",
        f"Drift and missing values are checked per month; calibration, recall and "
        f"fairness on the latest {window_size} months (the window), and only once "
        "the window is full (`n/a` before that). Thresholds: "
        "`configs/default.yaml` (fixed in `docs/DESIGN.md`).",
        "",
        "| Month | Scenario | Status | " + " | ".join(checks) + " |",
        "|---|---|---|" + "---:|" * len(checks),
    ]
    for r in report:
        cells = [_cell(c) for c in r["checks"]]
        lines.append(
            f"| {r['month']} | {r['scenario']} | **{r['status']}** | "
            + " | ".join(cells)
            + " |"
        )
    lines += ["", "## Warnings and alerts", ""]
    flagged = [
        (r, c) for r in report for c in r["checks"] if c["status"] in ("WARN", "ALERT")
    ]
    if not flagged:
        lines.append("None.")
    for r, c in flagged:
        detail = f" ({c['detail']})" if c["detail"] else ""
        lines.append(
            f"- Month {r['month']}, **{c['status']}**: {c['check']} = "
            f"{c['value']:.2f}{detail}"
        )
    return "\n".join(lines) + "\n"


def plot_report(report: list[dict[str, Any]], thresholds: dict) -> plt.Figure:
    """Score PSI and worst input PSI per month against the alert lines."""
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    months = [r["month"] for r in report]
    series = {
        # Log scale: month 6's PSI is ~40x the alert line. Floor stable months
        # (PSI ~0) at 0.001 so they can be drawn on a log axis.
        "Score drift": [max(r["checks"][0]["value"], 1e-3) for r in report],
        "Worst input drift": [max(r["checks"][1]["value"], 1e-3) for r in report],
    }
    for (label, values), colour in zip(series.items(), plots.SERIES_ORDER, strict=True):
        ax.plot(
            months,
            values,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=plots.SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
    for name, level in (
        ("warn", thresholds["psi"]["warn"]),
        ("alert", thresholds["psi"]["alert"]),
    ):
        ax.axhline(
            level,
            color=plots.MUTED,
            linewidth=1,
            linestyle="--" if name == "warn" else "-",
        )
        ax.text(
            months[-1] + 0.15,
            level,
            name,
            va="center",
            fontsize=8.5,
            color=plots.INK_SECONDARY,
        )
    ax.set_xticks(months, [f"M{m}" for m in months])
    ax.set_xlabel("Simulated month (M5: population shift, M6: pipeline bug)")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 30)
    ax.set_ylabel("Population Stability Index (log scale)")
    ax.legend(frameon=False, loc="upper left", labelcolor=plots.INK_SECONDARY)
    ax.set_title("Monitoring catches both simulated problems")
    return fig


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "artifacts/model"
    )
    parser.add_argument(
        "--log", type=Path, default=PROJECT_ROOT / "logs/replay_predictions.jsonl"
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports/monitoring")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = load_config()
    manifest = json.loads((args.out / "manifest.json").read_text())
    reference = json.loads((args.model_dir / "reference.json").read_text())
    scored = load_scored(args.log, args.out / "outcomes.jsonl", manifest)
    report = build_report(scored, manifest, reference, config)

    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = to_markdown(report, config["monitoring"]["performance_window_months"])
    (args.out / "report.md").write_text(markdown)
    fig = plot_report(report, config["monitoring"])
    plots.save(fig, args.out / "drift.png")
    print(markdown)


if __name__ == "__main__":
    main()
