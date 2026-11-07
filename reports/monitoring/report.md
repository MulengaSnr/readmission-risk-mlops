# Monitoring report

Drift and missing values are checked per month; calibration, recall and fairness on the latest 3 months (the window), and only once the window is full (`n/a` before that). Thresholds: `configs/default.yaml` (fixed in `docs/DESIGN.md`).

| Month | Scenario | Status | Score drift (PSI) | Input drift, worst feature (PSI) | Missing values, largest increase (pts) | Calibration, worst decile (pts) | Recall at 20% cut-off, drop vs training (pts) | Recall gap, ages 80+ vs 60-79 (pts) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | normal | **OK** | 0.00 | 0.00 | 0.12 | n/a | n/a | n/a |
| 2 | normal | **OK** | 0.00 | 0.00 | 0.06 | n/a | n/a | n/a |
| 3 | normal | **OK** | 0.00 | 0.00 | 0.79 | 1.88 | 2.08 | 2.78 |
| 4 | normal | **OK** | 0.00 | 0.00 | 0.32 | 1.26 | 0.49 | 1.00 |
| 5 | population shift (older, more prior stays) | **ALERT** | 0.15 WARN | 0.39 ALERT | 0.00 | 1.65 | -5.71 | -0.95 |
| 6 | pipeline bug (discharge destination missing) | **ALERT** | 0.49 ALERT | 10.50 ALERT | 95.28 ALERT | 1.77 | -6.26 | 2.02 |

## Warnings and alerts

- Month 5, **WARN**: Score drift (PSI) = 0.15
- Month 5, **ALERT**: Input drift, worst feature (PSI) = 0.39 (age)
- Month 6, **ALERT**: Score drift (PSI) = 0.49
- Month 6, **ALERT**: Input drift, worst feature (PSI) = 10.50 (discharge_disposition)
- Month 6, **ALERT**: Missing values, largest increase (pts) = 95.28 (discharge_disposition)
