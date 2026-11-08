# Model card: 30-day readmission risk, v2 (served model)

_Structure based on Mitchell et al. (2019), "Model Cards for Model Reporting". This card covers v2, the model behind the API. The development of the model, including tuning and the choice of XGBoost, is in [project 1's model card](https://github.com/MulengaSnr/readmission-risk-model/blob/v1.0/docs/MODEL_CARD.md)._

## Model details
- **Type:** XGBoost binary classifier in a scikit-learn pipeline: feature engineering, preprocessing (imputation, `log1p` on counts, scaling, one-hot encoding), then the model.
- **Settings:** project 1's tuned values, unchanged (`configs/default.yaml`). 659 trees, learning rate 0.019, maximum depth 5.
- **Difference from v1.0:** v2 is refitted **without `payer_code`**. Insurance type may stand in for income or social disadvantage, and project 1 found that removing it costs little.
- **Version:** label `v2`. The live service reports its git commit, MLflow run and data fingerprint at `GET /model-info`. The registry number differs between environments (CI starts a fresh registry on every release, so the live model is number 1), so identify a model by its label and commit.
- **Reproducibility:** CI retrains the model from code and checksum-verified data on each merge. The live model's metrics, risk cut-offs and training-data fingerprint (`e916fc8d62057cf1`) match the model validated locally.
- **File format:** skops, loaded with an allow-list of types held in the code (design decision D11), not pickle.
- **Owner:** Enock Mulenga
- **Date:** 2026-11-08

## Intended use
- **Primary use (illustrative):** rank adult inpatients with diabetes at discharge by risk of readmission within 30 days, so a care team with limited capacity can prioritise follow-up.
- **Users:** clinical and care-coordination staff, who make the final decision.
- **The score supports prioritisation only.** It must not be used to exclude anyone from care. At 20% capacity, about 62% of readmissions are not flagged.
- **Out of scope:**
  - Any use on real patients. This is a portfolio demo. Real use would need local validation, governance approval and a data protection impact assessment.
  - Decisions about insurance, payment, eligibility or discharge timing.
  - Patients outside the training population: children, admissions without a diabetes diagnosis, stays longer than 14 days.
  - Estimating who would benefit from an intervention. The model predicts risk, not treatment effect.

## Training and evaluation data
- **Source:** UCI Diabetes 130-US Hospitals for Years 1999–2008 (CC BY 4.0), de-identified. Details are in [data/README.md](../data/README.md).
- **Cohort and split:** identical to project 1, confirmed by matching fingerprints. Training set: 79,473 encounters. Test set: 19,867 encounters, 13,951 patients and 2,263 readmissions. No patient appears in both.
- **Known biases:** US hospitals only; ICD-9 coding; readmissions to hospitals outside the network are probably missed; race recording quality is unknown; no dates, so no temporal validation.

## Metrics
PR-AUC is the primary metric, because readmissions are the minority class and the task is to rank. The operating point is the top 20% of discharges by predicted risk.

| Metric (test set) | v2 | v1.0 (with payer) | No-skill baseline |
|---|---:|---:|---:|
| PR-AUC | **0.231** | 0.235 | 0.114 |
| ROC-AUC | 0.670 | 0.672 | 0.500 |
| Brier score | 0.0959 | 0.0958 | 0.101 |
| Readmissions reached, top 10% | 24.0% | 24.5% | 10% |
| Readmissions reached, top 20% | **37.9%** | 38.5% | 20% |
| Precision, top 20% | 21.6% | 21.9% | 11.4% |

- **Cross-validation:** PR-AUC 0.230 ± 0.014 (5-fold, patient-grouped), consistent with the test score.
- **Promotion checks:** the model becomes champion only if cross-validated PR-AUC is at least 0.20 and test PR-AUC is no more than 0.01 below v1.0's. v2 is 0.003 below.
- **Test-set use:** v1.0 was evaluated on this test set first. For v2 it is used only to check against that fixed limit and to report, not to choose between options.
- **Intervals:** overall bootstrap intervals were calculated for v1.0, not for v2. The v1.0 PR-AUC interval is 0.216–0.257.
- **Risk bands:** cut-offs come from out-of-fold training predictions (highest 10%: risk ≥ 19.7%; next 10%: ≥ 15.2%). On the test set they flag 10.1% and 19.9% cumulatively.

## Subgroup performance (fairness)
`make audit` applies project 1's audit method to v2: one top-20% cut-off for everyone and 1,000 patient-level bootstrap resamples. Recall is the share of each group's readmissions that are flagged. Race is not a model input. Results: [`reports/audit/fairness_test.csv`](../reports/audit/fairness_test.csv). The audit reports on a model already chosen, and no decision depended on it.

| Group | Readmissions | Recall | Gap vs reference (95% CI) | Calibration gap |
|---|---:|---:|---:|---:|
| Female (reference) | 1,207 | 36.6% | n/a | −0.2 pts |
| Male | 1,056 | 39.3% | +2.7 pts (−2.2 to +7.7) | +0.1 pts |
| Age 60–79 (reference) | 1,110 | 37.5% | n/a | +0.1 pts |
| Age <40 | 129 | 51.2% | +13.7 pts (−0.2 to +26.2) | 0.0 pts |
| Age 40–59 | 533 | 38.3% | +0.8 pts (−6.1 to +7.6) | 0.0 pts |
| Age 80+ | 491 | 34.8% | −2.7 pts (−8.0 to +3.2) | −0.6 pts |
| Caucasian (reference) | 1,731 | 38.5% | n/a | −0.2 pts |
| African American | 407 | 38.1% | −0.4 pts (−6.6 to +6.1) | +0.5 pts |
| Race unknown (small group) | 37 | 27.0% | −11.4 pts (−26.2 to +3.8) | +1.0 pts |

- **Calibration holds in every group shown,** within about 1 point: a score means about the same risk whatever the group.
- **Age 80+ remains the main concern.** It has the weakest ranking (ROC-AUC 0.624) and the lowest recall among adequately sized groups. The gap is smaller than v1.0's (−4.9 pts), but both intervals include zero, so this is not evidence of a real improvement.
- **Patients with missing race** are flagged 10.2% of the time, against about 20% overall.
- **Too small to judge:** Asian (11 readmissions), Other (34), Unknown (37) and Hispanic (43) each have fewer than 50 readmissions. Asian patients show the lowest recall (9.1%), and that needs investigating with more data rather than being read as either a finding or a clearance.

## Explainability
- **Each prediction returns its top three reasons,** from exact TreeSHAP with one-hot columns summed back to the original feature.
- **Global drivers were analysed for v1.0, not recalculated for v2:** prior inpatient stays and discharge destination were the largest. v1.0's ranking includes payer, which v2 does not use.
- **SHAP describes what the model uses, not causes.**

## Ethical considerations and UK GDPR
- **No real or personal data is used.** Live traffic to the demo comes only from whoever calls the public URL, and the API rejects unknown fields, so identifiers such as `patient_nbr` cannot be submitted. It cannot stop someone typing real clinical values into the permitted fields, so the demo must not be used with real patient details.
- **Health data is special-category data** under UK GDPR Article 9. A real deployment would need a lawful basis and an Article 9 condition, a data protection impact assessment and local information-governance approval. This is a summary, not legal advice.
- **Human in the loop:** the score prioritises outreach and makes no decision about an individual by itself, which is relevant to Article 22 on solely automated decisions.
- **Risk of reinforcing access patterns:** the model learns from past use of services, so patients who historically used fewer services, possibly because of access barriers, may be under-flagged.

## Limitations and risks
- Trained on US hospital data from 1999–2008 with ICD-9 codes. Performance in a UK setting is **unknown**.
- Modest discrimination (ROC-AUC 0.67), typical of models built on administrative data without social or post-discharge information.
- A random patient split, not a temporal one, because the data has no dates.
- Readmissions elsewhere are probably not recorded, so the labels likely undercount events.
- High risk does not mean most preventable.
- Monitoring has been demonstrated on simulated traffic only. The live service's prediction log is temporary and is lost when the app scales to zero.

## Monitoring
`make replay` and `make monitor` compare each month with a reference profile saved with the model. Thresholds were fixed before any data was seen ([DESIGN.md](DESIGN.md#7-monitoring-thresholds-initial)), and responses are in the [runbook](RUNBOOK.md#monitoring-alerts).

| Signal | Warn | Alert | Window |
|---|---|---|---|
| PSI on predicted risk or an input | ≥ 0.10 | ≥ 0.25 | Monthly |
| Missing values, increase vs training | +5 pts | +10 pts | Monthly |
| Calibration, worst decile | ≥ 3 pts | ≥ 5 pts | 3 months |
| Recall at 20% cut-off, drop vs training | −3 pts | −5 pts | 3 months |
| Recall gap, ages 80+ vs 60–79 | ≥ 7 pts | ≥ 10 pts | 3 months |

In the six-month simulation, normal months stayed OK, and both injected problems raised alerts that named their cause.
