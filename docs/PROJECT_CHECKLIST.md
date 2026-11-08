# Project checklist

Status for this repo. "Project 1" means the work was done in [`readmission-risk-model`](https://github.com/MulengaSnr/readmission-risk-model), which this project builds on.

## Plan
- [x] Business problem and decision-maker written down
- [x] Technical objectives and success metric (with baseline) defined
- [x] Dataset chosen; licence checked and recorded in `data/README.md`
- [x] Architecture sketched in README
- [ ] Issues created for each major task (each step was a pull request instead)

## Build
- [x] Reproducible data download / generation script
- [x] Data validation checks
- [x] EDA notebook with key findings (project 1)
- [x] Train / validation / test split decided **before** modelling; leakage reviewed (project 1; split reused)
- [x] Baseline model (project 1)
- [x] Candidate models + cross-validation (project 1; v2 cross-validated here)
- [x] Final evaluation on test set (second use, recorded in the README and model card)
- [x] Explainability + fairness analysis (`make audit`; per-prediction SHAP)
- [x] Unit tests for `src/` logic; CI green

## Document
- [x] README complete (two-minute test passes)
- [x] Model card
- [x] `reports/findings.md` for a non-technical reader
- [x] Screenshot / headline figure
- [x] Limitations and next steps

## Publish (only after approval)
- [ ] No secrets (`git log -p | grep -i key` reviewed), no real personal data
- [ ] Repo description + topics set
- [ ] Linked from profile README and landing page
