# Readmission Risk Model in Production

Taking a validated healthcare model from a laptop to a versioned, tested, monitored service: MLflow, FastAPI, Docker, GitHub Actions and Azure.

![CI](https://github.com/MulengaSnr/readmission-risk-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

**Status:** In development, step 2 of 7 (training pipeline). Nothing is deployed yet.

## The problem
[`readmission-risk-model`](https://github.com/MulengaSnr/readmission-risk-model) built a calibrated, explained and fairness-audited model that predicts 30-day readmission for patients with diabetes. A model on a laptop helps no one. A care team needs a service that scores a discharge on request, explains the score, can be traced to the exact model that produced it, and warns when something drifts. This project shows the difference between **developing** a model and **operating** one.

## Planned architecture
```
 train ──► MLflow (tracking + registry) ──► FastAPI service in Docker ──► Azure Container Apps
                                               │ prediction log
                                               ▼
                                    monitoring: drift, calibration, recall by group
```
Full design, decisions and alert thresholds: [docs/DESIGN.md](docs/DESIGN.md).

## Model v2: training, tracking and registry
`make train` builds one pipeline (project 1's feature engineering → preprocessing **without payer** → XGBoost with project 1's tuned settings). It scores the pipeline with 5-fold patient-grouped cross-validation on the same folds as project 1, refits it on the full training set and scores the test set. It then logs everything to MLflow (parameters, metrics, per-fold results, data fingerprints, code versions) and registers the model. The `champion` alias moves only if the model passes the promotion checks in `configs/default.yaml`.

| | Cross-validation PR-AUC | Test PR-AUC | Test ROC-AUC | Readmissions reached, top 20% |
|---|---:|---:|---:|---:|
| v1.0 (project 1, with payer) | 0.234 ± 0.015 | 0.235 | 0.672 | 38.5% |
| **v2 (this project, no payer)** | **0.230 ± 0.014** | **0.231** | **0.670** | **37.9%** |

- **The v2 test score is a second use of the test set.** v1.0 was evaluated on it first. The score is used only to check v2 against v1.0, against a limit fixed in advance (no more than 0.01 PR-AUC below v1.0), not to choose between options. v2 is 0.003 lower and passed.
- **The cross-validation score reproduces project 1:** 0.2335 − 0.0037 = 0.2298, as found by project 1's payer analysis.
- **Safe model files:** models are saved with skops, not pickle. When the service loads a model, it accepts only the object types listed in [`serialization.py`](src/readmission_mlops/models/serialization.py), held in our code rather than in the model file (design decision D11).
- `make export` writes the champion to `artifacts/model/` (`model.skops` + `metadata.json`) for the service image. `make mlflow-ui` lets you browse runs and the registry.

## Data and governance
- Public, de-identified UCI Diabetes 130-US Hospitals data (CC BY 4.0). No real or personal data is used. Production traffic is simulated by replaying held-out records.
- Data preparation reuses project 1's code, installed as a library pinned to its [`v1.0` release](https://github.com/MulengaSnr/readmission-risk-model/releases/tag/v1.0).

## Run it
```bash
git clone https://github.com/MulengaSnr/readmission-risk-mlops.git
cd readmission-risk-mlops
uv sync          # Python 3.12 + dependencies, including project 1 at v1.0
make data        # download (SHA-256 verified), clean, split: identical to project 1
make train       # cross-validate, fit, log to MLflow, register, promote if checks pass
make export      # champion model -> artifacts/model/
make check       # lint + tests
```

## Progress
- [x] 1. Scaffold, project 1 dependency, design
- [x] 2. Training pipeline with MLflow tracking and registry
- [ ] 3. FastAPI service with validation, explanations and tests
- [ ] 4. Docker image and container smoke test in CI
- [ ] 5. Monitoring and replay simulation
- [ ] 6. Azure deployment (free tier)
- [ ] 7. README, model card and runbook

## Author
**Enock Mulenga**, Data Scientist · [LinkedIn](https://www.linkedin.com/in/enock-mulenga-8472ba137) · [Portfolio](https://mulengasnr.github.io)
