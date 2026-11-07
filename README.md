# Readmission Risk Model in Production

Taking a validated healthcare model from a laptop to a versioned, tested, monitored service: MLflow, FastAPI, Docker, GitHub Actions and Azure.

![CI](https://github.com/MulengaSnr/readmission-risk-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

**Status:** In development, step 4 of 7 (container image and CI/CD). Not deployed to the cloud yet.

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

## Prediction API
`make serve` starts a FastAPI service on the exported champion model. Interactive documentation is at `/docs`.

| Endpoint | Returns |
|---|---|
| `POST /predict` | Probability of 30-day readmission, risk band, the **top 3 reasons** (exact TreeSHAP) and warnings |
| `POST /predict/batch` | The same for up to 1,000 encounters (used for replay in step 5) |
| `GET /health` | Whether a model is loaded, and its version |
| `GET /model-info` | Model version, MLflow run, data fingerprint, code versions, band cut-offs, offline metrics |

Example response for an 80–90-year-old with 4 inpatient stays in the previous year:
```json
{"probability": 0.3634, "risk_band": "highest 10%",
 "reasons": [{"label": "Inpatient stays, previous year", "value": "4", "effect": "raises risk", "shap_log_odds": 0.665},
             {"label": "Additional diagnosis group", "value": "Genitourinary (584)", "effect": "raises risk", "shap_log_odds": 0.143},
             {"label": "Secondary diagnosis group", "value": "Diabetes (250.02)", "effect": "raises risk", "shap_log_odds": 0.128}],
 "warnings": [], "request_id": "c2189cb9-…", "model_version": "2"}
```

- **Input validation:** closed categories (sex, age band, test results, medication status) must be known values, so a typo returns 422 instead of a silent guess. Open categories (admission type and source, discharge destination, specialty) accept new values with a warning. Diagnosis codes must be valid ICD-9. Length of stay must be 1–14 days, the training population. **Unknown fields are rejected, so identifiers can never be sent or logged.**
- **Risk bands** use cut-offs from out-of-fold training predictions (highest 10%: risk ≥ 19.7%; top 20%: ≥ 15.2%). On the test set they flag 10.1% and 19.9% of encounters.
- **Prediction log:** one JSON line per prediction (time, request ID, model version, inputs, score) feeds monitoring in step 5.
- **Speed:** about 90 ms per explained prediction on a laptop.

## Container and CI/CD
**Image** ([Dockerfile](Dockerfile)): a two-stage build. uv installs the locked runtime dependencies only (no MLflow, no test tools), and the slim runtime image holds the environment, the code and the model.
- Base images are **pinned by digest**, so rebuilds are reproducible.
- Runs as a **non-root** user, with a Docker `HEALTHCHECK`. `/health` returns 503 when no model is loaded, so a broken container is marked unhealthy.
- **850 MB**, down from 1.08 GB, by leaving out pyarrow (170 MB), which only reads parquet. Predictions on 500 real test encounters are **identical** with and without it.

**Pipeline** ([ci.yml](.github/workflows/ci.yml)):

| Job | When | What |
|---|---|---|
| `lint-and-test` | Every PR and push | ruff + pytest |
| `container` | Every PR and push | Builds the image around a tiny **synthetic** model, runs it and smoke-tests it: health, model info, prediction, identifier and typo rejection, non-root, Docker health check |
| `release` | Merges to `main` | Downloads the data (cached, re-verified by SHA-256), **retrains the real model**, fails the build if the promotion checks fail, exports it, builds and smoke-tests the image, and pushes it to GitHub Container Registry as `sha-<commit>` and `latest` |

Training is deterministic, so CI rebuilds exactly the model validated locally from code and data, and every image can be traced to a commit, a data checksum and passed promotion checks. Images from this private repo are private in the registry.

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
make serve       # API on http://127.0.0.1:8000 (docs at /docs)
make image       # build the Docker image around artifacts/model
make smoke       # run the image and smoke-test it
make check       # lint + tests
```

## Progress
- [x] 1. Scaffold, project 1 dependency, design
- [x] 2. Training pipeline with MLflow tracking and registry
- [x] 3. FastAPI service with validation, explanations and tests
- [x] 4. Docker image and container smoke test in CI
- [ ] 5. Monitoring and replay simulation
- [ ] 6. Azure deployment (free tier)
- [ ] 7. README, model card and runbook

## Author
**Enock Mulenga**, Data Scientist · [LinkedIn](https://www.linkedin.com/in/enock-mulenga-8472ba137) · [Portfolio](https://mulengasnr.github.io)
