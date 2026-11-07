# Readmission Risk Model in Production

Taking a validated healthcare model from a laptop to a versioned, tested, monitored service: MLflow, FastAPI, Docker, GitHub Actions and Azure.

![CI](https://github.com/MulengaSnr/readmission-risk-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

**Status:** In development, step 1 of 7 (scaffold and design). Nothing is deployed yet, and no results are claimed.

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

## Data and governance
- Public, de-identified UCI Diabetes 130-US Hospitals data (CC BY 4.0). No real or personal data is used. Production traffic is simulated by replaying held-out records.
- Data preparation reuses project 1's code, installed as a library pinned to its [`v1.0` release](https://github.com/MulengaSnr/readmission-risk-model/releases/tag/v1.0).

## Run it
```bash
git clone https://github.com/MulengaSnr/readmission-risk-mlops.git
cd readmission-risk-mlops
uv sync          # Python 3.12 + dependencies, including project 1 at v1.0
make check       # lint + tests
```

## Progress
- [x] 1. Scaffold, project 1 dependency, design
- [ ] 2. Training pipeline with MLflow tracking and registry
- [ ] 3. FastAPI service with validation, explanations and tests
- [ ] 4. Docker image and container smoke test in CI
- [ ] 5. Monitoring and replay simulation
- [ ] 6. Azure deployment (free tier)
- [ ] 7. README, model card and runbook

## Author
**Enock Mulenga**, Data Scientist · [LinkedIn](https://www.linkedin.com/in/enock-mulenga-8472ba137) · [Portfolio](https://mulengasnr.github.io)
