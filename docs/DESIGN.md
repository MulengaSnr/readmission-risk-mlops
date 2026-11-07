# Design: running the readmission risk model in production

_Status: agreed 2026-11-07, before implementation. Decisions are recorded with the reason and the alternatives considered, so later changes can be judged against them._

## 1. Problem
[`readmission-risk-model`](https://github.com/MulengaSnr/readmission-risk-model) produced a validated, calibrated and audited model, but it exists only as a file on a laptop. A care team needs it to:

- score a discharge on request and explain the score;
- be versioned, so every prediction can be traced to the model and data that produced it;
- be rebuilt, tested and deployed the same way every time;
- warn when inputs, scores or performance drift, including for groups where the model is known to be weaker.

This project shows the difference between **developing** a model and **operating** one.

## 2. Scope
**In scope:** a training pipeline with experiment tracking and a model registry; a prediction API with input validation and explanations; a container image; CI/CD; monitoring of drift, performance and fairness, with alert thresholds; deployment on the Azure free tier; a runbook.

**Out of scope:** real patients or real hospital systems; authentication beyond a demo-level API key; high availability; any spending beyond the free tier without approval.

## 3. Data
- Same public dataset as project 1 (UCI Diabetes 130-US Hospitals, CC BY 4.0). No real or personal data is used.
- The data is prepared with project 1's own functions (`fetch`, `clean`, `validate`, `patient_split`), installed as a library pinned to its `v1.0` release. With the same seed, the train/test split is identical to project 1's.
- **Production traffic is simulated** by replaying test-set encounters in batches. One batch is deliberately shifted, to show that the monitoring catches it.

## 4. Model
- **v2:** project 1's tuned XGBoost settings, refitted **without `payer_code`** (decision in project 1, PR #12 and issue #13). Training uses the full training set.
- Its test-set score is reported as a **second use of the test set**: v1.0 was already evaluated on it. It is used to check v2 against v1.0, not to choose between options.

## 5. Architecture
```
 train ──► MLflow tracking (params, CV metrics, artefacts) ──► registry: alias "champion"
                                                │ export
                                                ▼
 FastAPI service (Docker, non-root)
   POST /predict      risk probability, risk band, top-3 SHAP reasons, model version
   GET  /health       liveness and model loaded
   GET  /model-info   version, training data hash, CV and test metrics
        │ appends each prediction to a log (inputs + score, no identifiers)
        ▼
 monitoring job ──► input drift (PSI), score drift, calibration and recall once
                    outcomes arrive, recall for ages 80+ and missing race ──► report + alert flags

 GitHub Actions: lint + tests → build image → container smoke test → push to GHCR
                 → deploy to Azure Container Apps (OIDC, no stored secrets)
```

## 6. Decisions

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| D1 | Install project 1 as a library pinned to `v1.0` | One source of truth for cleaning and features; no copy-paste drift | Copy the code (two versions to maintain); a git submodule (awkward with `uv`) |
| D2 | MLflow with a local SQLite backend | Real tracking and registry API, free, runs anywhere and in CI | Hosted MLflow or Azure ML workspace (cost, account needed from day one) |
| D3 | Model built into the image at build time | The service starts with no network access to a registry; the image *is* the release | Load from the registry at start-up (needs an MLflow server in production) |
| D4 | FastAPI + Pydantic schemas | Validation and OpenAPI docs come for free; the de facto standard in UK job ads | Flask (manual validation); BentoML or MLflow serving (less to show, less control) |
| D5 | Explanations returned with each prediction | Staff need a reason, not just a number; project 1 showed SHAP is exact here | Global importance only |
| D6 | GitHub Container Registry for images | Free for this use; Azure Container Registry Basic costs money every month | Azure Container Registry; Docker Hub (rate limits) |
| D7 | Azure Container Apps, Consumption plan | Scales to zero; monthly free grant per subscription; Azure is the most-requested cloud in UK ads | App Service F1 (limited container support); AKS (cost and complexity) |
| D8 | OIDC federation from GitHub Actions to Azure | No long-lived cloud secrets stored in GitHub | A service-principal secret in repository secrets |
| D9 | Hand-written drift metrics (PSI, calibration by decile, recall by group) | Small, tested, explainable in an interview; no heavy dependency | Evidently or WhyLogs (useful, but hide the mechanics) |
| D10 | Monitoring as a batch job writing a report | Matches how outcomes arrive (30 days later); cheap | A real-time metrics stack (Prometheus and Grafana), out of proportion here |

## 7. Monitoring thresholds (initial)
Thresholds are fixed now and revisited only with evidence:

| Signal | Warn | Alert |
|---|---|---|
| PSI on predicted risk or a key input | ≥ 0.10 | ≥ 0.25 |
| Calibration: \|observed − predicted\| in any decile | ≥ 3 pts | ≥ 5 pts |
| Recall at 20% capacity, relative to training-time recall | −3 pts | −5 pts |
| Recall gap, ages 80+ vs 60–79 | ≥ 7 pts | ≥ 10 pts |
| Missing values per feature, relative to training | +5 pts | +10 pts |

PSI cut-offs follow the common credit-risk convention (< 0.1 stable, 0.1–0.25 moderate, > 0.25 significant shift). The others start from project 1's observed variation and are deliberately conservative.

## 8. Cost
The target is **£0**. GitHub Actions and Container Registry are free for this use. Azure Container Apps (Consumption plan) includes a monthly free grant per subscription: 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests. A demo app that scales to zero stays well inside it. A **£1 budget alert** will be set before anything is deployed, and no Azure resource is created without the owner's approval.

## 9. Delivery plan (one PR per step)
1. Scaffold, project 1 dependency, this design
2. Training pipeline with MLflow tracking and registry
3. FastAPI service with validation, explanations and tests
4. Docker image and a container smoke test in CI
5. Monitoring jobs and the replay simulation
6. Azure deployment (needs an Azure account and approval)
7. README, model card and runbook
