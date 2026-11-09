# Runbook

How to check, deploy, roll back and shut down the readmission risk API. Each procedure says whether it has actually been exercised on this deployment.

## The service

| | |
|---|---|
| **URL** | <https://ca-readmission-api.purplemushroom-33c1d15b.francecentral.azurecontainerapps.io> (docs at `/docs`) |
| **Azure subscription** | **Azure for Students.** Do not use `UoR-Standard`, the university's subscription. |
| **Region** | France Central. The subscription's policy does not allow UK South. |
| **Resource group** | `rg-readmission-mlops` |
| **Container App** | `ca-readmission-api`: 0.5 vCPU, 1 GiB, 0 to 1 copies, single active revision |
| **Environment** | `cae-readmission` (Consumption plan, log storage off) |
| **Deploy identity** | `id-readmission-deploy`: Contributor on the resource group only |
| **Image** | `ghcr.io/mulengasnr/readmission-risk-mlops:sha-<commit>` (public) |

Before any Azure change:

```bash
az account show --query name -o tsv      # must print: Azure for Students
az account set -s "Azure for Students"   # if it does not
```

Commands below assume:

```bash
APP="-n ca-readmission-api -g rg-readmission-mlops"
URL=https://ca-readmission-api.purplemushroom-33c1d15b.francecentral.azurecontainerapps.io
```

## Check health
_Exercised._

```bash
curl -s $URL/health       # {"status":"ok","model_version":"1"}
curl -s $URL/model-info   # git commit, MLflow run, data fingerprint, metrics
python3 scripts/smoke_test.py $URL --expect-label v2 --wait 240
```

The smoke test should end with every line `PASS`. Measured on 8 November 2026 from the VM: one cold start from zero copies took **6 seconds**, a warm `/health` took 85 ms, and a warm explained `/predict` had a median of 91 ms over 10 calls.

`/health` returns 503 if no model is loaded.

## Deploy
_Exercised: PR #6 deployed `sha-31b4892` through this path._

Merge a PR to `main`. If it changes only Markdown files, `docs/`, `reports/`, `notebooks/` or the licence, CI does not run on `main`, so nothing retrains or redeploys. Otherwise CI runs `lint-and-test`, `container`, `release` (retrains, applies the promotion checks and pushes the image) and `deploy` (logs in with OIDC, updates the app and smoke-tests the live URL). If an earlier job fails, nothing is deployed.

## Roll back
_Not drilled. It uses the same `az containerapp update` command that CI's deploy has exercised._

```bash
az containerapp update $APP --image ghcr.io/mulengasnr/readmission-risk-mlops:sha-<good-commit> -o none
az containerapp show $APP --query properties.template.containers[0].image -o tsv
python3 scripts/smoke_test.py $URL --expect-label v2 --wait 240
```

Find earlier tags in the repository's Actions run summaries. A rollback lasts only until the next merge to `main`, so revert the faulty commit as well.

## Troubleshooting
**There are no logs to read.** Log storage is off to stay at £0, and `az containerapp logs show` currently fails on this app with `KeyError: 'eventStreamEndpoint'` because Azure returns no log stream endpoint for it. The likely reason is that log storage is off, but this has not been proven. To investigate a failure, use the state below and reproduce it locally.

```bash
az containerapp show $APP --query '{status:properties.runningStatus,state:properties.provisioningState}' -o json
az containerapp revision list $APP -o table
az containerapp replica list $APP --query 'length(@)' -o tsv   # 0 is normal when idle

docker run --rm -p 8000:8000 ghcr.io/mulengasnr/readmission-risk-mlops:sha-<commit>
python3 scripts/smoke_test.py http://127.0.0.1:8000 --expect-label v2
```

| Symptom | Likely cause | Response |
|---|---|---|
| First request is slow | Normal: the app scaled to zero | Retry; the smoke test waits with `--wait` |
| CI `Log in to Azure` fails with `AADSTS700213` | The federated credential's subject does not match GitHub's | Compare it with the `subject claim` in the job log. This repo uses GitHub's ID-based format, which also changes if the repo is renamed or transferred. |
| Azure cannot pull the image | The GHCR package is private | Set the package to public in its GitHub settings; test with a Docker client that is not logged in |
| `release` fails at the promotion step | The model missed a promotion check | Do not override it. Read the reasons in the job output. |
| Subscription disabled | Student credit used up or expired | Renew it, or redeploy elsewhere. The image is portable. |

## Monitoring alerts
_Exercised in simulation only (`make replay`, `make monitor`). The live service's prediction log is in `/tmp` and is lost when the app scales to zero, so live traffic is not monitored._

| Alert | First check | Response |
|---|---|---|
| Missing values up | Is a data feed broken? | Fix the feed before trusting scores. Do not retrain on broken data. |
| Input or score drift (PSI) | Which feature does the alert name? Is the change real? | If it is a pipeline fault, fix it. If it is real, check calibration and recall before acting. |
| Calibration | Does it persist across the 3-month window? | Investigate; recalibrate or retrain on recent local data |
| Recall drop | Is capacity or case mix changing? | Investigate; retrain if sustained |
| Recall gap, ages 80+ | Known risk from the audit | Clinical review; consider a separate rule for this group; re-run `make audit` |

Drift alone is not a reason to retrain: in the simulation's month 5 the inputs shifted while calibration held.

## Cost and credentials
- **Target:** £0, expected to fall within Container Apps' monthly free grant. This has not yet been confirmed against a billing period.
- **Safeguards:** the subscription has a spending limit, the app is capped at one copy, and it scales to zero. The budget `budget-readmission-1` emails at 50% and 100% of 1 per month, in the subscription's billing currency. Delivery of that email has not been tested.
- **Credentials:** none are stored. GitHub logs in through OIDC, and the client, tenant and subscription IDs are ordinary repository variables, not secrets.
- **Availability:** this depends on the student account and its credit, so the demo will need moving when those end.

## Shut down
_Not exercised. This permanently deletes the app._

```bash
az group delete -n rg-readmission-mlops
SUB=$(az account show --query id -o tsv)
az rest --method delete --url "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.Consumption/budgets/budget-readmission-1?api-version=2023-05-01"
```

After this, remove the `deploy` job from CI, or merges to `main` will fail at that step.
