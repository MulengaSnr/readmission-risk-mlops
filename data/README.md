# Data

Raw and processed data are **not committed** (see `.gitignore`). Recreate them with `make data`, which uses project 1's download, cleaning and split functions (pinned to `readmission-risk` v1.0).

## Provenance and licences

| Dataset | Source (URL) | Licence | Retrieved | Contains personal data? | Notes |
|---|---|---|---|---|---|
| Diabetes 130-US Hospitals for Years 1999–2008 | <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008> | CC BY 4.0 | 2026-11-07 | No: public, de-identified, no names, dates or addresses | Archive SHA-256 pinned in `configs/default.yaml`. Full provenance, citation and data-protection notes in [project 1's data README](https://github.com/MulengaSnr/readmission-risk-model/blob/v1.0/data/README.md). |

## Folders
- `raw/` – untouched source files, never edited by hand
- `external/` – reference data (lookups, boundaries)
- `processed/` – outputs of `src/` pipelines; always reproducible from `raw/`

## Check against project 1
`make data` writes `processed/data_summary.json` with an order-independent fingerprint of each split. Both fingerprints (train `e916fc8d62057cf1`, test `fae7d6b65a0988f4`) match project 1's files exactly, so models from both projects are scored on identical data.

## Data protection
Only public, synthetic or properly licensed data is used. No real patient, client or employer data. Where a dataset relates to people, note the lawful basis or anonymisation status and any UK GDPR considerations.
