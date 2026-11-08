.PHONY: setup lint format test check data train export audit mlflow-ui serve image smoke replay monitor

setup:   ## Create the virtual environment and install dependencies
	uv sync

lint:    ## Static checks (PEP 8, imports, docstrings, common bugs)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix formatting and safe lint issues
	uv run ruff format .
	uv run ruff check --fix .

test:    ## Run the test suite with coverage
	uv run pytest --cov=src --cov-report=term-missing

check: lint test  ## Everything CI runs

data:    ## Download (SHA-256 verified), clean and split with project 1's code
	uv run python -m readmission_mlops.data.prepare

train:   ## Train v2, log to MLflow, register; promote to champion if gates pass
	uv run python -m readmission_mlops.models.train

export:  ## Export the champion model to artifacts/model for the service image
	uv run python -m readmission_mlops.models.export

audit:   ## Fairness audit of the exported champion on the test set
	uv run python -m readmission_mlops.models.audit

mlflow-ui: ## Browse runs and the registry at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1

serve:   ## Run the API locally on http://127.0.0.1:8000 (docs at /docs)
	uv run uvicorn readmission_mlops.service.app:app --host 127.0.0.1 --port 8000

image:   ## Build the API image around artifacts/model (run make export first)
	docker build -t readmission-api:local --build-arg GIT_COMMIT=$$(git rev-parse --short HEAD) .

smoke:   ## Run the image and smoke-test it, then stop it
	docker rm -f readmission-api >/dev/null 2>&1 || true
	docker run -d --name readmission-api -p 8000:8000 readmission-api:local
	uv run python scripts/smoke_test.py http://127.0.0.1:8000; status=$$?; \
	docker rm -f readmission-api >/dev/null; exit $$status

replay:  ## Simulate six months of traffic through the API (in-process)
	uv run python -m readmission_mlops.monitoring.replay

monitor: ## Monthly monitoring report from the prediction log and outcomes
	uv run python -m readmission_mlops.monitoring.report
