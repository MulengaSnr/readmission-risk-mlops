# Readmission risk API. Two stages: build the virtual environment with uv,
# then copy only the environment, code and model into a slim runtime image.
# Base images are pinned by digest so rebuilds are reproducible.

FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /bin/uv
# git: the project 1 dependency is installed from its v1.0 git tag.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so code changes do not invalidate this layer. pyarrow
# (170 MB) is only needed to read parquet files, which the API never does.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --no-install-project --no-editable \
    --no-install-package pyarrow
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable --no-install-package pyarrow

FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54
ARG MODEL_DIR=artifacts/model
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/MulengaSnr/readmission-risk-mlops" \
      org.opencontainers.image.description="Readmission risk API (FastAPI + XGBoost, explained predictions)" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system app && useradd --system --gid app --no-create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --chown=root:root ${MODEL_DIR}/ /app/model/
ENV PATH="/app/.venv/bin:${PATH}" \
    MODEL_DIR=/app/model \
    PREDICTION_LOG=/tmp/logs/predictions.jsonl \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "readmission_mlops.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
