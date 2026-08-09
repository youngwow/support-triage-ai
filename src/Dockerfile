ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.11.18

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app

RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY --link --from=builder /opt/venv /opt/venv
COPY --link src/ /app/src/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c 'import os,urllib.request as r; r.urlopen("http://127.0.0.1:8000" + os.environ.get("API_PREFIX", "/api/v1") + "/health/ready", timeout=3)'

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
