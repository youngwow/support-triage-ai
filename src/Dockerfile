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
    PYTHONPATH=/app \
    HF_HOME=/home/app/.cache/huggingface

# gcc: triton (torch's GPU kernels) JIT-compiles a C stub at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# The cache dir must exist and belong to uid 10001 BEFORE the named volume is
# mounted there, so the volume inherits ownership the non-root user can write.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app

WORKDIR /app

COPY --link --from=builder /opt/venv /opt/venv
COPY --link src/ /app/src/
# The knowledge base and the ticket fixtures ship with the image, so a bare
# `docker run` works. Compose bind-mounts ./data over this to make the corpus
# editable without a rebuild.
COPY --link data/ /app/data/

USER app

EXPOSE 8000

# First boot downloads the ~14 GB embedding model into the cache volume, so the
# probe must not start failing while that is still in progress.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
    CMD python -c 'import os,urllib.request as r; r.urlopen("http://127.0.0.1:8000" + os.environ.get("API_PREFIX", "/api/v1") + "/health/ready", timeout=3)'

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
