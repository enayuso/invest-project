# ================================
# invest-project
# ================================
# app.daemon now serves the live dashboard (FastAPI, port 8094 by default —
# see app.config's dashboard_port) with the minute-level poller running as a
# background task in its lifespan. dashboard_host is 127.0.0.1 on purpose
# (unauthenticated order placement), so `docker run -d` uses --network host
# (see pipeline.sh) instead of `-p` — a bridge-network `-p` publish can't
# reach a loopback-bound process inside the container at all. Override the
# command at `docker run` time to use the one-shot entrypoints instead
# (app.main, app.backtest, app.memory, app.execution) for manual/batch use.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Dependencies first for build-cache reuse, app code after. README.md is
# copied alongside pyproject.toml/uv.lock because hatchling (the build
# backend) requires the file its `readme` field points at to exist.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY app ./app
RUN uv sync --frozen

# Code version stamped onto every decision (memory_entries.git_commit) and
# backtest run (backtest_runs.git_commit). Baked in at BUILD time because it
# cannot be discovered at run time: .dockerignore excludes .git and this slim
# base image has no git binary, so the `git rev-parse` fallback in
# app.memory.store/_current_git_commit always failed here — every deployed
# decision was tagged NULL (0 of 580 rows, confirmed 2026-08-13). Empty
# default keeps a plain `docker build` with no --build-arg working exactly as
# before; pipeline.sh supplies the real value.
# 컨테이너 안에는 .git도 git도 없어서 실행 시점엔 알아낼 수 없음 — 빌드 때 굽는다.
ARG GIT_COMMIT=""
ENV GIT_COMMIT=$GIT_COMMIT

# Default: persistent daemon (app.daemon), meant for `docker run -d`. For
# one-shot manual runs, override the command instead, e.g.:
#   docker run --rm --env-file .env <image> uv run python -m app.main --ticker AAPL --date 2026-07-08
#   docker run --rm --env-file .env <image> uv run python -m app.backtest --ticker AAPL --start 2026-05-01 --end 2026-07-01
#   docker run --rm --env-file .env <image> uv run python -m app.memory resolve --ticker AAPL
#   docker run --rm --env-file .env <image> uv run python -m app.execution check-stops --ticker AAPL
CMD ["uv", "run", "python", "-m", "app.daemon"]
