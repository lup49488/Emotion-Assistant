FROM python:3.12-slim-bookworm AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Build into a virtualenv so the compilers stay out of the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt


FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf_cache \
    PATH="/opt/venv/bin:$PATH"

# Must match the owner of the bind-mounted host directories, otherwise the
# container cannot write to data/, users/, exports/ or logs/. Override with
# `docker compose build --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g)`.
ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid ${APP_GID} app \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /usr/sbin/nologin app

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY . ./
RUN mkdir -p /app/data /app/users /app/exports /app/logs /app/knowledge_base/documents /app/style_base/documents \
    && chown -R ${APP_UID}:${APP_GID} /app

USER app

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
