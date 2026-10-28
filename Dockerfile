FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data LOG_DIR=/app/runtime/logs \
    APP_DB_PATH=/app/runtime/application.sqlite3 UPLOAD_DIR=/app/runtime/uploads
RUN useradd --create-home --uid 1000 stacklens
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=stacklens:stacklens alembic.ini ./
COPY --chown=stacklens:stacklens migrations/ ./migrations/
COPY --chown=stacklens:stacklens data/ ./data/
COPY --chown=stacklens:stacklens evals/ ./evals/
RUN mkdir -p /app/runtime && chown -R stacklens:stacklens /app
USER stacklens
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/health')"
ENTRYPOINT ["stacklens"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
