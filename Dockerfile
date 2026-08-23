FROM python:3.13-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

FROM python:3.13-slim
WORKDIR /app
RUN useradd --create-home appuser \
 && mkdir -p /app/data \
 && chown -R appuser:appuser /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
ENV TOOLEVO_DB_PATH=/app/data/engine.db
VOLUME ["/app/data"]
EXPOSE 8000
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"
CMD ["uvicorn", "tool_evolution.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
