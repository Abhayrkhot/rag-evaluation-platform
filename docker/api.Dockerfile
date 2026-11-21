# RAG agent API (FastAPI + LangGraph). Encoders run in-process on CPU in this image; GPU
# deployments use the same package on top of a CUDA torch (see slurm/ for the Explorer setup).
FROM python:3.12-slim AS build
RUN pip install --no-cache-dir "uv==0.9.*"
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra models --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra models

FROM python:3.12-slim
RUN useradd -m -u 10001 app
WORKDIR /app
COPY --from=build /app /app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/models
USER app
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"
CMD ["rageval", "serve", "--config", "/config/serve.yaml", "--host", "0.0.0.0", "--port", "8080"]
