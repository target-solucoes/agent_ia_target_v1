# =============================================================================
# Analytics Chatbot - Dockerfile Multi-stage
# =============================================================================

# ---------------------------
# Stage 1: Builder
# ---------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/install \
        langgraph==1.0.2 \
        langchain==1.0.3 \
        langchain-core==1.0.2 \
        langchain-google-genai \
        langgraph-checkpoint \
        pyarrow \
        pyyaml \
        rapidfuzz \
        unidecode \
        pydantic \
        pydantic-settings \
        python-dotenv \
        duckdb \
        pandas \
        psutil \
        pandera \
        plotly \
        kaleido \
        streamlit \
        scipy \
        supabase \
        rich

# ---------------------------
# Stage 2: Runtime
# ---------------------------
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local/lib/python3.11/site-packages

COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
COPY data/mappings/ ./data/mappings/
COPY data/datasets/ ./data/datasets/
COPY app.py ./
COPY pyproject.toml ./

RUN mkdir -p logs data/output/graphics && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV DATASET_PATH=data/datasets/DadosComercial_resumido_v02.parquet

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
