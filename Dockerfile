# ============================================================================
# PyTorchSandbox Dockerfile
# Mechanistic Interpretability OpenEnv Benchmark
# ============================================================================
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv && uv --version

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system torch && \
    uv pip install --system openenv-core fastapi uvicorn pydantic openai gradio numpy

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user 2>/dev/null || true

# Copy application source
COPY . .
RUN chown -R user:user /app
RUN chmod 755 /etc/ssl/certs && chmod 644 /etc/ssl/certs/ca-certificates.crt

# Environment variables
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    ENABLE_WEB_INTERFACE=true \
    SSL_CERT_FILE=/usr/local/lib/python3.11/site-packages/certifi/cacert.pem \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
