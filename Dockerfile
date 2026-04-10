# ============================================================================
# PyTorchSandbox Dockerfile
# Mechanistic Interpretability OpenEnv Benchmark
# ============================================================================
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml server/requirements.txt ./

# Install torch CPU-only first (much smaller package ~200MB vs 800MB+)
# This prevents uv from resolving to full torch with CUDA
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system openenv-core fastapi uvicorn pydantic openai gradio numpy

# Copy application source
COPY --chown=user:user . .

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user 2>/dev/null || true

# Environment variables
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    ENABLE_WEB_INTERFACE=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
