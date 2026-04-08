FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up user and permissions for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    ENABLE_WEB_INTERFACE=true

WORKDIR $HOME/app

# Install Python dependencies natively
COPY pyproject.toml server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . .

# Expose port 8000
EXPOSE 8000

# Start Uvicorn server directly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
