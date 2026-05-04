# =============================================================================
# Developer DNA Matrix — Phase 3
# Dockerfile for fully reproducible containerized pipeline
#
# Build:  docker build -t ddm-phase3 .
# Run:    docker run --rm -v $(pwd)/outputs:/app/outputs ddm-phase3
# =============================================================================

FROM python:3.10-slim

LABEL maintainer="DDM Project"
LABEL description="Developer DNA Matrix Phase 3 — Reproducible ML Pipeline"
LABEL version="3.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY phase3_pipeline.py .
COPY src/ ./src/
COPY data/ ./data/

# Create output directories
RUN mkdir -p outputs figures reports

# Run the full Phase 3 pipeline on container start
CMD ["python", "phase3_pipeline.py"]
