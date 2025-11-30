# Multi-stage Dockerfile for ML project with DVC and MLflow support
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML/DVC/MLflow dependencies if not in requirements
RUN pip install --no-cache-dir \
    dvc[s3]==3.64.0 \
    mlflow==2.22.2 \
    boto3==1.41.5

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models reports mlruns

# Set up DVC (initialize if .dvc doesn't exist)
RUN if [ ! -d ".dvc" ]; then dvc init --no-scm; fi

# Expose MLflow port
EXPOSE 5000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    # Pull data from DVC if available\n\
    if [ -f "data/raw/publications.csv.dvc" ]; then\n\
    echo "Pulling data from DVC..."\n\
    dvc pull || echo "DVC pull failed, continuing..."\n\
    fi\n\
    \n\
    # Check if MLflow server should be started\n\
    if [ "$1" = "mlflow-server" ]; then\n\
    echo "Starting MLflow server..."\n\
    mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri file:./mlruns \
    --default-artifact-root ./mlartifacts\n\
    elif [ "$1" = "train" ]; then\n\
    echo "Starting model training..."\n\
    python scripts/preprocess_data.py\n\
    python scripts/train_model.py \
    --input data/processed/publications_processed.csv \
    --model-output models/classifier.pkl \
    --metrics metrics.json\n\
    elif [ "$1" = "preprocess" ]; then\n\
    echo "Running data preprocessing..."\n\
    python scripts/preprocess_data.py\n\
    elif [ "$1" = "dvc-status" ]; then\n\
    echo "Checking DVC status..."\n\
    dvc status\n\
    dvc dag\n\
    else\n\
    # Execute the command passed to docker run\n\
    exec "$@"\n\
    fi' > /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set up health check for MLflow server
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "researchhub/main.py"]

# Multi-stage for development
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    jupyterlab \
    pytest \
    black \
    flake8

# Expose Jupyter port
EXPOSE 8888

# Override entrypoint for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Production stage
FROM base as production

# Remove unnecessary files
RUN find . -type f -name "*.pyc" -delete && \
    find . -type d -name "__pycache__" -delete && \
    rm -rf .git .pytest_cache .mypy_cache

# Run as non-root user for security
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app

USER mluser

# Health check for production
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1
