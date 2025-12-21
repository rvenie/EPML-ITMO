# Многоэтапный Dockerfile для ML проекта с Poetry, DVC и MLflow
FROM python:3.11-slim AS base

# Установка переменных окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
RUN pip install poetry==2.2.1

# Установка рабочей директории
WORKDIR /app

# Копирование файлов Poetry для установки зависимостей
COPY pyproject.toml poetry.lock README.md ./

# Установка зависимостей через Poetry
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Копирование файлов конфигурации
COPY params.yaml dvc.yaml ./
COPY .dvc/ .dvc/

# Копирование исходного кода
COPY researchhub/ researchhub/
COPY scripts/ scripts/

# Установка текущего проекта
RUN poetry install --only-root && rm -rf $POETRY_CACHE_DIR

# Создание необходимых директорий
RUN mkdir -p data/raw data/processed models reports mlruns mlartifacts

# Открытие портов для MLflow и Jupyter
EXPOSE 3000 8888

# Создание entrypoint скрипта
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    echo "=== Research Agents Hub Container Starting ==="\n\
    echo "Using params from: $(pwd)/params.yaml"\n\
    \n\
    # Функция для чтения параметров из params.yaml\n\
    get_param() {\n\
    python -c "import yaml; config=yaml.safe_load(open(\"params.yaml\")); print(config[\"$1\"][\"$2\"])"\n\
    }\n\
    \n\
    # Функция для настройки MLflow из params.yaml\n\
    setup_mlflow() {\n\
    export MLFLOW_EXPERIMENT_NAME=$(get_param "mlflow" "experiment_name")\n\
    export MLFLOW_TRACKING_URI=$(get_param "mlflow" "tracking_uri")\n\
    echo "MLflow experiment: $MLFLOW_EXPERIMENT_NAME"\n\
    echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"\n\
    }\n\
    \n\
    # Проверка доступности данных\n\
    check_data() {\n\
    if [ -f ".dvc/config" ]; then\n\
    echo "DVC configuration found"\n\
    poetry run dvc remote list\n\
    echo "Attempting to pull data..."\n\
    poetry run dvc pull || echo "Warning: DVC pull failed, continuing without remote data"\n\
    else\n\
    echo "No DVC configuration found"\n\
    fi\n\
    }\n\
    \n\
    # Валидация параметров\n\
    validate_params() {\n\
    if [ ! -f "params.yaml" ]; then\n\
    echo "Error: params.yaml not found!"\n\
    exit 1\n\
    fi\n\
    echo "✓ params.yaml found and loaded"\n\
    \n\
    # Показать основные параметры\n\
    echo "Model algorithm: $(get_param \"train\" \"algorithm\")"\n\
    echo "Test size: $(get_param \"train\" \"test_size\")"\n\
    echo "Random state: $(get_param \"train\" \"random_state\")"\n\
    }\n\
    \n\
    case "$1" in\n\
    "mlflow-server")\n\
    echo "🚀 Starting MLflow server..."\n\
    validate_params\n\
    setup_mlflow\n\
    poetry run mlflow server \\\n\
    --host 0.0.0.0 \\\n\
    --port 3000 \\\n\
    --backend-store-uri file:./mlruns \\\n\
    --default-artifact-root ./mlartifacts\n\
    ;;\n\
    "pipeline")\n\
    echo "🔄 Running full DVC pipeline..."\n\
    validate_params\n\
    check_data\n\
    setup_mlflow\n\
    poetry run dvc repro\n\
    echo "✅ Pipeline completed!"\n\
    ;;\n\
    "train")\n\
    echo "🎯 Training model with parameters from params.yaml..."\n\
    validate_params\n\
    setup_mlflow\n\
    poetry run python scripts/train_model.py \\\n\
    --input data/processed/features.csv \\\n\
    --model-output models/classifier.pkl \\\n\
    --metrics metrics.json \\\n\
    --params params.yaml\n\
    ;;\n\
    "preprocess")\n\
    echo "🔧 Running data preprocessing..."\n\
    validate_params\n\
    poetry run python scripts/preprocess_data.py \\\n\
    --input data/raw/publications.csv \\\n\
    --output data/processed/publications_processed.csv \\\n\
    --metadata data/processed/processing_metadata.yaml\n\
    ;;\n\
    "feature-engineering")\n\
    echo "⚙️ Running feature engineering..."\n\
    validate_params\n\
    poetry run python scripts/feature_engineering.py \\\n\
    --input data/processed/publications_processed.csv \\\n\
    --output data/processed/features.csv \\\n\
    --params params.yaml\n\
    ;;\n\
    "evaluate")\n\
    echo "📊 Evaluating model..."\n\
    validate_params\n\
    poetry run python scripts/evaluate_model.py \\\n\
    --model models/classifier.pkl \\\n\
    --data data/processed/features.csv \\\n\
    --output reports/evaluation.json\n\
    ;;\n\
    "dvc-status")\n\
    echo "📋 Checking DVC status..."\n\
    poetry run dvc status\n\
    poetry run dvc dag\n\
    ;;\n\
    "params-info")\n\
    echo "📄 Current parameters:"\n\
    cat params.yaml\n\
    ;;\n\
    "jupyter")\n\
    echo "📓 Starting Jupyter Lab..."\n\
    poetry run jupyter lab \\\n\
    --ip=0.0.0.0 \\\n\
    --port=8888 \\\n\
    --no-browser \\\n\
    --allow-root \\\n\
    --NotebookApp.token=\"\" \\\n\
    --NotebookApp.password=\"\"\n\
    ;;\n\
    "bash")\n\
    echo "🐚 Starting interactive bash shell..."\n\
    exec /bin/bash\n\
    ;;\n\
    *)\n\
    echo "Available commands:"\n\
    echo "  mlflow-server    - Start MLflow tracking server"\n\
    echo "  pipeline         - Run full DVC pipeline"\n\
    echo "  train           - Train model only"\n\
    echo "  preprocess      - Run data preprocessing"\n\
    echo "  feature-engineering - Run feature engineering"\n\
    echo "  evaluate        - Evaluate trained model"\n\
    echo "  dvc-status      - Show DVC pipeline status"\n\
    echo "  params-info     - Show current parameters"\n\
    echo "  jupyter         - Start Jupyter Lab"\n\
    echo "  bash            - Interactive shell"\n\
    echo ""\n\
    echo "Or run custom command: $@"\n\
    exec "$@"\n\
    ;;\n\
    esac' > /app/entrypoint.sh

# Делаем entrypoint исполняемым
RUN chmod +x /app/entrypoint.sh

# Настройка проверки здоровья
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Команда по умолчанию
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["params-info"]

# === DEVELOPMENT STAGE ===
FROM base AS development

# Установка dev зависимостей
RUN poetry install && rm -rf $POETRY_CACHE_DIR

# Установка дополнительных dev инструментов
RUN poetry run pip install \
    jupyter \
    jupyterlab \
    ipywidgets

# Копирование дополнительных dev файлов
COPY notebooks/ notebooks/
COPY tests/ tests/
COPY Makefile README.md ./

# Команда по умолчанию для разработки
CMD ["jupyter"]

# === PRODUCTION STAGE ===
FROM base AS production

# Удаление ненужных файлов для продакшена
RUN find . -type f -name "*.pyc" -delete && \
    find . -type d -name "__pycache__" -delete && \
    rm -rf tests/ notebooks/ .git .pytest_cache .mypy_cache

# Создание пользователя для безопасности
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app

USER mluser

# Оптимизированная проверка здоровья для продакшена
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD poetry run python -c "import researchhub; print('OK')" || exit 1

# Команда по умолчанию для продакшена
CMD ["pipeline"]
