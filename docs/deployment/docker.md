# Docker Deployment

Руководство по развёртыванию ResearchHub с помощью Docker.

## Обзор

Docker обеспечивает:

- Изоляцию окружения
- Воспроизводимость
- Простоту развёртывания
- Масштабируемость

## Docker образ

### Dockerfile

```dockerfile
# Базовый образ
FROM python:3.11-slim as base

# Метаданные
LABEL maintainer="Research Team"
LABEL description="ResearchHub ML Pipeline"

# Переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.7.1

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
RUN pip install poetry==$POETRY_VERSION

# Копирование зависимостей
COPY pyproject.toml poetry.lock ./

# Установка зависимостей
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Копирование кода
COPY . .

# Установка проекта
RUN poetry install --no-interaction --no-ansi

# Точка входа
ENTRYPOINT ["python", "-m"]
CMD ["researchhub.main"]
```

### Сборка

```bash
# Сборка образа
docker build -t research-agets-hub:latest .

# С тегом версии
docker build -t research-agets-hub:v1.0.0 .

# Без кэша
docker build --no-cache -t research-agets-hub:latest .
```

## Docker Compose

### docker-compose.yml

```yaml
version: '3.8'

services:
  # MLflow Tracking Server
  mlflow-server:
    image: research-agets-hub:latest
    container_name: mlflow-server
    ports:
      - "3000:3000"
    volumes:
      - ./mlruns:/app/mlruns
      - mlflow-artifacts:/app/artifacts
    command: >
      mlflow server
      --host 0.0.0.0
      --port 3000
      --backend-store-uri file:///app/mlruns
      --default-artifact-root /app/artifacts
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ML Application
  ml-app:
    image: research-agets-hub:latest
    container_name: ml-app
    depends_on:
      mlflow-server:
        condition: service_healthy
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./mlruns:/app/mlruns
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:3000
    command: ["researchhub.main"]

  # DVC Pipeline Runner
  pipeline-runner:
    image: research-agets-hub:latest
    container_name: pipeline-runner
    volumes:
      - ./:/app
    command: ["dvc", "repro"]
    profiles:
      - pipeline

  # Model Training
  model-training:
    image: research-agets-hub:latest
    container_name: model-training
    depends_on:
      - mlflow-server
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./mlruns:/app/mlruns
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:3000
    command: >
      python scripts/train_model.py
      --input data/processed/publications_processed.csv
      --model-output models/classifier.pkl
    profiles:
      - training

  # Jupyter Development
  jupyter-dev:
    image: research-agets-hub:latest
    container_name: jupyter-dev
    ports:
      - "8888:8888"
    volumes:
      - ./:/app
    command: >
      jupyter lab
      --ip=0.0.0.0
      --port=8888
      --no-browser
      --allow-root
    profiles:
      - development

volumes:
  mlflow-artifacts:
```

## Команды запуска

### Основные сервисы

```bash
# Запуск MLflow сервера
docker-compose up -d mlflow-server

# Запуск ML приложения
docker-compose up ml-app

# Запуск всех основных сервисов
docker-compose up -d
```

### Специальные профили

```bash
# Запуск DVC pipeline
docker-compose --profile pipeline up pipeline-runner

# Запуск обучения
docker-compose --profile training up model-training

# Jupyter для разработки
docker-compose --profile development up jupyter-dev
```

### Остановка

```bash
# Остановка всех сервисов
docker-compose down

# С удалением volumes
docker-compose down -v

# Только конкретный сервис
docker-compose stop mlflow-server
```

## Полный workflow

### 1. Сборка и запуск

```bash
# Сборка образов
docker-compose build

# Запуск MLflow
docker-compose up -d mlflow-server

# Проверка запуска
docker-compose ps
curl http://localhost:3000/
```

### 2. Запуск pipeline

```bash
# Полный DVC pipeline
docker-compose --profile pipeline up pipeline-runner

# Или отдельные этапы
docker-compose run --rm ml-app dvc repro fetch_data
docker-compose run --rm ml-app dvc repro preprocess
docker-compose run --rm ml-app dvc repro train
```

### 3. Эксперименты

```bash
# Запуск экспериментов
docker-compose run --rm ml-app python scripts/run_experiments.py

# Просмотр результатов
open http://localhost:3000
```

## Переменные окружения

### .env файл

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow-server:3000
MLFLOW_EXPERIMENT_NAME=docker_experiments

# Data
DATA_DIR=/app/data
MODELS_DIR=/app/models

# Logging
LOG_LEVEL=INFO
```

### Использование в compose

```yaml
services:
  ml-app:
    env_file:
      - .env
    environment:
      - EXTRA_VAR=value
```

## Volumes и данные

### Persistent volumes

```yaml
volumes:
  mlflow-artifacts:
    driver: local
  
  data-volume:
    driver: local
    driver_opts:
      type: none
      device: /path/to/data
      o: bind
```

### Bind mounts

```yaml
services:
  ml-app:
    volumes:
      - ./data:/app/data:ro          # Read-only
      - ./models:/app/models:rw      # Read-write
      - ./mlruns:/app/mlruns
```

## Health checks

```yaml
services:
  mlflow-server:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Логирование

### Просмотр логов

```bash
# Все сервисы
docker-compose logs

# Конкретный сервис
docker-compose logs mlflow-server

# Follow режим
docker-compose logs -f ml-app

# Последние N строк
docker-compose logs --tail=100 ml-app
```

### Настройка логирования

```yaml
services:
  ml-app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Отладка

### Интерактивная оболочка

```bash
# Запуск bash в контейнере
docker-compose run --rm ml-app bash

# В запущенном контейнере
docker-compose exec ml-app bash
```

### Проверка сети

```bash
# Просмотр сетей
docker network ls

# Инспекция сети
docker network inspect research-agets-hub_default
```

## Production рекомендации

!!! tip "Лучшие практики"

    1. **Используйте multi-stage builds** для уменьшения размера образа
    2. **Не храните секреты в образе** — используйте secrets или env files
    3. **Настройте health checks** для всех сервисов
    4. **Ограничьте ресурсы** через resource limits
    5. **Используйте non-root user** в контейнере

### Пример resource limits

```yaml
services:
  ml-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Следующие шаги

- [DVC Pipeline](dvc.md)
- [MLflow Server](mlflow.md)
- [ClearML](clearml.md)
