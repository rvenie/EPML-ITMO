# Конфигурация

Руководство по настройке параметров ResearchHub.

## Структура конфигурации

Проект использует многоуровневую систему конфигурации:

```
config/
├── pipeline_config.py      # Pydantic модели валидации
├── simple_composer.py      # Генератор конфигураций
├── monitoring.py           # Настройки мониторинга
└── generated/              # Сгенерированные конфиги
    ├── randomforestclassifier_medium_config.yaml
    ├── svm_small_config.yaml
    └── ...
```

## Файл params.yaml

Основной файл параметров для DVC пайплайна:

```yaml
# params.yaml
mlflow:
  experiment_name: "research_publications_classification"
  run_name: "baseline_model"
  tracking_uri: "file:./mlruns"
  tags:
    project: "research_agents_hub"
    domain: "digital_pathology"
    model_type: "classification"
    data_version: "v1.0"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  test_size: 0.2
  random_state: 42

model:
  algorithm: "RandomForestClassifier"
  n_estimators: 100
  max_depth: 10

features:
  tfidf_max_features: 5000
  ngram_range: [1, 2]
```

## Pydantic конфигурации

### Модели валидации

Все конфигурации проходят строгую валидацию через Pydantic:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class DataConfig(BaseModel):
    """Конфигурация данных."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = 42

class ModelConfig(BaseModel):
    """Конфигурация модели."""
    algorithm: Literal["RandomForestClassifier", "SVM", "LogisticRegression"]
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = Field(default=10, ge=1, le=100)
    
    @field_validator('n_estimators')
    def validate_estimators(cls, v):
        if v < 10:
            raise ValueError('n_estimators должен быть >= 10')
        return v

class PipelineConfig(BaseModel):
    """Полная конфигурация пайплайна."""
    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig
```

### Использование

```python
from config.pipeline_config import PipelineConfig

# Загрузка из YAML
config = PipelineConfig.from_yaml("config/generated/rf_config.yaml")

# Валидация проходит автоматически
print(config.model.algorithm)  # RandomForestClassifier
```

## Генерация конфигураций

Используйте `simple_composer.py` для автоматической генерации:

```bash
poetry run python config/simple_composer.py
```

Это создаст конфигурации для всех комбинаций алгоритмов и размеров данных:

| Алгоритм | Small | Medium |
|----------|-------|--------|
| RandomForest | ✅ | ✅ |
| SVM | ✅ | ✅ |
| LogisticRegression | ✅ | ✅ |

## Переменные окружения

### .env файл

```bash
# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=research_publications_classification

# DVC
DVC_REMOTE=local_storage

# ClearML
CLEARML_WEB_HOST=http://localhost:8080
CLEARML_API_HOST=http://localhost:8008
CLEARML_FILES_HOST=http://localhost:8081

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Загрузка в коде

```python
import os
from dotenv import load_dotenv

load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
```

## Конфигурация DVC

### dvc.yaml

Определяет этапы пайплайна:

```yaml
stages:
  fetch_data:
    cmd: python scripts/fetch_arxiv_data.py --output-dir data/raw --max-results 100
    deps:
      - scripts/fetch_arxiv_data.py
    outs:
      - data/raw/arxiv_publications.csv
    params:
      - data.raw_dir

  preprocess:
    cmd: python scripts/preprocess_data.py --input ${data.raw_dir}/arxiv_publications.csv
    deps:
      - scripts/preprocess_data.py
      - data/raw/arxiv_publications.csv
    outs:
      - data/processed/publications_processed.csv
    params:
      - data

  train:
    cmd: python scripts/train_model.py --input data/processed/publications_processed.csv
    deps:
      - scripts/train_model.py
      - data/processed/publications_processed.csv
    outs:
      - models/classifier.pkl
    params:
      - model
      - features
    metrics:
      - metrics.json:
          cache: false
```

### DVC Remote

```bash
# Локальный remote (по умолчанию)
dvc remote add -d local_storage ../dvc_storage

# S3 (опционально)
dvc remote add s3_remote s3://bucket-name/path

# Google Cloud Storage
dvc remote add gcs_remote gs://bucket-name/path
```

## Конфигурация Docker

### docker-compose.yml

```yaml
version: '3.8'

services:
  mlflow-server:
    image: research-agets-hub:latest
    ports:
      - "3000:3000"
    volumes:
      - ./mlruns:/app/mlruns
    command: mlflow-server
    
  ml-app:
    image: research-agets-hub:latest
    depends_on:
      - mlflow-server
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:3000
```

## Примеры конфигураций

### Минимальная конфигурация

```yaml
# minimal_config.yaml
model:
  algorithm: LogisticRegression
data:
  test_size: 0.2
```

### Конфигурация для продакшена

```yaml
# production_config.yaml
mlflow:
  experiment_name: "production_classifier"
  tracking_uri: "http://mlflow-server:3000"
  tags:
    environment: "production"
    version: "1.0.0"

model:
  algorithm: RandomForestClassifier
  n_estimators: 500
  max_depth: 20
  
features:
  tfidf_max_features: 10000
  ngram_range: [1, 3]

data:
  test_size: 0.15
  random_state: 42
```

## Следующие шаги

- [Обзор пользовательского руководства](../user-guide/overview.md)
- [Запуск экспериментов](../user-guide/experiments.md)
- [Развёртывание](../deployment/docker.md)
