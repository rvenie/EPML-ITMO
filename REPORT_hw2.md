## Введение
Настроена комплексная система версионирования данных и моделей для ML проекта с использованием DVC и MLflow. Реализован полный цикл: от версионирования датасетов до отслеживания экспериментов и регистрации моделей.

## Выбранные инструменты

### Версионирование данных: DVC (Data Version Control)
**Обоснование выбора:**
- Git-подобный интерфейс для работы с большими файлами
- Поддержка различных типов remote storage (локальное, S3, GCS)
- Интеграция с существующим Git workflow
- Возможность создания воспроизводимых ML пайплайнов

### Версионирование моделей: MLflow
**Обоснование выбора:**
- Комплексное решение для ML lifecycle management
- Model Registry для управления версиями моделей
- Веб-интерфейс для анализа экспериментов
- Автоматическое логирование параметров, метрик и артефактов

## Настройка DVC для версионирования данных

### Инициализация DVC
```bash
# Инициализация DVC в проекте
dvc init

# Настройка Google Drive remote storage для воспроизводимости
dvc remote add -d gdrive_storage gdrive://133mVXq1xIxJMEDuN_7haphoHf8ZqD9g0

# Включение автоматического добавления .dvc файлов в Git
dvc config core.autostage true
```

### Подготовка данных
Создан датасет из 51 публикации по цифровой патологии:
```bash
# Создание raw данных
echo "title,authors,journal,year,doi,abstract,keywords,cited_by,methodology,dataset_used" > data/raw/publications.csv
# ... добавлены 51 запись научных публикаций
```

### Версионирование данных с помощью DVC
```bash
# Добавление исходных данных в DVC
dvc add data/raw/publications.csv
dvc add data/raw/data_metadata.yaml

# Результат: созданы .dvc файлы для отслеживания
# data/raw/publications.csv.dvc
# data/raw/data_metadata.yaml.dvc
```

### Предобработка данных
Создан скрипт `scripts/preprocess_data.py` со следующими функциями:
- Очистка и нормализация текстовых данных
- Извлечение дополнительных признаков (количество авторов, длина текста)
- Категоризация журналов по типам
- Вычисление impact score на основе цитирований

```bash
# Запуск предобработки
python scripts/preprocess_data.py

# Результат:
# 2025-11-30 16:45:51,617 - INFO - Final dataset shape: (51, 21)
# 2025-11-30 16:45:51,618 - INFO - Data preprocessing completed successfully!
```

### Версионирование обработанных данных
```bash
# Добавление обработанных данных в DVC
dvc add data/processed/publications_processed.csv
dvc add data/processed/processing_metadata.yaml

# Проверка статуса DVC
dvc status
# Data and pipelines are up to date.
```

## Настройка MLflow для версионирования моделей

### Конфигурация MLflow
В файле `params.yaml` настроены параметры эксперимента:
```yaml
mlflow:
  experiment_name: "research_publications_classification"
  run_name: "baseline_model"
  tracking_uri: "file:./mlruns"
  tags:
    project: "research_agents_hub"
    domain: "digital_pathology" 
    model_type: "classification"
    data_version: "v1.0"
```

### Запуск MLflow сервера
```bash
# Запуск tracking server
nohup mlflow server --host 127.0.0.1 --port 3000 --backend-store-uri file:./mlruns > mlflow.log 2>&1 &

# Проверка доступности
curl http://127.0.0.1:3000/
# <!doctype html><html lang="en"><head><meta charset="utf-8"/>...
```


### Обучение модели с MLflow логированием
Создан скрипт `scripts/train_model.py` с полной интеграцией MLflow:

```bash
# Обучение модели
python scripts/train_model.py \
    --input data/processed/publications_processed.csv \
    --model-output models/classifier.pkl \
    --metrics metrics.json

# Результаты обучения:
# 2025-11-30 16:46:13,132 - INFO - Cross-validation scores: [0.71428571 0.84615385 0.76923077]
# 2025-11-30 16:46:13,132 - INFO - Mean CV score: 0.7766 (+/- 0.1082)
# Successfully registered model 'research_publications_classification_model'.
# Created version '1' of model 'research_publications_classification_model'.
```

### Версионирование обученной модели
```bash
# Добавление модели в DVC
dvc add models/classifier.pkl

# Результат: создан models/classifier.pkl.dvc для отслеживания
```

## Результаты работы системы версионирования

### MLflow Model Registry
- **Зарегистрированная модель:** research_publications_classification_model
- **Версия:** 1
- **MLflow Run ID:** 3e0b59a18e4d4d2386a2e214affc35bd

### Логированные параметры (15 параметров):
- algorithm: RandomForestClassifier
- n_estimators: 100
- max_depth: 10
- test_size: 0.2
- tfidf_max_features: 5000
- random_state: 42
- и др.

### Логированные метрики:
- cv_mean: 0.7766
- test_accuracy: 0.9091
- test_precision: 0.8333
- test_recall: 0.9091  
- test_f1_score: 0.8678

### Структура версионированных данных и моделей
```
data/
├── raw/
│   ├── publications.csv.dvc          # DVC отслеживает исходные данные
│   └── data_metadata.yaml.dvc        # DVC отслеживает метаданные
└── processed/
    ├── publications_processed.csv.dvc # DVC отслеживает обработанные данные
    └── processing_metadata.yaml.dvc   # DVC отслеживает метаданные обработки

models/
├── classifier.pkl.dvc                 # DVC отслеживает обученную модель
└── classifier_metadata.yaml          # Метаданные модели

mlruns/                               # MLflow tracking данные
├── 0/                                # Эксперименты
├── 513200582704195422/              # ID эксперимента
│   └── 3e0b59a18e4d4d2386a2e214affc35bd/ # Run с моделью и метриками
└── models/                          # Model Registry
    └── research_publications_classification_model/
```
![MLflow](pics/mlflow1.png)
![MLflow](pics/mlflow2.png)
![MLflow](pics/mlflow3.png)

## Контейнеризация для воспроизводимости

### Docker настройка
Создан многоэтапный Dockerfile:
```dockerfile
# Базовая среда с Python и зависимостями
FROM python:3.11-slim as base
# Установка DVC, MLflow и ML зависимостей
RUN pip install dvc[s3]==3.64.0 mlflow==2.22.2
```

### Docker Compose для оркестрации
```yaml
services:
  mlflow-server:
    ports: ["3000:3000"]
    command: ["mlflow-server"]
    
  ml-app:
    depends_on: [mlflow-server]
    command: ["train"]
```

## Проверка воспроизводимости

### Локальный запуск
```bash
# Установка зависимостей через Poetry
poetry install

# Получение данных из DVC remote
poetry run dvc pull

# Воспроизведение результатов
poetry run python scripts/preprocess_data.py
poetry run python scripts/train_model.py --input data/processed/publications_processed.csv --model-output models/classifier.pkl --metrics metrics.json
```

### Docker запуск
```bash
# Сборка и запуск через Docker Compose
docker-compose up -d mlflow-server
docker-compose run --rm ml-app train

# Результат: модель обучена, метрики в MLflow UI
```

### Верификация результатов
```bash
# Проверка DVC статуса
dvc status
# Data and pipelines are up to date.

```
![DVC](pics/dvc.png)

## Инструкция по воспроизводимости 
```bash
# 1. Клонирование репозитория
git clone <repository_url>
cd research_agets_hub

# 2. Запуск MLflow сервера
docker-compose up -d mlflow-server

# 3. Проверка доступности MLflow UI (должно открыться в браузере)
open http://localhost:3000

# 4. Запуск полного обучения модели
docker-compose run --rm ml-app train
```

### Пошаговая инструкци
### Шаг 1: Сборка и запуск сервисов
```bash
# Сборка базового образа
docker-compose build mlflow-server

# Запуск MLflow сервера в фоне
docker-compose up -d mlflow-server

# Проверка статуса сервиса
docker-compose ps
```

**Ожидаемый результат**: 
```
NAME            COMMAND           SERVICE          STATUS    PORTS
mlflow-server   "mlflow-server"   mlflow-server    running   0.0.0.0:3000->3000/tcp
```

#### Шаг 2: Верификация MLflow UI

```bash
# Проверка доступности API
curl -f http://localhost:3000/api/2.0/mlflow/experiments/search

# Проверка веб-интерфейса (должен вернуть HTML)
curl -s http://localhost:3000 | head -n 5
```

**Ожидаемый результат**: JSON ответ для API и HTML для веб-интерфейса



### Способ через Poetry

Если Docker недоступен:

```bash
# 1. Установка Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 2. Установка зависимостей
poetry install

# 3. Проверка конфигурации DVC
poetry run dvc remote list

# 4. Запуск MLflow сервера (в фоне)
nohup poetry run mlflow server --host 127.0.0.1 --port 3000 --backend-store-uri file:./mlruns > mlflow.log 2>&1 &


## Git workflow для версионирования

### Структура веток
- **main** — стабильная версия с базовой настройкой
- **feature/setup-workspace** — настройка рабочего окружения
- **feature/setup-dvc-mlflow** — текущая ветка с системой версионирования

### Коммиты версионирования
```bash
git add .dvc/config data/raw/*.dvc data/processed/*.dvc models/*.dvc
git commit -m "feat: ..."
```
