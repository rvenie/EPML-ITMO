# DVC Pipeline

Руководство по работе с DVC для версионирования данных и автоматизации пайплайнов.

## Обзор DVC

DVC (Data Version Control) обеспечивает:

- Версионирование больших файлов данных
- Воспроизводимые ML пайплайны
- Кэширование результатов
- Совместную работу над данными

## Структура DVC в проекте

```
research_agets_hub/
├── .dvc/
│   ├── config          # Конфигурация DVC
│   └── .gitignore
├── dvc.yaml            # Определение пайплайна
├── dvc.lock            # Зафиксированные версии
├── params.yaml         # Параметры пайплайна
└── data/
    ├── raw/
    │   └── *.dvc       # DVC tracking файлы
    └── processed/
        └── *.dvc
```

## Конфигурация

### .dvc/config

```ini
[core]
    autostage = true
    remote = local_storage

[remote "local_storage"]
    url = ../dvc_storage
```

### Настройка remote

```bash
# Локальный remote
dvc remote add -d local_storage ../dvc_storage

# S3
dvc remote add s3_remote s3://bucket-name/dvc-storage
dvc remote modify s3_remote access_key_id <YOUR_KEY>
dvc remote modify s3_remote secret_access_key <YOUR_SECRET>

# Google Cloud Storage
dvc remote add gcs_remote gs://bucket-name/dvc-storage

# Установка remote по умолчанию
dvc remote default local_storage
```

## Pipeline (dvc.yaml)

### Структура пайплайна

```yaml
stages:
  fetch_data:
    cmd: python scripts/fetch_arxiv_data.py --output-dir data/raw --max-results 100
    deps:
      - scripts/fetch_arxiv_data.py
    params:
      - data.raw_dir
    outs:
      - data/raw/arxiv_publications.csv
      - data/raw/arxiv_metadata.yaml

  preprocess:
    cmd: >
      python scripts/preprocess_data.py
      --input data/raw/arxiv_publications.csv
      --output data/processed/publications_processed.csv
    deps:
      - scripts/preprocess_data.py
      - data/raw/arxiv_publications.csv
    params:
      - data
    outs:
      - data/processed/publications_processed.csv
      - data/processed/processing_metadata.yaml

  train:
    cmd: >
      python scripts/train_model.py
      --input data/processed/publications_processed.csv
      --model-output models/classifier.pkl
      --metrics metrics.json
    deps:
      - scripts/train_model.py
      - data/processed/publications_processed.csv
    params:
      - model
      - features
    outs:
      - models/classifier.pkl
    metrics:
      - metrics.json:
          cache: false
```

### Типы зависимостей

| Тип | Описание | Пример |
|-----|----------|--------|
| `deps` | Файлы-зависимости | scripts, data |
| `params` | Параметры из params.yaml | model.n_estimators |
| `outs` | Выходные файлы | models, processed data |
| `metrics` | Метрики (JSON/YAML) | metrics.json |
| `plots` | Данные для визуализации | plots.csv |

## Команды DVC

### Запуск пайплайна

```bash
# Полный запуск
dvc repro

# Конкретный этап
dvc repro train

# Принудительный перезапуск
dvc repro --force

# Сухой запуск (показать что будет выполнено)
dvc repro --dry
```

### Версионирование данных

```bash
# Добавление файла в DVC
dvc add data/raw/dataset.csv

# Результат: создан data/raw/dataset.csv.dvc

# Коммит в Git
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "Add dataset v1"

# Загрузка в remote
dvc push

# Получение данных
dvc pull
```

### Статус и информация

```bash
# Статус пайплайна
dvc status

# Граф зависимостей
dvc dag

# Метрики
dvc metrics show

# Сравнение метрик
dvc metrics diff
```

## Параметры (params.yaml)

```yaml
# params.yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  test_size: 0.2
  random_state: 42

model:
  algorithm: RandomForestClassifier
  n_estimators: 100
  max_depth: 10

features:
  tfidf_max_features: 5000
  ngram_range:
    - 1
    - 2

mlflow:
  experiment_name: research_publications_classification
  tracking_uri: file:./mlruns
```

### Использование параметров

```python
import yaml

with open('params.yaml') as f:
    params = yaml.safe_load(f)

n_estimators = params['model']['n_estimators']
```

## Эксперименты с DVC

### Запуск эксперимента

```bash
# Изменение параметра и запуск
dvc exp run --set-param model.n_estimators=200

# Очередь экспериментов
dvc exp run --queue --set-param model.n_estimators=100
dvc exp run --queue --set-param model.n_estimators=200
dvc exp run --queue --set-param model.n_estimators=500
dvc exp run --run-all
```

### Просмотр экспериментов

```bash
# Список экспериментов
dvc exp show

# Сравнение
dvc exp diff exp1 exp2

# Применение эксперимента
dvc exp apply exp-abc123
```

## Кэширование

### Как работает кэш

DVC автоматически кэширует результаты каждого этапа. При повторном запуске:

1. DVC проверяет хеши входных данных
2. Если ничего не изменилось — использует кэш
3. Если изменилось — перезапускает этап

### Управление кэшем

```bash
# Просмотр кэша
du -sh .dvc/cache

# Очистка неиспользуемого кэша
dvc gc --workspace

# Полная очистка
dvc gc --all-commits
```

## Интеграция с Git

### Workflow

```bash
# 1. Внесение изменений в данные
# (данные автоматически отслеживаются)

# 2. Запуск пайплайна
dvc repro

# 3. Коммит изменений
git add dvc.lock params.yaml metrics.json
git commit -m "Update model with new parameters"

# 4. Push данных и кода
git push
dvc push
```

### Переключение версий

```bash
# Переключение на предыдущую версию
git checkout v1.0.0
dvc checkout

# Возврат к последней версии
git checkout main
dvc checkout
```

## CI/CD интеграция

### GitHub Actions пример

```yaml
name: DVC Pipeline

on:
  push:
    branches: [main]

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install Poetry
        uses: snok/install-poetry@v1
      
      - name: Install dependencies
        run: poetry install
      
      - name: Pull DVC data
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: poetry run dvc pull
      
      - name: Run pipeline
        run: poetry run dvc repro
      
      - name: Push results
        run: poetry run dvc push
```

## Визуализация

### DAG пайплайна

```bash
# ASCII граф
dvc dag

# Mermaid формат
dvc dag --md
```

```
+------------+
| fetch_data |
+------------+
      |
      v
+------------+
| preprocess |
+------------+
      |
      v
+------------+
|   train    |
+------------+
```

## Советы

!!! tip "Лучшие практики"

    1. **Версионируйте params.yaml** — для воспроизводимости
    2. **Используйте remote** — для совместной работы
    3. **Регулярно делайте dvc push** — после изменений данных
    4. **Документируйте изменения** — в commit сообщениях

!!! warning "Осторожно"

    - Не добавляйте большие файлы напрямую в Git
    - Проверяйте статус перед `dvc repro --force`
    - Убедитесь что remote настроен перед `dvc push`

## Следующие шаги

- [Docker Deployment](docker.md)
- [MLflow Server](mlflow.md)
- [Эксперименты](../user-guide/experiments.md)
