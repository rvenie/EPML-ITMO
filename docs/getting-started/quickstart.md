# Быстрый старт

Это руководство поможет вам начать работу с ResearchHub за несколько минут.

## Предварительные требования

!!! info "Системные требования"

    - Python 3.11+
    - Poetry (для управления зависимостями)
    - Docker (опционально, для контейнеризации)
    - Git

## Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/your-username/research-agets-hub.git
cd research-agets-hub
```

## Шаг 2: Установка зависимостей

```bash
# Установка Poetry (если не установлен)
curl -sSL https://install.python-poetry.org | python3 -

# Установка зависимостей проекта
poetry install

# Активация виртуального окружения
poetry shell
```

## Шаг 3: Настройка pre-commit хуков

```bash
poetry run pre-commit install
```

Это обеспечит автоматическую проверку качества кода при каждом коммите.

## Шаг 4: Запуск пайплайна

### Автоматический запуск (рекомендуется)

```bash
# Запуск полного пайплайна через DVC
poetry run dvc repro
```

### Пошаговый запуск

```bash
# 1. Загрузка данных из ArXiv
poetry run python scripts/fetch_arxiv_data.py \
    --query "cat:eess.IV OR cat:cs.CV OR cat:q-bio.QM" \
    --max-results 100

# 2. Предобработка данных
poetry run python scripts/preprocess_data.py \
    --input data/raw/arxiv_publications.csv \
    --output data/processed/publications_processed.csv

# 3. Обучение модели
poetry run python scripts/train_model.py \
    --input data/processed/publications_processed.csv \
    --model-output models/classifier.pkl
```

## Шаг 5: Запуск MLflow UI

```bash
# В отдельном терминале
poetry run mlflow server --host 127.0.0.1 --port 3000 --backend-store-uri file:./mlruns
```

Откройте браузер: [http://localhost:3000](http://localhost:3000)

## Шаг 6: Запуск экспериментов

```bash
# Запуск серии экспериментов
poetry run python scripts/run_experiments.py
```

## Проверка результатов

После выполнения пайплайна:

| Артефакт | Расположение | Описание |
|----------|--------------|----------|
| Данные | `data/processed/` | Обработанные данные |
| Модели | `models/` | Обученные модели |
| Метрики | `metrics.json` | Результаты оценки |
| Эксперименты | `mlruns/` | MLflow логи |

## Что дальше?

- [Подробная установка](installation.md) — детали настройки окружения
- [Конфигурация](configuration.md) — настройка параметров
- [Работа с данными](../user-guide/data.md) — загрузка и обработка данных
- [Обучение моделей](../user-guide/training.md) — обучение и оценка моделей

## Устранение проблем

??? question "Poetry не найден"

    Убедитесь, что Poetry добавлен в PATH:
    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

??? question "Ошибки при установке зависимостей"

    Попробуйте обновить Poetry:
    ```bash
    poetry self update
    poetry lock --no-update
    poetry install
    ```

??? question "DVC не может получить данные"

    Проверьте конфигурацию remote:
    ```bash
    dvc remote list
    dvc pull -v
    ```
