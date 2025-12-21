# ResearchHub

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

мультиагентная система, которая автоматизирует мониторинг и анализ научных публикаций в области цифровой патологии и анализа WSI (Whole Slide Imaging) данных. Система работает как умный исследовательский ассистент, который непрерывно отслеживает новые разработки в области анализа гистопатологических изображений.

--------

## Установка и настройка окружения

### Требования
- Python 3.11+
- Poetry для управления зависимостями

### Быстрый старт

1. **Установите Poetry** (если еще не установлен):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Клонируйте репозиторий**:
```bash
git clone <repository-url>
cd research_agets_hub
```

3. **Установите зависимости**:
```bash
poetry install
```

4. **Активируйте виртуальное окружение**:
```bash
poetry shell
```

### Управление зависимостями

- **Добавить новую зависимость**:
```bash
poetry add package-name
```

- **Добавить dev-зависимость**:
```bash
poetry add --group dev package-name
```

- **Обновить зависимости**:
```bash
poetry update
```

- **Экспорт в requirements.txt** (если нужно для CI/CD):
```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### Инструменты разработки

В проекте настроены следующие инструменты качества кода:
- **Ruff** - линтер и форматтер
- **MyPy** - статическая типизация  
- **Bandit** - анализ безопасности
- **Pre-commit hooks** - автопроверки при коммитах

Запуск проверок:
```bash
poetry run ruff check .
poetry run mypy .
poetry run bandit -r researchhub/
```

--------

## 🐳 Запуск с Docker

### Быстрый старт с Docker Compose

**1. Запуск MLflow сервера:**
```bash
docker-compose up mlflow-server
```
Откройте http://localhost:3000 для доступа к MLflow UI.

**2. Запуск полного ML pipeline:**
```bash
docker-compose --profile training up model-training
```

**3. Запуск среды разработки с Jupyter:**
```bash
docker-compose --profile development up jupyter-dev
```
Откройте http://localhost:8888 для доступа к Jupyter Lab.

### Доступные Docker команды

```bash
# Проверить текущие параметры
docker-compose run --rm ml-app params-info

# Запустить полный DVC pipeline
docker-compose run --rm ml-app pipeline

# Предобработка данных
docker-compose run --rm ml-app preprocess

# Обучение модели
docker-compose run --rm ml-app train

# Оценка модели
docker-compose run --rm ml-app evaluate

# Проверка статуса DVC
docker-compose run --rm ml-app dvc-status

# Интерактивный режим
docker-compose run --rm ml-app bash
```

### Сборка образов

```bash
# Базовый образ для продакшена
docker build -t research-hub:base .

# Образ для разработки
docker build --target development -t research-hub:dev .

# Продакшен образ
docker build --target production -t research-hub:prod .
```

### Особенности Docker конфигурации

- ✅ **params.yaml** явно прокидывается в контейнер
- ✅ **Poetry** используется для управления зависимостями
- ✅ **DVC с Google Drive** настроен для получения данных
- ✅ **MLflow** интегрирован с параметрами из params.yaml
- ✅ Многостадийная сборка для разных окружений

--------

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Dockerfile
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Poetry configuration with dependencies, dev tools (ruff, mypy, bandit)
│                         and project metadata. Use `poetry install` to set up environment.
├── poetry.lock        <- Lock file with exact versions for reproducible builds
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│

│
└── researchhub   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes researchhub a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
