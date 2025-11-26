# ResearchHub

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

мультиагентная система, которая автоматизирует мониторинг и анализ научных публикаций в области цифровой патологии и анализа WSI (Whole Slide Imaging) данных. Система работает как умный исследовательский ассистент, который непрерывно отслеживает новые разработки в области анализа гистопатологических изображений.

--------

## Введение
Рабочее место Data Scientist настроено по актуальным стандартам: Cookiecutter, Poetry, Docker, линтеры и pre-commit.

## Структура проекта
Сгенерировано с помощью:
```
pip install cookiecutter-data-science
ccds
```
В корне репозитория находятся: `Dockerfile`, Makefile и все основные папки проекта.

## Инструменты и их настройка
- **Poetry** — менеджер зависимостей и виртуального окружения. Используется для установки и фиксации зависимостей, а также экспорта `requirements.txt` для Docker.
- **Pre-commit** — автоматическая проверка качества кода при каждом коммите.
- **Ruff** — линтер и авто-форматтер Python-кода, заменяет Black и Isort.
- **MyPy** — статический анализ типов, повышает стабильность кода.
- **Bandit** — анализ на уязвимости и безопасность.
- **Docker** — контейнеризация проекта для быстрой передачи и запуска на другой машине.
- **Makefile** — для удобства командной автоматизации (запуск тестов, линтинга и build).

Конфигурация инструментов хранится в файлах `.pre-commit-config.yaml`, `pyproject.toml`, `Dockerfile`.

## Управление зависимостями
Использовал Poetry:
```
poetry init --name research-agets-hub -n
poetry add pandas numpy scikit-learn
poetry add --group dev ruff mypy bandit pre-commit
poetry self add poetry-plugin-export
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Развертывание и запуск проекта

**Шаги:**
1. Клонировать репозиторий:
    ```
    git clone <адрес_репозитория>
    cd research_agets_hub
    ```
2. Установить Poetry, зависимости и активировать среду:
    ```
    pip install poetry
    poetry install
    ```
3. Инициализировать pre-commit-хуки:
    ```
    poetry run pre-commit install
    ```
4. Запуск проверки кода вручную:
    ```
    poetry run pre-commit run --all-files
    ```
5. Использование Makefile (если добавлен):
    ```
    make lint   # Проверить стиль кода
    make test   # Запуск тестов, если они добавлены
    ```

**Docker:**
```
docker build -t research-agets-hub .
docker run --rm research-agets-hub
```

## Проверка качества кода
Инициализация pre-commit хуков:
```
poetry run pre-commit install
poetry run pre-commit run --all-files
```
Результат (все проверки прошли):
![проверки при коммите](check_screenshot.png)

## Git workflow (ветвление)
- **main** — стабильная версия проекта.
- **feature/** — ветки для новых функций или экспериментов.
- Все фичи вливаются в main после проверки линтеров и успешного тестирования через pull-request.

Пример:
```
git checkout -b feature/название-фичи
# разработка
git add . && git commit -m "Feature: краткое описание"
git checkout main 
git merge feature/название-фичи
```


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
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         researchhub and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
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
