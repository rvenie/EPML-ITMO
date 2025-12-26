# Установка

Подробное руководство по установке и настройке ResearchHub.

## Системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| Python | 3.11 | 3.11+ |
| RAM | 4 GB | 8 GB+ |
| Диск | 2 GB | 10 GB+ |
| OS | Linux, macOS, Windows | Linux, macOS |

## Способы установки

### 1. Poetry (рекомендуется)

Poetry — современный менеджер зависимостей Python, обеспечивающий воспроизводимые сборки.

#### Установка Poetry

=== "Linux/macOS"

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

=== "Windows (PowerShell)"

    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

#### Установка проекта

```bash
# Клонирование
git clone https://github.com/your-username/research-agets-hub.git
cd research-agets-hub

# Установка зависимостей
poetry install

# Установка с dev-зависимостями
poetry install --with dev

# Активация окружения
poetry shell
```

### 2. Docker

Docker обеспечивает полную изоляцию и воспроизводимость.

```bash
# Сборка образа
docker build -t research-agets-hub .

# Или через docker-compose
docker-compose build
```

## Настройка инструментов

### Pre-commit хуки

Pre-commit обеспечивает автоматическую проверку качества кода.

```bash
# Установка хуков
poetry run pre-commit install

# Запуск вручную
poetry run pre-commit run --all-files
```

!!! info "Включённые проверки"

    - **Ruff** — линтинг и форматирование
    - **MyPy** — проверка типов
    - **Bandit** — анализ безопасности

### DVC (Data Version Control)

DVC уже сконфигурирован в проекте. При необходимости настройте remote:

```bash
# Проверка конфигурации
dvc remote list

# Получение данных (если remote настроен)
dvc pull
```

### MLflow

MLflow используется для трекинга экспериментов.

```bash
# Запуск локального сервера
poetry run mlflow server \
    --host 127.0.0.1 \
    --port 3000 \
    --backend-store-uri file:./mlruns

# Доступ: http://localhost:3000
```

### ClearML (опционально)

```bash
# Запуск ClearML сервера
make clearml-server

# Настройка SDK
clearml-init
```

## Проверка установки

Выполните следующие команды для проверки:

```bash
# Проверка Python версии
python --version  # Должна быть 3.11+

# Проверка Poetry
poetry --version

# Проверка установки пакетов
poetry run python -c "import researchhub; print('OK')"

# Проверка DVC
poetry run dvc version

# Проверка MLflow
poetry run mlflow --version

# Запуск тестов
poetry run pytest tests/ -v
```

## Структура зависимостей

### Основные зависимости

```toml
[project.dependencies]
pandas = ">=2.3.3,<3.0.0"
numpy = ">=2.3.5,<3.0.0"
scikit-learn = ">=1.7.2,<2.0.0"
dvc = ">=3.0.0,<4.0.0"
mlflow = ">=2.8.0,<3.0.0"
pydantic = ">=2.12.5,<3.0.0"
clearml = ">=2.1.0,<3.0.0"
```

### Dev-зависимости

```toml
[project.optional-dependencies.dev]
ruff = ">=0.4.4"
mypy = ">=1.10.0"
bandit = ">=1.7.8"
pre-commit = ">=3.7.1"
pytest = ">=7.4.0"
pytest-cov = ">=4.1.0"
```

## Переменные окружения

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MLFLOW_TRACKING_URI` | URI MLflow сервера | `file:./mlruns` |
| `DVC_REMOTE` | Имя DVC remote | `local_storage` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |

## Устранение проблем

??? bug "ImportError: No module named 'researchhub'"

    Убедитесь, что установка прошла корректно:
    ```bash
    poetry install
    ```

??? bug "DVC: Unable to find remote"

    Настройте локальный remote:
    ```bash
    dvc remote add -d local_storage ../dvc_storage
    ```

??? bug "MLflow: Connection refused"

    Убедитесь, что сервер запущен:
    ```bash
    poetry run mlflow server --host 127.0.0.1 --port 3000
    ```

## Следующие шаги

После успешной установки:

1. [Настройте конфигурацию](configuration.md)
2. [Изучите быстрый старт](quickstart.md)
3. [Запустите первые эксперименты](../user-guide/experiments.md)
