# MLflow Server

Руководство по развёртыванию и настройке MLflow Tracking Server.

## Обзор

MLflow предоставляет:

- Трекинг экспериментов
- Model Registry
- Веб-интерфейс для анализа
- API для интеграции

## Запуск сервера

### Локальный запуск

```bash
# Базовый запуск
mlflow server \
    --host 127.0.0.1 \
    --port 3000 \
    --backend-store-uri file:./mlruns

# С артефактами
mlflow server \
    --host 127.0.0.1 \
    --port 3000 \
    --backend-store-uri file:./mlruns \
    --default-artifact-root ./mlflow-artifacts

# В фоне
nohup mlflow server \
    --host 127.0.0.1 \
    --port 3000 \
    --backend-store-uri file:./mlruns \
    > mlflow.log 2>&1 &
```

### Docker запуск

```bash
# Через docker-compose
docker-compose up -d mlflow-server

# Напрямую через Docker
docker run -d \
    --name mlflow-server \
    -p 3000:3000 \
    -v $(pwd)/mlruns:/app/mlruns \
    research-agets-hub:latest \
    mlflow server --host 0.0.0.0 --port 3000
```

## Конфигурация

### Параметры сервера

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--host` | Хост для привязки | `0.0.0.0` |
| `--port` | Порт сервера | `3000` |
| `--backend-store-uri` | Хранилище метаданных | `file:./mlruns` |
| `--default-artifact-root` | Директория артефактов | `./artifacts` |
| `--workers` | Количество воркеров | `4` |

### Backend Store варианты

```bash
# Локальная файловая система
--backend-store-uri file:./mlruns

# SQLite
--backend-store-uri sqlite:///mlflow.db

# PostgreSQL
--backend-store-uri postgresql://user:pass@host:5432/mlflow

# MySQL
--backend-store-uri mysql://user:pass@host:3306/mlflow
```

### Artifact Store варианты

```bash
# Локальная директория
--default-artifact-root ./mlflow-artifacts

# S3
--default-artifact-root s3://bucket-name/mlflow-artifacts

# GCS
--default-artifact-root gs://bucket-name/mlflow-artifacts

# Azure Blob
--default-artifact-root wasbs://container@account.blob.core.windows.net/mlflow
```

## Подключение клиента

### В коде

```python
import mlflow

# Установка tracking URI
mlflow.set_tracking_uri("http://localhost:3000")

# Или через переменную окружения
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:3000"
```

### В params.yaml

```yaml
mlflow:
  tracking_uri: "http://localhost:3000"
  experiment_name: "my_experiment"
```

## Веб-интерфейс

### Доступ

После запуска сервера откройте: http://localhost:3000

### Возможности UI

| Раздел | Описание |
|--------|----------|
| **Experiments** | Список экспериментов |
| **Runs** | Запуски с метриками |
| **Compare** | Сравнение запусков |
| **Models** | Model Registry |
| **Artifacts** | Файлы и модели |

### Поиск и фильтрация

```
# По метрикам
metrics.accuracy > 0.8

# По параметрам
params.algorithm = "RandomForestClassifier"

# По тегам
tags.environment = "production"

# Комбинированный
metrics.accuracy > 0.8 AND params.algorithm = "SVC"
```

## Model Registry

### Регистрация модели

```python
# Автоматическая регистрация
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="my_classifier"
)

# Ручная регистрация
mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="my_classifier"
)
```

### Стадии модели

| Стадия | Описание |
|--------|----------|
| `None` | Новая версия |
| `Staging` | Тестирование |
| `Production` | Продакшен |
| `Archived` | Архив |

### Переключение стадий

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Переход в Production
client.transition_model_version_stage(
    name="my_classifier",
    version="1",
    stage="Production"
)
```

## Production развёртывание

### Docker Compose для продакшена

```yaml
version: '3.8'

services:
  mlflow-server:
    image: research-agets-hub:latest
    container_name: mlflow-server
    ports:
      - "3000:3000"
    volumes:
      - mlflow-data:/app/mlruns
      - mlflow-artifacts:/app/artifacts
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@db:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/artifacts
    command: >
      mlflow server
      --host 0.0.0.0
      --port 3000
      --workers 4
    depends_on:
      - db
    restart: always

  db:
    image: postgres:15
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=password

volumes:
  mlflow-data:
  mlflow-artifacts:
  postgres-data:
```

### Nginx reverse proxy

```nginx
server {
    listen 80;
    server_name mlflow.example.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Мониторинг

### Health check

```bash
# Проверка доступности
curl http://localhost:3000/health

# API check
curl http://localhost:3000/api/2.0/mlflow/experiments/search
```

### Логирование

```bash
# Просмотр логов Docker
docker-compose logs -f mlflow-server

# Логи в файл
mlflow server ... > mlflow.log 2>&1
```

## Безопасность

### Базовая аутентификация

```bash
# Установка mlflow-auth
poetry add "mlflow[auth]"

# Запуск с аутентификацией
poetry run mlflow server --app-name basic-auth
```

### Настройка через nginx

```nginx
location / {
    auth_basic "MLflow";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:3000;
}
```

## Резервное копирование

### Backup метаданных

```bash
# SQLite
cp mlflow.db mlflow.db.backup

# PostgreSQL
pg_dump -U mlflow mlflow > mlflow_backup.sql
```

### Backup артефактов

```bash
# Локальные артефакты
tar -czf artifacts_backup.tar.gz ./mlflow-artifacts/

# S3
aws s3 sync s3://bucket/mlflow-artifacts ./backup/
```

## Интеграция с проектом

### Makefile команды

```makefile
.PHONY: mlflow-server
mlflow-server:
	poetry run mlflow server \
		--host 127.0.0.1 \
		--port 3000 \
		--backend-store-uri file:./mlruns

.PHONY: mlflow-ui
mlflow-ui:
	poetry run mlflow ui --port 3000
```

### Использование

```bash
# Запуск через Makefile
make mlflow-server

# Или через poetry
poetry run mlflow server --host 127.0.0.1 --port 3000
```

## Устранение проблем

??? bug "Connection refused"

    Убедитесь что сервер запущен:
    ```bash
    # Проверка процесса
    ps aux | grep mlflow
    
    # Проверка порта
    lsof -i :3000
    ```

??? bug "Artifact not found"

    Проверьте пути и права:
    ```bash
    # Права на директорию
    ls -la ./mlruns/
    
    # Настройка artifact root
    --default-artifact-root $(pwd)/artifacts
    ```

??? bug "Database locked (SQLite)"

    Используйте PostgreSQL для многопоточности:
    ```bash
    --backend-store-uri postgresql://user:pass@host/db
    ```

## Следующие шаги

- [ClearML](clearml.md)
- [Docker](docker.md)
- [Эксперименты](../user-guide/experiments.md)
