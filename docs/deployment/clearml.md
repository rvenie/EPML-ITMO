# ClearML Deployment

Руководство по развёртыванию и использованию ClearML для MLOps.

## Обзор

ClearML предоставляет:

- Полнофункциональный MLOps платформу
- Автоматический трекинг экспериментов
- Model Registry
- Data Management
- Orchestration и Pipelines

## Архитектура ClearML

```
┌─────────────────────────────────────────────────────────────┐
│                    ClearML Server                            │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  WebServer  │  APIServer  │ FileServer  │   Databases      │
│   :8080     │   :8008     │   :8081     │ Mongo, ES, Redis │
└─────────────┴─────────────┴─────────────┴──────────────────┘
                           │
                    ClearML SDK
                           │
              ┌────────────┴────────────┐
              │    Python Application    │
              └─────────────────────────┘
```

## Запуск сервера

### Docker Compose

```bash
# Переход в директорию конфигурации
cd clearml/config

# Запуск всех сервисов
docker-compose -f docker-compose-clearml.yml up -d

# Или через Makefile
make clearml-server
```

### Состав сервисов

| Сервис | Порт | Описание |
|--------|------|----------|
| `webserver` | 8080 | Веб-интерфейс |
| `apiserver` | 8008 | REST API |
| `fileserver` | 8081 | Хранение файлов |
| `elasticsearch` | 9200 | Поиск и индексация |
| `mongo` | 27017 | База данных |
| `redis` | 6379 | Кэширование |

### docker-compose-clearml.yml

```yaml
version: '3.6'

services:
  apiserver:
    image: allegroai/clearml:latest
    container_name: clearml-apiserver
    ports:
      - "8008:8008"
    depends_on:
      - mongo
      - elasticsearch
      - redis
    environment:
      CLEARML__APISERVER__MONGO__HOST: mongo
      CLEARML__APISERVER__ES__HOSTS: elasticsearch
      CLEARML__APISERVER__REDIS__HOST: redis

  webserver:
    image: allegroai/clearml:latest
    container_name: clearml-webserver
    ports:
      - "8080:8080"
    depends_on:
      - apiserver

  fileserver:
    image: allegroai/clearml:latest
    container_name: clearml-fileserver
    ports:
      - "8081:8081"
    volumes:
      - clearml-fileserver:/data

  mongo:
    image: mongo:4.4
    volumes:
      - clearml-mongo:/data/db

  elasticsearch:
    image: elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
    volumes:
      - clearml-elastic:/usr/share/elasticsearch/data

  redis:
    image: redis:6
    volumes:
      - clearml-redis:/data

volumes:
  clearml-mongo:
  clearml-elastic:
  clearml-redis:
  clearml-fileserver:
```

## Настройка SDK

### Получение credentials

1. Откройте http://localhost:8080
2. Создайте аккаунт (первый вход)
3. Settings → Workspace → Create new credentials
4. Сохраните `access_key` и `secret_key`

### Инициализация

```bash
# Запуск настройки
clearml-init
```

Введите:
- API Host: `http://localhost:8008`
- Web Host: `http://localhost:8080`
- File Host: `http://localhost:8081`
- Access Key: (из шага выше)
- Secret Key: (из шага выше)

### clearml.conf

```conf
# ~/clearml.conf
api {
    web_server: http://localhost:8080
    api_server: http://localhost:8008
    files_server: http://localhost:8081
    credentials {
        access_key = "YOUR_ACCESS_KEY"
        secret_key = "YOUR_SECRET_KEY"
    }
}
```

## Использование

### Базовый эксперимент

```python
from clearml import Task

# Инициализация задачи
task = Task.init(
    project_name="ResearchHub",
    task_name="Training Experiment",
    task_type=Task.TaskTypes.training
)

# Параметры логируются автоматически
params = {
    "learning_rate": 0.01,
    "epochs": 100,
    "batch_size": 32
}
task.connect(params)

# Обучение
for epoch in range(params["epochs"]):
    loss = train_epoch(model, data)
    
    # Логирование метрик
    task.get_logger().report_scalar(
        title="Training",
        series="Loss",
        value=loss,
        iteration=epoch
    )

# Завершение
task.close()
```

### Автоматический трекинг

```python
from clearml import Task

# ClearML автоматически трекает:
# - TensorFlow/Keras
# - PyTorch
# - scikit-learn
# - XGBoost
# - и другие

task = Task.init(
    project_name="ResearchHub",
    task_name="Auto-tracked Training",
    auto_connect_frameworks=True  # Автотрекинг
)

# sklearn автоматически отслеживается
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)  # Параметры логируются автоматически
```

### Model Registry

```python
from clearml import Model

# Сохранение модели
output_model = Model.create(
    task=task,
    name="classifier_v1",
    framework="scikit-learn"
)
output_model.update_weights(weights_filename="model.pkl")
output_model.publish()

# Загрузка модели
loaded_model = Model(model_id="abc123")
local_path = loaded_model.get_local_copy()
```

## Pipeline

### ML Pipeline

```python
from clearml.pipelines import PipelineController

pipe = PipelineController(
    name="ML Training Pipeline",
    project="ResearchHub"
)

# Добавление этапов
pipe.add_step(
    name="fetch_data",
    base_task_project="ResearchHub",
    base_task_name="Data Fetch Template"
)

pipe.add_step(
    name="preprocess",
    parents=["fetch_data"],
    base_task_project="ResearchHub",
    base_task_name="Preprocess Template"
)

pipe.add_step(
    name="train",
    parents=["preprocess"],
    base_task_project="ResearchHub",
    base_task_name="Training Template"
)

# Запуск
pipe.start()
```

### Scheduler

```python
from clearml.pipelines import PipelineScheduler

scheduler = PipelineScheduler(
    name="Daily Training",
    project="ResearchHub"
)

# Ежедневный запуск
scheduler.schedule(
    pipeline_controller=pipe,
    target_project="ResearchHub",
    cron_schedule="0 2 * * *"  # 2:00 AM daily
)
```

## Веб-интерфейс

### Основные разделы

| Раздел | Описание |
|--------|----------|
| **Projects** | Список проектов |
| **Experiments** | Эксперименты проекта |
| **Models** | Зарегистрированные модели |
| **Datasets** | Управление данными |
| **Pipelines** | Пайплайны и DAG |
| **Workers** | Агенты выполнения |

### Функции UI

- Compare experiments (сравнение экспериментов)
- Scalars (графики метрик)
- Plots (визуализации)
- Debug Samples (примеры данных)
- Console (логи в реальном времени)

## Мониторинг

### Проверка здоровья

```python
from clearml.pipelines import PipelineMonitor

monitor = PipelineMonitor(project_name="ResearchHub")

# Проверка серверов
health = monitor.check_servers_health()
print(health)

# Статистика пайплайнов
stats = monitor.get_pipeline_statistics()
print(stats)
```

### Алерты

```python
from clearml import Task

task = Task.init(...)

# Webhook при завершении
task.set_completed_callback(
    callback_function=send_notification
)
```

## Makefile команды

```makefile
# Запуск сервера
.PHONY: clearml-server
clearml-server:
	cd clearml/config && docker-compose -f docker-compose-clearml.yml up -d

# Остановка
.PHONY: clearml-stop
clearml-stop:
	cd clearml/config && docker-compose -f docker-compose-clearml.yml down

# Тестовый эксперимент
.PHONY: clearml-test
clearml-test:
	python clearml/pipelines/pipeline_scheduler.py test
```

## Интеграция с проектом

### Структура файлов

```
clearml/
├── config/
│   ├── docker-compose-clearml.yml
│   ├── clearml.conf
│   └── scheduler_config.json
├── experiments/
│   ├── experiment_runner.py
│   └── experiment_comparison.py
├── models/
│   └── model_manager.py
└── pipelines/
    ├── ml_pipeline.py
    ├── pipeline_scheduler.py
    └── pipeline_monitor.py
```

### Использование в коде

```python
from clearml.pipelines.ml_pipeline import MLPipeline

# Создание пайплайна
pipeline = MLPipeline(project_name="ResearchHub")

# Запуск эксперимента
result = pipeline.run_training_experiment(
    input_file="data/processed/publications_processed.csv"
)

print(f"Task ID: {result['task_id']}")
print(f"Metrics: {result['metrics']}")
```

## Устранение проблем

??? bug "Connection Error"

    Проверьте что сервер запущен:
    ```bash
    docker-compose ps
    curl http://localhost:8008/debug.ping
    ```

??? bug "Authentication Failed"

    Проверьте credentials в ~/clearml.conf:
    ```bash
    cat ~/clearml.conf
    clearml-init  # Перенастройка
    ```

??? bug "Task не появляется в UI"

    Проверьте подключение:
    ```python
    from clearml import Task
    task = Task.init(project_name="Test", task_name="Test")
    print(task.id)  # Должен вывести ID
    ```

## Сравнение с MLflow

| Функция | ClearML | MLflow |
|---------|---------|--------|
| Автотрекинг | ✅ | Частично |
| Data Versioning | ✅ | ❌ |
| Remote Agents | ✅ | ❌ |
| Pipelines | ✅ | ❌ |
| UI | Продвинутый | Базовый |
| Self-hosted | ✅ | ✅ |

## Следующие шаги

- [Docker](docker.md)
- [MLflow](mlflow.md)
- [DVC Pipeline](dvc.md)
