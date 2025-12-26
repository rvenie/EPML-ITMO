# API Reference

Документация программного интерфейса ResearchHub.

## Модули

### Core модули

| Модуль | Описание |
|--------|----------|
| [`mlflow_utils`](mlflow-utils.md) | Утилиты для работы с MLflow |
| [`decorators`](decorators.md) | Декораторы для автоматического логирования |
| [`features`](features.md) | Извлечение и обработка признаков |

### Modeling

```python
from researchhub.modeling import train, predict
```

| Модуль | Описание |
|--------|----------|
| `modeling.train` | Обучение моделей |
| `modeling.predict` | Предсказания |

## Быстрый старт

### Импорт модулей

```python
# MLflow утилиты
from researchhub.mlflow_utils import (
    MLflowExperimentManager,
    MLflowModelRegistry,
    mlflow_experiment,
    mlflow_run_context,
    search_runs_by_metrics,
    get_experiment_leaderboard,
)

# Декораторы
from researchhub.decorators import (
    mlflow_track,
    log_execution_time,
    log_model_metrics,
    log_dataset_info,
    track_ml_experiment,
)

# Конфигурация
from researchhub.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)
```

## Классы

### MLflowExperimentManager

Основной класс для управления экспериментами.

```python
manager = MLflowExperimentManager(tracking_uri="file:./mlruns")

# Создание эксперимента
exp_id = manager.create_experiment_if_not_exists("my_experiment")

# Получение запусков
runs = manager.get_experiment_runs(
    experiment_name="my_experiment",
    filter_string="metrics.accuracy > 0.8"
)

# Лучший результат
best = manager.get_best_run("my_experiment", metric_name="accuracy")

# Сравнение
df = manager.compare_runs(run_ids=["run1", "run2"])

# Экспорт
manager.export_experiment_results("my_experiment", "results.csv")
```

### MLflowModelRegistry

Управление Model Registry.

```python
registry = MLflowModelRegistry()

# Регистрация
version = registry.register_model(
    model_uri="runs:/abc123/model",
    model_name="classifier",
    tags={"env": "prod"}
)

# Получение версии
latest = registry.get_latest_model_version("classifier", stage="Production")

# Переход стадии
registry.transition_model_version_stage(
    model_name="classifier",
    version="1",
    stage="Production"
)
```

## Декораторы

### @mlflow_track

Полный трекинг эксперимента.

```python
@mlflow_track(
    experiment_name="training",
    run_name="model_v1",
    tags={"algorithm": "RF"},
    log_params=True,
    log_artifacts=True,
    log_model=True,
    auto_log=False
)
def train_model(X, y, **params):
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model
```

### @log_execution_time

Логирование времени выполнения.

```python
@log_execution_time
def process_data(df):
    # Обработка...
    return processed_df
```

### @log_model_metrics

Логирование метрик.

```python
@log_model_metrics(metrics_to_log=["accuracy", "f1_score"], prefix="test_")
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred, average="weighted")
    }
```

### @log_dataset_info

Логирование информации о данных.

```python
@log_dataset_info(log_shape=True, log_missing=True, log_stats=True)
def load_data(filepath):
    return pd.read_csv(filepath)
```

## Контекстные менеджеры

### mlflow_run_context

```python
from researchhub.mlflow_utils import mlflow_run_context

with mlflow_run_context(
    experiment_name="preprocessing",
    run_name="feature_engineering",
    tags={"stage": "prep"}
):
    # Код выполняется внутри MLflow run
    processed = process_features(data)
    mlflow.log_metric("features_count", processed.shape[1])
```

## Функции

### search_runs_by_metrics

Поиск по метрикам.

```python
from researchhub.mlflow_utils import search_runs_by_metrics

df = search_runs_by_metrics(
    experiment_name="training",
    metric_thresholds={
        "accuracy": (0.8, ">="),
        "f1_score": (0.7, ">=")
    }
)
```

### get_experiment_leaderboard

Топ результатов.

```python
from researchhub.mlflow_utils import get_experiment_leaderboard

top5 = get_experiment_leaderboard(
    experiment_name="training",
    metric="accuracy",
    top_n=5
)
```

### quick_compare_algorithms

Сравнение алгоритмов.

```python
from researchhub.mlflow_utils import quick_compare_algorithms

comparison = quick_compare_algorithms(
    experiment_name="training",
    algorithms=["RandomForest", "SVM", "LogisticRegression"]
)
```

## Типы

```python
from typing import Any, Callable
from mlflow.entities import Experiment, Run
import pandas as pd

# Типичные типы возврата
def get_runs() -> list[Run]: ...
def compare() -> pd.DataFrame: ...
def get_best() -> Run | None: ...
```

## Исключения

```python
# Стандартные исключения Python
ValueError  # Неверные параметры
FileNotFoundError  # Файл не найден
ConnectionError  # Проблемы с MLflow server
```

## См. также

- [MLflow Utils детально](mlflow-utils.md)
- [Декораторы детально](decorators.md)
- [Features API](features.md)
