# MLflow Utils API

Полная документация модуля `researchhub.mlflow_utils`.

## Обзор

Модуль предоставляет высокоуровневый API для работы с MLflow:

- Управление экспериментами
- Поиск и фильтрация запусков
- Сравнение результатов
- Model Registry

## Классы

### MLflowExperimentManager

Основной класс для управления экспериментами MLflow.

#### Инициализация

```python
from researchhub.mlflow_utils import MLflowExperimentManager

manager = MLflowExperimentManager(
    tracking_uri="file:./mlruns"  # или "http://localhost:3000"
)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `tracking_uri` | `str` | `"file:./mlruns"` | URI MLflow tracking server |

#### Методы

##### create_experiment_if_not_exists

Создаёт эксперимент, если он не существует.

```python
experiment_id = manager.create_experiment_if_not_exists("my_experiment")
```

**Параметры:**

- `experiment_name` (str): Название эксперимента

**Возвращает:** `str` — ID эксперимента

##### list_experiments

Возвращает список всех экспериментов.

```python
experiments = manager.list_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")
```

**Возвращает:** `list[Experiment]`

##### get_experiment_runs

Получает запуски эксперимента с фильтрацией.

```python
runs = manager.get_experiment_runs(
    experiment_name="my_experiment",
    filter_string="metrics.accuracy > 0.8",
    order_by=["metrics.accuracy DESC"],
    max_results=100
)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `experiment_name` | `str` | — | Название эксперимента |
| `filter_string` | `str` | `""` | MLflow фильтр |
| `order_by` | `list[str]` | `None` | Сортировка |
| `max_results` | `int` | `1000` | Макс. количество |

**Возвращает:** `list[Run]`

##### get_best_run

Находит лучший запуск по метрике.

```python
best_run = manager.get_best_run(
    experiment_name="my_experiment",
    metric_name="accuracy",
    maximize=True  # False для минимизации
)

if best_run:
    print(f"Best accuracy: {best_run.data.metrics['accuracy']}")
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `experiment_name` | `str` | — | Название эксперимента |
| `metric_name` | `str` | — | Название метрики |
| `maximize` | `bool` | `True` | Максимизация или минимизация |

**Возвращает:** `Run | None`

##### compare_runs

Сравнивает запуски по метрикам.

```python
comparison_df = manager.compare_runs(
    run_ids=["abc123", "def456", "ghi789"],
    metrics=["accuracy", "f1_score", "precision", "recall"]
)

print(comparison_df)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `run_ids` | `list[str]` | — | ID запусков |
| `metrics` | `list[str]` | `None` | Метрики для сравнения |

**Возвращает:** `pd.DataFrame`

##### export_experiment_results

Экспортирует результаты в файл.

```python
# CSV
manager.export_experiment_results(
    experiment_name="my_experiment",
    output_file="results.csv",
    format="csv"
)

# JSON
manager.export_experiment_results(
    experiment_name="my_experiment",
    output_file="results.json",
    format="json"
)

# Excel
manager.export_experiment_results(
    experiment_name="my_experiment",
    output_file="results.xlsx",
    format="excel"
)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `experiment_name` | `str` | — | Название эксперимента |
| `output_file` | `str` | — | Путь к файлу |
| `format` | `str` | `"csv"` | Формат: csv, json, excel |

---

### MLflowModelRegistry

Класс для работы с Model Registry.

#### Инициализация

```python
from researchhub.mlflow_utils import MLflowModelRegistry

registry = MLflowModelRegistry(tracking_uri="file:./mlruns")
```

#### Методы

##### register_model

Регистрирует модель в Registry.

```python
version = registry.register_model(
    model_uri="runs:/abc123/model",
    model_name="classifier",
    tags={"algorithm": "RF", "version": "1.0"}
)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model_uri` | `str` | — | URI модели |
| `model_name` | `str` | — | Название модели |
| `tags` | `dict[str, Any]` | `None` | Теги |

**Возвращает:** `str` — версия модели

##### get_latest_model_version

Получает последнюю версию модели.

```python
version = registry.get_latest_model_version(
    model_name="classifier",
    stage="Production"  # "None", "Staging", "Production", "Archived"
)
```

**Возвращает:** `str | None`

##### transition_model_version_stage

Переводит модель в новую стадию.

```python
registry.transition_model_version_stage(
    model_name="classifier",
    version="1",
    stage="Production",
    archive_existing_versions=True
)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model_name` | `str` | — | Название модели |
| `version` | `str` | — | Версия |
| `stage` | `str` | — | Новая стадия |
| `archive_existing_versions` | `bool` | `False` | Архивировать старые |

---

## Функции

### mlflow_experiment

Декоратор для создания эксперимента.

```python
from researchhub.mlflow_utils import mlflow_experiment

@mlflow_experiment(
    experiment_name="training",
    run_name="rf_v1",
    tags={"algorithm": "RF"},
    auto_log=True
)
def train(X, y, **params):
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model
```

### mlflow_run_context

Контекстный менеджер для MLflow run.

```python
from researchhub.mlflow_utils import mlflow_run_context

with mlflow_run_context(
    experiment_name="preprocessing",
    run_name="feature_eng",
    tags={"stage": "prep"},
    nested=False
):
    # Код внутри MLflow run
    data = process(raw_data)
    mlflow.log_metric("rows", len(data))
```

### search_runs_by_metrics

Поиск запусков по пороговым значениям метрик.

```python
from researchhub.mlflow_utils import search_runs_by_metrics

df = search_runs_by_metrics(
    experiment_name="training",
    metric_thresholds={
        "accuracy": (0.8, ">="),
        "f1_score": (0.7, ">="),
        "loss": (0.5, "<=")
    },
    tracking_uri="file:./mlruns"
)
```

**Операторы:** `>=`, `>`, `<=`, `<`, `=`

### get_experiment_leaderboard

Получает топ результатов эксперимента.

```python
from researchhub.mlflow_utils import get_experiment_leaderboard

top10 = get_experiment_leaderboard(
    experiment_name="training",
    metric="test_accuracy",
    top_n=10,
    tracking_uri="file:./mlruns"
)

print(top10[['run_name', 'accuracy', 'algorithm']])
```

### quick_compare_algorithms

Быстрое сравнение алгоритмов.

```python
from researchhub.mlflow_utils import quick_compare_algorithms

comparison = quick_compare_algorithms(
    experiment_name="training",
    algorithms=["RandomForestClassifier", "SVC", "LogisticRegression"],
    tracking_uri="file:./mlruns"
)
```

### create_experiment_summary_report

Создаёт HTML отчёт.

```python
from researchhub.mlflow_utils import create_experiment_summary_report

create_experiment_summary_report(
    experiment_name="training",
    output_file="report.html",
    tracking_uri="file:./mlruns"
)
```

---

## Примеры использования

### Полный workflow

```python
from researchhub.mlflow_utils import (
    MLflowExperimentManager,
    MLflowModelRegistry,
    mlflow_run_context
)

# 1. Создание менеджера
manager = MLflowExperimentManager()

# 2. Обучение с трекингом
with mlflow_run_context("training", "model_v1"):
    model = train_model(X, y)
    mlflow.log_metric("accuracy", evaluate(model, X_test, y_test))
    mlflow.sklearn.log_model(model, "model")

# 3. Поиск лучшего результата
best = manager.get_best_run("training", "accuracy")

# 4. Регистрация модели
registry = MLflowModelRegistry()
registry.register_model(
    f"runs:/{best.info.run_id}/model",
    "production_model"
)

# 5. Деплой
registry.transition_model_version_stage(
    "production_model", "1", "Production"
)
```

### Автоматизированные эксперименты

```python
experiments = {
    "rf_100": {"n_estimators": 100},
    "rf_200": {"n_estimators": 200},
    "rf_500": {"n_estimators": 500},
}

for name, params in experiments.items():
    with mlflow_run_context("hyperparameter_tuning", name):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

# Сравнение результатов
leaderboard = get_experiment_leaderboard(
    "hyperparameter_tuning", 
    metric="accuracy"
)
print(leaderboard)
```
