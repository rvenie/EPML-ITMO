# Decorators API

Документация модуля `researchhub.decorators` — декораторы для автоматического логирования в MLflow.

## Обзор

Модуль предоставляет декораторы для:

- Автоматического трекинга экспериментов
- Логирования времени выполнения
- Записи метрик и параметров
- Сохранения артефактов

## Основные декораторы

### @mlflow_track

Полнофункциональный декоратор для трекинга ML экспериментов.

```python
from researchhub.decorators import mlflow_track

@mlflow_track(
    experiment_name="my_experiment",
    run_name="training_v1",
    tags={"algorithm": "RF", "version": "1.0"},
    log_params=True,
    log_artifacts=True,
    log_model=True,
    auto_log=False
)
def train_model(X, y, n_estimators=100, max_depth=10):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    model.fit(X, y)
    return model
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `experiment_name` | `str` | — | Название эксперимента |
| `run_name` | `str \| None` | `None` | Название запуска |
| `tags` | `dict[str, Any] \| None` | `None` | Теги для запуска |
| `log_params` | `bool` | `True` | Логировать параметры функции |
| `log_artifacts` | `bool` | `True` | Логировать артефакты |
| `log_model` | `bool` | `True` | Логировать модель |
| `auto_log` | `bool` | `False` | Автологирование sklearn |

**Автоматически логируется:**

- Параметры функции (args, kwargs)
- Время выполнения
- Статус выполнения (success/failed)
- Результат (если dict — как метрики)
- Модель (если sklearn)
- Traceback при ошибке

---

### @log_execution_time

Логирует время выполнения функции.

```python
from researchhub.decorators import log_execution_time

@log_execution_time
def preprocess_data(df):
    # Обработка данных
    return processed_df

# При активном MLflow run логируется:
# metric: preprocess_data_execution_time = X.XX seconds
```

**Поведение:**

- Если есть активный MLflow run — логирует метрику
- Всегда выводит в лог время выполнения
- При ошибке — логирует время до ошибки

---

### @log_model_metrics

Автоматическое логирование метрик из результата функции.

```python
from researchhub.decorators import log_model_metrics

@log_model_metrics(
    metrics_to_log=["accuracy", "f1_score", "precision"],
    prefix="test_"
)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted")
    }

# Логируется:
# test_accuracy, test_f1_score, test_precision
# (recall не логируется, т.к. не в списке)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `metrics_to_log` | `list[str] \| None` | `None` | Какие метрики логировать (None = все) |
| `prefix` | `str` | `""` | Префикс для имён метрик |

---

### @log_dataset_info

Логирует информацию о DataFrame.

```python
from researchhub.decorators import log_dataset_info

@log_dataset_info(
    log_shape=True,
    log_dtypes=True,
    log_missing=True,
    log_stats=True
)
def load_data(filepath):
    return pd.read_csv(filepath)

# Логируется:
# - result_rows, result_columns
# - result_dtype_int64, result_dtype_object, ...
# - result_missing_values, result_missing_percentage
# - result_numeric_columns, result_mean_of_means, result_mean_of_stds
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `log_shape` | `bool` | `True` | Логировать размерность |
| `log_dtypes` | `bool` | `True` | Логировать типы данных |
| `log_missing` | `bool` | `True` | Логировать пропуски |
| `log_stats` | `bool` | `True` | Логировать статистику |

---

### @save_artifacts

Сохраняет указанные файлы как артефакты.

```python
from researchhub.decorators import save_artifacts

@save_artifacts("model.pkl", "metrics.json", "reports/")
def train_and_save(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Сохранение файлов
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": 0.85}, f)
    
    return model

# После выполнения артефакты загружаются в MLflow
```

**Параметры:**

- `*artifact_paths` (str): Пути к файлам/директориям

---

### @handle_exceptions

Обработка исключений с логированием в MLflow.

```python
from researchhub.decorators import handle_exceptions

@handle_exceptions(
    log_traceback=True,
    reraise=True
)
def risky_operation():
    # Код который может упасть
    pass

# При ошибке логируется:
# - error_type: ExceptionClassName
# - error_message: "Error details"
# - traceback: "Full traceback..." (если log_traceback=True)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `log_traceback` | `bool` | `True` | Логировать полный traceback |
| `reraise` | `bool` | `True` | Пробрасывать исключение дальше |

---

### @conditional_log

Условное логирование в зависимости от результата.

```python
from researchhub.decorators import conditional_log

def is_good_result(result, *args, **kwargs):
    return result.get("accuracy", 0) > 0.9

@conditional_log(
    condition_func=is_good_result,
    log_on_true={
        "tags": {"quality": "high"},
        "metrics": {"high_quality": 1}
    },
    log_on_false={
        "tags": {"quality": "low"},
        "metrics": {"high_quality": 0}
    }
)
def evaluate(model, X, y):
    return {"accuracy": accuracy_score(y, model.predict(X))}
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `condition_func` | `Callable` | Функция проверки условия |
| `log_on_true` | `dict` | Что логировать при True |
| `log_on_false` | `dict` | Что логировать при False |

---

## Составные декораторы

### @track_ml_experiment

Комбинированный декоратор для полного трекинга.

```python
from researchhub.decorators import track_ml_experiment

@track_ml_experiment(
    experiment_name="training",
    run_name="full_pipeline",
    auto_log=True,
    log_dataset=True,
    save_model=True
)
def full_training_pipeline(df, **params):
    X, y = prepare_features(df)
    model = train_model(X, y, **params)
    metrics = evaluate(model, X_test, y_test)
    return model, metrics
```

Включает:

- `@log_dataset_info`
- `@log_execution_time`
- `@handle_exceptions`
- `@mlflow_track`

---

### @quick_experiment

Быстрый декоратор для простых экспериментов.

```python
from researchhub.decorators import quick_experiment

@quick_experiment("quick_test")
def simple_train(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model
```

Эквивалентно:

```python
@track_ml_experiment(
    experiment_name="quick_test",
    auto_log=True,
    log_dataset=True,
    save_model=True
)
```

---

## Примеры использования

### Полный ML pipeline

```python
from researchhub.decorators import (
    mlflow_track,
    log_execution_time,
    log_model_metrics,
    log_dataset_info
)

@log_dataset_info()
def load_data(path):
    return pd.read_csv(path)

@log_execution_time
def preprocess(df):
    # Preprocessing...
    return X, y

@log_model_metrics(prefix="cv_")
def cross_validate(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    return {"accuracy_mean": scores.mean(), "accuracy_std": scores.std()}

@mlflow_track(
    experiment_name="pipeline",
    log_model=True
)
def train(X, y, **params):
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model

# Использование
with mlflow.start_run():
    df = load_data("data.csv")
    X, y = preprocess(df)
    cv_results = cross_validate(RandomForestClassifier(), X, y)
    model = train(X, y, n_estimators=100)
```

### Комбинация с контекстным менеджером

```python
from researchhub.mlflow_utils import mlflow_run_context
from researchhub.decorators import log_execution_time

@log_execution_time
def stage_1(data):
    return process(data)

@log_execution_time
def stage_2(data):
    return transform(data)

# Все декорированные функции логируют в один run
with mlflow_run_context("multi_stage", "full_run"):
    result1 = stage_1(data)
    result2 = stage_2(result1)
```

---

## Вспомогательные функции

Внутренние функции модуля (не для прямого использования):

| Функция | Описание |
|---------|----------|
| `_log_function_params` | Логирование параметров функции |
| `_serialize_param` | Сериализация параметра для MLflow |
| `_log_metrics_from_dict` | Извлечение метрик из dict |
| `_log_sklearn_model` | Логирование sklearn модели |
| `_log_function_artifacts` | Логирование артефактов |
| `_log_dataframe_info` | Логирование информации о DataFrame |
