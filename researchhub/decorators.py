#!/usr/bin/env python3
"""
Декораторы для автоматического логирования ML экспериментов в MLflow.
Предоставляет удобные декораторы для логирования параметров, метрик, артефактов и времени выполнения.
"""

import functools
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def mlflow_track(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    log_params: bool = True,
    log_artifacts: bool = True,
    log_model: bool = True,
    auto_log: bool = False,
):
    """
    Декоратор для автоматического трекинга ML экспериментов.

    Args:
        experiment_name: Название эксперимента
        run_name: Название запуска (если None, будет использовано имя функции)
        tags: Теги для запуска
        log_params: Логировать параметры функции
        log_artifacts: Логировать артефакты
        log_model: Логировать модель (если возвращается)
        auto_log: Включить автоматическое логирование sklearn
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Устанавливаем эксперимент
            mlflow.set_experiment(experiment_name)

            # Включаем автологирование если нужно
            if auto_log:
                mlflow.sklearn.autolog()

            actual_run_name = run_name or f"{func.__name__}_{int(time.time())}"

            with mlflow.start_run(run_name=actual_run_name):
                start_time = time.time()

                try:
                    # Устанавливаем теги
                    mlflow.set_tag("function_name", func.__name__)
                    mlflow.set_tag("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))

                    if tags:
                        for key, value in tags.items():
                            mlflow.set_tag(key, str(value))

                    # Логируем параметры функции
                    if log_params:
                        _log_function_params(func, args, kwargs)

                    # Выполняем функцию
                    result = func(*args, **kwargs)

                    # Логируем время выполнения
                    execution_time = time.time() - start_time
                    mlflow.log_metric("execution_time_seconds", execution_time)

                    # Логируем результат если это метрики
                    if isinstance(result, dict):
                        _log_metrics_from_dict(result)

                    # Логируем модель если это sklearn модель
                    if log_model and hasattr(result, "fit"):
                        _log_sklearn_model(result, func.__name__)

                    # Логируем артефакты если есть
                    if log_artifacts:
                        _log_function_artifacts(result)

                    mlflow.set_tag("status", "success")
                    logger.info(
                        f"✅ Успешно завершен запуск {actual_run_name} "
                        f"за {execution_time:.2f}s"
                    )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    mlflow.log_metric("execution_time_seconds", execution_time)
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))
                    mlflow.set_tag("traceback", traceback.format_exc())

                    logger.error(f"❌ Ошибка в запуске {actual_run_name}: {e}")
                    raise

        return wrapper

    return decorator


def log_execution_time(func: Callable) -> Callable:
    """Декоратор для логирования времени выполнения функции."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Логируем время если есть активный MLflow run
            if mlflow.active_run():
                mlflow.log_metric(f"{func.__name__}_execution_time", execution_time)

            logger.info(f"Функция {func.__name__} выполнена за {execution_time:.4f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            if mlflow.active_run():
                mlflow.log_metric(f"{func.__name__}_execution_time", execution_time)
                mlflow.set_tag(f"{func.__name__}_error", str(e))
            raise

    return wrapper


def log_model_metrics(
    metrics_to_log: Optional[List[str]] = None,
    prefix: str = "",
):
    """
    Декоратор для логирования метрик модели.

    Args:
        metrics_to_log: Список метрик для логирования
        prefix: Префикс для названий метрик
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if not mlflow.active_run():
                logger.warning("Нет активного MLflow run для логирования метрик")
                return result

            # Если результат - словарь с метриками
            if isinstance(result, dict):
                for key, value in result.items():
                    if metrics_to_log is None or key in metrics_to_log:
                        metric_name = f"{prefix}{key}" if prefix else key
                        if isinstance(value, (int, float, np.number)):
                            mlflow.log_metric(metric_name, float(value))

            # Если результат - число (единственная метрика)
            elif isinstance(result, (int, float, np.number)):
                metric_name = f"{prefix}{func.__name__}" if prefix else func.__name__
                mlflow.log_metric(metric_name, float(result))

            return result

        return wrapper

    return decorator


def log_dataset_info(
    log_shape: bool = True,
    log_dtypes: bool = True,
    log_missing: bool = True,
    log_stats: bool = True,
):
    """
    Декоратор для логирования информации о датасете.

    Args:
        log_shape: Логировать размерность данных
        log_dtypes: Логировать типы данных
        log_missing: Логировать пропуски
        log_stats: Логировать базовую статистику
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if not mlflow.active_run():
                return result

            # Ищем DataFrame в результате или аргументах
            dataframes = []

            # Проверяем результат
            if isinstance(result, pd.DataFrame):
                dataframes.append(("result", result))
            elif isinstance(result, tuple):
                for i, item in enumerate(result):
                    if isinstance(item, pd.DataFrame):
                        dataframes.append((f"result_{i}", item))

            # Проверяем аргументы
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    dataframes.append((f"arg_{i}", arg))

            # Логируем информацию о каждом DataFrame
            for name, df in dataframes:
                _log_dataframe_info(
                    df, name, log_shape, log_dtypes, log_missing, log_stats
                )

            return result

        return wrapper

    return decorator


def save_artifacts(*artifact_paths: str):
    """
    Декоратор для сохранения артефактов.

    Args:
        artifact_paths: Пути к файлам/директориям для сохранения
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if not mlflow.active_run():
                logger.warning("Нет активного MLflow run для сохранения артефактов")
                return result

            # Сохраняем указанные артефакты
            for path in artifact_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    if path_obj.is_file():
                        mlflow.log_artifact(str(path_obj))
                    elif path_obj.is_dir():
                        mlflow.log_artifacts(str(path_obj))
                    logger.info(f"Сохранен артефакт: {path}")
                else:
                    logger.warning(f"Артефакт не найден: {path}")

            return result

        return wrapper

    return decorator


def handle_exceptions(
    log_traceback: bool = True,
    reraise: bool = True,
):
    """
    Декоратор для обработки исключений с логированием в MLflow.

    Args:
        log_traceback: Логировать полный traceback
        reraise: Повторно выбрасывать исключение
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if mlflow.active_run():
                    mlflow.set_tag("error_type", type(e).__name__)
                    mlflow.set_tag("error_message", str(e))

                    if log_traceback:
                        mlflow.set_tag("traceback", traceback.format_exc())

                logger.error(f"Ошибка в функции {func.__name__}: {e}")

                if reraise:
                    raise

        return wrapper

    return decorator


def conditional_log(
    condition_func: Callable[..., bool],
    log_on_true: Dict[str, Any] = None,
    log_on_false: Dict[str, Any] = None,
):
    """
    Декоратор для условного логирования.

    Args:
        condition_func: Функция для проверки условия
        log_on_true: Что логировать если условие истинно
        log_on_false: Что логировать если условие ложно
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if not mlflow.active_run():
                return result

            # Проверяем условие
            condition_result = condition_func(result, *args, **kwargs)

            logs_to_apply = log_on_true if condition_result else log_on_false

            if logs_to_apply:
                # Логируем метрики
                if "metrics" in logs_to_apply:
                    for key, value in logs_to_apply["metrics"].items():
                        mlflow.log_metric(key, value)

                # Логируем параметры
                if "params" in logs_to_apply:
                    for key, value in logs_to_apply["params"].items():
                        mlflow.log_param(key, value)

                # Логируем теги
                if "tags" in logs_to_apply:
                    for key, value in logs_to_apply["tags"].items():
                        mlflow.set_tag(key, value)

            return result

        return wrapper

    return decorator


# Вспомогательные функции


def _log_function_params(func: Callable, args: tuple, kwargs: dict):
    """Логирует параметры функции."""
    try:
        # Получаем имена параметров функции
        func_args = func.__code__.co_varnames[: func.__code__.co_argcount]

        # Логируем позиционные аргументы
        for i, (arg_name, arg_value) in enumerate(zip(func_args, args, strict=False)):
            if not arg_name.startswith("_"):  # Пропускаем приватные параметры
                param_value = _serialize_param(arg_value)
                if param_value is not None:
                    mlflow.log_param(f"arg_{arg_name}", param_value)

        # Логируем именованные аргументы
        for key, value in kwargs.items():
            if not key.startswith("_"):
                param_value = _serialize_param(value)
                if param_value is not None:
                    mlflow.log_param(f"kwarg_{key}", param_value)

    except Exception as e:
        logger.warning(f"Не удалось залогировать параметры функции: {e}")


def _serialize_param(value: Any) -> Optional[str]:
    """Сериализует параметр для логирования."""
    if value is None:
        return "None"
    elif isinstance(value, (str, int, float, bool)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        if len(value) < 10:  # Логируем только короткие списки
            return str(value)
        else:
            return f"List/Tuple of length {len(value)}"
    elif isinstance(value, dict):
        if len(value) < 5:  # Логируем только маленькие словари
            return json.dumps(value, default=str)[:500]  # Ограничиваем длину
        else:
            return f"Dict with {len(value)} keys"
    elif hasattr(value, "__class__"):
        return f"Object of type {type(value).__name__}"
    else:
        return str(type(value))


def _log_metrics_from_dict(result_dict: dict):
    """Логирует метрики из словаря результатов."""
    for key, value in result_dict.items():
        if isinstance(value, (int, float, np.number)):
            mlflow.log_metric(key, float(value))
        elif isinstance(value, dict):  # Вложенные метрики
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (int, float, np.number)):
                    mlflow.log_metric(f"{key}_{nested_key}", float(nested_value))


def _log_sklearn_model(model: BaseEstimator, model_name: str):
    """Логирует sklearn модель."""
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"models/{model_name}",
            registered_model_name=f"{model_name}_model",
        )
        logger.info(f"Модель {model_name} залогирована в MLflow")
    except Exception as e:
        logger.warning(f"Не удалось залогировать модель {model_name}: {e}")


def _log_function_artifacts(result: Any):
    """Логирует артефакты из результата функции."""
    # Если результат - путь к файлу
    if isinstance(result, (str, Path)):
        path = Path(result)
        if path.exists() and path.is_file():
            mlflow.log_artifact(str(path))

    # Если результат - DataFrame, сохраняем как CSV
    elif isinstance(result, pd.DataFrame):
        temp_path = f"temp_dataframe_{int(time.time())}.csv"
        result.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path)
        Path(temp_path).unlink(missing_ok=True)


def _log_dataframe_info(
    df: pd.DataFrame,
    name: str,
    log_shape: bool,
    log_dtypes: bool,
    log_missing: bool,
    log_stats: bool,
):
    """Логирует информацию о DataFrame."""
    prefix = f"{name}_"

    if log_shape:
        mlflow.log_param(f"{prefix}rows", df.shape[0])
        mlflow.log_param(f"{prefix}columns", df.shape[1])

    if log_dtypes:
        dtype_counts = df.dtypes.value_counts().to_dict()
        for dtype, count in dtype_counts.items():
            mlflow.log_param(f"{prefix}dtype_{dtype}", count)

    if log_missing:
        missing_count = df.isnull().sum().sum()
        missing_percentage = (missing_count / (df.shape[0] * df.shape[1])) * 100
        mlflow.log_metric(f"{prefix}missing_values", missing_count)
        mlflow.log_metric(f"{prefix}missing_percentage", missing_percentage)

    if log_stats:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            mlflow.log_metric(f"{prefix}numeric_columns", len(numeric_columns))

            # Базовая статистика для числовых колонок
            stats = df[numeric_columns].describe()
            mlflow.log_metric(f"{prefix}mean_of_means", stats.loc["mean"].mean())
            mlflow.log_metric(f"{prefix}mean_of_stds", stats.loc["std"].mean())


# Удобные составные декораторы


def track_ml_experiment(
    experiment_name: str,
    run_name: Optional[str] = None,
    auto_log: bool = True,
    log_dataset: bool = True,
    save_model: bool = True,
):
    """
    Комбинированный декоратор для полного трекинга ML эксперимента.

    Args:
        experiment_name: Название эксперимента
        run_name: Название запуска
        auto_log: Включить автологирование
        log_dataset: Логировать информацию о данных
        save_model: Сохранять модель
    """

    def decorator(func: Callable) -> Callable:
        # Применяем множественные декораторы
        decorated_func = func

        if log_dataset:
            decorated_func = log_dataset_info()(decorated_func)

        decorated_func = log_execution_time(decorated_func)
        decorated_func = handle_exceptions()(decorated_func)
        decorated_func = mlflow_track(
            experiment_name=experiment_name,
            run_name=run_name,
            auto_log=auto_log,
            log_model=save_model,
        )(decorated_func)

        return decorated_func

    return decorator


def quick_experiment(experiment_name: str):
    """Быстрый декоратор для простых экспериментов."""
    return track_ml_experiment(
        experiment_name=experiment_name,
        auto_log=True,
        log_dataset=True,
        save_model=True,
    )
