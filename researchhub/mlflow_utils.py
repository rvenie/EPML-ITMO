#!/usr/bin/env python3
"""
Утилиты для работы с MLflow экспериментами.
Предоставляет функции для поиска, фильтрации, сравнения и управления экспериментами.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.tracking
import pandas as pd
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowExperimentManager:
    """Класс для управления MLflow экспериментами."""

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """
        Инициализация менеджера экспериментов.

        Args:
            tracking_uri: URI для подключения к MLflow tracking server
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

    def create_experiment_if_not_exists(self, experiment_name: str) -> str:
        """
        Создает эксперимент если он не существует.

        Args:
            experiment_name: Название эксперимента

        Returns:
            ID эксперимента
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                return experiment.experiment_id
        except Exception:
            pass

        experiment_id = self.client.create_experiment(experiment_name)
        logger.info(
            f"Создан новый эксперимент: {experiment_name} (ID: {experiment_id})"
        )
        return experiment_id

    def list_experiments(self) -> List[Experiment]:
        """Возвращает список всех экспериментов."""
        return self.client.search_experiments()

    def get_experiment_runs(
        self,
        experiment_name: str,
        filter_string: str = "",
        order_by: List[str] = None,
        max_results: int = 1000,
    ) -> List[Run]:
        """
        Получает запуски эксперимента с фильтрацией.

        Args:
            experiment_name: Название эксперимента
            filter_string: Строка фильтрации MLflow
            order_by: Список полей для сортировки
            max_results: Максимальное количество результатов

        Returns:
            Список запусков
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Эксперимент '{experiment_name}' не найден")
            return []

        return self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or [],
            max_results=max_results,
        )

    def get_best_run(
        self, experiment_name: str, metric_name: str, maximize: bool = True
    ) -> Optional[Run]:
        """
        Находит лучший запуск по указанной метрике.

        Args:
            experiment_name: Название эксперимента
            metric_name: Название метрики
            maximize: True для максимизации, False для минимизации

        Returns:
            Лучший запуск или None
        """
        order_by = f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"
        runs = self.get_experiment_runs(
            experiment_name=experiment_name, order_by=[order_by], max_results=1
        )

        return runs[0] if runs else None

    def compare_runs(
        self, run_ids: List[str], metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Сравнивает запуски по указанным метрикам.

        Args:
            run_ids: Список ID запусков
            metrics: Список метрик для сравнения

        Returns:
            DataFrame с результатами сравнения
        """
        if not metrics:
            metrics = ["accuracy", "f1_score", "precision", "recall"]

        comparison_data = []

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                run_data = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                }

                # Добавляем метрики
                for metric in metrics:
                    metric_key = (
                        f"test_{metric}"
                        if f"test_{metric}" in run.data.metrics
                        else metric
                    )
                    run_data[metric] = run.data.metrics.get(metric_key, None)

                # Добавляем основные параметры
                run_data.update(
                    {
                        "algorithm": run.data.params.get("algorithm", "Unknown"),
                        "test_size": run.data.params.get("test_size", None),
                        "random_state": run.data.params.get("random_state", None),
                    }
                )

                comparison_data.append(run_data)

            except Exception as e:
                logger.error(f"Ошибка при получении данных для run {run_id}: {e}")

        return pd.DataFrame(comparison_data)

    def export_experiment_results(
        self, experiment_name: str, output_file: str, format: str = "csv"
    ):
        """
        Экспортирует результаты эксперимента в файл.

        Args:
            experiment_name: Название эксперимента
            output_file: Путь к выходному файлу
            format: Формат файла ("csv", "json", "excel")
        """
        runs = self.get_experiment_runs(experiment_name)

        if not runs:
            logger.warning(f"Нет запусков в эксперименте '{experiment_name}'")
            return

        # Собираем данные
        export_data = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "run_name": run.data.tags.get("mlflow.runName", ""),
            }

            # Добавляем все метрики
            for key, value in run.data.metrics.items():
                run_data[f"metric_{key}"] = value

            # Добавляем все параметры
            for key, value in run.data.params.items():
                run_data[f"param_{key}"] = value

            # Добавляем теги
            for key, value in run.data.tags.items():
                if not key.startswith("mlflow."):
                    run_data[f"tag_{key}"] = value

            export_data.append(run_data)

        # Экспортируем в нужном формате
        df = pd.DataFrame(export_data)

        if format.lower() == "csv":
            df.to_csv(output_file, index=False)
        elif format.lower() == "json":
            df.to_json(output_file, orient="records", indent=2)
        elif format.lower() == "excel":
            df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")

        logger.info(f"Результаты экспортированы в {output_file}")

    def delete_experiment(self, experiment_name: str, confirm: bool = False):
        """
        Удаляет эксперимент (только для тестирования).

        Args:
            experiment_name: Название эксперимента
            confirm: Подтверждение удаления
        """
        if not confirm:
            logger.warning("Для удаления эксперимента установите confirm=True")
            return

        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment:
            self.client.delete_experiment(experiment.experiment_id)
            logger.info(f"Эксперимент '{experiment_name}' удален")
        else:
            logger.warning(f"Эксперимент '{experiment_name}' не найден")


def mlflow_experiment(
    experiment_name: str,
    run_name: str = None,
    tags: Dict[str, Any] = None,
    auto_log: bool = True,
):
    """
    Декоратор для автоматического создания MLflow эксперимента.

    Args:
        experiment_name: Название эксперимента
        run_name: Название запуска
        tags: Теги для запуска
        auto_log: Включить автоматическое логирование
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Настраиваем MLflow
            mlflow.set_experiment(experiment_name)

            # Включаем автологирование если нужно
            if auto_log:
                mlflow.sklearn.autolog()

            with mlflow.start_run(run_name=run_name):
                # Устанавливаем теги
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)

                # Логируем параметры функции
                func_args = func.__code__.co_varnames[: func.__code__.co_argcount]
                for i, arg in enumerate(func_args):
                    if i < len(args):
                        mlflow.log_param(f"arg_{arg}", args[i])

                for key, value in kwargs.items():
                    mlflow.log_param(f"kwarg_{key}", value)

                # Выполняем функцию
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    mlflow.log_metric("execution_time", time.time() - start_time)
                    mlflow.set_tag("status", "success")
                    return result
                except Exception as e:
                    mlflow.log_metric("execution_time", time.time() - start_time)
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))
                    raise

        return wrapper

    return decorator


@contextmanager
def mlflow_run_context(
    experiment_name: str,
    run_name: str = None,
    tags: Dict[str, Any] = None,
    nested: bool = False,
):
    """
    Контекстный менеджер для MLflow запуска.

    Args:
        experiment_name: Название эксперимента
        run_name: Название запуска
        tags: Теги для запуска
        nested: Создать вложенный запуск
    """
    mlflow.set_experiment(experiment_name)

    start_time = time.time()
    with mlflow.start_run(run_name=run_name, nested=nested):
        try:
            # Устанавливаем теги
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            mlflow.set_tag("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))

            yield

            # Логируем время выполнения
            execution_time = time.time() - start_time
            mlflow.log_metric("execution_time", execution_time)
            mlflow.set_tag("status", "completed")

        except Exception as e:
            execution_time = time.time() - start_time
            mlflow.log_metric("execution_time", execution_time)
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            logger.error(f"Ошибка в MLflow запуске: {e}")
            raise


class MLflowModelRegistry:
    """Класс для работы с MLflow Model Registry."""

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

    def register_model(
        self, model_uri: str, model_name: str, tags: Dict[str, Any] = None
    ) -> str:
        """
        Регистрирует модель в Model Registry.

        Args:
            model_uri: URI модели
            model_name: Название модели
            tags: Теги для модели

        Returns:
            Версия модели
        """
        result = mlflow.register_model(model_uri, model_name)

        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    model_name, result.version, key, value
                )

        logger.info(f"Модель {model_name} версии {result.version} зарегистрирована")
        return result.version

    def get_latest_model_version(
        self, model_name: str, stage: str = "None"
    ) -> Optional[str]:
        """
        Получает последнюю версию модели.

        Args:
            model_name: Название модели
            stage: Стадия модели ("None", "Staging", "Production", "Archived")

        Returns:
            Версия модели или None
        """
        try:
            latest_versions = self.client.get_latest_versions(
                model_name, stages=[stage]
            )
            return latest_versions[0].version if latest_versions else None
        except Exception as e:
            logger.error(f"Ошибка при получении версии модели {model_name}: {e}")
            return None

    def transition_model_version_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False,
    ):
        """
        Переводит версию модели в новую стадию.

        Args:
            model_name: Название модели
            version: Версия модели
            stage: Новая стадия
            archive_existing_versions: Архивировать существующие версии
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

        logger.info(f"Модель {model_name} v{version} переведена в стадию {stage}")


def search_runs_by_metrics(
    experiment_name: str,
    metric_thresholds: Dict[str, Tuple[float, str]] = None,
    tracking_uri: str = "file:./mlruns",
) -> pd.DataFrame:
    """
    Ищет запуски по пороговым значениям метрик.

    Args:
        experiment_name: Название эксперимента
        metric_thresholds: Словарь {метрика: (значение, оператор)}
        tracking_uri: URI трекинга

    Returns:
        DataFrame с результатами поиска
    """
    manager = MLflowExperimentManager(tracking_uri)

    # Строим фильтр
    filter_conditions = []
    if metric_thresholds:
        for metric, (threshold, operator) in metric_thresholds.items():
            if operator in [">=", ">", "<=", "<", "="]:
                filter_conditions.append(f"metrics.{metric} {operator} {threshold}")

    filter_string = " and ".join(filter_conditions) if filter_conditions else ""

    runs = manager.get_experiment_runs(
        experiment_name=experiment_name,
        filter_string=filter_string,
        order_by=["metrics.test_accuracy DESC"],
    )

    # Конвертируем в DataFrame
    runs_data = []
    for run in runs:
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "status": run.info.status,
        }

        # Добавляем метрики
        for key, value in run.data.metrics.items():
            run_info[key] = value

        # Добавляем параметры
        for key, value in run.data.params.items():
            run_info[f"param_{key}"] = value

        runs_data.append(run_info)

    return pd.DataFrame(runs_data)


def create_experiment_summary_report(
    experiment_name: str,
    output_file: str = "experiment_summary.html",
    tracking_uri: str = "file:./mlruns",
):
    """
    Создает HTML отчет с результатами эксперимента.

    Args:
        experiment_name: Название эксперимента
        output_file: Путь к выходному HTML файлу
        tracking_uri: URI трекинга
    """
    manager = MLflowExperimentManager(tracking_uri)
    runs = manager.get_experiment_runs(experiment_name)

    if not runs:
        logger.warning(f"Нет запусков в эксперименте '{experiment_name}'")
        return

    # Создаем DataFrame для анализа
    df = manager.compare_runs([run.info.run_id for run in runs])

    # Создаем HTML отчет
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Отчет по эксперименту: {experiment_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #2E8B57; }}
            .best {{ background-color: #90EE90; }}
        </style>
    </head>
    <body>
        <h1>Отчет по эксперименту: {experiment_name}</h1>
        <h2>Общая информация</h2>
        <ul>
            <li>Всего запусков: {len(runs)}</li>
            <li>Дата создания отчета: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</li>
        </ul>

        <h2>Результаты экспериментов</h2>
        {df.to_html(classes="experiment-table", escape=False, index=False)}

        <h2>Лучшие результаты</h2>
    """

    # Добавляем информацию о лучших результатах
    if not df.empty:
        if "accuracy" in df.columns:
            best_accuracy = df.loc[df["accuracy"].idxmax()]
            html_content += f"<p><strong>Лучшая точность:</strong> {best_accuracy['accuracy']:.4f} (Run: {best_accuracy['run_name']})</p>"

        if "f1_score" in df.columns:
            best_f1 = df.loc[df["f1_score"].idxmax()]
            html_content += f"<p><strong>Лучший F1-score:</strong> {best_f1['f1_score']:.4f} (Run: {best_f1['run_name']})</p>"

    html_content += """
    </body>
    </html>
    """

    # Сохраняем отчет
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML отчет сохранен в {output_file}")


# Удобные функции для быстрого доступа
def quick_compare_algorithms(
    experiment_name: str, algorithms: List[str], tracking_uri: str = "file:./mlruns"
) -> pd.DataFrame:
    """Быстрое сравнение алгоритмов в эксперименте."""
    manager = MLflowExperimentManager(tracking_uri)

    all_runs = []
    for algorithm in algorithms:
        filter_string = f"params.algorithm = '{algorithm}'"
        runs = manager.get_experiment_runs(
            experiment_name=experiment_name, filter_string=filter_string, max_results=10
        )
        all_runs.extend([run.info.run_id for run in runs])

    return manager.compare_runs(all_runs)


def get_experiment_leaderboard(
    experiment_name: str,
    metric: str = "test_accuracy",
    top_n: int = 10,
    tracking_uri: str = "file:./mlruns",
) -> pd.DataFrame:
    """Получает топ результатов эксперимента."""
    manager = MLflowExperimentManager(tracking_uri)

    runs = manager.get_experiment_runs(
        experiment_name=experiment_name,
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n,
    )

    return manager.compare_runs([run.info.run_id for run in runs])
