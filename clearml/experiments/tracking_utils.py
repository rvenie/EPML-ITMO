#!/usr/bin/env python3
"""
ClearML Auto Logger - утилиты для автоматического логирования
Предоставляет удобный интерфейс для логирования экспериментов ML в ClearML
"""

import logging
import platform
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from clearml import Task
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class ClearMLAutoLogger:
    """Автоматический логгер для ClearML экспериментов."""

    def __init__(self, task: Task):
        """
        Инициализация автологгера.

        Args:
            task: ClearML Task объект
        """
        self.task = task
        self.logger = task.get_logger()

    def log_system_info(self) -> None:
        """Логирует системную информацию."""
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        }

        for key, value in system_info.items():
            self.task.connect_configuration(
                name=f"system_{key}",
                configuration=value,
            )

        logger.info(f"Logged system info: {system_info}")

    def log_model_params(self, model: Any) -> None:
        """
        Логирует параметры модели.

        Args:
            model: Обученная модель sklearn
        """
        if hasattr(model, "get_params"):
            params = model.get_params()
            self.task.connect(params, name="model_params")
            logger.info(f"Logged model parameters: {len(params)} params")

    def log_dataset_info(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str,  # noqa: N803
    ) -> None:
        """
        Логирует информацию о датасете.

        Args:
            X: Матрица признаков
            y: Целевой вектор
            split_name: Название сплита (train/test/val)
        """
        dataset_info = {
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "classes": len(np.unique(y)),
            "class_distribution": {
                str(cls): int(count)
                for cls, count in zip(*np.unique(y, return_counts=True), strict=False)
            },
        }

        self.task.connect(dataset_info, name=f"dataset_{split_name}")

        # Логируем распределение классов как серию
        for cls, count in dataset_info["class_distribution"].items():
            self.logger.report_scalar(
                f"Dataset {split_name}",
                f"class_{cls}_count",
                value=count,
                iteration=0,
            )

        logger.info(f"Logged dataset info for {split_name}: {dataset_info}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str] | None = None,
    ) -> None:
        """
        Логирует confusion matrix.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            class_names: Названия классов
        """
        cm = confusion_matrix(y_true, y_pred)

        # Создаем визуализацию
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names if class_names else "auto",
            yticklabels=class_names if class_names else "auto",
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # Логируем в ClearML
        self.logger.report_matplotlib_figure(
            title="Confusion Matrix",
            series="confusion_matrix",
            figure=plt.gcf(),
            iteration=0,
        )
        plt.close()

        logger.info("Logged confusion matrix")

    def log_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str] | None = None,
    ) -> None:
        """
        Логирует classification report.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            class_names: Названия классов
        """
        report_dict = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        # Логируем метрики для каждого класса
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    self.logger.report_scalar(
                        f"Classification Report - {class_name}",
                        metric_name,
                        value=value,
                        iteration=0,
                    )

        logger.info("Logged classification report")

    def log_feature_importance(
        self, model: Any, feature_names: list[str] | None = None
    ) -> None:
        """
        Логирует важность признаков (если доступно).

        Args:
            model: Обученная модель
            feature_names: Названия признаков
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            # Создаем визуализацию топ-20 признаков
            top_n = min(20, len(importances))
            indices = np.argsort(importances)[-top_n:]

            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), importances[indices])
            if feature_names and len(feature_names) == len(importances):
                plt.yticks(range(top_n), [feature_names[i] for i in indices])
            else:
                plt.yticks(range(top_n), [f"Feature {i}" for i in indices])
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importances")
            plt.tight_layout()

            # Логируем в ClearML
            self.logger.report_matplotlib_figure(
                title="Feature Importance",
                series="top_features",
                figure=plt.gcf(),
                iteration=0,
            )
            plt.close()

            # Логируем статистику важности
            self.logger.report_scalar(
                "Feature Importance",
                "mean",
                value=float(importances.mean()),
                iteration=0,
            )
            self.logger.report_scalar(
                "Feature Importance",
                "max",
                value=float(importances.max()),
                iteration=0,
            )

            logger.info("Logged feature importance")
        else:
            logger.info("Model doesn't have feature_importances_ attribute")

    def create_experiment_summary(self, metrics: dict[str, float]) -> dict[str, Any]:
        """
        Создает сводку эксперимента.

        Args:
            metrics: Словарь метрик

        Returns:
            Словарь с информацией об эксперименте
        """
        summary = {
            "experiment_id": self.task.id,
            "experiment_name": self.task.name,
            "project": self.task.get_project_name(),
            "metrics": metrics,
            "status": "completed",
        }

        self.task.connect(summary, name="experiment_summary")

        logger.info(f"Created experiment summary: {summary['experiment_id']}")

        return summary
