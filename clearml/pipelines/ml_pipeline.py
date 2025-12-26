#!/usr/bin/env python3
"""
ClearML ML Pipeline с полным трекингом экспериментов и управлением моделями
Включает автоматическое логирование, версионирование и сравнение экспериментов
"""

import json
import logging
import pickle  # nosec B403
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from clearml import Dataset, Model, PipelineController, Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClearMLExperimentTracker:
    """Система трекинга экспериментов и управления моделями ClearML."""

    def __init__(
        self, project_name: str = "ResearchHub", task_name: str = "ML Experiment"
    ):
        """
        Инициализация трекера экспериментов.

        Args:
            project_name: Имя проекта ClearML
            task_name: Имя задачи/эксперимента
        """
        self.project_name = project_name
        self.task_name = task_name
        self.task = None
        self.model = None

    def init_experiment(self, experiment_params: dict[str, Any]) -> Task:
        """
        Инициализация нового эксперимента с автоматическим логированием.

        Args:
            experiment_params: Параметры эксперимента

        Returns:
            ClearML Task объект
        """
        self.task = Task.init(
            project_name=self.project_name,
            task_name=f"{self.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=Task.TaskTypes.training,
            auto_connect_frameworks=True,
            auto_connect_arg_parser=True,
        )

        # Логируем параметры эксперимента
        for param_name, param_value in experiment_params.items():
            self.task.set_parameter(param_name, param_value)

        # Добавляем теги для группировки экспериментов
        self.task.add_tags(
            [
                f"model_type_{experiment_params.get('model_type', 'RandomForest')}",
                f"dataset_version_{experiment_params.get('dataset_version', 'v1')}",
            ]
        )

        logger.info(f"Эксперимент инициализирован: {self.task.id}")
        return self.task

    def log_dataset_info(self, dataset_path: str, dataset_stats: dict[str, Any]):
        """
        Логирование информации о датасете.

        Args:
            dataset_path: Путь к датасету
            dataset_stats: Статистики датасета
        """
        if not self.task:
            return

        # Логируем статистики датасета
        for stat_name, stat_value in dataset_stats.items():
            try:
                # Ensure value is numeric
                numeric_value = float(stat_value) if stat_value is not None else 0.0
                self.task.get_logger().report_scalar(
                    "Dataset Stats", stat_name, numeric_value, 0
                )
            except (ValueError, TypeError):
                # Skip non-numeric values
                pass

        # Try to register dataset in ClearML (optional, may fail on some configs)
        try:
            dataset = Dataset.create(
                dataset_name="research_publications",
                dataset_project=self.project_name,
                parent_datasets=None,
            )
            dataset.add_files(path=dataset_path)
            dataset.upload()
            dataset.finalize()
            logger.info(f"Датасет зарегистрирован: {dataset.id}")
        except Exception as e:
            logger.warning(f"Не удалось зарегистрировать датасет (не критично): {e}")
            # Continue without dataset registration - stats are already logged

    def log_training_metrics(self, metrics: dict[str, float], epoch: int = 0):
        """
        Логирование метрик обучения.

        Args:
            metrics: Словарь с метриками
            epoch: Номер эпохи/итерации
        """
        if not self.task:
            return

        logger_obj = self.task.get_logger()

        for metric_name, metric_value in metrics.items():
            logger_obj.report_scalar(
                "Training Metrics", metric_name, metric_value, epoch
            )

        logger.info(f"Метрики логированы для эпохи {epoch}: {metrics}")

    def register_model(
        self,
        model_data: dict[str, Any],
        model_path: str,
        model_metadata: dict[str, Any],
    ) -> Model | None:
        """
        Регистрация модели с версионированием и метаданными.

        Args:
            model_data: Данные модели (модель + векторизатор)
            model_path: Путь для сохранения модели
            model_metadata: Метаданные модели

        Returns:
            ClearML Model объект или None
        """
        if not self.task:
            return None

        # Сохраняем модель на диск
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        # Логируем метаданные как параметры модели
        for meta_key, meta_value in model_metadata.items():
            try:
                self.task.set_parameter(f"model_metadata/{meta_key}", meta_value)
            except Exception:
                pass  # nosec B110 - metadata logging is non-critical

        # Регистрируем модель через OutputModel (более стабильный API)
        try:
            from clearml import OutputModel

            model_name = (
                f"research_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            output_model = OutputModel(
                task=self.task,
                framework="scikit-learn",
                name=model_name,
            )

            # Загружаем веса модели
            output_model.update_weights(
                weights_filename=model_path,
                auto_delete_file=False,
            )

            # Добавляем метаданные
            output_model.update_design(
                config_dict={
                    "accuracy": model_metadata.get("accuracy"),
                    "f1_score": model_metadata.get("f1_score"),
                    "train_samples": model_metadata.get("train_samples"),
                    "training_date": model_metadata.get("training_date"),
                }
            )

            logger.info(f"Модель зарегистрирована: {output_model.id}")
            return output_model

        except Exception as e:
            logger.warning(f"Не удалось зарегистрировать модель через OutputModel: {e}")
            # Fallback: просто загружаем как артефакт
            try:
                self.task.upload_artifact(
                    name="trained_model",
                    artifact_object=model_path,
                )
                logger.info(f"Модель загружена как артефакт: {model_path}")
            except Exception as e2:
                logger.warning(f"Не удалось загрузить модель как артефакт: {e2}")
            return None

    def compare_with_baseline(
        self, current_metrics: dict[str, float], baseline_model_id: str = None
    ):
        """
        Сравнение текущего эксперимента с базовой моделью.

        Args:
            current_metrics: Метрики текущей модели
            baseline_model_id: ID базовой модели для сравнения
        """
        if not self.task or not baseline_model_id:
            return

        try:
            # Получаем базовую модель
            baseline_model = Model(model_id=baseline_model_id)
            baseline_task = Task.get_task(task_id=baseline_model.task)

            # Получаем метрики базовой модели (упрощенная версия)
            baseline_accuracy = (
                baseline_task.get_parameter("model_metadata/accuracy") or 0.0
            )
            current_accuracy = current_metrics.get("accuracy", 0.0)

            # Логируем сравнение
            improvement = current_accuracy - float(baseline_accuracy)
            self.task.get_logger().report_scalar(
                "Model Comparison", "accuracy_improvement", improvement, 0
            )
            self.task.get_logger().report_scalar(
                "Model Comparison", "baseline_accuracy", float(baseline_accuracy), 0
            )

            logger.info(
                f"Сравнение с базовой моделью: улучшение точности на {improvement:.4f}"
            )

        except Exception as e:
            logger.warning(f"Не удалось сравнить с базовой моделью: {e}")


class MLPipeline:
    """Основной ML пайплайн с интеграцией ClearML."""

    def __init__(self, project_name: str = "ResearchHub"):
        """
        Инициализация пайплайна.

        Args:
            project_name: Имя проекта ClearML
        """
        self.project_name = project_name
        self.experiment_tracker = None

    def run_training_experiment(
        self, input_file: str = "data/processed/publications_processed.csv"
    ) -> dict[str, Any]:
        """
        Запуск полного эксперимента обучения с трекингом.

        Args:
            input_file: Путь к входным данным

        Returns:
            Результаты эксперимента
        """
        # Параметры эксперимента
        experiment_params = {
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_features": 1000,
            "test_size": 0.2,
            "random_state": 42,
            "dataset_version": "v1.0",
        }

        # Инициализация трекера экспериментов
        self.experiment_tracker = ClearMLExperimentTracker(
            self.project_name, "ML Training"
        )
        task = self.experiment_tracker.init_experiment(experiment_params)

        try:
            # 1. Загрузка и анализ данных
            logger.info("Загрузка данных...")
            df = self._load_or_create_data(input_file)

            # Determine target column for stats
            target_col = None
            for col in [
                "category_encoded",
                "label",
                "target",
                "abstract_category",
                "citation_category",
            ]:
                if col in df.columns:
                    target_col = col
                    break

            # Статистики датасета
            classes_count = 2  # default
            if target_col and target_col in df.columns:
                try:
                    classes_count = int(df[target_col].nunique())
                except (ValueError, TypeError):
                    classes_count = 2

            dataset_stats = {
                "total_samples": len(df),
                "features_count": df.shape[1] - 1,
                "classes_count": classes_count,
                "missing_values": int(df.isnull().sum().sum()),
            }

            self.experiment_tracker.log_dataset_info(input_file, dataset_stats)

            # 2. Подготовка данных
            logger.info("Подготовка данных для обучения...")
            # Combine text columns for vectorization
            text_columns = []
            if "title" in df.columns:
                text_columns.append(df["title"].fillna(""))
            if "summary" in df.columns:
                text_columns.append(df["summary"].fillna(""))
            if "abstract" in df.columns:
                text_columns.append(df["abstract"].fillna(""))

            if text_columns:
                X_text = text_columns[0]  # noqa: N806
                for col in text_columns[1:]:
                    X_text = X_text + " " + col  # noqa: N806
            else:
                # Fallback: use first string column
                X_text = df.iloc[:, 0].astype(str)  # noqa: N806

            # Get target variable
            if "category_encoded" in df.columns:
                y = df["category_encoded"]
            elif "abstract_category" in df.columns:
                # Encode categorical target
                from sklearn.preprocessing import LabelEncoder

                le = LabelEncoder()
                y = le.fit_transform(df["abstract_category"])
            elif "label" in df.columns:
                y = df["label"]
            elif "target" in df.columns:
                y = df["target"]
            else:
                y = df.iloc[:, -1]

            # Векторизация
            vectorizer = TfidfVectorizer(
                max_features=experiment_params["max_features"], stop_words="english"
            )
            X = vectorizer.fit_transform(X_text).toarray()  # noqa: N806

            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
                X,
                y,
                test_size=experiment_params["test_size"],
                random_state=experiment_params["random_state"],
            )

            # 3. Обучение модели
            logger.info("Обучение модели...")
            model = RandomForestClassifier(
                n_estimators=experiment_params["n_estimators"],
                random_state=experiment_params["random_state"],
            )
            model.fit(X_train, y_train)

            # 4. Оценка модели
            logger.info("Оценка модели...")
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Вычисление метрик
            train_metrics = {
                "accuracy": accuracy_score(y_train, y_pred_train),
                "f1_score": f1_score(y_train, y_pred_train, average="weighted"),
            }

            test_metrics = {
                "accuracy": accuracy_score(y_test, y_pred_test),
                "f1_score": f1_score(y_test, y_pred_test, average="weighted"),
            }

            # Логирование метрик
            self.experiment_tracker.log_training_metrics(train_metrics, epoch=0)
            self.experiment_tracker.log_training_metrics(test_metrics, epoch=1)

            # 5. Регистрация модели
            logger.info("Регистрация модели...")
            model_data = {"model": model, "vectorizer": vectorizer}
            model_path = f"models/research_classifier_{task.id}.pkl"

            model_metadata = {
                "accuracy": test_metrics["accuracy"],
                "f1_score": test_metrics["f1_score"],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X.shape[1],
                "training_date": datetime.now().isoformat(),
            }

            registered_model = self.experiment_tracker.register_model(
                model_data, model_path, model_metadata
            )

            # 6. Сохранение отчета
            self._save_experiment_report(test_metrics, model_metadata, task.id)

            logger.info("Эксперимент завершен успешно")
            return {
                "task_id": task.id,
                "model_id": registered_model.id if registered_model else None,
                "metrics": test_metrics,
                "model_path": model_path,
            }

        except Exception as e:
            import traceback

            logger.error(f"Ошибка в эксперименте: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if task:
                task.mark_failed()
            raise

    def _load_or_create_data(self, input_file: str) -> pd.DataFrame:
        """Загрузка данных или создание тестовых данных."""
        if Path(input_file).exists():
            return pd.read_csv(input_file)

        # Создание тестовых данных
        logger.info("Создание тестовых данных...")
        test_data = {
            "title": [
                "Machine Learning Applications in Medical Imaging",
                "Deep Learning for Natural Language Processing",
                "Computer Vision in Autonomous Vehicles",
                "Reinforcement Learning in Robotics",
                "Neural Networks for Time Series Prediction",
            ],
            "summary": [
                "Research on ML applications in medical image analysis and diagnosis",
                "Study of deep learning techniques for NLP tasks and language modeling",
                "Investigation of computer vision methods for autonomous driving",
                "Analysis of RL algorithms for robotic control and navigation",
                "Development of neural network models for time series forecasting",
            ],
            "category_encoded": [0, 1, 2, 3, 1],
        }

        df = pd.DataFrame(test_data)
        Path(input_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(input_file, index=False)

        logger.info(f"Тестовые данные созданы: {input_file}")
        return df

    def _save_experiment_report(
        self, metrics: dict[str, float], metadata: dict[str, Any], task_id: str
    ):
        """Сохранение отчета об эксперименте."""
        report = {
            "experiment_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "metadata": metadata,
            "model_performance": {
                "accuracy_threshold_passed": metrics.get("accuracy", 0) > 0.7,
                "f1_threshold_passed": metrics.get("f1_score", 0) > 0.7,
                "overall_quality": "good"
                if all(
                    [metrics.get("accuracy", 0) > 0.7, metrics.get("f1_score", 0) > 0.7]
                )
                else "needs_improvement",
            },
        }

        report_path = Path(f"reports/experiment_report_{task_id}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Отчет сохранен: {report_path}")

    def create_pipeline(self) -> PipelineController:
        """Создание ClearML пайплайна для автоматического выполнения."""
        logger.info("Создание автоматического пайплайна...")

        pipe = PipelineController(
            name="Research ML Pipeline",
            project=self.project_name,
            version="1.0.0",
            add_pipeline_tags=True,
        )

        # Единый этап - полный ML эксперимент
        pipe.add_step(
            name="ml_experiment",
            base_task_project=self.project_name,
            base_task_name="ML Training",
            parameter_override={
                "input_file": "data/processed/publications_processed.csv",
                "project_name": self.project_name,
            },
            pre_execute_callback=lambda controller, node, params: logger.info(
                f"Запуск эксперимента: {node.name}"
            ),
            post_execute_callback=lambda controller, node, params: logger.info(
                f"Эксперимент завершен: {node.name}"
            ),
        )

        return pipe

    def run_pipeline(self, queue_name: str = "default") -> str:
        """Запуск пайплайна в очереди."""
        try:
            logger.info("Запуск ML пайплайна...")

            pipe = self.create_pipeline()
            pipe.start(queue=queue_name)

            logger.info(f"Пайплайн запущен: {pipe.id}")
            return pipe.id

        except Exception as e:
            logger.error(f"Ошибка запуска пайплайна: {e}")
            raise


def main():
    """Главная функция для демонстрации системы."""
    logger.info("=== Запуск ClearML ML Pipeline ===")

    try:
        # Создание пайплайна
        pipeline = MLPipeline("ResearchHub")

        # Запуск эксперимента
        result = pipeline.run_training_experiment()

        logger.info("=== Результаты эксперимента ===")
        logger.info(f"ID задачи: {result['task_id']}")
        logger.info(f"ID модели: {result['model_id']}")
        logger.info(f"Метрики: {result['metrics']}")
        logger.info("Веб интерфейс: http://localhost:8080")

    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
        raise


if __name__ == "__main__":
    main()
