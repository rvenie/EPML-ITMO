#!/usr/bin/env python3
"""
Менеджер моделей для ClearML
Управление версионированием, регистрацией и сравнением моделей
"""

import logging
from datetime import datetime
from typing import Any

from clearml import Model

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class ClearMLModelManager:
    """Менеджер моделей для ClearML с автоматическим версионированием."""

    def __init__(self, project_name: str = "ResearchHub"):
        """
        Инициализация менеджера моделей.

        Args:
            project_name: Имя проекта в ClearML
        """
        self.project_name = project_name
        logger.info(f"Инициализация ClearML Model Manager для проекта: {project_name}")

    def register_model(
        self,
        model_path: str,
        model_name: str,
        task_id: str,
        metadata: dict[str, Any],
        tags: list[str] | None = None,
        auto_version: bool = True,
    ) -> Model:
        """
        Регистрирует модель в ClearML с автоматическим версионированием.

        Args:
            model_path: Путь к файлу модели
            model_name: Имя модели
            task_id: ID задачи ClearML
            metadata: Метаданные модели
            tags: Теги для модели
            auto_version: Автоматическое версионирование

        Returns:
            Объект зарегистрированной модели ClearML
        """
        try:
            # Автоматическое версионирование
            if auto_version:
                existing_models = Model.query(
                    project_name=self.project_name, model_name=model_name
                )
                if existing_models:
                    version_num = len(existing_models) + 1
                    model_name = f"{model_name}_v{version_num}"
                    metadata["version"] = version_num
                    metadata["base_model"] = model_name.replace(f"_v{version_num}", "")
                else:
                    metadata["version"] = 1
                    metadata["base_model"] = model_name

            logger.info(f"Регистрация модели: {model_name}")

            # Расширенные метаданные
            enhanced_metadata = {
                "registration_date": datetime.now().isoformat(),
                "project": self.project_name,
                "framework": "scikit-learn",
                "task_id": task_id,
                **metadata,
            }

            # Создаем модель в ClearML
            model = Model(
                uri=model_path,
                name=model_name,
                project=self.project_name,
                framework="scikit-learn",
                task_id=task_id,
            )

            # Добавляем расширенные метаданные
            for key, value in enhanced_metadata.items():
                model.set_metadata(key, str(value))

            # Добавляем теги
            if tags:
                for tag in tags:
                    model.add_tag(tag)

            # Добавляем системные теги
            model.add_tag(f"algorithm_{enhanced_metadata.get('algorithm', 'unknown')}")

            # Точность из метрик
            test_accuracy = enhanced_metadata.get("test_accuracy", 0)
            if isinstance(test_accuracy, int | float) and test_accuracy > 0:
                model.add_tag(f"accuracy_{test_accuracy:.3f}")

            # Версия
            version = enhanced_metadata.get("version", 1)
            model.add_tag(f"v{version}")

            # Статус качества модели
            if isinstance(test_accuracy, int | float):
                if test_accuracy >= 0.9:
                    model.add_tag("high_quality")
                elif test_accuracy >= 0.8:
                    model.add_tag("medium_quality")
                else:
                    model.add_tag("low_quality")

            # Публикуем модель
            model.publish()

            logger.info(f"Модель успешно зарегистрирована: {model.id}")
            return model

        except Exception as e:
            logger.error(f"Ошибка регистрации модели: {e}")
            raise

    def create_model_version(
        self,
        base_model_name: str,
        new_model_path: str,
        task_id: str,
        performance_metrics: dict[str, float],
        improvement_notes: str = "",
    ) -> Model:
        """
        Создает новую версию существующей модели.

        Args:
            base_model_name: Базовое имя модели
            new_model_path: Путь к новой версии модели
            task_id: ID задачи обучения
            performance_metrics: Метрики производительности
            improvement_notes: Заметки об улучшениях

        Returns:
            Новая версия модели
        """
        try:
            # Получаем следующий номер версии
            existing_models = Model.query(
                project_name=self.project_name, model_name=base_model_name
            )

            next_version = len(existing_models) + 1
            versioned_name = f"{base_model_name}_v{next_version}"

            logger.info(f"Создание версии {next_version} для модели {base_model_name}")

            # Метаданные для новой версии
            metadata = {
                "base_model": base_model_name,
                "version": next_version,
                "created_date": datetime.now().isoformat(),
                "improvement_notes": improvement_notes,
                "performance": performance_metrics,
            }

            # Регистрируем новую версию
            model = self.register_model(
                model_path=new_model_path,
                model_name=versioned_name,
                task_id=task_id,
                metadata=metadata,
                tags=[f"v{next_version}", "auto_version"],
            )

            return model

        except Exception as e:
            logger.error(f"Ошибка создания версии модели: {e}")
            raise

    def compare_models(self, model_names: list[str]) -> dict[str, Any]:
        """
        Сравнивает производительность нескольких моделей.

        Args:
            model_names: Список имен моделей для сравнения

        Returns:
            Словарь с результатами сравнения
        """
        try:
            logger.info(f"Сравнение моделей: {model_names}")

            comparison_results = {
                "models": {},
                "comparison_date": datetime.now().isoformat(),
                "best_model": None,
                "best_accuracy": 0.0,
            }

            for model_name in model_names:
                # Найти последнюю версию модели
                models = Model.query(
                    project_name=self.project_name, model_name=model_name
                )

                if not models:
                    logger.warning(f"Модель {model_name} не найдена")
                    continue

                latest_model = models[0]  # Последняя модель
                metadata = latest_model.get_all_metadata()

                # Извлекаем метрики производительности
                performance = metadata.get("performance", {})
                test_accuracy = float(performance.get("test_accuracy", 0.0))

                comparison_results["models"][model_name] = {
                    "model_id": latest_model.id,
                    "accuracy": test_accuracy,
                    "f1_score": float(performance.get("test_f1_score", 0.0)),
                    "precision": float(performance.get("test_precision", 0.0)),
                    "recall": float(performance.get("test_recall", 0.0)),
                    "created_date": metadata.get("created_date"),
                    "version": metadata.get("version", "1.0.0"),
                }

                # Отслеживаем лучшую модель
                if test_accuracy > comparison_results["best_accuracy"]:
                    comparison_results["best_accuracy"] = test_accuracy
                    comparison_results["best_model"] = model_name

            logger.info(
                f"Лучшая модель: {comparison_results['best_model']} "
                f"(accuracy: {comparison_results['best_accuracy']:.4f})"
            )

            return comparison_results

        except Exception as e:
            logger.error(f"Ошибка сравнения моделей: {e}")
            raise

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """
        Получает подробную информацию о модели.

        Args:
            model_name: Имя модели

        Returns:
            Информация о модели
        """
        try:
            models = Model.query(project_name=self.project_name, model_name=model_name)

            if not models:
                raise ValueError(f"Модель {model_name} не найдена")

            model = models[0]
            metadata = model.get_all_metadata()

            info = {
                "model_id": model.id,
                "name": model.name,
                "framework": model.framework,
                "uri": model.uri,
                "tags": model.tags,
                "created": model.created,
                "last_update": model.last_update,
                "metadata": metadata,
            }

            return info

        except Exception as e:
            logger.error(f"Ошибка получения информации о модели: {e}")
            raise

    def list_models(self) -> list[dict[str, Any]]:
        """
        Возвращает список всех моделей в проекте.

        Returns:
            Список моделей с базовой информацией
        """
        try:
            models = Model.query(project_name=self.project_name)

            models_info = []
            for model in models:
                metadata = model.get_all_metadata()
                performance = metadata.get("performance", {})

                model_info = {
                    "name": model.name,
                    "id": model.id,
                    "created": model.created,
                    "accuracy": performance.get("test_accuracy", 0.0),
                    "version": metadata.get("version", "1.0.0"),
                    "tags": model.tags,
                }
                models_info.append(model_info)

            # Сортируем по дате создания (новые первые)
            models_info.sort(key=lambda x: x["created"], reverse=True)

            logger.info(f"Найдено {len(models_info)} моделей в проекте")
            return models_info

        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            raise

    def auto_register_from_training(
        self,
        model_path: str,
        model_name: str,
        task_id: str,
        training_metrics: dict[str, float],
        model_params: dict[str, Any],
        training_data_info: dict[str, Any],
    ) -> Model:
        """
        Автоматически регистрирует модель после обучения.

        Args:
            model_path: Путь к файлу модели
            model_name: Базовое имя модели
            task_id: ID задачи обучения
            training_metrics: Метрики обучения
            model_params: Параметры модели
            training_data_info: Информация о данных обучения

        Returns:
            Зарегистрированная модель
        """
        try:
            logger.info(f"Автоматическая регистрация модели из обучения: {model_name}")

            # Полные метаданные модели
            metadata = {
                "algorithm": model_params.get("algorithm", "unknown"),
                "training_date": datetime.now().isoformat(),
                # Метрики производительности
                "test_accuracy": training_metrics.get("accuracy", 0),
                "test_f1_score": training_metrics.get("f1_score", 0),
                "test_precision": training_metrics.get("precision", 0),
                "test_recall": training_metrics.get("recall", 0),
                # Информация о валидации
                "cv_mean": training_metrics.get("cv_mean", 0),
                "cv_std": training_metrics.get("cv_std", 0),
                # Параметры модели
                "model_params": model_params,
                # Данные обучения
                "training_samples": training_data_info.get("training_samples", 0),
                "test_samples": training_data_info.get("test_samples", 0),
                "features_count": training_data_info.get("features", 0),
                "classes_count": training_data_info.get("classes", 0),
            }

            # Теги
            tags = ["auto_registered", "from_training"]
            if training_metrics.get("accuracy", 0) >= 0.9:
                tags.append("high_performance")

            # Регистрируем модель с автоматическим версионированием
            model = self.register_model(
                model_path=model_path,
                model_name=model_name,
                task_id=task_id,
                metadata=metadata,
                tags=tags,
                auto_version=True,
            )

            logger.info(f"Модель автоматически зарегистрирована: {model.name}")
            return model

        except Exception as e:
            logger.error(f"Ошибка автоматической регистрации модели: {e}")
            raise


def main():
    """Демонстрация работы менеджера моделей."""

    # Создаем экземпляр менеджера
    model_manager = ClearMLModelManager("ResearchHub")

    try:
        # Список всех моделей
        models = model_manager.list_models()
        print("\n=== Список моделей ===")
        if models:
            for model in models:
                print(f"Модель: {model['name']}")
                print(f"  ID: {model['id'][:8]}...")
                print(f"  Точность: {model['accuracy']:.4f}")
                print(f"  Версия: {model['version']}")
                print(f"  Создана: {model['created']}")
                print(f"  Теги: {', '.join(model['tags'])}")
                print("-" * 50)

            # Сравнение моделей (если есть)
            if len(models) >= 2:
                model_names = [model["name"] for model in models[:3]]  # Топ 3
                comparison = model_manager.compare_models(model_names)

                print("\n=== Сравнение моделей ===")
                print(f"Лучшая модель: {comparison['best_model']}")
                print(f"Лучшая точность: {comparison['best_accuracy']:.4f}")

                for name, metrics in comparison["models"].items():
                    print(f"\n{name}:")
                    print(f"  Точность: {metrics['accuracy']:.4f}")
                    print(f"  F1-score: {metrics['f1_score']:.4f}")
                    print(f"  Версия: {metrics['version']}")
        else:
            print("Модели не найдены. Сначала запустите обучение:")
            print("make clearml-experiments")

        print("\n=== Статистика проекта ===")
        print(f"Всего моделей: {len(models)}")
        if models:
            avg_accuracy = sum(m["accuracy"] for m in models) / len(models)
            print(f"Средняя точность: {avg_accuracy:.4f}")
            best_model = max(models, key=lambda x: x["accuracy"])
            print(f"Лучшая модель: {best_model['name']} ({best_model['accuracy']:.4f})")

    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {e}")


if __name__ == "__main__":
    main()
