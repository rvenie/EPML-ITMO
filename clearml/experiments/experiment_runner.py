#!/usr/bin/env python3
"""
Скрипт запуска экспериментов ClearML
Автоматизированные эксперименты с различными параметрами и алгоритмами
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from clearml import Task

# Добавляем путь к корневой директории проекта
sys.path.append(str(Path(__file__).parent.parent.parent))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiments.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class ClearMLExperimentRunner:
    """Запускатель экспериментов для ClearML."""

    def __init__(self, project_name: str = "ResearchHub"):
        """
        Инициализация запускателя экспериментов.

        Args:
            project_name: Имя проекта в ClearML
        """
        self.project_name = project_name
        self.base_params_file = "params.yaml"

    def load_base_params(self) -> dict[str, Any]:
        """Загружает базовые параметры из файла."""
        try:
            with open(self.base_params_file) as f:
                params = yaml.safe_load(f)
            logger.info(f"Загружены базовые параметры из {self.base_params_file}")
            return params
        except Exception as e:
            logger.error(f"Ошибка загрузки параметров: {e}")
            raise

    def create_experiment_variants(self) -> list[dict[str, Any]]:
        """
        Создает варианты экспериментов с различными параметрами.

        Returns:
            Список конфигураций экспериментов
        """
        base_params = self.load_base_params()

        # Варианты алгоритмов для экспериментов
        algorithm_variants = [
            {
                "name": "RandomForest_Default",
                "algorithm": "RandomForestClassifier",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                },
            },
            {
                "name": "RandomForest_Optimized",
                "algorithm": "RandomForestClassifier",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                },
            },
            {
                "name": "LogisticRegression_Default",
                "algorithm": "LogisticRegression",
                "params": {
                    "C": 1.0,
                    "max_iter": 1000,
                    "solver": "liblinear",
                },
            },
            {
                "name": "LogisticRegression_Regularized",
                "algorithm": "LogisticRegression",
                "params": {
                    "C": 0.1,
                    "max_iter": 2000,
                    "solver": "liblinear",
                },
            },
        ]

        # Создаем конфигурации экспериментов
        experiments = []
        for variant in algorithm_variants:
            experiment_params = base_params.copy()

            # Обновляем параметры алгоритма
            experiment_params["train"]["algorithm"] = variant["algorithm"]

            # Обновляем специфичные параметры алгоритма
            if variant["algorithm"] == "RandomForestClassifier":
                experiment_params["train"]["random_forest"].update(variant["params"])
            elif variant["algorithm"] == "LogisticRegression":
                experiment_params["train"]["logistic_regression"].update(
                    variant["params"]
                )

            experiments.append(
                {
                    "name": variant["name"],
                    "params": experiment_params,
                    "description": f"Эксперимент с {variant['algorithm']}: {variant['name']}",
                }
            )

        return experiments

    def run_single_experiment(
        self, experiment_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Запускает один эксперимент.

        Args:
            experiment_config: Конфигурация эксперимента

        Returns:
            Результаты эксперимента
        """
        experiment_name = experiment_config["name"]
        logger.info(f"Запуск эксперимента: {experiment_name}")

        # Инициализация ClearML Task
        task = Task.init(
            project_name=self.project_name,
            task_name=f"Experiment_{experiment_name}",
            task_type=Task.TaskTypes.training,
        )

        # Подключаем параметры эксперимента
        params = experiment_config["params"]
        task.connect(params, name="experiment_params")

        # Добавляем теги
        task.add_tags(
            [
                experiment_name,
                params["train"]["algorithm"],
                f"test_size_{params['train']['test_size']}",
                "automated_experiment",
            ]
        )

        try:
            # Создаем временный файл параметров для этого эксперимента
            temp_params_file = f"temp_params_{experiment_name}.yaml"
            with open(temp_params_file, "w") as f:
                yaml.dump(params, f, default_flow_style=False)

            # Запускаем обучение модели
            from scripts.train_model import train_model

            model_output = f"models/experiment_{experiment_name}_model.pkl"
            metrics_output = f"models/experiment_{experiment_name}_metrics.json"

            # Проверяем существование входных данных
            input_file = params.get("data", {}).get(
                "processed", "data/processed/publications_processed.csv"
            )
            if not os.path.exists(input_file):
                logger.warning(
                    f"Файл данных не найден: {input_file}, используем тестовые данные"
                )
                input_file = "data/test/test_publications.csv"

            model, metrics = train_model(
                data_file=input_file,
                params_file=temp_params_file,
                model_output=model_output,
                metrics_output=metrics_output,
            )

            # Загружаем метрики
            with open(metrics_output) as f:
                experiment_metrics = json.load(f)

            # Логируем результаты в ClearML
            logger_clearml = task.get_logger()

            test_metrics = experiment_metrics["test_metrics"]
            for metric_name, value in test_metrics.items():
                logger_clearml.report_scalar(
                    "Final Results", metric_name, float(value), iteration=0
                )

            # Сохраняем артефакты
            task.upload_artifact("final_model", artifact_object=model_output)
            task.upload_artifact("experiment_metrics", artifact_object=metrics_output)

            # Очищаем временный файл
            os.remove(temp_params_file)

            results = {
                "experiment_name": experiment_name,
                "task_id": task.id,
                "metrics": test_metrics,
                "status": "completed",
                "completion_time": datetime.now().isoformat(),
            }

            logger.info(f"Эксперимент {experiment_name} завершен успешно")
            logger.info(f"Точность: {test_metrics.get('accuracy', 0):.4f}")

            return results

        except Exception as e:
            logger.error(f"Ошибка в эксперименте {experiment_name}: {e}")

            # Отмечаем задачу как неудачную
            task.mark_failed(status_reason=str(e))

            return {
                "experiment_name": experiment_name,
                "task_id": task.id,
                "status": "failed",
                "error": str(e),
                "completion_time": datetime.now().isoformat(),
            }

    def run_experiment_suite(self) -> dict[str, Any]:
        """
        Запускает серию экспериментов.

        Returns:
            Общие результаты всех экспериментов
        """
        logger.info("Запуск серии экспериментов ClearML...")

        # Создаем главную задачу для серии экспериментов
        suite_task = Task.init(
            project_name=self.project_name,
            task_name="Experiment Suite",
            task_type=Task.TaskTypes.optimizer,
        )

        experiments = self.create_experiment_variants()
        results = []

        suite_logger = suite_task.get_logger()

        for i, experiment in enumerate(experiments):
            logger.info(f"Прогресс: {i + 1}/{len(experiments)}")

            result = self.run_single_experiment(experiment)
            results.append(result)

            # Логируем промежуточные результаты
            if result["status"] == "completed":
                accuracy = result["metrics"].get("accuracy", 0)
                suite_logger.report_scalar(
                    "Experiment Results", "accuracy", accuracy, iteration=i
                )

        # Анализ результатов
        successful_results = [r for r in results if r["status"] == "completed"]
        failed_results = [r for r in results if r["status"] == "failed"]

        if successful_results:
            best_experiment = max(
                successful_results, key=lambda x: x["metrics"].get("accuracy", 0)
            )

            # Логируем лучший результат
            suite_logger.report_scalar(
                "Suite Summary",
                "best_accuracy",
                best_experiment["metrics"]["accuracy"],
                iteration=0,
            )

            logger.info(f"Лучший эксперимент: {best_experiment['experiment_name']}")
            logger.info(
                f"Лучшая точность: {best_experiment['metrics']['accuracy']:.4f}"
            )

        # Сводка результатов
        suite_summary = {
            "suite_task_id": suite_task.id,
            "total_experiments": len(experiments),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "best_experiment": best_experiment["experiment_name"]
            if successful_results
            else None,
            "best_accuracy": best_experiment["metrics"]["accuracy"]
            if successful_results
            else None,
            "execution_time": datetime.now().isoformat(),
            "detailed_results": results,
        }

        # Сохраняем сводку
        summary_file = (
            f"reports/experiment_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs("reports", exist_ok=True)

        with open(summary_file, "w") as f:
            json.dump(suite_summary, f, indent=2, ensure_ascii=False)

        suite_task.upload_artifact(
            "experiment_suite_summary", artifact_object=summary_file
        )

        logger.info(f"Серия экспериментов завершена. Сводка сохранена: {summary_file}")

        return suite_summary

    def compare_experiment_results(self, task_ids: list[str]) -> dict[str, Any]:
        """
        Сравнивает результаты нескольких экспериментов.

        Args:
            task_ids: Список ID задач для сравнения

        Returns:
            Результаты сравнения
        """
        logger.info(f"Сравнение экспериментов: {task_ids}")

        comparison_data = {
            "comparison_date": datetime.now().isoformat(),
            "experiments": {},
            "best_experiment": None,
            "best_metric_value": 0.0,
        }

        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)

                # Получаем метрики из логов
                scalars = task.get_reported_scalars()

                # Извлекаем основные метрики
                test_metrics = scalars.get("Test Metrics", {})
                accuracy = 0.0

                if "accuracy" in test_metrics:
                    accuracy = (
                        test_metrics["accuracy"]["y"][-1]
                        if test_metrics["accuracy"]["y"]
                        else 0.0
                    )

                experiment_info = {
                    "task_name": task.name,
                    "task_id": task_id,
                    "accuracy": accuracy,
                    "status": task.get_status(),
                    "created": str(task.data.created),
                }

                comparison_data["experiments"][task.name] = experiment_info

                # Отслеживаем лучший результат
                if accuracy > comparison_data["best_metric_value"]:
                    comparison_data["best_metric_value"] = accuracy
                    comparison_data["best_experiment"] = task.name

            except Exception as e:
                logger.error(f"Ошибка обработки задачи {task_id}: {e}")

        logger.info(f"Лучший эксперимент: {comparison_data['best_experiment']}")
        logger.info(f"Лучшая точность: {comparison_data['best_metric_value']:.4f}")

        return comparison_data


def main():
    """Главная функция для демонстрации."""
    try:
        logger.info("Демонстрация ClearML Experiment Runner")

        # Создаем экземпляр запускателя экспериментов
        runner = ClearMLExperimentRunner("ResearchHub")

        # Запускаем серию экспериментов
        logger.info("Запуск серии экспериментов...")
        suite_results = runner.run_experiment_suite()

        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ СЕРИИ ЭКСПЕРИМЕНТОВ")
        print("=" * 60)
        print(f"Всего экспериментов: {suite_results['total_experiments']}")
        print(f"Успешных: {suite_results['successful']}")
        print(f"Неудачных: {suite_results['failed']}")

        if suite_results["best_experiment"]:
            print(f"Лучший эксперимент: {suite_results['best_experiment']}")
            print(f"Лучшая точность: {suite_results['best_accuracy']:.4f}")

        print("=" * 60)

        logger.info("Демонстрация завершена успешно")

    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {e}")


if __name__ == "__main__":
    main()
