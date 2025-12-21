#!/usr/bin/env python3
"""
Скрипт для проведения серии ML экспериментов с разными алгоритмами и параметрами.
Автоматически запускает 15+ экспериментов и логирует результаты в MLflow.
"""

import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from train_model import train_model

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiments.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_base_params(params_file: str = "params.yaml") -> Dict[str, Any]:
    """Загружает базовые параметры из файла."""
    with open(params_file, "r") as f:
        return yaml.safe_load(f)


def create_experiment_configs() -> List[Dict[str, Any]]:
    """Создает список конфигураций для различных экспериментов."""
    base_params = load_base_params()

    experiments = []

    # Эксперименты с Random Forest
    rf_configs = [
        # Базовая конфигурация
        {
            "name": "RF_baseline",
            "algorithm": "RandomForestClassifier",
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
        # Больше деревьев
        {
            "name": "RF_more_trees",
            "algorithm": "RandomForestClassifier",
            "params": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
        # Глубже деревья
        {
            "name": "RF_deeper",
            "algorithm": "RandomForestClassifier",
            "params": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
        # Более консервативные параметры
        {
            "name": "RF_conservative",
            "algorithm": "RandomForestClassifier",
            "params": {"n_estimators": 150, "max_depth": 7, "min_samples_split": 10},
            "feature_params": {"tfidf_max_features": 3000, "ngram_range": [1, 2]},
        },
        # Больше признаков
        {
            "name": "RF_more_features",
            "algorithm": "RandomForestClassifier",
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
            "feature_params": {"tfidf_max_features": 10000, "ngram_range": [1, 3]},
        },
    ]

    # Эксперименты с SVM
    svm_configs = [
        # Базовая конфигурация
        {
            "name": "SVM_baseline",
            "algorithm": "SVM",
            "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
        # Линейное ядро
        {
            "name": "SVM_linear",
            "algorithm": "SVM",
            "params": {"kernel": "linear", "C": 1.0, "gamma": "scale"},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
        # Высокий C
        {
            "name": "SVM_high_C",
            "algorithm": "SVM",
            "params": {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
        # Низкий C
        {
            "name": "SVM_low_C",
            "algorithm": "SVM",
            "params": {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
            "feature_params": {"tfidf_max_features": 3000, "ngram_range": [1, 2]},
        },
        # Полиномиальное ядро
        {
            "name": "SVM_poly",
            "algorithm": "SVM",
            "params": {"kernel": "poly", "C": 1.0, "gamma": "scale", "degree": 3},
            "feature_params": {"tfidf_max_features": 3000, "ngram_range": [1, 2]},
        },
    ]

    # Эксперименты с Logistic Regression
    lr_configs = [
        # Базовая конфигурация
        {
            "name": "LR_baseline",
            "algorithm": "LogisticRegression",
            "params": {"penalty": "l2", "C": 1.0, "solver": "liblinear"},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
        # L1 регуляризация
        {
            "name": "LR_l1_penalty",
            "algorithm": "LogisticRegression",
            "params": {"penalty": "l1", "C": 1.0, "solver": "liblinear"},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
        # Высокая регуляризация
        {
            "name": "LR_high_reg",
            "algorithm": "LogisticRegression",
            "params": {"penalty": "l2", "C": 0.1, "solver": "liblinear"},
            "feature_params": {"tfidf_max_features": 3000, "ngram_range": [1, 2]},
        },
        # Низкая регуляризация
        {
            "name": "LR_low_reg",
            "algorithm": "LogisticRegression",
            "params": {"penalty": "l2", "C": 10.0, "solver": "liblinear"},
            "feature_params": {"tfidf_max_features": 7000, "ngram_range": [1, 2]},
        },
        # Другой солвер
        {
            "name": "LR_lbfgs",
            "algorithm": "LogisticRegression",
            "params": {"penalty": "l2", "C": 1.0, "solver": "lbfgs"},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 2]},
        },
    ]

    # Эксперименты с разными настройками признаков
    feature_experiments = [
        # Только униграммы
        {
            "name": "RF_unigrams_only",
            "algorithm": "RandomForestClassifier",
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
            "feature_params": {"tfidf_max_features": 5000, "ngram_range": [1, 1]},
        },
        # Расширенные n-граммы
        {
            "name": "LR_extended_ngrams",
            "algorithm": "LogisticRegression",
            "params": {"penalty": "l2", "C": 1.0, "solver": "liblinear"},
            "feature_params": {"tfidf_max_features": 8000, "ngram_range": [1, 4]},
        },
    ]

    # Собираем все эксперименты
    all_configs = rf_configs + svm_configs + lr_configs + feature_experiments

    # Создаем полные параметры для каждого эксперимента
    for i, config in enumerate(all_configs):
        params = copy.deepcopy(base_params)

        # Обновляем основные параметры обучения
        params["train"]["algorithm"] = config["algorithm"]

        # Обновляем параметры алгоритма
        if config["algorithm"] == "RandomForestClassifier":
            params["train"]["random_forest"].update(config["params"])
        elif config["algorithm"] == "SVM":
            params["train"]["svm"].update(config["params"])
        elif config["algorithm"] == "LogisticRegression":
            params["train"]["logistic_regression"].update(config["params"])

        # Обновляем параметры признаков
        params["feature_engineering"].update(config["feature_params"])

        # Обновляем MLflow параметры
        params["mlflow"]["run_name"] = f"exp_{i + 1:02d}_{config['name']}"
        params["mlflow"]["tags"]["experiment_type"] = config["name"]
        params["mlflow"]["tags"]["algorithm"] = config["algorithm"]

        experiments.append(
            {
                "name": config["name"],
                "params": params,
                "description": f"Эксперимент {i + 1}: {config['name']} с параметрами {config['params']}",
            }
        )

    return experiments


def run_single_experiment(
    exp_config: Dict[str, Any], data_file: str, base_output_dir: str
) -> Dict[str, Any]:
    """Запускает один эксперимент и возвращает результаты."""
    exp_name = exp_config["name"]
    params = exp_config["params"]

    logger.info(f"Запуск эксперимента: {exp_name}")
    logger.info(f"Описание: {exp_config['description']}")

    # Создаем временный файл параметров
    temp_params_file = f"temp_params_{exp_name}.yaml"
    with open(temp_params_file, "w") as f:
        yaml.dump(params, f, default_flow_style=False)

    # Создаем пути для выходных файлов
    output_dir = Path(base_output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_output = output_dir / "model.pkl"
    metrics_output = output_dir / "metrics.json"

    try:
        # Запускаем обучение
        start_time = time.time()
        model, test_metrics = train_model(
            data_file=data_file,
            params_file=temp_params_file,
            model_output=str(model_output),
            metrics_output=str(metrics_output),
        )
        end_time = time.time()

        # Загружаем результаты
        with open(metrics_output, "r") as f:
            metrics = json.load(f)

        result = {
            "experiment_name": exp_name,
            "algorithm": params["train"]["algorithm"],
            "status": "success",
            "training_time": end_time - start_time,
            "test_accuracy": metrics["test_metrics"]["accuracy"],
            "test_f1_score": metrics["test_metrics"]["f1_score"],
            "cv_mean_score": metrics["cross_validation"]["mean_score"],
            "cv_std_score": metrics["cross_validation"]["std_score"],
            "params": params["train"],
            "feature_params": params["feature_engineering"],
            "model_path": str(model_output),
            "metrics_path": str(metrics_output),
        }

        logger.info(f"✅ Эксперимент {exp_name} завершен успешно")
        logger.info(f"   Точность на тесте: {result['test_accuracy']:.4f}")
        logger.info(f"   F1-score: {result['test_f1_score']:.4f}")
        logger.info(f"   Время обучения: {result['training_time']:.2f}s")

    except Exception as e:
        logger.error(f"❌ Ошибка в эксперименте {exp_name}: {str(e)}")
        result = {
            "experiment_name": exp_name,
            "algorithm": params["train"]["algorithm"],
            "status": "failed",
            "error": str(e),
            "params": params["train"],
            "feature_params": params["feature_engineering"],
        }

    finally:
        # Удаляем временный файл
        Path(temp_params_file).unlink(missing_ok=True)

    return result


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Анализирует результаты всех экспериментов."""
    successful_results = [r for r in results if r["status"] == "success"]

    if not successful_results:
        return {"status": "no_successful_experiments"}

    # Сортируем по test_accuracy
    sorted_by_accuracy = sorted(
        successful_results, key=lambda x: x["test_accuracy"], reverse=True
    )
    sorted_by_f1 = sorted(
        successful_results, key=lambda x: x["test_f1_score"], reverse=True
    )

    # Статистика по алгоритмам
    algorithm_stats = {}
    for result in successful_results:
        alg = result["algorithm"]
        if alg not in algorithm_stats:
            algorithm_stats[alg] = {"count": 0, "accuracies": [], "f1_scores": []}
        algorithm_stats[alg]["count"] += 1
        algorithm_stats[alg]["accuracies"].append(result["test_accuracy"])
        algorithm_stats[alg]["f1_scores"].append(result["test_f1_score"])

    # Средние метрики по алгоритмам
    for alg, stats in algorithm_stats.items():
        stats["mean_accuracy"] = sum(stats["accuracies"]) / len(stats["accuracies"])
        stats["mean_f1_score"] = sum(stats["f1_scores"]) / len(stats["f1_scores"])
        stats["best_accuracy"] = max(stats["accuracies"])
        stats["best_f1_score"] = max(stats["f1_scores"])

    analysis = {
        "total_experiments": len(results),
        "successful_experiments": len(successful_results),
        "failed_experiments": len(results) - len(successful_results),
        "best_accuracy_experiment": sorted_by_accuracy[0],
        "best_f1_experiment": sorted_by_f1[0],
        "algorithm_statistics": algorithm_stats,
        "top_5_by_accuracy": sorted_by_accuracy[:5],
        "top_5_by_f1": sorted_by_f1[:5],
    }

    return analysis


def save_experiment_summary(
    results: List[Dict[str, Any]], analysis: Dict[str, Any], output_file: str
):
    """Сохраняет сводку результатов экспериментов."""
    summary = {
        "experiment_run_date": datetime.now().isoformat(),
        "summary": analysis,
        "detailed_results": results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Сводка экспериментов сохранена в {output_file}")


def print_summary(analysis: Dict[str, Any]):
    """Выводит краткую сводку результатов в консоль."""
    print("\n" + "=" * 80)
    print("СВОДКА ЭКСПЕРИМЕНТОВ")
    print("=" * 80)

    print(f"Всего экспериментов: {analysis['total_experiments']}")
    print(f"Успешных: {analysis['successful_experiments']}")
    print(f"Неудачных: {analysis['failed_experiments']}")

    print("\n🏆 ЛУЧШИЙ ПО ТОЧНОСТИ:")
    best_acc = analysis["best_accuracy_experiment"]
    print(f"   Эксперимент: {best_acc['experiment_name']}")
    print(f"   Алгоритм: {best_acc['algorithm']}")
    print(f"   Точность: {best_acc['test_accuracy']:.4f}")
    print(f"   F1-score: {best_acc['test_f1_score']:.4f}")

    print("\n🎯 ЛУЧШИЙ ПО F1-SCORE:")
    best_f1 = analysis["best_f1_experiment"]
    print(f"   Эксперимент: {best_f1['experiment_name']}")
    print(f"   Алгоритм: {best_f1['algorithm']}")
    print(f"   Точность: {best_f1['test_accuracy']:.4f}")
    print(f"   F1-score: {best_f1['test_f1_score']:.4f}")

    print("\n📊 СТАТИСТИКА ПО АЛГОРИТМАМ:")
    for alg, stats in analysis["algorithm_statistics"].items():
        print(f"   {alg}:")
        print(f"      Количество экспериментов: {stats['count']}")
        print(f"      Средняя точность: {stats['mean_accuracy']:.4f}")
        print(f"      Лучшая точность: {stats['best_accuracy']:.4f}")
        print(f"      Средний F1-score: {stats['mean_f1_score']:.4f}")
        print(f"      Лучший F1-score: {stats['best_f1_score']:.4f}")

    print("\n" + "=" * 80)


def main():
    """Основная функция для запуска серии экспериментов."""
    logger.info("Начало серии ML экспериментов")

    # Настройки
    data_file = "data/processed/publications_processed.csv"
    output_dir = "experiments"
    summary_file = "experiments_summary.json"

    # Проверяем наличие данных
    if not Path(data_file).exists():
        logger.error(f"Файл данных не найден: {data_file}")
        return

    # Создаем список экспериментов
    logger.info("Создание конфигураций экспериментов...")
    experiment_configs = create_experiment_configs()
    logger.info(f"Создано {len(experiment_configs)} экспериментов")

    # Создаем выходную директорию
    Path(output_dir).mkdir(exist_ok=True)

    # Запускаем эксперименты
    results = []
    total_start_time = time.time()

    for i, config in enumerate(experiment_configs, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Эксперимент {i}/{len(experiment_configs)}")
        logger.info(f"{'=' * 60}")

        result = run_single_experiment(config, data_file, output_dir)
        results.append(result)

        # Промежуточный отчет каждые 5 экспериментов
        if i % 5 == 0:
            successful = sum(1 for r in results if r["status"] == "success")
            logger.info(f"Промежуточный итог: {successful}/{i} экспериментов успешно")

    total_time = time.time() - total_start_time
    logger.info(f"\nВсе эксперименты завершены за {total_time:.2f} секунд")

    # Анализируем результаты
    logger.info("Анализ результатов...")
    analysis = analyze_results(results)

    # Сохраняем результаты
    save_experiment_summary(results, analysis, summary_file)

    # Выводим сводку
    print_summary(analysis)

    # Выводим инструкции для просмотра MLflow
    print("\n📈 Для просмотра результатов в MLflow запустите:")
    print("   mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 3000")
    print("   Затем откройте: http://127.0.0.1:3000")

    logger.info("Серия экспериментов завершена!")


if __name__ == "__main__":
    main()
