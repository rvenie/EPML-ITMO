#!/usr/bin/env python3
"""
Простая система композиции конфигураций для разных алгоритмов ML.
Генерирует конфигурации с валидацией через Pydantic.
"""

import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импортируем только PipelineConfig для валидации
from config.pipeline_config import PipelineConfig

# Базовые конфигурации для разных алгоритмов
ALGORITHM_CONFIGS = {
    "RandomForestClassifier": {
        "train": {
            "algorithm": "RandomForestClassifier",
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "oob_score": True,
            },
        },
        "mlflow": {
            "experiment_name": "random_forest_experiment",
            "tags": {"algorithm": "RandomForest", "complexity": "medium"},
        },
    },
    "SVM": {
        "train": {
            "algorithm": "SVM",
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
            },
        },
        "mlflow": {
            "experiment_name": "svm_experiment",
            "tags": {"algorithm": "SVM", "complexity": "high"},
        },
    },
    "LogisticRegression": {
        "train": {
            "algorithm": "LogisticRegression",
            "logistic_regression": {
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 1000,
                "solver": "liblinear",
            },
        },
        "mlflow": {
            "experiment_name": "logistic_regression_experiment",
            "tags": {"algorithm": "LogisticRegression", "complexity": "low"},
        },
    },
}

# Конфигурации для разных размеров данных
DATA_SIZE_CONFIGS = {
    "small": {
        "data": {"max_results": 50},
        "feature_engineering": {"tfidf_max_features": 1000},
    },
    "medium": {
        "data": {"max_results": 100},
        "feature_engineering": {"tfidf_max_features": 5000},
    },
    "large": {
        "data": {"max_results": 500},
        "feature_engineering": {"tfidf_max_features": 10000},
    },
}


def load_base_config(config_path="params.yaml"):
    """Загрузка базовой конфигурации"""
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ Файл {config_path} не найден, используется минимальная конфигурация")
        return get_minimal_config()


def get_minimal_config():
    """Минимальная базовая конфигурация"""
    return {
        "data": {
            "query": "cat:eess.IV OR cat:cs.CV OR cat:q-bio.QM",
            "max_results": 100,
            "source": "arxiv",
        },
        "feature_engineering": {
            "tfidf_max_features": 5000,
            "text_columns": ["title", "abstract"],
            "categorical_columns": ["journal_category"],
            "numerical_columns": ["year", "author_count"],
        },
        "train": {"test_size": 0.2, "random_state": 42},
        "mlflow": {"tracking_uri": "file:./mlruns"},
        "evaluate": {
            "target_column": "arxiv_categories",
            "metrics": ["accuracy", "precision", "recall", "f1_score"],
        },
    }


def deep_merge_dicts(dict1, dict2):
    """Глубокое объединение словарей"""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def compose_config(algorithm, data_size="medium", base_config_path="params.yaml"):
    """Создание составной конфигурации"""
    # Загрузка базовой конфигурации
    base_config = load_base_config(base_config_path)

    # Получение конфигураций для композиции
    algorithm_config = ALGORITHM_CONFIGS.get(algorithm, {})
    data_size_config = DATA_SIZE_CONFIGS.get(data_size, {})

    # Объединение конфигураций (специфичные перезаписывают базовые)
    result = deep_merge_dicts(base_config, algorithm_config)
    result = deep_merge_dicts(result, data_size_config)

    # Добавление метаданных
    result["_metadata"] = {
        "algorithm": algorithm,
        "data_size": data_size,
        "generated_by": "simple_composer.py",
    }

    return result


def validate_config(config):
    """
    Валидация конфигурации через Pydantic модели

    Args:
        config: Словарь с конфигурацией

    Returns:
        tuple: (bool, str) - статус валидации и сообщение
    """
    try:
        # Используем Pydantic для валидации
        PipelineConfig(**config)
        return True, "Конфигурация валидна (проверено Pydantic)"
    except Exception as e:
        return False, f"Ошибка валидации Pydantic: {str(e)}"


def save_config(config, output_path):
    """Сохранение конфигурации в файл"""
    try:
        # Создание директории если не существует
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        return False


def generate_all_configs(base_config_path="params.yaml"):
    """Генерация всех конфигураций для экспериментов"""
    print("🔧 Генерация конфигураций для разных алгоритмов...")

    # Создание директории
    configs_dir = Path("config/generated")
    configs_dir.mkdir(exist_ok=True, parents=True)

    success_count = 0
    total_count = 0

    # Генерация для каждого алгоритма
    for algorithm in ALGORITHM_CONFIGS.keys():
        for data_size in ["small", "medium"]:
            total_count += 1

            print(f"📝 Создание конфигурации: {algorithm} ({data_size})")

            # Создание конфигурации
            config = compose_config(algorithm, data_size, base_config_path)

            # Валидация
            is_valid, message = validate_config(config)
            if not is_valid:
                print(f"❌ Ошибка валидации {algorithm}: {message}")
                continue

            # Сохранение
            filename = f"{algorithm.lower()}_{data_size}_config.yaml"
            config_path = configs_dir / filename

            if save_config(config, config_path):
                print(f"✅ Сохранена: {config_path}")
                success_count += 1
            else:
                print(f"❌ Ошибка сохранения: {config_path}")

    print(f"\n📊 Результат: {success_count}/{total_count} конфигураций создано успешно")

    # Создание сводки
    create_config_summary(configs_dir)


def create_config_summary(configs_dir):
    """Создание сводного файла с описанием конфигураций"""
    summary = {
        "generated_configs": {
            "description": "Автоматически сгенерированные конфигурации для разных алгоритмов",
            "algorithms": list(ALGORITHM_CONFIGS.keys()),
            "data_sizes": list(DATA_SIZE_CONFIGS.keys()),
            "total_configs": len(list(configs_dir.glob("*.yaml"))),
            "usage": "Используйте эти конфигурации с automated_pipeline.py --config <path>",
        }
    }

    summary_path = configs_dir / "README.yaml"
    save_config(summary, summary_path)
    print(f"📋 Сводка сохранена: {summary_path}")


def list_generated_configs():
    """Список всех сгенерированных конфигураций"""
    configs_dir = Path("config/generated")

    if not configs_dir.exists():
        print("❌ Директория конфигураций не существует")
        return

    config_files = list(configs_dir.glob("*.yaml"))

    if not config_files:
        print("❌ Конфигурации не найдены")
        return

    print("📁 Сгенерированные конфигурации:")
    for config_file in sorted(config_files):
        if config_file.name != "README.yaml":
            print(f"   • {config_file.name}")


def main():
    """Главная функция"""
    print("🚀 Простая система композиции конфигураций")
    print("=" * 50)

    # Генерация всех конфигураций
    generate_all_configs()

    print("\n" + "=" * 50)

    # Список созданных конфигураций
    list_generated_configs()

    print("\n✅ Генерация конфигураций завершена!")
    print(
        "💡 Используйте: python scripts/automated_pipeline.py --config config/generated/<filename>"
    )


if __name__ == "__main__":
    main()
