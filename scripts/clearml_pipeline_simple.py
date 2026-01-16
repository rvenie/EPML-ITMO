#!/usr/bin/env python3
"""
ClearML Pipeline с Pydantic валидацией параметров.
Версия с полной валидацией для консистентности с остальным проектом.

Использование:
    python scripts/clearml_pipeline_validated.py
    python scripts/clearml_pipeline_validated.py --config config/pipeline_clearml.yaml
"""

import argparse
import pickle  # nosec B403
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Dataset, Task
from clearml.automation import PipelineDecorator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Добавляем путь к researchhub модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from researchhub.pipeline_config_simple import PipelineConfig, load_pipeline_config


# ========================================
# Шаг 1: Загрузка данных
# ========================================
@PipelineDecorator.component(
    return_values=["train_data", "test_data", "data_version", "dataset_id"],
    cache=True,
    task_type=Task.TaskTypes.data_processing,
)
def step_load_data(config_dict: dict, use_clearml_dataset: bool = True):
    """Загружает и разделяет данные с Pydantic валидацией."""
    from researchhub.pipeline_config_simple import DataConfig

    # Валидируем конфигурацию данных
    data_config = DataConfig(**config_dict)

    # Добавляем информативные теги
    task = Task.current_task()
    if task:
        task.add_tags(
            [
                "stage:data_loading",
                "source:clearml_pipeline.py",
                f"test_size:{data_config.test_size}",
            ]
        )

    # Пытаемся загрузить из ClearML Dataset
    dataset_id = None
    if use_clearml_dataset:
        try:
            print("📦 Trying to load from ClearML Dataset...")
            dataset = Dataset.get(
                dataset_name="ResearchHub Publications",
                dataset_project="researchhub",
                dataset_version="1.0",
            )
            dataset_id = dataset.id
            dataset_path = dataset.get_local_copy()

            # Ищем CSV файл в датасете
            csv_files = list(Path(dataset_path).rglob("*.csv"))
            if csv_files:
                data_file = csv_files[0]
                print(f"✅ Loaded from ClearML Dataset: {dataset_id}")
                print(f"   File: {data_file}")
                df = pd.read_csv(data_file)

                # Добавляем dataset_id в теги
                if task:
                    task.add_tags([f"dataset_id:{dataset_id[:8]}"])
            else:
                raise FileNotFoundError("No CSV files in dataset")

        except Exception as e:
            print(f"⚠️  Could not load from ClearML Dataset: {e}")
            print(f"📊 Fallback: Loading from local file: {data_config.data_path}")
            df = pd.read_csv(data_config.data_path)
    else:
        print(f"📊 Loading data from {data_config.data_path}")
        df = pd.read_csv(data_config.data_path)

    print(f"✅ Loaded {len(df)} records")

    # Определяем текстовые колонки
    text_columns = [
        col for col in ["title", "summary", "abstract"] if col in df.columns
    ]

    # Объединяем текст
    df["text"] = df[text_columns].fillna("").apply(lambda x: " ".join(x), axis=1)

    # Определяем целевую переменную
    if data_config.target_column and data_config.target_column in df.columns:
        target_column = data_config.target_column
    else:
        # Создаем синтетическую на основе author_count_category
        if "author_count_category" in df.columns:
            target_column = "author_count_category"
        else:
            # Fallback: создаем из abstract_length
            df["synthetic_target"] = pd.cut(
                df["abstract_length"], bins=3, labels=["short", "medium", "long"]
            )
            target_column = "synthetic_target"

    # Проверяем количество классов
    unique_classes = df[target_column].nunique()
    print(f"📊 Target: {target_column}, unique classes: {unique_classes}")

    if unique_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {unique_classes}")

    # Разделяем с валидированными параметрами
    train_df, test_df = train_test_split(
        df[["text", target_column]],
        test_size=data_config.test_size,
        random_state=data_config.random_state,
        stratify=df[target_column],
    )

    # Data version
    import hashlib
    import tempfile

    data_hash = hashlib.md5(
        df.to_csv(index=False).encode(), usedforsecurity=False
    ).hexdigest()[:8]

    # Добавляем data_version в теги
    if task:
        task.add_tags([f"data_version:{data_hash}"])

    print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"📦 Data version: {data_hash}")
    if dataset_id:
        print(f"📦 Dataset ID: {dataset_id}")

    # Создаем дочерний Dataset с train/test сплитами
    processed_dataset_id = None
    if dataset_id:
        try:
            print("📦 Creating processed dataset with train/test splits...")
            parent_dataset = Dataset.get(dataset_id=dataset_id)

            # Создаем временные файлы для train/test
            with tempfile.TemporaryDirectory() as tmpdir:
                train_path = Path(tmpdir) / "train.csv"
                test_path = Path(tmpdir) / "test.csv"

                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)

                # Создаем новую версию датасета с обработанными данными
                processed_dataset = Dataset.create(
                    dataset_name="ResearchHub Publications",
                    dataset_project="researchhub",
                    dataset_version=f"1.1-{data_hash[:8]}",  # Версия с hash сплита
                    parent_datasets=[parent_dataset],
                    description=f"Preprocessed dataset with train/test split (test_size={data_config.test_size}, random_state={data_config.random_state})",
                )

                # Добавляем train/test файлы
                processed_dataset.add_files(path=str(train_path))
                processed_dataset.add_files(path=str(test_path))

                # Получаем logger для метаданных
                logger = processed_dataset.get_logger()

                # 1. Preview train/test
                logger.report_table(
                    title="Train Split Preview",
                    series="First 5 rows",
                    table_plot=train_df.head(5),
                    iteration=0,
                )

                logger.report_table(
                    title="Test Split Preview",
                    series="First 5 rows",
                    table_plot=test_df.head(5),
                    iteration=1,
                )

                # 2. Распределение классов в train/test
                train_dist = train_df[target_column].value_counts().to_dict()
                test_dist = test_df[target_column].value_counts().to_dict()

                logger.report_histogram(
                    title="Class Distribution - Train",
                    series="train",
                    values=list(train_dist.values()),
                    xlabels=list(train_dist.keys()),
                    yaxis="Number of samples",
                    iteration=0,
                )

                logger.report_histogram(
                    title="Class Distribution - Test",
                    series="test",
                    values=list(test_dist.values()),
                    xlabels=list(test_dist.keys()),
                    yaxis="Number of samples",
                    iteration=1,
                )

                # 3. Статистика сплитов
                split_stats = {
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    "train_ratio": len(train_df) / len(df),
                    "test_ratio": len(test_df) / len(df),
                    "total_samples": len(df),
                    "target_column": target_column,
                    "test_size": data_config.test_size,
                    "random_state": data_config.random_state,
                }

                for key, value in split_stats.items():
                    if isinstance(value, (int, float)):
                        logger.report_single_value(name=key, value=value)

                # 4. Текстовое описание
                logger.report_text(
                    f"Preprocessing Statistics:\n"
                    f"- Train samples: {split_stats['train_samples']}\n"
                    f"- Test samples: {split_stats['test_samples']}\n"
                    f"- Train ratio: {split_stats['train_ratio']:.2%}\n"
                    f"- Test ratio: {split_stats['test_ratio']:.2%}\n"
                    f"- Target: {target_column}\n"
                    f"- Classes: {len(train_dist)} unique\n"
                    f"- Data hash: {data_hash}"
                )

                # Загружаем и финализируем
                processed_dataset.upload()
                processed_dataset.finalize()

                processed_dataset_id = processed_dataset.id

                # Добавляем dataset_id в теги
                if task:
                    task.add_tags([f"processed_dataset_id:{processed_dataset_id[:8]}"])

                print(f"✅ Processed dataset created: {processed_dataset_id}")
                print(f"   Version: 1.1-{data_hash[:8]}")

        except Exception as e:
            print(f"⚠️  Could not create processed dataset: {e}")
            import traceback

            traceback.print_exc()
            print("   Continuing with local data only...")

    return (
        train_df.to_dict(),
        test_df.to_dict(),
        data_hash,
        processed_dataset_id or dataset_id,
    )


# ========================================
# Шаги 2-5: Обучение моделей
# ========================================
@PipelineDecorator.component(
    return_values=["model_data", "metrics"],
    cache=True,
    task_type=Task.TaskTypes.training,
)
def step_train_model(train_data: dict, test_data: dict, model_config: dict):
    """Обучает модель с Pydantic валидацией параметров."""
    model_type = model_config["model_type"]
    print(f"🚀 Training {model_type} model")

    # Получаем task и переименовываем его
    task = Task.current_task()
    if task:
        # Переименовываем task на основе типа модели
        task.set_name(f"train_{model_type}")
        # Информативные теги
        task.add_tags(
            [
                "stage:training",
                f"model:{model_type}",
                "source:clearml_pipeline.py",
                "task_type:classification",
            ]
        )

    # Получаем logger для метрик
    logger = task.get_logger() if task else None

    # Восстанавливаем DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Векторизация
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X_train = vectorizer.fit_transform(train_df["text"]).toarray()
    X_test = vectorizer.transform(test_df["text"]).toarray()

    # Получаем целевую переменную
    target_col = [col for col in train_df.columns if col != "text"][0]

    # Label encoding для строковых меток
    le = LabelEncoder()
    if train_df[target_col].dtype == "object":
        y_train = le.fit_transform(train_df[target_col])
        y_test = le.transform(test_df[target_col])
    else:
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

    print(f"📊 Classes: {len(np.unique(y_train))} train, {len(np.unique(y_test))} test")

    # Создаем модель с валидированными параметрами
    params = model_config["params"]

    if model_type == "LogisticRegression":
        from researchhub.pipeline_config_simple import LogisticRegressionConfig

        validated_params = LogisticRegressionConfig(**params)
        model = LogisticRegression(
            C=validated_params.C,
            max_iter=validated_params.max_iter,
            solver=validated_params.solver,
            random_state=42,
        )
    elif model_type == "RandomForest":
        from researchhub.pipeline_config_simple import RandomForestConfig

        validated_params = RandomForestConfig(**params)
        model = RandomForestClassifier(
            n_estimators=validated_params.n_estimators,
            max_depth=validated_params.max_depth,
            min_samples_split=validated_params.min_samples_split,
            random_state=42,
        )
    elif model_type == "GradientBoosting":
        from researchhub.pipeline_config_simple import GradientBoostingConfig

        validated_params = GradientBoostingConfig(**params)
        model = GradientBoostingClassifier(
            n_estimators=validated_params.n_estimators,
            learning_rate=validated_params.learning_rate,
            max_depth=validated_params.max_depth,
            random_state=42,
        )
    elif model_type == "SVC":
        from researchhub.pipeline_config_simple import SVCConfig

        validated_params = SVCConfig(**params)
        model = SVC(
            kernel=validated_params.kernel,
            C=validated_params.C,
            gamma=validated_params.gamma,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Обучение
    model.fit(X_train, y_train)

    # Оценка на train
    y_pred_train = model.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, y_pred_train))
    train_f1 = float(f1_score(y_train, y_pred_train, average="weighted"))

    # Оценка на test
    y_pred = model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # Логируем метрики в Scalars
    if logger:
        # Train metrics
        logger.report_scalar("accuracy", "train", value=train_accuracy, iteration=0)
        logger.report_scalar("f1_score", "train", value=train_f1, iteration=0)

        # Test metrics
        logger.report_scalar("accuracy", "test", value=test_accuracy, iteration=0)
        logger.report_scalar("f1_score", "test", value=test_f1, iteration=0)

        # Overfitting check
        overfitting = train_accuracy - test_accuracy
        logger.report_scalar(
            "overfitting", "train-test-gap", value=overfitting, iteration=0
        )

    # Добавляем accuracy в теги для быстрой фильтрации
    if task:
        task.add_tags([f"test_accuracy:{test_accuracy:.3f}"])

    metrics = {
        "accuracy": test_accuracy,
        "f1_weighted": test_f1,
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "model_type": model_type,
    }

    print(
        f"✅ {model_type}: test_acc={test_accuracy:.4f}, train_acc={train_accuracy:.4f}"
    )

    # Сериализация
    model_data = {"model": model, "vectorizer": vectorizer}
    model_bytes = pickle.dumps(model_data)

    return model_bytes, metrics


# ========================================
# Шаг 6: Сравнение моделей
# ========================================
@PipelineDecorator.component(
    return_values=["best_model_data", "best_model_name", "comparison"],
    cache=False,
    task_type=Task.TaskTypes.optimizer,
)
def step_evaluate(
    logreg_result: tuple, rf_result: tuple, gb_result: tuple, svc_result: tuple
):
    """Сравнивает модели и выбирает лучшую."""
    print("📊 Comparing models...")

    # Добавляем теги
    task = Task.current_task()
    if task:
        task.add_tags(
            [
                "stage:evaluation",
                "source:clearml_pipeline.py",
                "task_type:optimizer",
            ]
        )

    logger = task.get_logger() if task else None

    models = {
        "LogisticRegression": {"data": logreg_result[0], "metrics": logreg_result[1]},
        "RandomForest": {"data": rf_result[0], "metrics": rf_result[1]},
        "GradientBoosting": {"data": gb_result[0], "metrics": gb_result[1]},
        "SVC": {"data": svc_result[0], "metrics": svc_result[1]},
    }

    comparison = {
        name: result["metrics"]["accuracy"] for name, result in models.items()
    }
    best_name = max(comparison, key=comparison.get)
    best_model_data = models[best_name]["data"]

    # Логируем сравнение моделей
    if logger:
        for i, (name, accuracy) in enumerate(
            sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        ):
            logger.report_scalar(
                "model_comparison", "accuracy", value=accuracy, iteration=i
            )
            # Также логируем как таблицу
            logger.report_scalar("leaderboard", name, value=accuracy, iteration=0)

    # Добавляем результаты в теги
    if task:
        task.add_tags(
            [
                f"best_model:{best_name}",
                f"best_accuracy:{comparison[best_name]:.3f}",
            ]
        )

    print(f"🏆 Best: {best_name} (accuracy={comparison[best_name]:.4f})")
    print(f"📊 All: {comparison}")

    return best_model_data, best_name, comparison


# ========================================
# Шаг 7: Регистрация модели
# ========================================
@PipelineDecorator.component(
    return_values=["model_id"], cache=False, task_type=Task.TaskTypes.controller
)
def step_register(best_model_data: bytes, best_model_name: str, data_version: str):
    """Регистрирует лучшую модель."""
    print(f"📦 Registering {best_model_name}")

    # Добавляем информативные теги
    task = Task.current_task()
    if task:
        task.add_tags(
            [
                "stage:model_registry",
                "source:clearml_pipeline.py",
                f"registered_model:{best_model_name}",
                f"data_version:{data_version}",
            ]
        )

    model_path = Path("models") / f"{best_model_name}_best.pkl"
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, "wb") as f:
        f.write(best_model_data)

    task = Task.current_task()
    if task:
        task.upload_artifact(
            name=f"best_model_{best_model_name}", artifact_object=str(model_path)
        )
        task.set_parameter("best_model_name", best_model_name)
        task.set_parameter("data_version", data_version)

        # Логируем как text (не scalar, т.к. это строки)
        logger = task.get_logger()
        logger.report_text(
            f"Best Model: {best_model_name}\nData Version: {data_version}"
        )

    print(f"✅ Registered: {best_model_name}")
    return task.id


# ========================================
# Pipeline Controller
# ========================================
@PipelineDecorator.pipeline(
    name="researchhub-pipeline",
    project="researchhub",
    version="1.0",
    default_queue="default",
)
def run_ml_pipeline(config: PipelineConfig):
    """ML Pipeline с автоматическим трекингом, валидацией и сравнением моделей."""
    print("=" * 80)
    print("🚀 ClearML ML Pipeline")
    print("=" * 80)

    # Добавляем теги к главному pipeline task
    pipeline_task = Task.current_task()
    if pipeline_task:
        pipeline_task.add_tags(
            [
                "orchestrator:clearml",
                "source:clearml_pipeline.py",
                f"pipeline_version:{config.clearml.pipeline_version}",
                f"project:{config.clearml.project_name}",
            ]
        )

    # Шаг 1: Загрузка данных (из ClearML Dataset или локально)
    train_data, test_data, data_version, dataset_id = step_load_data(
        config.data.model_dump(), use_clearml_dataset=True
    )

    # Шаги 2-5: Обучение моделей параллельно
    # Используем один компонент, но в UI будут видны имена моделей через теги
    logreg_result = step_train_model(
        train_data,
        test_data,
        model_config={
            "model_type": "LogisticRegression",
            "params": config.logreg.model_dump(),
        },
    )

    rf_result = step_train_model(
        train_data,
        test_data,
        model_config={"model_type": "RandomForest", "params": config.rf.model_dump()},
    )

    gb_result = step_train_model(
        train_data,
        test_data,
        model_config={
            "model_type": "GradientBoosting",
            "params": config.gb.model_dump(),
        },
    )

    svc_result = step_train_model(
        train_data,
        test_data,
        model_config={"model_type": "SVC", "params": config.svc.model_dump()},
    )

    # Шаг 6: Сравнение
    best_model_data, best_model_name, comparison = step_evaluate(
        logreg_result, rf_result, gb_result, svc_result
    )

    # Шаг 7: Регистрация
    model_id = step_register(best_model_data, best_model_name, data_version)

    print("=" * 80)
    print(f"✅ Pipeline completed! Best: {best_model_name}")
    print("=" * 80)

    return model_id


# ========================================
# Main
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ClearML Pipeline with Pydantic validation"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (optional)",
    )
    args = parser.parse_args()

    # Загружаем конфигурацию с валидацией
    config = load_pipeline_config(args.config)

    print("📝 Loaded configuration:")
    print(f"  Project: {config.clearml.project_name}")
    print(f"  Pipeline: {config.clearml.pipeline_name}")
    print(f"  Data: {config.data.data_path}")
    print("  Models: LogReg, RF, GB, SVC")
    print()

    # Запускаем локально
    PipelineDecorator.run_locally()

    # Выполняем pipeline
    result = run_ml_pipeline(config)

    print(f"\n✅ Pipeline completed. Model ID: {result}")
