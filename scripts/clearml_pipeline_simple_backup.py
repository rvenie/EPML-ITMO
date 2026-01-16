#!/usr/bin/env python3
"""
Простой ClearML Pipeline без overengineering.
Использует PipelineDecorator для создания реального DAG из 7 шагов.

Основан на лучших практиках из официальной документации ClearML.
"""

import pickle  # nosec B403
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Task
from clearml.automation import PipelineDecorator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# ========================================
# Шаг 1: Загрузка данных
# ========================================
@PipelineDecorator.component(
    return_values=["train_data", "test_data", "data_version"],
    cache=True,
    task_type=Task.TaskTypes.data_processing,
)
def step_load_data(data_path: str = "data/processed/publications_processed.csv"):
    """Загружает и разделяет данные."""
    print(f"📊 Loading data from {data_path}")

    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} records")

    # Определяем текстовые и целевые колонки
    text_columns = [
        col for col in ["title", "summary", "abstract"] if col in df.columns
    ]
    target_column = None
    for col in ["category_encoded", "label", "target", "abstract_category"]:
        if col in df.columns:
            target_column = col
            break

    # Объединяем текст
    df["text"] = df[text_columns].fillna("").apply(lambda x: " ".join(x), axis=1)

    # Создаем целевую переменную на основе author_count_category
    # Если она существует и имеет множественные классы
    if "author_count_category" in df.columns:
        target_column = "author_count_category"
    else:
        # Создаем синтетическую целевую переменную на основе длины abstract
        df["synthetic_target"] = pd.cut(
            df["abstract_length"], bins=3, labels=["short", "medium", "long"]
        )
        target_column = "synthetic_target"

    # Проверяем количество классов
    unique_classes = df[target_column].nunique()
    print(f"📊 Target column: {target_column}, unique classes: {unique_classes}")

    if unique_classes < 2:
        raise ValueError(
            f"Target has only {unique_classes} class(es). Need at least 2 for classification."
        )

    # Разделяем на train/test с стратификацией
    train_df, test_df = train_test_split(
        df[["text", target_column]],
        test_size=0.2,
        random_state=42,
        stratify=df[target_column],  # Сохраняем распределение классов
    )

    # Вычисляем версию данных (MD5 hash)
    import hashlib

    data_hash = hashlib.md5(
        df.to_csv(index=False).encode(), usedforsecurity=False
    ).hexdigest()[:8]

    print(f"✅ Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    print(f"📦 Data version: {data_hash}")

    return train_df.to_dict(), test_df.to_dict(), data_hash


# ========================================
# Шаги 2-5: Обучение моделей (параллельно)
# ========================================
@PipelineDecorator.component(
    return_values=["model_data", "metrics"],
    cache=True,
    task_type=Task.TaskTypes.training,
)
def step_train_model(
    train_data: dict, test_data: dict, model_type: str, model_params: dict
):
    """Обучает одну модель."""
    print(f"🚀 Training {model_type} model")

    # Восстанавливаем DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Векторизация
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X_train = vectorizer.fit_transform(train_df["text"]).toarray()
    X_test = vectorizer.transform(test_df["text"]).toarray()

    # Получаем имя целевой колонки (первая не-text колонка)
    target_col = [col for col in train_df.columns if col != "text"][0]

    # Кодируем целевую переменную если она строковая
    le = LabelEncoder()
    if train_df[target_col].dtype == "object":
        y_train = le.fit_transform(train_df[target_col])
        y_test = le.transform(test_df[target_col])
    else:
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

    print(
        f"📊 Target classes: {len(np.unique(y_train))} in train, {len(np.unique(y_test))} in test"
    )

    # Создаем модель
    if model_type == "LogisticRegression":
        # Используем solver который поддерживает multiclass
        model_params_fixed = model_params.copy()
        model_params_fixed["solver"] = "lbfgs"  # Вместо liblinear
        model_params_fixed["max_iter"] = 1000
        model = LogisticRegression(**model_params_fixed, random_state=42)
    elif model_type == "RandomForest":
        model = RandomForestClassifier(**model_params, random_state=42)
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(**model_params, random_state=42)
    elif model_type == "SVC":
        model = SVC(**model_params, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Обучение
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "model_type": model_type,
    }

    print(f"✅ {model_type}: accuracy={metrics['accuracy']:.4f}")

    # Сериализуем модель
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
    logreg_result: tuple,
    rf_result: tuple,
    gb_result: tuple,
    svc_result: tuple,
):
    """Сравнивает модели и выбирает лучшую."""
    print("📊 Comparing models...")

    # Распаковываем результаты
    models = {
        "LogisticRegression": {"data": logreg_result[0], "metrics": logreg_result[1]},
        "RandomForest": {"data": rf_result[0], "metrics": rf_result[1]},
        "GradientBoosting": {"data": gb_result[0], "metrics": gb_result[1]},
        "SVC": {"data": svc_result[0], "metrics": svc_result[1]},
    }

    # Сравниваем по accuracy
    comparison = {
        name: result["metrics"]["accuracy"] for name, result in models.items()
    }
    best_name = max(comparison, key=comparison.get)
    best_model_data = models[best_name]["data"]

    print(f"🏆 Best model: {best_name} (accuracy={comparison[best_name]:.4f})")
    print(f"📊 All results: {comparison}")

    return best_model_data, best_name, comparison


# ========================================
# Шаг 7: Регистрация лучшей модели
# ========================================
@PipelineDecorator.component(
    return_values=["model_id"],
    cache=False,
    task_type=Task.TaskTypes.controller,
)
def step_register(best_model_data: bytes, best_model_name: str, data_version: str):
    """Регистрирует лучшую модель в ClearML Model Registry."""
    print(f"📦 Registering model: {best_model_name}")

    # Сохраняем модель во временный файл
    model_path = Path("models") / f"{best_model_name}_best.pkl"
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, "wb") as f:
        f.write(best_model_data)

    # Получаем текущий Task
    task = Task.current_task()

    # Загружаем как артефакт
    task.upload_artifact(
        name=f"best_model_{best_model_name}",
        artifact_object=str(model_path),
    )

    # Добавляем метаданные
    task.set_parameter("best_model_name", best_model_name)
    task.set_parameter("data_version", data_version)

    print(f"✅ Model registered: {best_model_name}")
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
def run_ml_pipeline(data_path: str = "data/processed/publications_processed.csv"):
    """Главный pipeline с 7 шагами и параллельным обучением."""

    print("=" * 80)
    print("🚀 Starting ClearML ML Pipeline")
    print("=" * 80)

    # Шаг 1: Загрузка данных
    train_data, test_data, data_version = step_load_data(data_path)

    # Шаги 2-5: Обучение 4 моделей параллельно
    logreg_result = step_train_model(
        train_data,
        test_data,
        model_type="LogisticRegression",
        model_params={"C": 1.0},  # solver будет установлен автоматически
    )

    rf_result = step_train_model(
        train_data,
        test_data,
        model_type="RandomForest",
        model_params={"n_estimators": 50, "max_depth": 8},
    )

    gb_result = step_train_model(
        train_data,
        test_data,
        model_type="GradientBoosting",
        model_params={"n_estimators": 50, "learning_rate": 0.05},
    )

    svc_result = step_train_model(
        train_data,
        test_data,
        model_type="SVC",
        model_params={
            "kernel": "rbf",
            "C": 1.0,
        },  # rbf лучше для multiclass с малым датасетом
    )

    # Шаг 6: Сравнение моделей
    best_model_data, best_model_name, comparison = step_evaluate(
        logreg_result, rf_result, gb_result, svc_result
    )

    # Шаг 7: Регистрация лучшей модели
    model_id = step_register(best_model_data, best_model_name, data_version)

    print("=" * 80)
    print("✅ Pipeline completed successfully!")
    print(f"🏆 Best model: {best_model_name}")
    print(f"📦 Model ID: {model_id}")
    print("=" * 80)

    return model_id


# ========================================
# Main execution
# ========================================
if __name__ == "__main__":
    # Создаем pipeline controller task
    PipelineDecorator.run_locally()

    # Запускаем pipeline
    result = run_ml_pipeline()

    print(f"\n✅ Pipeline completed. Model ID: {result}")
