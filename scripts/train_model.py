#!/usr/bin/env python3
"""
Скрипт обучения модели с интеграцией MLflow
Данный скрипт обучает ML модели для классификации научных публикаций и логирует
все данные в MLflow.
"""

import argparse
import json
import logging
import pickle  # nosec

# Импортируем наши утилиты автоматического логирования
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# MLflow
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml  # type: ignore

# ClearML
from clearml import Task

sys.path.append(str(Path(__file__).parent.parent / "clearml" / "experiments"))
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ML библиотеки
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from tracking_utils import ClearMLAutoLogger

# Добавляем путь к модулю моделей
sys.path.append(str(Path(__file__).parent.parent / "clearml" / "models"))
from model_manager import ClearMLModelManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_params(params_file: str) -> dict[str, Any]:
    """Загружает параметры из YAML файла."""
    try:
        with open(params_file) as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded parameters from {params_file}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise


def load_data(data_file: str) -> pd.DataFrame:
    """Загружает обработанные данные."""
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} records from {data_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def create_features(
    df: pd.DataFrame, params: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Создает матрицу признаков и целевой вектор."""
    feature_params = params["feature_engineering"]

    # Текстовые признаки
    text_columns = feature_params["text_columns"]
    text_data = df[text_columns].fillna("").apply(lambda x: " ".join(x), axis=1)

    # TF-IDF векторизация
    tfidf = TfidfVectorizer(
        max_features=feature_params["tfidf_max_features"],
        ngram_range=tuple(feature_params["ngram_range"]),
        min_df=feature_params["min_df"],
        max_df=feature_params["max_df"],
        lowercase=feature_params["lowercase"],
        stop_words=feature_params["stop_words"],
    )

    text_features = tfidf.fit_transform(text_data).toarray()
    logger.info(f"Created {text_features.shape[1]} text features")

    # Числовые признаки
    numerical_cols = feature_params["numerical_columns"]
    numerical_features = df[numerical_cols].fillna(0).values

    # Категориальные признаки (one-hot кодирование)
    categorical_cols = feature_params["categorical_columns"]
    categorical_features = pd.get_dummies(df[categorical_cols]).values

    # Объединяем все признаки
    X = np.hstack([text_features, numerical_features, categorical_features])  # noqa: N806

    # Целевая переменная
    target_col = params["evaluate"]["target_column"]
    y = df[target_col].values

    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {np.unique(y, return_counts=True)}")

    return X, y, tfidf


def get_model(algorithm: str, params: dict[str, Any]):
    """Возвращает экземпляр модели на основе алгоритма и параметров."""
    if algorithm == "RandomForestClassifier":
        rf_params = params["train"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_split=rf_params["min_samples_split"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            max_features=rf_params["max_features"],
            bootstrap=rf_params["bootstrap"],
            oob_score=rf_params["oob_score"],
            random_state=params["train"]["random_state"],
            n_jobs=-1,
        )
    elif algorithm == "SVM":
        svm_params = params["train"]["svm"]
        return SVC(
            kernel=svm_params["kernel"],
            C=svm_params["C"],
            gamma=svm_params["gamma"],
            probability=svm_params["probability"],
            random_state=params["train"]["random_state"],
        )
    elif algorithm == "LogisticRegression":
        lr_params = params["train"]["logistic_regression"]
        return LogisticRegression(
            penalty=lr_params["penalty"],
            C=lr_params["C"],
            max_iter=lr_params["max_iter"],
            solver=lr_params["solver"],
            random_state=params["train"]["random_state"],
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:  # noqa: N803
    """Оценивает модель и возвращает метрики."""
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Добавляем ROC AUC для многоклассовой классификации если возможно
    if y_pred_proba is not None and len(np.unique(y_test)) > 2:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_test, y_pred_proba, multi_class="ovr", average="weighted"
            )
        except ValueError:
            logger.warning("Could not calculate ROC AUC score")

    return metrics


def train_model(
    data_file: str, params_file: str, model_output: str, metrics_output: str
):
    """Основная функция обучения с логированием в MLflow и ClearML."""

    # Загружаем параметры и данные
    params = load_params(params_file)
    df = load_data(data_file)

    # Инициализация ClearML Task
    task = Task.init(
        project_name="ResearchHub",
        task_name="Model Training Experiment",
        task_type=Task.TaskTypes.training,
    )

    # Получаем ClearML логгер
    clearml_logger = task.get_logger()

    # Инициализируем автоматический логгер
    auto_logger = ClearMLAutoLogger(task)

    # Настройка MLflow
    mlflow_params = params["mlflow"]

    # Инициализируем менеджер моделей
    model_manager = ClearMLModelManager(
        mlflow_params.get("experiment_name", "ResearchHub")
    )
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run(run_name=mlflow_params["run_name"]):
        # Логируем параметры в ClearML
        train_params = params["train"]
        task.connect(train_params, name="training_params")
        task.connect(params["feature_engineering"], name="feature_params")

        # Логируем параметры в MLflow
        mlflow.log_params(
            {
                "algorithm": train_params["algorithm"],
                "test_size": train_params["test_size"],
                "random_state": train_params["random_state"],
                "cv_folds": train_params["cross_validation"]["folds"],
            }
        )

        # Логируем специфические параметры алгоритма
        if train_params["algorithm"] == "RandomForestClassifier":
            mlflow.log_params(params["train"]["random_forest"])
        elif train_params["algorithm"] == "SVM":
            mlflow.log_params(params["train"]["svm"])
        elif train_params["algorithm"] == "LogisticRegression":
            mlflow.log_params(params["train"]["logistic_regression"])

        # Логируем параметры предобработки признаков
        mlflow.log_params(params["feature_engineering"])

        # Добавляем теги
        for key, value in mlflow_params["tags"].items():
            mlflow.set_tag(key, value)

        # Создаем признаки
        logger.info("Creating features...")
        X, y, tfidf = create_features(df, params)  # noqa: N806

        # Логируем системную информацию
        auto_logger.log_system_info()

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
            X,
            y,
            test_size=train_params["test_size"],
            random_state=train_params["random_state"],
        )

        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")

        # Инициализируем и обучаем модель
        logger.info(f"Training {train_params['algorithm']} model...")
        model = get_model(train_params["algorithm"], params)

        # Логируем параметры модели
        auto_logger.log_model_params(model)

        # Логируем информацию о датасетах
        auto_logger.log_dataset_info(X_train, y_train, "train")
        auto_logger.log_dataset_info(X_test, y_test, "test")

        model.fit(X_train, y_train)

        # Кросс-валидация
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=train_params["cross_validation"]["folds"],
            scoring=train_params["cross_validation"]["scoring"],
            n_jobs=-1,
        )

        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(
            f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        # Оцениваем на тестовой выборке
        logger.info("Evaluating model on test set...")
        test_metrics = evaluate_model(model, X_test, y_test)

        # Используем автологгер для создания визуализаций
        y_pred = model.predict(X_test)

        # Получаем названия признаков и классов если возможно
        feature_names = None
        class_names = None
        if hasattr(model, "classes_"):
            class_names = [str(c) for c in model.classes_]

        # Логируем confusion matrix и classification report
        auto_logger.log_confusion_matrix(y_test, y_pred, class_names)
        auto_logger.log_classification_report(y_test, y_pred, class_names)

        # Логируем важность признаков
        auto_logger.log_feature_importance(model, feature_names)

        # Логируем метрики в ClearML
        clearml_logger.report_scalar(
            "Cross Validation", "Mean Score", cv_scores.mean(), iteration=0
        )
        clearml_logger.report_scalar(
            "Cross Validation", "Std Score", cv_scores.std(), iteration=0
        )

        for metric_name, metric_value in test_metrics.items():
            clearml_logger.report_scalar(
                "Test Metrics", metric_name, metric_value, iteration=0
            )

        # Логируем метрики в MLflow
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        # Логируем важность признаков если доступно
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
            # ClearML
            clearml_logger.report_scalar(
                "Feature Importance", "Mean", feature_importance.mean(), iteration=0
            )
            clearml_logger.report_scalar(
                "Feature Importance", "Max", feature_importance.max(), iteration=0
            )
            # MLflow
            mlflow.log_metric("mean_feature_importance", feature_importance.mean())
            mlflow.log_metric("max_feature_importance", feature_importance.max())

        # Создаем сигнатуру модели для MLflow
        signature = infer_signature(X_train, model.predict(X_train))

        # Логируем модель в MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name=f"{mlflow_params['experiment_name']}_model",
        )

        # Сохраняем модель локально
        model_path = Path(model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": model,
            "tfidf_vectorizer": tfidf,
            "feature_columns": {
                "text_columns": params["feature_engineering"]["text_columns"],
                "numerical_columns": params["feature_engineering"]["numerical_columns"],
                "categorical_columns": params["feature_engineering"][
                    "categorical_columns"
                ],
            },
            "target_column": params["evaluate"]["target_column"],
            "training_date": datetime.now().isoformat(),
            "model_version": "1.0.0",
        }

        with open(model_output, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_output}")

        # Сохраняем метрики
        all_metrics = {
            "cross_validation": {
                "mean_score": float(cv_scores.mean()),
                "std_score": float(cv_scores.std()),
                "scores": cv_scores.tolist(),
            },
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "model_info": {
                "algorithm": train_params["algorithm"],
                "training_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "features": int(X.shape[1]),
                "classes": len(np.unique(y)),
            },
            "training_date": datetime.now().isoformat(),
        }

        with open(metrics_output, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_output}")

        # Создаем метаданные модели
        metadata_file = model_output.replace(".pkl", "_metadata.yaml")
        metadata = {
            "model_name": f"{mlflow_params['experiment_name']}_model",
            "model_version": "1.0.0",
            "algorithm": train_params["algorithm"],
            "training_date": datetime.now().isoformat(),
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "data_version": mlflow_params["tags"]["data_version"],
            "performance": {
                "cv_accuracy": float(cv_scores.mean()),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_f1_score": float(test_metrics["f1_score"]),
            },
            "hyperparameters": train_params,
            "feature_engineering": params["feature_engineering"],
            "data_info": {
                "training_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "features": int(X.shape[1]),
                "target_classes": len(np.unique(y)),
            },
        }

        with open(metadata_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(f"Model metadata saved to {metadata_file}")

        # Логируем артефакты в ClearML
        task.upload_artifact("metrics", artifact_object=metrics_output)
        task.upload_artifact("metadata", artifact_object=metadata_file)
        task.upload_artifact("model", artifact_object=model_output)

        # Логируем артефакты в MLflow
        mlflow.log_artifact(metrics_output, "metrics")
        mlflow.log_artifact(metadata_file, "metadata")

        # Создаем сводку эксперимента
        experiment_summary = auto_logger.create_experiment_summary(test_metrics)

        # Автоматическая регистрация модели в ClearML
        try:
            model_name = f"{train_params['algorithm']}_model"

            # Подготавливаем информацию о данных для регистрации
            training_data_info = {
                "training_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "features": int(X.shape[1]),
                "classes": len(np.unique(y)),
            }

            # Регистрируем модель с полными метаданными
            registered_model = model_manager.auto_register_from_training(
                model_path=model_output,
                model_name=model_name,
                task_id=task.id,
                training_metrics=test_metrics,
                model_params=train_params,
                training_data_info=training_data_info,
            )

            logger.info(
                f"Модель автоматически зарегистрирована: {registered_model.name}"
            )

        except Exception as e:
            logger.error(f"Ошибка регистрации модели: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"ClearML task ID: {task.id}")
        logger.info(
            f"Experiment summary created: {experiment_summary.get('experiment_id', 'N/A')}"
        )

        return model, test_metrics


def main():
    """Главная функция с парсингом аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Train ML model with MLflow logging")
    parser.add_argument(
        "--input", type=str, required=True, help="Input processed data CSV file"
    )
    parser.add_argument(
        "--model-output", type=str, required=True, help="Output model pickle file"
    )
    parser.add_argument(
        "--metrics", type=str, required=True, help="Output metrics JSON file"
    )
    parser.add_argument(
        "--params", type=str, default="params.yaml", help="Parameters YAML file"
    )

    args = parser.parse_args()

    # Обучаем модель
    train_model(args.input, args.params, args.model_output, args.metrics)


if __name__ == "__main__":
    main()
