#!/usr/bin/env python3
"""
Скрипт обучения модели с интеграцией MLflow и ClearML
Данный скрипт обучает ML модели для классификации научных публикаций и логирует
все данные в MLflow и ClearML.
"""

import argparse
import json
import logging
import pickle  # nosec B403
import sys
from datetime import datetime
from pathlib import Path

# Добавляем корневую директорию в путь для импорта config
sys.path.insert(0, str(Path(__file__).parent.parent))

# MLflow
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml  # type: ignore

# ClearML
from clearml import Task
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

# Импорт Pydantic моделей для валидации
from config.pipeline_config import PipelineConfig, load_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_params(params_file: str) -> PipelineConfig:
    """
    Загружает и валидирует параметры из YAML файла через Pydantic.

    Args:
        params_file: Путь к файлу конфигурации

    Returns:
        PipelineConfig: Валидированная конфигурация
    """
    try:
        # Используем Pydantic для загрузки и валидации
        config = load_config(params_file)
        logger.info(f"✅ Loaded and validated parameters from {params_file}")
        logger.info(f"   Algorithm: {config.train.algorithm}")
        logger.info(f"   Experiment: {config.mlflow.experiment_name}")
        return config
    except Exception as e:
        logger.error(f"❌ Error loading/validating parameters: {e}")
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
    df: pd.DataFrame, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Создает матрицу признаков и целевой вектор.

    Args:
        df: DataFrame с данными
        config: Валидированная Pydantic конфигурация

    Returns:
        tuple: (X, y, tfidf_vectorizer)
    """
    # Используем Pydantic модель напрямую
    feature_config = config.feature_engineering

    # Текстовые признаки
    text_columns = feature_config.text_columns
    text_data = df[text_columns].fillna("").apply(lambda x: " ".join(x), axis=1)

    # TF-IDF векторизация с параметрами из Pydantic
    tfidf = TfidfVectorizer(
        max_features=feature_config.tfidf_max_features,
        ngram_range=tuple(feature_config.ngram_range),
        min_df=feature_config.min_df,
        max_df=feature_config.max_df,
        lowercase=feature_config.lowercase,
        stop_words=feature_config.stop_words,
    )

    text_features = tfidf.fit_transform(text_data).toarray()
    logger.info(f"Created {text_features.shape[1]} text features")

    # Числовые признаки
    numerical_cols = feature_config.numerical_columns
    numerical_features = df[numerical_cols].fillna(0).values

    # Категориальные признаки (one-hot кодирование)
    categorical_cols = feature_config.categorical_columns
    categorical_features = pd.get_dummies(df[categorical_cols]).values

    # Объединяем все признаки
    X = np.hstack([text_features, numerical_features, categorical_features])  # noqa: N806

    # Целевая переменная из Pydantic config
    target_col = config.evaluate.target_column
    y = df[target_col].values

    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {np.unique(y, return_counts=True)}")

    return X, y, tfidf


def get_model(algorithm: str, config: PipelineConfig):
    """
    Возвращает экземпляр модели на основе алгоритма и Pydantic конфигурации.

    Args:
        algorithm: Название алгоритма
        config: Валидированная Pydantic конфигурация

    Returns:
        Экземпляр модели sklearn
    """
    train_config = config.train

    if algorithm == "RandomForestClassifier":
        # Используем Pydantic модель RandomForestConfig
        rf = train_config.random_forest
        return RandomForestClassifier(
            n_estimators=rf.n_estimators,
            max_depth=rf.max_depth,
            min_samples_split=rf.min_samples_split,
            min_samples_leaf=rf.min_samples_leaf,
            max_features=rf.max_features,
            bootstrap=rf.bootstrap,
            oob_score=rf.oob_score,
            random_state=train_config.random_state,
            n_jobs=-1,
        )
    elif algorithm == "SVM":
        # Параметры SVM из config
        svm_dict = config.model_dump().get("train", {}).get("svm", {})
        return SVC(
            kernel=svm_dict.get("kernel", "rbf"),
            C=svm_dict.get("C", 1.0),
            gamma=svm_dict.get("gamma", "scale"),
            probability=svm_dict.get("probability", True),
            random_state=train_config.random_state,
        )
    elif algorithm == "LogisticRegression":
        # Параметры LR из config
        lr_dict = config.model_dump().get("train", {}).get("logistic_regression", {})
        return LogisticRegression(
            penalty=lr_dict.get("penalty", "l2"),
            C=lr_dict.get("C", 1.0),
            max_iter=lr_dict.get("max_iter", 1000),
            solver=lr_dict.get("solver", "liblinear"),
            random_state=train_config.random_state,
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
    data_file: str,
    params_file: str,
    model_output: str,
    metrics_output: str,
    algorithm: str | None = None,
):
    """Основная функция обучения с логированием в MLflow и ClearML."""

    # Загружаем и валидируем параметры через Pydantic
    config = load_params(params_file)
    df = load_data(data_file)

    # Переопределяем алгоритм если передан через командную строку
    if algorithm:
        # Обновляем конфигурацию (Pydantic позволяет это)
        config.train.algorithm = algorithm
        logger.info(f"✏️ Algorithm overridden from command line: {algorithm}")

    # Инициализация ClearML Task с информативными тегами
    task = Task.init(
        project_name="researchhub",
        task_name=f"train_{config.train.algorithm}",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks=True,  # Автологирование sklearn, pandas
    )

    # Добавляем информативные теги
    task.add_tags(
        [
            "stage:training",
            f"model:{config.train.algorithm}",
            "source:train_model.py",
            "task_type:classification",
            f"mlflow_experiment:{config.mlflow.experiment_name}",
        ]
    )

    # Получаем ClearML логгер
    clearml_logger = task.get_logger()

    # Настройка MLflow через Pydantic config
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    # Формируем уникальное имя для каждого запуска на основе алгоритма
    # Если run_name не задан или равен "baseline_model", генерируем автоматически
    if config.mlflow.run_name and config.mlflow.run_name != "baseline_model":
        run_name = config.mlflow.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.train.algorithm}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # Логируем параметры в ClearML
        train_params_dict = config.train.model_dump()
        task.connect(train_params_dict, name="training_params")
        task.connect(config.feature_engineering.model_dump(), name="feature_params")

        # Логируем параметры из Pydantic модели в MLflow
        mlflow.log_params(
            {
                "algorithm": config.train.algorithm,
                "test_size": config.train.test_size,
                "random_state": config.train.random_state,
                "cv_folds": config.train.cross_validation.get("folds", 5),
            }
        )

        # Логируем специфические параметры алгоритма из Pydantic
        if config.train.algorithm == "RandomForestClassifier":
            mlflow.log_params(config.train.random_forest.model_dump())
        elif config.train.algorithm == "SVM":
            svm_params = config.model_dump().get("train", {}).get("svm", {})
            mlflow.log_params(svm_params)
        elif config.train.algorithm == "LogisticRegression":
            lr_params = (
                config.model_dump().get("train", {}).get("logistic_regression", {})
            )
            mlflow.log_params(lr_params)

        # Логируем параметры предобработки признаков из Pydantic
        mlflow.log_params(
            config.feature_engineering.model_dump(
                exclude={"text_columns", "categorical_columns", "numerical_columns"}
            )
        )

        # Добавляем теги из Pydantic config
        for key, value in config.mlflow.tags.items():
            mlflow.set_tag(key, value)

        # Создаем признаки используя Pydantic config
        logger.info("Creating features...")
        X, y, tfidf = create_features(df, config)  # noqa: N806

        # Разделяем данные по параметрам из Pydantic
        X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
            X,
            y,
            test_size=config.train.test_size,
            random_state=config.train.random_state,
        )

        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")

        # Логируем информацию о датасетах в ClearML
        train_info = {
            "samples": int(X_train.shape[0]),
            "features": int(X_train.shape[1]),
            "classes": len(np.unique(y_train)),
        }
        test_info = {
            "samples": int(X_test.shape[0]),
            "features": int(X_test.shape[1]),
            "classes": len(np.unique(y_test)),
        }
        task.connect(train_info, name="dataset_train")
        task.connect(test_info, name="dataset_test")

        # Логируем распределение классов
        train_dist = pd.Series(y_train).value_counts().to_dict()
        test_dist = pd.Series(y_test).value_counts().to_dict()
        for cls, count in train_dist.items():
            clearml_logger.report_scalar(
                "Dataset Train", f"class_{cls}_count", count, 0
            )
        for cls, count in test_dist.items():
            clearml_logger.report_scalar("Dataset Test", f"class_{cls}_count", count, 0)

        # Инициализируем и обучаем модель используя Pydantic config
        logger.info(f"Training {config.train.algorithm} model...")
        model = get_model(config.train.algorithm, config)

        # Логируем параметры модели
        if hasattr(model, "get_params"):
            model_params = model.get_params()
            task.connect(model_params, name="model_params")

        model.fit(X_train, y_train)

        # Кросс-валидация с параметрами из Pydantic
        cv_folds = config.train.cross_validation.get("folds", 5)
        cv_scoring = config.train.cross_validation.get("scoring", "accuracy")
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=cv_scoring,
            n_jobs=-1,
        )

        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(
            f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        # Оцениваем на тестовой выборке
        logger.info("Evaluating model on test set...")
        test_metrics = evaluate_model(model, X_test, y_test)

        # Предсказания и визуализации
        y_pred = model.predict(X_test)

        # Получаем названия классов если возможно
        class_names = None
        if hasattr(model, "classes_"):
            class_names = [str(c) for c in model.classes_]

        # Логируем confusion matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        clearml_logger.report_matplotlib_figure(
            title="Confusion Matrix",
            series="confusion_matrix",
            figure=plt.gcf(),
            iteration=0,
        )
        plt.close()

        # Логируем важность признаков если доступно
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_n = min(20, len(importances))
            indices = np.argsort(importances)[-top_n:]

            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [f"Feature {i}" for i in indices])
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importances")
            plt.tight_layout()
            clearml_logger.report_matplotlib_figure(
                title="Feature Importance",
                series="top_features",
                figure=plt.gcf(),
                iteration=0,
            )
            plt.close()

        # Логируем метрики в ClearML (train и test)
        train_metrics = evaluate_model(model, X_train, y_train)

        # Train metrics
        for _metric_name, _metric_value in train_metrics.items():
            clearml_logger.report_scalar(
                "accuracy", "train", train_metrics.get("accuracy", 0), 0
            )
            clearml_logger.report_scalar(
                "f1_score", "train", train_metrics.get("f1_score", 0), 0
            )

        # Test metrics
        for _metric_name, _metric_value in test_metrics.items():
            clearml_logger.report_scalar(
                "accuracy", "test", test_metrics.get("accuracy", 0), 0
            )
            clearml_logger.report_scalar(
                "f1_score", "test", test_metrics.get("f1_score", 0), 0
            )
            clearml_logger.report_scalar(
                "precision", "test", test_metrics.get("precision", 0), 0
            )
            clearml_logger.report_scalar(
                "recall", "test", test_metrics.get("recall", 0), 0
            )

        # Cross-validation
        clearml_logger.report_scalar(
            "Cross Validation", "Mean Score", cv_scores.mean(), iteration=0
        )
        clearml_logger.report_scalar(
            "Cross Validation", "Std Score", cv_scores.std(), iteration=0
        )

        # Overfitting check
        overfitting = train_metrics.get("accuracy", 0) - test_metrics.get("accuracy", 0)
        clearml_logger.report_scalar("overfitting", "train-test-gap", overfitting, 0)

        # Добавляем accuracy в теги
        task.add_tags([f"test_accuracy:{test_metrics.get('accuracy', 0):.3f}"])

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
            registered_model_name=f"{config.mlflow.experiment_name}_model",
        )

        # Сохраняем модель локально
        model_path = Path(model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": model,
            "tfidf_vectorizer": tfidf,
            "feature_columns": {
                "text_columns": config.feature_engineering.text_columns,
                "numerical_columns": config.feature_engineering.numerical_columns,
                "categorical_columns": config.feature_engineering.categorical_columns,
            },
            "target_column": config.evaluate.target_column,
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
                "algorithm": config.train.algorithm,
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

        # Создаем метаданные модели из Pydantic config
        metadata_file = model_output.replace(".pkl", "_metadata.yaml")
        metadata = {
            "model_name": f"{config.mlflow.experiment_name}_model",
            "model_version": "1.0.0",
            "algorithm": config.train.algorithm,
            "training_date": datetime.now().isoformat(),
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "data_version": config.mlflow.tags.get("data_version", "unknown"),
            "performance": {
                "cv_accuracy": float(cv_scores.mean()),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_f1_score": float(test_metrics["f1_score"]),
            },
            "hyperparameters": config.train.model_dump(),
            "feature_engineering": config.feature_engineering.model_dump(),
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

        # Регистрация модели в ClearML Model Registry
        try:
            from clearml import OutputModel

            model_name = f"{config.train.algorithm}_model"

            # Создаем OutputModel
            output_model = OutputModel(
                task=task,
                framework="scikit-learn",
                name=model_name,
            )

            # Загружаем веса модели
            output_model.update_weights(
                weights_filename=model_output,
                auto_delete_file=False,
            )

            # Добавляем метаданные
            output_model.update_design(
                config_dict={
                    "algorithm": config.train.algorithm,
                    "test_accuracy": test_metrics.get("accuracy", 0),
                    "test_f1_score": test_metrics.get("f1_score", 0),
                    "cv_mean": float(cv_scores.mean()),
                    "training_samples": int(X_train.shape[0]),
                    "test_samples": int(X_test.shape[0]),
                    "features": int(X.shape[1]),
                    "classes": len(np.unique(y)),
                    "training_date": datetime.now().isoformat(),
                }
            )

            # Публикуем модель
            output_model.publish()

            logger.info(f"✅ Model registered in ClearML: {output_model.id}")

        except Exception as e:
            logger.warning(f"⚠️  Could not register model in ClearML: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"ClearML task ID: {task.id}")
        logger.info(
            f"ClearML results: http://localhost:8090/projects/researchhub/experiments/{task.id}"
        )

        return model, test_metrics


def main():
    """Главная функция с парсингом аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Train ML model with MLflow and ClearML logging"
    )
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
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["RandomForestClassifier", "SVM", "LogisticRegression"],
        help="ML algorithm to use (overrides params.yaml)",
    )

    args = parser.parse_args()

    # Обучаем модель
    train_model(
        args.input, args.params, args.model_output, args.metrics, args.algorithm
    )


if __name__ == "__main__":
    main()
