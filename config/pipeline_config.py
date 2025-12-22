"""
Pydantic модели для валидации конфигураций ML пайплайна.
Обеспечивают типизацию и валидацию параметров из params.yaml.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Конфигурация для получения данных"""

    query: str = Field(..., description="Запрос для ArXiv API")
    max_results: int = Field(
        default=100, ge=1, le=2000, description="Максимальное количество результатов"
    )
    source: str = Field(default="arxiv", description="Источник данных")
    output_format: str = Field(default="csv", description="Формат выходных данных")

    @validator("source")
    def validate_source(cls, v):
        """Проверка поддерживаемых источников данных"""
        allowed_sources = ["arxiv", "pubmed", "semantic_scholar"]
        if v not in allowed_sources:
            raise ValueError(f"Источник должен быть одним из: {allowed_sources}")
        return v


class FeatureEngineeringConfig(BaseModel):
    """Конфигурация для инженерии признаков"""

    # TF-IDF параметры
    tfidf_max_features: int = Field(default=5000, ge=100, le=50000)
    ngram_range: List[int] = Field(default=[1, 2], min_items=2, max_items=2)
    min_df: int = Field(default=2, ge=1)
    max_df: float = Field(default=0.95, gt=0, le=1)
    use_tfidf: bool = Field(default=True)
    lowercase: bool = Field(default=True)
    stop_words: str = Field(default="english")

    # Колонки для обработки
    text_columns: List[str] = Field(..., description="Текстовые колонки для обработки")
    categorical_columns: List[str] = Field(..., description="Категориальные колонки")
    numerical_columns: List[str] = Field(..., description="Числовые колонки")

    @validator("ngram_range")
    def validate_ngram_range(cls, v):
        """Проверка корректности диапазона n-грамм"""
        if len(v) != 2 or v[0] > v[1] or v[0] < 1:
            raise ValueError(
                "ngram_range должен быть [min, max] где min >= 1 и min <= max"
            )
        return v


class RandomForestConfig(BaseModel):
    """Конфигурация для Random Forest"""

    n_estimators: int = Field(default=100, ge=1, le=1000)
    max_depth: Optional[int] = Field(default=10, ge=1)
    min_samples_split: int = Field(default=5, ge=2)
    min_samples_leaf: int = Field(default=2, ge=1)
    max_features: str = Field(default="sqrt")
    bootstrap: bool = Field(default=True)
    oob_score: bool = Field(default=True)

    @validator("max_features")
    def validate_max_features(cls, v):
        """Проверка корректности max_features"""
        allowed_values = ["sqrt", "log2", "auto", None]
        if v not in allowed_values and not isinstance(v, (int, float)):
            raise ValueError(
                f"max_features должен быть одним из: {allowed_values} или числом"
            )
        return v


class TrainingConfig(BaseModel):
    """Конфигурация для обучения модели"""

    algorithm: str = Field(..., description="Алгоритм машинного обучения")
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = Field(default=42)

    # Параметры кросс-валидации
    cross_validation: Dict[str, Any] = Field(default_factory=dict)

    # Параметры алгоритмов
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)

    @validator("algorithm")
    def validate_algorithm(cls, v):
        """Проверка поддерживаемых алгоритмов"""
        supported_algorithms = ["RandomForestClassifier", "SVM", "LogisticRegression"]
        if v not in supported_algorithms:
            raise ValueError(f"Алгоритм должен быть одним из: {supported_algorithms}")
        return v


class MLflowConfig(BaseModel):
    """Конфигурация для MLflow"""

    experiment_name: str = Field(..., description="Название эксперимента")
    run_name: Optional[str] = Field(default=None, description="Название запуска")
    tracking_uri: str = Field(
        default="file:./mlruns", description="URI для tracking server"
    )

    # Теги эксперимента
    tags: Dict[str, str] = Field(default_factory=dict)

    # Артефакты для логирования
    log_artifacts: List[str] = Field(default_factory=list)

    @validator("tracking_uri")
    def validate_tracking_uri(cls, v):
        """Проверка корректности tracking URI"""
        if not (
            v.startswith("file:") or v.startswith("http") or v.startswith("sqlite")
        ):
            raise ValueError("tracking_uri должен начинаться с file:, http или sqlite")
        return v


class EvaluationConfig(BaseModel):
    """Конфигурация для оценки модели"""

    target_column: str = Field(..., description="Целевая колонка")
    metrics: List[str] = Field(..., description="Метрики для оценки")

    # Параметры визуализации
    plot_style: str = Field(default="seaborn")
    dpi: int = Field(default=300, ge=72, le=600)
    figure_size: List[int] = Field(default=[12, 8], min_items=2, max_items=2)

    # Параметры отчета
    include_classification_report: bool = Field(default=True)
    include_confusion_matrix: bool = Field(default=True)
    include_feature_importance: bool = Field(default=True)
    top_features_to_show: int = Field(default=20, ge=1, le=100)

    @validator("metrics")
    def validate_metrics(cls, v):
        """Проверка поддерживаемых метрик"""
        supported_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for metric in v:
            if metric not in supported_metrics:
                raise ValueError(
                    f"Метрика {metric} не поддерживается. Доступны: {supported_metrics}"
                )
        return v


class PipelineConfig(BaseModel):
    """Главная конфигурация ML пайплайна"""

    data: DataConfig
    feature_engineering: FeatureEngineeringConfig
    train: TrainingConfig
    evaluate: EvaluationConfig
    mlflow: MLflowConfig

    # Дополнительные параметры
    reproducibility: Dict[str, Any] = Field(default_factory=dict)
    resources: Dict[str, Any] = Field(default_factory=dict)
    versioning: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Настройки Pydantic модели"""

        # Разрешить дополнительные поля для обратной совместимости
        extra = "allow"
        # Валидировать поля при присвоении
        validate_assignment = True
        # Использовать enum значения
        use_enum_values = True


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """
    Загрузка и валидация конфигурации из YAML файла

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        PipelineConfig: Валидированная конфигурация

    Raises:
        ValidationError: Если конфигурация не прошла валидацию
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return PipelineConfig(**config_data)


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """
    Проверка корректности файла конфигурации

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        bool: True если конфигурация корректна
    """
    try:
        load_config(config_path)
        return True
    except Exception as e:
        print(f"Ошибка валидации конфигурации: {e}")
        return False


# Пример использования валидации для разных алгоритмов
ALGORITHM_CONFIGS = {
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    },
    "SVM": {
        "kernel": ["rbf", "linear", "poly"],
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    "LogisticRegression": {
        "penalty": ["l1", "l2", "elasticnet"],
        "C": [0.1, 1.0, 10.0],
        "solver": ["liblinear", "saga", "lbfgs"],
    },
}
