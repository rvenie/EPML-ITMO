"""
Pydantic конфигурация для ClearML Pipeline.
Валидация параметров моделей и настроек pipeline.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Конфигурация загрузки данных."""

    data_path: str = Field(
        default="data/processed/publications_processed.csv",
        description="Путь к обработанным данным",
    )
    test_size: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Доля тестовой выборки"
    )
    random_state: int = Field(default=42, description="Random seed")
    target_column: str | None = Field(
        default="author_count_category",
        description="Целевая колонка (None = автоопределение)",
    )


class LogisticRegressionConfig(BaseModel):
    """Параметры Logistic Regression."""

    C: float = Field(default=1.0, gt=0, description="Inverse regularization strength")
    max_iter: int = Field(default=1000, gt=0, description="Maximum iterations")
    solver: Literal["lbfgs", "saga", "newton-cg"] = Field(
        default="lbfgs", description="Solver algorithm"
    )


class RandomForestConfig(BaseModel):
    """Параметры Random Forest."""

    n_estimators: int = Field(default=50, ge=10, le=1000, description="Number of trees")
    max_depth: int | None = Field(
        default=8, ge=1, le=50, description="Maximum tree depth"
    )
    min_samples_split: int = Field(
        default=2, ge=2, description="Minimum samples to split"
    )


class GradientBoostingConfig(BaseModel):
    """Параметры Gradient Boosting."""

    n_estimators: int = Field(default=50, ge=10, le=1000, description="Number of trees")
    learning_rate: float = Field(default=0.05, gt=0, le=1, description="Learning rate")
    max_depth: int = Field(default=3, ge=1, le=20, description="Maximum tree depth")


class SVCConfig(BaseModel):
    """Параметры SVC."""

    kernel: Literal["linear", "rbf", "poly"] = Field(
        default="rbf", description="Kernel type"
    )
    C: float = Field(default=1.0, gt=0, description="Regularization parameter")
    gamma: Literal["scale", "auto"] | float = Field(
        default="scale", description="Kernel coefficient"
    )


class ClearMLConfig(BaseModel):
    """Конфигурация ClearML."""

    project_name: str = Field(default="researchhub", description="Имя проекта ClearML")
    pipeline_name: str = Field(
        default="researchhub-pipeline", description="Имя pipeline"
    )
    pipeline_version: str = Field(default="1.0", description="Версия pipeline")
    default_queue: str = Field(default="default", description="Очередь для выполнения")


class PipelineConfig(BaseModel):
    """Полная конфигурация ML Pipeline."""

    data: DataConfig = Field(default_factory=DataConfig)
    logreg: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)
    rf: RandomForestConfig = Field(default_factory=RandomForestConfig)
    gb: GradientBoostingConfig = Field(default_factory=GradientBoostingConfig)
    svc: SVCConfig = Field(default_factory=SVCConfig)
    clearml: ClearMLConfig = Field(default_factory=ClearMLConfig)

    class Config:
        """Pydantic config."""

        validate_assignment = True
        extra = "forbid"


def load_pipeline_config(config_path: str | None = None) -> PipelineConfig:
    """
    Загружает конфигурацию pipeline.

    Args:
        config_path: Путь к YAML файлу конфигурации (опционально)

    Returns:
        PipelineConfig: Валидированная конфигурация
    """
    if config_path:
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f)
        return PipelineConfig(**data)

    # Возвращаем дефолтную конфигурацию
    return PipelineConfig()
