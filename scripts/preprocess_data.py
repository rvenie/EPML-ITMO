#!/usr/bin/env python3
"""
Скрипт предобработки данных для датасета научных публикаций
Данный скрипт обрабатывает сырые данные публикаций и подготавливает их
для обучения ML модели.
"""

import argparse
import hashlib
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # type: ignore

# ClearML Dataset
try:
    from clearml import Dataset

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False

# Добавляем корневую директорию в путь для импорта config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорт Pydantic моделей для валидации
from config.pipeline_config import PipelineConfig, load_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preprocessing.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Очищает и нормализует текстовые данные.

    Args:
        text: Исходная текстовая строка

    Returns:
        Очищенная текстовая строка
    """
    if not isinstance(text, str):
        return ""

    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем специальные символы и лишние пробелы
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Убираем пробелы в начале и конце
    text = text.strip()

    return text


def extract_keywords_list(keywords: str) -> list:
    """
    Извлекает ключевые слова из строки, разделенной запятыми.

    Args:
        keywords: Строка ключевых слов, разделенных запятыми

    Returns:
        Список очищенных ключевых слов
    """
    if not isinstance(keywords, str):
        return []

    # Разделяем по запятой и очищаем каждое ключевое слово
    keyword_list = [kw.strip() for kw in keywords.split(",")]
    return [kw for kw in keyword_list if kw]  # Удаляем пустые ключевые слова


def categorize_journal(journal: str) -> str:
    """
    Категоризирует журнал по типу на основе названия.

    Args:
        journal: Название журнала

    Returns:
        Категория журнала
    """
    journal_lower = journal.lower()

    if any(term in journal_lower for term in ["nature", "science", "cell"]):
        return "top_tier"
    elif any(term in journal_lower for term in ["ieee", "acm", "springer"]):
        return "technical"
    elif any(term in journal_lower for term in ["medical", "medicine", "clinical"]):
        return "medical"
    else:
        return "other"


def calculate_impact_score(cited_by: int, year: int) -> float:
    """
    Вычисляет простой индекс влияния на основе цитирований и года публикации.

    Args:
        cited_by: Количество цитирований
        year: Год публикации

    Returns:
        Индекс влияния
    """
    current_year = datetime.now().year
    years_since_publication = max(1, current_year - year)

    # Нормализуем цитирования по годам с момента публикации
    citations_per_year = cited_by / years_since_publication

    # Простой индекс влияния (может быть улучшен более сложными метриками)
    impact_score = np.log1p(citations_per_year) * (1 / np.sqrt(years_since_publication))

    return float(round(impact_score, 3))


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
        logger.info(
            f"✅ Loaded and validated preprocessing parameters from {params_file}"
        )
        logger.info(f"   Text columns: {config.feature_engineering.text_columns}")
        logger.info(
            f"   Categorical columns: {config.feature_engineering.categorical_columns}"
        )
        logger.info(
            f"   Numerical columns: {config.feature_engineering.numerical_columns}"
        )
        return config
    except Exception as e:
        logger.error(f"❌ Error loading/validating parameters: {e}")
        raise


def preprocess_data(
    input_file: str,
    output_file: str,
    metadata_file: str | None = None,
    config: PipelineConfig | None = None,
) -> None:
    """
    Основная функция предобработки с использованием Pydantic конфигурации.

    Args:
        input_file: Путь к входному CSV файлу
        output_file: Путь к выходному обработанному CSV файлу
        metadata_file: Опциональный путь для сохранения метаданных обработки
        config: Валидированная Pydantic конфигурация (опционально)
    """
    logger.info(f"Starting data preprocessing: {input_file} -> {output_file}")

    # Загружаем данные
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records from {input_file}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Сохраняем исходную форму для метаданных
    original_shape = df.shape

    # Шаги предобработки
    logger.info("Cleaning text columns...")
    df["title_cleaned"] = df["title"].apply(clean_text)
    df["abstract_cleaned"] = df["abstract"].apply(clean_text)

    logger.info("Processing keywords...")
    df["keywords_list"] = df["keywords"].apply(extract_keywords_list)
    df["num_keywords"] = df["keywords_list"].apply(len)

    logger.info("Categorizing journals...")
    df["journal_category"] = df["journal"].apply(categorize_journal)

    logger.info("Calculating impact scores...")
    df["impact_score"] = df.apply(
        lambda row: calculate_impact_score(row["cited_by"], row["year"]), axis=1
    )

    logger.info("Creating additional features...")
    df["title_length"] = df["title"].str.len()
    df["abstract_length"] = df["abstract"].str.len()
    df["author_count"] = df["authors"].str.count(",") + 1

    # Create text length categories
    df["abstract_category"] = pd.cut(
        df["abstract_length"],
        bins=[0, 100, 300, 500, float("inf")],
        labels=["short", "medium", "long", "very_long"],
    )

    # Create citation categories
    df["citation_category"] = pd.cut(
        df["cited_by"],
        bins=[0, 50, 100, 200, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )

    # Create year categories
    df["year_category"] = pd.cut(
        df["year"],
        bins=[0, 2020, 2022, 2024, float("inf")],
        labels=["old", "recent", "very_recent", "latest"],
    )

    # Create author count categories
    df["author_count_category"] = pd.cut(
        df["author_count"],
        bins=[0, 3, 6, 10, float("inf")],
        labels=["few", "several", "many", "very_many"],
    )

    # Remove rows with missing critical data
    initial_count = len(df)
    df = df.dropna(subset=["title", "abstract", "doi"])
    final_count = len(df)

    if initial_count != final_count:
        logger.info(
            f"Removed {initial_count - final_count} rows with missing critical data"
        )

    # Save processed data
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        logger.info(f"Final dataset shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return

    # Create and save metadata
    if metadata_file:
        processing_metadata = {
            "processing_date": datetime.now().isoformat(),
            "input_file": input_file,
            "output_file": output_file,
            "original_shape": {
                "rows": int(original_shape[0]),
                "columns": int(original_shape[1]),
            },
            "processed_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "new_columns": [
                "title_cleaned",
                "abstract_cleaned",
                "keywords_list",
                "num_keywords",
                "journal_category",
                "impact_score",
                "title_length",
                "abstract_length",
                "author_count",
                "abstract_category",
                "citation_category",
                "year_category",
                "author_count_category",
            ],
            "processing_steps": [
                "Text cleaning and normalization",
                "Keyword extraction and counting",
                "Journal categorization",
                "Impact score calculation",
                "Feature engineering",
                "Missing data removal",
            ],
            "quality_metrics": {
                "completeness": float(
                    df.notna().sum().sum() / (df.shape[0] * df.shape[1])
                ),
                "rows_processed": int(final_count),
                "rows_removed": int(initial_count - final_count),
            },
        }

        # Добавляем параметры из Pydantic config если доступны
        if config:
            processing_metadata["config_parameters"] = {
                "text_columns": config.feature_engineering.text_columns,
                "categorical_columns": config.feature_engineering.categorical_columns,
                "numerical_columns": config.feature_engineering.numerical_columns,
                "tfidf_max_features": config.feature_engineering.tfidf_max_features,
                "ngram_range": config.feature_engineering.ngram_range,
                "min_df": config.feature_engineering.min_df,
                "max_df": config.feature_engineering.max_df,
            }

        try:
            with open(metadata_file, "w") as f:
                yaml.dump(processing_metadata, f, default_flow_style=False)
            logger.info(f"Saved processing metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    # Upload processed data to ClearML Dataset (optional)
    if CLEARML_AVAILABLE:
        try:
            logger.info("📦 Uploading processed data to ClearML Dataset...")

            # Compute hash for versioning
            with open(output_file, "rb") as f:
                data_hash = hashlib.md5(
                    f.to_csv(index=False).encode(), usedforsecurity=False
                ).hexdigest()[:8]

            # Try to find parent dataset (raw data)
            parent_dataset = None
            try:
                datasets = Dataset.list_datasets(
                    dataset_project="researchhub",
                    dataset_name="ArXiv Raw Publications",
                )
                if datasets:
                    parent_dataset = datasets[0]  # Get latest version
            except (IndexError, KeyError):
                parent_dataset = None  # No parent dataset found

            # Create processed dataset
            dataset = Dataset.create(
                dataset_name="ResearchHub Publications",
                dataset_project="researchhub",
                dataset_version=f"1.0-{data_hash[:8]}",
                parent_datasets=[parent_dataset] if parent_dataset else None,
                description=f"Preprocessed ArXiv publications dataset. Processing date: {datetime.now().isoformat()}",
            )

            # Add processed file
            dataset.add_files(path=output_file)
            if metadata_file and Path(metadata_file).exists():
                dataset.add_files(path=metadata_file)

            # Add metadata
            logger_ds = dataset.get_logger()

            # Preview
            logger_ds.report_table(
                title="Processed Dataset Preview",
                series="First 10 rows",
                table_plot=df.head(10),
                iteration=0,
            )

            # Statistics
            stats = {
                "original_rows": int(original_shape[0]),
                "processed_rows": int(df.shape[0]),
                "original_columns": int(original_shape[1]),
                "processed_columns": int(df.shape[1]),
                "rows_removed": int(original_shape[0] - df.shape[0]),
                "data_hash": data_hash,
            }

            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    logger_ds.report_single_value(name=key, value=value)

            # Distribution of new categories
            if "abstract_category" in df.columns:
                cat_dist = df["abstract_category"].value_counts().to_dict()
                logger_ds.report_histogram(
                    title="Abstract Category Distribution",
                    series="abstract_category",
                    values=list(cat_dist.values()),
                    xlabels=list(cat_dist.keys()),
                    yaxis="Number of samples",
                    iteration=0,
                )

            if "author_count_category" in df.columns:
                auth_dist = df["author_count_category"].value_counts().to_dict()
                logger_ds.report_histogram(
                    title="Author Count Category Distribution",
                    series="author_count_category",
                    values=list(auth_dist.values()),
                    xlabels=list(auth_dist.keys()),
                    yaxis="Number of samples",
                    iteration=1,
                )

            # Upload and finalize
            dataset.upload()
            dataset.finalize()

            logger.info(f"✅ Processed dataset uploaded: {dataset.id}")
            logger.info(f"   Version: 1.0-{data_hash[:8]}")

        except Exception as e:
            logger.warning(f"⚠️  Could not upload to ClearML: {e}")
            logger.info("   Continuing without ClearML upload...")

    logger.info("Data preprocessing completed successfully!")


def main():
    """Главная функция с парсингом аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Preprocess research publications data with Pydantic validation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/publications.csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/publications_processed.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/processing_metadata.yaml",
        help="Processing metadata output file",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="Parameters YAML file with Pydantic validation",
    )

    args = parser.parse_args()

    # Загружаем и валидируем параметры через Pydantic
    config = load_params(args.params)

    # Создаем выходную директорию если она не существует
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Запускаем предобработку с Pydantic конфигурацией
    preprocess_data(args.input, args.output, args.metadata, config)


if __name__ == "__main__":
    main()
