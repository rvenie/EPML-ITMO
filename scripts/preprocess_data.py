#!/usr/bin/env python3
"""
Скрипт предобработки данных для датасета научных публикаций
Данный скрипт обрабатывает сырые данные публикаций и подготавливает их
для обучения ML модели.
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # type: ignore

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

    return round(impact_score, 3)


def preprocess_data(
    input_file: str, output_file: str, metadata_file: str | None = None
) -> None:
    """
    Основная функция предобработки.

    Args:
        input_file: Путь к входному CSV файлу
        output_file: Путь к выходному обработанному CSV файлу
        metadata_file: Опциональный путь для сохранения метаданных обработки
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

        try:
            with open(metadata_file, "w") as f:
                yaml.dump(processing_metadata, f, default_flow_style=False)
            logger.info(f"Saved processing metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    logger.info("Data preprocessing completed successfully!")


def main():
    """Главная функция с парсингом аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Preprocess research publications data"
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

    args = parser.parse_args()

    # Создаем выходную директорию если она не существует
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Запускаем предобработку
    preprocess_data(args.input, args.output, args.metadata)


if __name__ == "__main__":
    main()
