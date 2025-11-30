#!/usr/bin/env python3
"""
Data Preprocessing Script for Research Publications Dataset
This script processes raw publication data and prepares it for ML model training.
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preprocessing.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_keywords_list(keywords: str) -> list:
    """
    Extract keywords from comma-separated string.

    Args:
        keywords: Comma-separated keywords string

    Returns:
        List of cleaned keywords
    """
    if not isinstance(keywords, str):
        return []

    # Split by comma and clean each keyword
    keyword_list = [kw.strip() for kw in keywords.split(",")]
    return [kw for kw in keyword_list if kw]  # Remove empty keywords


def categorize_journal(journal: str) -> str:
    """
    Categorize journal by type based on name.

    Args:
        journal: Journal name

    Returns:
        Journal category
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
    Calculate a simple impact score based on citations and recency.

    Args:
        cited_by: Number of citations
        year: Publication year

    Returns:
        Impact score
    """
    current_year = datetime.now().year
    years_since_publication = max(1, current_year - year)

    # Normalize citations by years since publication
    citations_per_year = cited_by / years_since_publication

    # Simple impact score (can be enhanced with more sophisticated metrics)
    impact_score = np.log1p(citations_per_year) * (1 / np.sqrt(years_since_publication))

    return round(impact_score, 3)


def preprocess_data(
    input_file: str, output_file: str, metadata_file: Optional[str] = None
) -> None:
    """
    Main preprocessing function.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output processed CSV file
        metadata_file: Optional path to save processing metadata
    """
    logger.info(f"Starting data preprocessing: {input_file} -> {output_file}")

    # Load data
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records from {input_file}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Store original shape for metadata
    original_shape = df.shape

    # Preprocessing steps
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
    """Main function with command line argument parsing."""
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

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    preprocess_data(args.input, args.output, args.metadata)


if __name__ == "__main__":
    main()
