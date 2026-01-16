#!/usr/bin/env python3
"""
Скрипт загрузки датасета в ClearML Dataset.
Создает версионированный датасет с метаданными.

Использование:
    python scripts/upload_dataset.py
    python scripts/upload_dataset.py --version 1.0
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from clearml import Dataset


def create_dataset(
    data_path: str = "data/processed/publications_processed.csv",
    dataset_name: str = "ResearchHub Publications",
    dataset_project: str = "researchhub",
    version: str = "1.0",
    description: str = "Processed ArXiv publications dataset",
):
    """
    Создает и загружает датасет в ClearML.

    Args:
        data_path: Путь к CSV файлу
        dataset_name: Имя датасета
        dataset_project: Проект в ClearML
        version: Версия датасета
        description: Описание датасета
    """
    print(f"📊 Загрузка датасета: {data_path}")

    # Проверяем существование файла
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Файл не найден: {data_path}")

    # Читаем датасет для получения метаданных
    df = pd.read_csv(data_path)
    print(f"✅ Загружено {len(df)} записей, {df.shape[1]} колонок")

    # Вычисляем hash для версионирования
    with open(data_path, "rb") as f:
        file_hash = hashlib.md5(f.read(), usedforsecurity=False).hexdigest()[:16]

    # Создаем ClearML Dataset
    print(f"📦 Создание ClearML Dataset версии {version}")
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=version,
        description=f"{description}\nMD5: {file_hash}",
    )

    # Добавляем файл
    dataset.add_files(path=data_path)
    print(f"✅ Добавлен файл: {data_path}")

    # Получаем logger для метаданных
    logger = dataset.get_logger()

    # 1. Preview датасета (первые 10 строк)
    print("📝 Добавление метаданных...")
    logger.report_table(
        title="Dataset Preview",
        series="First 10 rows",
        table_plot=df.head(10),
        iteration=0,
    )

    # 2. Статистика по колонкам
    stats = {
        "total_records": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
    }

    for key, value in stats.items():
        logger.report_single_value(name=key, value=value)

    # 3. Распределение по целевой переменной (если есть)
    target_columns = [
        "abstract_category",
        "citation_category",
        "author_count_category",
    ]

    for col in target_columns:
        if col in df.columns:
            distribution = df[col].value_counts().to_dict()

            # Гистограмма распределения
            logger.report_histogram(
                title=f"{col} Distribution",
                series=col,
                values=list(distribution.values()),
                xlabels=list(distribution.keys()),
                yaxis="Number of samples",
                iteration=0,
            )

            print(f"  - {col}: {len(distribution)} уникальных значений")

    # 4. Распределение по годам (если есть)
    if "year" in df.columns:
        year_dist = df["year"].value_counts().sort_index().to_dict()
        logger.report_histogram(
            title="Publications by Year",
            series="year",
            values=list(year_dist.values()),
            xlabels=[str(y) for y in year_dist.keys()],
            yaxis="Number of publications",
            iteration=0,
        )

    # 5. Добавляем метаданные как parameters
    dataset.get_logger().report_text(
        f"Dataset Statistics:\n"
        f"- Total records: {stats['total_records']}\n"
        f"- Columns: {stats['columns']}\n"
        f"- Missing values: {stats['missing_values']}\n"
        f"- Memory: {stats['memory_usage_mb']:.2f} MB\n"
        f"- Hash: {file_hash}"
    )

    # Загружаем датасет в ClearML
    print("☁️  Загрузка в ClearML...")
    dataset.upload()

    # Финализируем
    dataset.finalize()

    print("✅ Датасет загружен успешно!")
    print(f"   ID: {dataset.id}")
    print(f"   Версия: {version}")
    print(f"   Hash: {file_hash}")
    print("\n📊 Доступ к датасету:")
    print("   dataset = Dataset.get(")
    print(f'       dataset_name="{dataset_name}",')
    print(f'       dataset_project="{dataset_project}",')
    print(f'       dataset_version="{version}",')
    print("   )")

    return dataset


def update_dataset_version(
    base_dataset_id: str,
    data_path: str,
    new_version: str,
    changes_description: str = "Updated dataset",
):
    """
    Создает новую версию датасета на основе существующего.

    Args:
        base_dataset_id: ID родительского датасета
        data_path: Путь к новым данным
        new_version: Новая версия
        changes_description: Описание изменений
    """
    print(f"🔄 Создание новой версии датасета: {new_version}")

    # Получаем родительский датасет
    parent_dataset = Dataset.get(dataset_id=base_dataset_id)

    # Создаем новую версию
    dataset = Dataset.create(
        dataset_name=parent_dataset.name,
        dataset_project=parent_dataset.project,
        dataset_version=new_version,
        parent_datasets=[parent_dataset],
        description=changes_description,
    )

    # Добавляем новый файл
    dataset.add_files(path=data_path)

    # Загружаем метаданные (как в create_dataset)
    df = pd.read_csv(data_path)
    logger = dataset.get_logger()

    logger.report_table(
        title="Dataset Preview", series="Updated data", table_plot=df.head(10)
    )

    # Загружаем и финализируем
    dataset.upload()
    dataset.finalize()

    print(f"✅ Новая версия создана: {new_version}")
    return dataset


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Upload dataset to ClearML")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/publications_processed.csv",
        help="Path to dataset CSV file",
    )
    parser.add_argument("--version", type=str, default="1.0", help="Dataset version")
    parser.add_argument(
        "--name",
        type=str,
        default="ResearchHub Publications",
        help="Dataset name",
    )
    parser.add_argument(
        "--project", type=str, default="researchhub", help="ClearML project"
    )

    args = parser.parse_args()

    # Создаем и загружаем датасет
    dataset = create_dataset(
        data_path=args.data_path,
        dataset_name=args.name,
        dataset_project=args.project,
        version=args.version,
    )

    print(f"\n✅ Готово! Dataset ID: {dataset.id}")


if __name__ == "__main__":
    main()
