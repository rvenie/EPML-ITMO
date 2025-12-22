#!/usr/bin/env python3
"""
Автоматизированный ML пайплайн с валидацией конфигурации и мониторингом.
Простой скрипт для запуска всех этапов обработки данных и обучения модели.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import logging
import multiprocessing as mp
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from config.monitoring import MonitoredStage, pipeline_monitor
from config.pipeline_config import load_config, validate_config_file


def setup_logging():
    """Настройка базового логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def run_dvc_stage(stage_name: str) -> dict[str, any]:
    """
    Запуск одного этапа DVC пайплайна

    Args:
        stage_name: Название этапа в dvc.yaml

    Returns:
        Dict с результатами выполнения
    """
    try:
        # Запуск конкретного этапа DVC
        result = subprocess.run(
            ["dvc", "repro", stage_name], capture_output=True, text=True, check=True
        )

        return {
            "stage": stage_name,
            "status": "success",
            "output": result.stdout,
            "metrics": {"exit_code": result.returncode},
        }

    except subprocess.CalledProcessError as e:
        return {
            "stage": stage_name,
            "status": "failed",
            "error": e.stderr,
            "metrics": {"exit_code": e.returncode},
        }


def run_sequential_pipeline(config_path: str = "params.yaml"):
    """
    Последовательный запуск всех этапов пайплайна

    Args:
        config_path: Путь к файлу конфигурации
    """
    # Валидация конфигурации
    print("🔍 Проверка конфигурации...")
    if not validate_config_file(config_path):
        print("❌ Ошибка валидации конфигурации")
        return False

    config = load_config(config_path)
    print(f"✅ Конфигурация валидна для {config.mlflow.experiment_name}")

    # Запуск пайплайна с мониторингом
    pipeline_monitor.start_pipeline("Automated ML Pipeline")

    # Этапы пайплайна в правильном порядке
    stages = ["fetch_data", "preprocess", "train"]

    success = True
    for stage in stages:
        with MonitoredStage(pipeline_monitor, stage, f"Выполнение этапа {stage}"):
            print(f"\n▶️ Запуск этапа: {stage}")
            result = run_dvc_stage(stage)

            if result["status"] == "success":
                print(f"✅ Этап {stage} завершен успешно")
            else:
                print(f"❌ Ошибка на этапе {stage}: {result['error']}")
                success = False
                break

    pipeline_monitor.complete_pipeline()
    return success


def run_parallel_pipeline(config_path: str = "params.yaml"):
    """
    Параллельный запуск независимых этапов пайплайна

    Args:
        config_path: Путь к файлу конфигурации
    """
    # Валидация конфигурации
    print("🔍 Проверка конфигурации...")
    if not validate_config_file(config_path):
        print("❌ Ошибка валидации конфигурации")
        return False

    config = load_config(config_path)
    print(f"✅ Конфигурация валидна для {config.mlflow.experiment_name}")

    pipeline_monitor.start_pipeline("Parallel ML Pipeline")

    # Этапы с их зависимостями
    pipeline_stages = {
        "fetch_data": [],  # Нет зависимостей
        "preprocess": ["fetch_data"],  # Зависит от fetch_data
        "train": ["preprocess"],  # Зависит от preprocess
    }

    completed_stages = set()
    success = True

    # Выполняем этапы, соблюдая зависимости
    while len(completed_stages) < len(pipeline_stages) and success:
        # Находим этапы, готовые к выполнению
        ready_stages = [
            stage
            for stage, deps in pipeline_stages.items()
            if stage not in completed_stages
            and all(dep in completed_stages for dep in deps)
        ]

        if not ready_stages:
            break

        # Запускаем готовые этапы параллельно
        with ProcessPoolExecutor(
            max_workers=min(len(ready_stages), mp.cpu_count())
        ) as executor:
            # Отправляем задачи на выполнение
            future_to_stage = {}
            for stage in ready_stages:
                pipeline_monitor.start_stage(stage, f"Параллельное выполнение {stage}")
                future = executor.submit(run_dvc_stage, stage)
                future_to_stage[future] = stage

            # Собираем результаты
            for future in as_completed(future_to_stage):
                stage = future_to_stage[future]
                result = future.result()

                if result["status"] == "success":
                    pipeline_monitor.complete_stage(stage, result.get("metrics"))
                    completed_stages.add(stage)
                    print(f"✅ Этап {stage} завершен успешно")
                else:
                    pipeline_monitor.fail_stage(
                        stage, result.get("error", "Неизвестная ошибка")
                    )
                    print(f"❌ Ошибка на этапе {stage}")
                    success = False

    pipeline_monitor.complete_pipeline()
    return success


def check_cache_status():
    """Проверка статуса кэша DVC"""
    print("🔍 Проверка кэша DVC...")
    try:
        result = subprocess.run(
            ["dvc", "status"], capture_output=True, text=True, check=True
        )

        if "Data and pipelines are up to date" in result.stdout:
            print("✅ Все данные и пайплайны актуальны (используется кэш)")
            return True
        else:
            print("🔄 Обнаружены изменения, требуется выполнение этапов")
            print(result.stdout)
            return False

    except subprocess.CalledProcessError:
        print("⚠️ Не удалось проверить статус DVC")
        return False


def main():
    """Главная функция запуска пайплайна"""
    parser = argparse.ArgumentParser(description="Автоматизированный ML пайплайн")
    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Путь к файлу конфигурации (по умолчанию: params.yaml)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Запуск с параллельным выполнением независимых этапов",
    )
    parser.add_argument(
        "--force", action="store_true", help="Принудительный запуск без проверки кэша"
    )

    args = parser.parse_args()

    setup_logging()

    print("🚀 Запуск автоматизированного ML пайплайна")
    print(f"📁 Конфигурация: {args.config}")
    print(f"⚡ Режим: {'Параллельный' if args.parallel else 'Последовательный'}")

    # Проверка кэша (если не принудительный запуск)
    if not args.force:
        if check_cache_status():
            print("✅ Пайплайн актуален, выполнение не требуется")
            return

    # Выбор режима выполнения
    if args.parallel:
        success = run_parallel_pipeline(args.config)
    else:
        success = run_sequential_pipeline(args.config)

    # Итоговый результат
    if success:
        print("\n🎉 Пайплайн выполнен успешно!")
        print("📊 Проверить результаты можно в MLflow UI: http://localhost:3000")
    else:
        print("\n💥 Пайплайн завершился с ошибками")
        print("📋 Подробности в файлах логов")
        sys.exit(1)


if __name__ == "__main__":
    main()
