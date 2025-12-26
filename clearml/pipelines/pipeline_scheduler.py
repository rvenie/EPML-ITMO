#!/usr/bin/env python3
"""
Планировщик автоматических запусков ClearML пайплайнов
Простая система для запуска и мониторинга ML пайплайнов по расписанию
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from clearml import Task
from ml_pipeline import MLPipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline_scheduler.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Планировщик автоматических запусков ML пайплайнов."""

    def __init__(self, config_path: str = "clearml/config/scheduler_config.json"):
        """
        Инициализация планировщика.

        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.running_pipelines = {}
        self.last_run_file = Path("pipeline_scheduler_state.json")

    def _load_config(self) -> dict:
        """Загрузка конфигурации планировщика."""
        default_config = {
            "schedule": {
                "enabled": True,
                "interval_hours": 24,
            },
            "monitoring": {
                "check_interval_minutes": 15,
                "max_runtime_hours": 4,
            },
            "notifications": {
                "notify_on_success": True,
                "notify_on_failure": True,
            },
            "pipeline": {
                "project_name": "ResearchHub",
                "queue_name": "default",
            },
        }

        if Path(self.config_path).exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Конфигурация загружена: {self.config_path}")
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации: {e}")
        else:
            self._save_config(default_config)

        return default_config

    def _save_config(self, config: dict):
        """Сохранение конфигурации."""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")

    def should_start_pipeline(self) -> bool:
        """Проверяет, нужно ли запускать новый пайплайн."""
        if not self.config["schedule"]["enabled"]:
            return False

        if not self.last_run_file.exists():
            return True

        try:
            with open(self.last_run_file) as f:
                state = json.load(f)
                last_run = datetime.fromisoformat(state.get("last_run", "1900-01-01"))
                interval = timedelta(hours=self.config["schedule"]["interval_hours"])
                return datetime.now() - last_run >= interval
        except Exception as e:
            logger.warning(f"Ошибка чтения состояния: {e}")
            return True

    def start_pipeline(self) -> str | None:
        """Запускает новый ML пайплайн."""
        try:
            logger.info("Запуск нового ML пайплайна...")

            pipeline = MLPipeline(self.config["pipeline"]["project_name"])
            result = pipeline.run_training_experiment()

            if result and result.get("task_id"):
                task_id = result["task_id"]
                self.running_pipelines[task_id] = {
                    "start_time": datetime.now().isoformat(),
                    "status": "running",
                }

                self._update_state()
                self._send_notification(
                    "Пайплайн запущен",
                    f"ML пайплайн запущен успешно. ID: {task_id}",
                )
                return task_id

        except Exception as e:
            error_msg = f"Ошибка запуска пайплайна: {e}"
            logger.error(error_msg)
            self._send_notification("Ошибка запуска", error_msg)

        return None

    def check_running_pipelines(self):
        """Проверяет статус запущенных пайплайнов."""
        if not self.running_pipelines:
            return

        completed = []

        for task_id, info in self.running_pipelines.items():
            try:
                task = Task.get_task(task_id=task_id)
                current_status = task.get_status()

                # Проверка таймаута
                start_time = datetime.fromisoformat(info["start_time"])
                max_runtime = timedelta(
                    hours=self.config["monitoring"]["max_runtime_hours"]
                )

                if datetime.now() - start_time > max_runtime:
                    logger.warning(f"Пайплайн {task_id} превысил максимальное время")
                    self._send_notification(
                        "Таймаут пайплайна",
                        f"Пайплайн {task_id} работает слишком долго",
                    )
                    completed.append(task_id)
                    continue

                # Проверка завершения
                if current_status in ["completed", "failed", "stopped"]:
                    runtime = datetime.now() - start_time

                    if (
                        current_status == "completed"
                        and self.config["notifications"]["notify_on_success"]
                    ):
                        self._send_notification(
                            "Пайплайн завершен",
                            f"Пайплайн {task_id} выполнен успешно за {runtime}",
                        )
                    elif (
                        current_status in ["failed", "stopped"]
                        and self.config["notifications"]["notify_on_failure"]
                    ):
                        self._send_notification(
                            "Ошибка пайплайна",
                            f"Пайплайн {task_id} завершился с ошибкой: {current_status}",
                        )

                    completed.append(task_id)

                logger.info(f"Пайплайн {task_id}: {current_status}")

            except Exception as e:
                logger.error(f"Ошибка проверки пайплайна {task_id}: {e}")

        # Удаляем завершенные пайплайны
        for task_id in completed:
            del self.running_pipelines[task_id]

        if completed:
            self._update_state()

    def _update_state(self):
        """Обновляет состояние планировщика."""
        state = {
            "last_run": datetime.now().isoformat(),
            "running_pipelines": list(self.running_pipelines.keys()),
        }

        try:
            with open(self.last_run_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def _send_notification(self, title: str, message: str):
        """Отправляет уведомление через логи."""
        log_message = f"{title}: {message}"
        logger.info(log_message)


def main_scheduler_loop():
    """Основной цикл планировщика."""
    logger.info("Запуск планировщика ClearML пайплайнов...")

    scheduler = PipelineScheduler()
    logger.info("Планировщик готов к работе")

    try:
        while True:
            # Проверяем запуск нового пайплайна
            if scheduler.should_start_pipeline():
                scheduler.start_pipeline()

            # Мониторим запущенные пайплайны
            scheduler.check_running_pipelines()

            # Ждем до следующей проверки
            check_interval = scheduler.config["monitoring"]["check_interval_minutes"]
            time.sleep(check_interval * 60)

    except KeyboardInterrupt:
        logger.info("Планировщик остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise


def run_test_pipeline():
    """Запуск одного тестового пайплайна."""
    logger.info("Запуск тестового пайплайна...")

    scheduler = PipelineScheduler()
    task_id = scheduler.start_pipeline()

    if task_id:
        logger.info("✅ Тестовый пайплайн выполнен успешно!")
        logger.info(f"   Task ID: {task_id}")
        logger.info("   Результаты: http://localhost:8080")
        logger.info("   Проект: ResearchHub")
    else:
        logger.error("❌ Не удалось запустить тестовый пайплайн")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test_pipeline()
    else:
        main_scheduler_loop()
