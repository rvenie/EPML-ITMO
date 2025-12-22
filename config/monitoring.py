"""
Система мониторинга выполнения ML пайплайна.
Отслеживает прогресс, логирует ошибки и отправляет уведомления.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class PipelineMonitor:
    """Система мониторинга выполнения ML пайплайна"""

    def __init__(self, log_file: str = "pipeline.log"):
        """
        Инициализация системы мониторинга

        Args:
            log_file: Путь к файлу логов
        """
        self.log_file = log_file
        self.start_time = None
        self.stages_status = {}
        self.current_stage = None

        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def start_pipeline(self, pipeline_name: str = "ML Pipeline"):
        """Начало выполнения пайплайна"""
        self.start_time = time.time()
        self.logger.info(f"🚀 Запуск пайплайна: {pipeline_name}")
        self.logger.info(
            f"⏰ Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def start_stage(self, stage_name: str, description: str = ""):
        """Начало выполнения этапа"""
        self.current_stage = stage_name
        self.stages_status[stage_name] = {
            "status": "running",
            "start_time": time.time(),
            "description": description,
        }
        self.logger.info(f"▶️ Начало этапа: {stage_name}")
        if description:
            self.logger.info(f"📄 Описание: {description}")

    def complete_stage(self, stage_name: str, metrics: Optional[Dict[str, Any]] = None):
        """Успешное завершение этапа"""
        if stage_name not in self.stages_status:
            self.logger.warning(f"⚠️ Этап {stage_name} не был запущен")
            return

        duration = time.time() - self.stages_status[stage_name]["start_time"]
        self.stages_status[stage_name].update(
            {
                "status": "completed",
                "end_time": time.time(),
                "duration": duration,
                "metrics": metrics or {},
            }
        )

        self.logger.info(f"✅ Этап завершен: {stage_name}")
        self.logger.info(f"⏱️ Время выполнения: {duration:.2f} секунд")

        if metrics:
            self.logger.info("📊 Метрики:")
            for key, value in metrics.items():
                self.logger.info(f"   {key}: {value}")

    def fail_stage(self, stage_name: str, error: str):
        """Ошибка на этапе выполнения"""
        if stage_name not in self.stages_status:
            self.logger.error(f"❌ Ошибка в неизвестном этапе: {stage_name}")
            return

        duration = time.time() - self.stages_status[stage_name]["start_time"]
        self.stages_status[stage_name].update(
            {
                "status": "failed",
                "end_time": time.time(),
                "duration": duration,
                "error": error,
            }
        )

        self.logger.error(f"❌ Ошибка на этапе: {stage_name}")
        self.logger.error(f"🔍 Детали ошибки: {error}")
        self.logger.error(f"⏱️ Время до ошибки: {duration:.2f} секунд")

    def complete_pipeline(self):
        """Завершение пайплайна"""
        if self.start_time is None:
            self.logger.warning("⚠️ Пайплайн не был запущен")
            return

        total_duration = time.time() - self.start_time
        successful_stages = sum(
            1 for s in self.stages_status.values() if s["status"] == "completed"
        )
        failed_stages = sum(
            1 for s in self.stages_status.values() if s["status"] == "failed"
        )

        self.logger.info("🏁 Пайплайн завершен")
        self.logger.info(f"⏱️ Общее время выполнения: {total_duration:.2f} секунд")
        self.logger.info(f"✅ Успешных этапов: {successful_stages}")
        self.logger.info(f"❌ Неудачных этапов: {failed_stages}")

        # Сохранение отчета
        self.save_report()

        # Уведомление о результатах
        self.send_notification(successful_stages, failed_stages, total_duration)

    def save_report(self):
        """Сохранение отчета о выполнении"""
        report = {
            "pipeline_execution": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": time.time() - self.start_time,
                "stages": self.stages_status,
            }
        }

        report_file = Path("reports/pipeline_execution_report.yaml")
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"📋 Отчет сохранен: {report_file}")

    def send_notification(
        self, successful_stages: int, failed_stages: int, duration: float
    ):
        """Отправка уведомления о результатах"""
        if failed_stages == 0:
            status = "SUCCESS"
            emoji = "✅"
        else:
            status = "FAILED"
            emoji = "❌"

        message = f"""
{emoji} ПАЙПЛАЙН {status}
━━━━━━━━━━━━━━━━━━━━
📊 Статистика:
   • Успешные этапы: {successful_stages}
   • Неудачные этапы: {failed_stages}
   • Время выполнения: {duration:.2f}с
━━━━━━━━━━━━━━━━━━━━
        """.strip()

        self.logger.info(message)

        # Сохранение уведомления в файл для внешних систем
        notification_file = Path("reports/notifications.log")
        notification_file.parent.mkdir(exist_ok=True)

        with open(notification_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - {status} - {duration:.2f}s\n")

    def get_stage_status(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Получение статуса конкретного этапа"""
        return self.stages_status.get(stage_name)

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Получение сводки по пайплайну"""
        if not self.stages_status:
            return {"status": "not_started"}

        statuses = [s["status"] for s in self.stages_status.values()]
        total_duration = sum(s.get("duration", 0) for s in self.stages_status.values())

        return {
            "total_stages": len(self.stages_status),
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "running": statuses.count("running"),
            "total_duration": total_duration,
            "stages": self.stages_status,
        }


def create_monitor_decorator(monitor: PipelineMonitor):
    """
    Создание декоратора для автоматического мониторинга функций

    Args:
        monitor: Экземпляр PipelineMonitor

    Returns:
        Декоратор для функций
    """

    def decorator(stage_name: str, description: str = ""):
        def wrapper(func):
            def inner(*args, **kwargs):
                monitor.start_stage(stage_name, description)
                try:
                    result = func(*args, **kwargs)
                    # Извлечение метрик из результата, если они есть
                    metrics = None
                    if isinstance(result, dict) and "metrics" in result:
                        metrics = result["metrics"]
                    monitor.complete_stage(stage_name, metrics)
                    return result
                except Exception as e:
                    monitor.fail_stage(stage_name, str(e))
                    raise

            return inner

        return wrapper

    return decorator


# Контекстный менеджер для автоматического мониторинга
class MonitoredStage:
    """Контекстный менеджер для автоматического мониторинга этапов"""

    def __init__(
        self, monitor: PipelineMonitor, stage_name: str, description: str = ""
    ):
        self.monitor = monitor
        self.stage_name = stage_name
        self.description = description

    def __enter__(self):
        self.monitor.start_stage(self.stage_name, self.description)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.monitor.complete_stage(self.stage_name)
        else:
            self.monitor.fail_stage(self.stage_name, str(exc_val))
        return False  # Не подавлять исключения


# Глобальный экземпляр монитора для удобства использования
pipeline_monitor = PipelineMonitor()
monitor_stage = create_monitor_decorator(pipeline_monitor)
