#!/usr/bin/env python3
"""
Система мониторинга ClearML пайплайнов
Отслеживает состояние серверов и выполнение пайплайнов
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests
from clearml import Task

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline_monitor.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ClearMLMonitor:
    """Мониторинг системы ClearML."""

    def __init__(self, project_name: str = "ResearchHub"):
        """
        Инициализация монитора.

        Args:
            project_name: Имя проекта ClearML
        """
        self.project_name = project_name
        self.servers = {
            "api_server": "http://localhost:8008",
            "web_server": "http://localhost:8080",
            "files_server": "http://localhost:8081",
        }
        self.metrics_file = Path("pipeline_metrics.json")

    def check_servers_health(self) -> dict[str, bool]:
        """Проверяет доступность ClearML серверов."""
        logger.info("Проверка доступности ClearML серверов...")

        health_status = {}

        for server_name, server_url in self.servers.items():
            try:
                if "8080" in server_url:
                    response = requests.get(server_url, timeout=10)
                else:
                    response = requests.get(f"{server_url}/debug.ping", timeout=10)
                # 200 = OK, 401 = requires auth (server is running)
                health_status[server_name] = response.status_code in (200, 401)

                status_icon = "✅" if health_status[server_name] else "❌"
                auth_note = " (auth)" if response.status_code == 401 else ""
                logger.info(f"{status_icon} {server_name}: {server_url}{auth_note}")

            except Exception as e:
                health_status[server_name] = False
                logger.error(f"❌ {server_name} недоступен: {e}")

        return health_status

    def get_pipeline_statistics(self) -> dict:
        """Получает статистику выполнения пайплайнов."""
        try:
            # Получаем задачи проекта
            tasks = Task.get_tasks(
                project_name=self.project_name,
                task_filter={"status": ["completed", "failed", "stopped", "running"]},
            )

            stats = {
                "total_tasks": len(tasks),
                "completed": 0,
                "failed": 0,
                "running": 0,
                "stopped": 0,
                "success_rate": 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Подсчет статусов
            for task in tasks:
                status = task.get_status()
                if status in stats:
                    stats[status] += 1

            # Процент успеха
            if stats["total_tasks"] > 0:
                success_rate = stats["completed"] / stats["total_tasks"] * 100
                stats["success_rate"] = round(success_rate, 2)

            logger.info(f"Статистика пайплайнов: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}

    def save_metrics(self, health_status: dict[str, bool], pipeline_stats: dict):
        """Сохраняет метрики мониторинга."""
        try:
            # Загружаем существующие метрики
            existing_metrics = []
            if self.metrics_file.exists():
                with open(self.metrics_file, encoding="utf-8") as f:
                    existing_metrics = json.load(f)

            # Добавляем новые метрики
            new_metric = {
                "timestamp": datetime.now().isoformat(),
                "servers_health": health_status,
                "pipeline_stats": pipeline_stats,
            }
            existing_metrics.append(new_metric)

            # Оставляем только последние 50 записей
            existing_metrics = existing_metrics[-50:]

            # Сохраняем
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(existing_metrics, f, indent=2, ensure_ascii=False)

            logger.info(f"Метрики сохранены: {self.metrics_file}")

        except Exception as e:
            logger.error(f"Ошибка сохранения метрик: {e}")

    def generate_report(self) -> str:
        """Генерирует отчет о состоянии системы."""
        logger.info("Генерация отчета о состоянии системы...")

        # Проверяем состояние
        health_status = self.check_servers_health()
        pipeline_stats = self.get_pipeline_statistics()

        # Формируем отчет
        report_lines = [
            "=" * 60,
            "ОТЧЕТ О СОСТОЯНИИ СИСТЕМЫ CLEARML",
            "=" * 60,
            f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "🖥️  СОСТОЯНИЕ СЕРВЕРОВ:",
        ]

        for server, status in health_status.items():
            icon = "✅" if status else "❌"
            status_text = "OK" if status else "НЕДОСТУПЕН"
            report_lines.append(f"   {icon} {server}: {status_text}")

        report_lines.append("")

        if pipeline_stats:
            report_lines.extend(
                [
                    "📊 СТАТИСТИКА ПАЙПЛАЙНОВ:",
                    f"   Всего задач: {pipeline_stats.get('total_tasks', 0)}",
                    f"   Завершено успешно: {pipeline_stats.get('completed', 0)}",
                    f"   Ошибки: {pipeline_stats.get('failed', 0)}",
                    f"   Выполняется: {pipeline_stats.get('running', 0)}",
                    f"   Остановлено: {pipeline_stats.get('stopped', 0)}",
                    f"   Процент успеха: {pipeline_stats.get('success_rate', 0)}%",
                    "",
                ]
            )

        # Рекомендации
        recommendations = self._get_recommendations(health_status, pipeline_stats)
        report_lines.extend(
            [
                "💡 РЕКОМЕНДАЦИИ:",
                recommendations,
                "=" * 60,
            ]
        )

        report_text = "\n".join(report_lines)

        # Сохраняем отчет
        report_file = (
            f"reports/health_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        )
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Отчет сохранен: {report_file}")
        return report_text

    def _get_recommendations(self, health_status: dict[str, bool], stats: dict) -> str:
        """Генерирует рекомендации по состоянию системы."""
        recommendations = []

        # Проверяем серверы
        failed_servers = [name for name, status in health_status.items() if not status]
        if failed_servers:
            recommendations.append(
                f"   ⚠️  Проверить серверы: {', '.join(failed_servers)}"
            )

        # Проверяем статистику
        if stats:
            success_rate = stats.get("success_rate", 100)
            if success_rate < 80:
                recommendations.append("   ⚠️  Низкий процент успешных выполнений")

            running_count = stats.get("running", 0)
            if running_count > 5:
                recommendations.append("   ⚠️  Много одновременно выполняющихся задач")

        if not recommendations:
            recommendations.append("   ✅ Система работает нормально")

        return "\n".join(recommendations)

    def run_monitoring_cycle(self, interval_minutes: int = 5):
        """Запускает цикл мониторинга."""
        logger.info(f"Запуск мониторинга (интервал: {interval_minutes} минут)")

        try:
            while True:
                # Проверяем состояние
                health_status = self.check_servers_health()
                pipeline_stats = self.get_pipeline_statistics()

                # Сохраняем метрики
                self.save_metrics(health_status, pipeline_stats)

                # Генерируем отчет каждый час
                if datetime.now().minute == 0:
                    report = self.generate_report()
                    logger.info(f"\n{report}")

                # Ждем до следующей проверки
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Мониторинг остановлен пользователем")
        except Exception as e:
            logger.error(f"Критическая ошибка мониторинга: {e}")
            raise


def main():
    """Главная функция."""
    import sys

    monitor = ClearMLMonitor("ResearchHub")

    if len(sys.argv) > 1 and sys.argv[1] == "report":
        # Генерация одного отчета
        report = monitor.generate_report()
        print(report)
    else:
        # Непрерывный мониторинг
        monitor.run_monitoring_cycle()


if __name__ == "__main__":
    main()
