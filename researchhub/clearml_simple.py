"""
Простая интеграция ClearML без оверинжиниринга.
Использует ClearML SDK напрямую для логирования экспериментов.
"""

import os
from pathlib import Path
from typing import Any

from clearml import Task
from dotenv import load_dotenv


def load_clearml_env() -> None:
    """Загружает переменные окружения для ClearML из .env.clearml если он существует."""
    env_file = Path(__file__).parent.parent / ".env.clearml"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def init_clearml_task(
    task_name: str,
    project_name: str = "ResearchHub",
    task_type: str = Task.TaskTypes.training,
    tags: list[str] | None = None,
    reuse_last_task_id: bool = True,
) -> Task:
    """
    Инициализирует ClearML Task. Простая обертка над Task.init().

    Args:
        task_name: Имя задачи
        project_name: Имя проекта
        task_type: Тип задачи
        tags: Теги для группировки
        reuse_last_task_id: Переиспользовать последнюю задачу при перезапуске

    Returns:
        ClearML Task объект
    """
    # Загружаем env переменные
    load_clearml_env()

    # Если нет credentials, ClearML работает в offline режиме (для разработки)
    if not os.getenv("CLEARML_API_ACCESS_KEY"):
        print("⚠️  ClearML API credentials not found. Running in offline mode.")
        print("   To enable ClearML tracking:")
        print("   1. Start ClearML server: make clearml-server-up")
        print("   2. Get credentials from http://localhost:8090")
        print("   3. Run: clearml-init")

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        reuse_last_task_id=reuse_last_task_id,
        auto_connect_frameworks=True,  # Автологирование sklearn, pandas, etc
    )

    if tags:
        task.add_tags(tags)

    return task


def log_metrics(task: Task, metrics: dict[str, Any], iteration: int = 0) -> None:
    """Логирует метрики в ClearML."""
    if not task:
        return

    logger = task.get_logger()
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.report_scalar("metrics", metric_name, metric_value, iteration)


def log_params(task: Task, params: dict[str, Any]) -> None:
    """Логирует параметры в ClearML."""
    if not task:
        return

    task.connect(params)


def upload_artifact(task: Task, name: str, artifact_object: Any) -> None:
    """Загружает артефакт в ClearML."""
    if not task:
        return

    task.upload_artifact(name=name, artifact_object=artifact_object)
