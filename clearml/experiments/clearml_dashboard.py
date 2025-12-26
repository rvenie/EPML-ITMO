#!/usr/bin/env python3
"""
Встроенный дашборд ClearML для анализа экспериментов
Использует встроенные возможности ClearML Logger для создания дашбордов
"""

import logging
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from clearml import Task

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка стиля графиков
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ClearMLDashboard:
    """Встроенный дашборд ClearML для анализа экспериментов."""

    def __init__(self, project_name: str = "ResearchHub"):
        """
        Инициализация дашборда.

        Args:
            project_name: Имя проекта в ClearML
        """
        self.project_name = project_name
        self.dashboard_task = None
        self.dashboard_logger = None

    def create_dashboard_task(self) -> Task:
        """
        Создает задачу ClearML для дашборда.

        Returns:
            ClearML задача для дашборда
        """
        self.dashboard_task = Task.init(
            project_name=self.project_name,
            task_name=f"Experiments Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            task_type=Task.TaskTypes.monitor,
        )

        self.dashboard_logger = self.dashboard_task.get_logger()

        # Добавляем теги
        self.dashboard_task.add_tags(["dashboard", "analysis", "monitoring"])

        logger.info(f"Создана задача дашборда: {self.dashboard_task.id}")
        return self.dashboard_task

    def collect_experiments_data(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Собирает данные экспериментов из ClearML.

        Args:
            limit: Максимальное количество экспериментов

        Returns:
            Список данных экспериментов
        """
        try:
            # Получаем задачи проекта
            tasks = Task.get_tasks(
                project_name=self.project_name,
                task_filter={"status": ["completed", "failed"]},
                order_by=["-created"],
            )[:limit]

            experiments = []
            for task in tasks:
                # Пропускаем саму задачу дашборда
                if self.dashboard_task and task.id == self.dashboard_task.id:
                    continue

                exp_data = self._extract_task_data(task)
                if exp_data:
                    experiments.append(exp_data)

            logger.info(f"Собрано данных по {len(experiments)} экспериментам")
            return experiments

        except Exception as e:
            logger.error(f"Ошибка сбора данных экспериментов: {e}")
            return []

    def _extract_task_data(self, task: Task) -> dict[str, Any] | None:
        """
        Извлекает данные из задачи ClearML.

        Args:
            task: ClearML задача

        Returns:
            Словарь с данными задачи
        """
        try:
            data = {
                "task_id": task.id,
                "name": task.name,
                "status": task.get_status(),
                "created": task.data.created,
                "completed": task.data.completed,
                "runtime_minutes": 0,
                "metrics": {},
                "parameters": {},
            }

            # Вычисляем время выполнения
            if task.data.created and task.data.completed:
                runtime = task.data.completed - task.data.created
                data["runtime_minutes"] = runtime.total_seconds() / 60

            # Извлекаем параметры
            try:
                params = task.get_parameters()
                data["parameters"] = params or {}
            except Exception:
                pass

            # Извлекаем метрики
            try:
                scalars = task.get_reported_scalars()
                metrics = {}

                for title, series_dict in scalars.items():
                    for series, metric_data in series_dict.items():
                        if metric_data.get("y"):
                            # Берем последнее значение
                            final_value = metric_data["y"][-1]
                            metric_key = (
                                f"{title}_{series}" if title != series else series
                            )
                            metrics[metric_key] = float(final_value)

                data["metrics"] = metrics
            except Exception:
                pass

            return data

        except Exception as e:
            logger.error(f"Ошибка извлечения данных задачи {task.id}: {e}")
            return None

    def generate_overview_metrics(self, experiments: list[dict[str, Any]]):
        """
        Генерирует обзорные метрики в дашборд.

        Args:
            experiments: Список данных экспериментов
        """
        if not experiments:
            return

        # Общая статистика
        total_experiments = len(experiments)
        completed = len([e for e in experiments if e["status"] == "completed"])
        success_rate = completed / total_experiments if total_experiments > 0 else 0

        # Логируем основные метрики
        self.dashboard_logger.report_scalar(
            title="Обзор экспериментов",
            series="Всего экспериментов",
            value=total_experiments,
            iteration=0,
        )

        self.dashboard_logger.report_scalar(
            title="Обзор экспериментов",
            series="Успешно завершено",
            value=completed,
            iteration=0,
        )

        self.dashboard_logger.report_scalar(
            title="Обзор экспериментов",
            series="Процент успеха (%)",
            value=success_rate * 100,
            iteration=0,
        )

        # Средние времена выполнения
        runtimes = [
            e["runtime_minutes"] for e in experiments if e["runtime_minutes"] > 0
        ]
        if runtimes:
            avg_runtime = np.mean(runtimes)
            self.dashboard_logger.report_scalar(
                title="Производительность",
                series="Среднее время выполнения (мин)",
                value=avg_runtime,
                iteration=0,
            )

        logger.info("Обзорные метрики добавлены в дашборд")

    def create_metrics_comparison_charts(self, experiments: list[dict[str, Any]]):
        """
        Создает графики сравнения метрик.

        Args:
            experiments: Список данных экспериментов
        """
        if not experiments:
            return

        # Собираем все метрики
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp["metrics"].keys())

        # Фильтруем важные метрики
        important_metrics = [
            m
            for m in all_metrics
            if any(
                keyword in m.lower()
                for keyword in ["accuracy", "f1", "precision", "recall"]
            )
        ]

        for metric in important_metrics:
            values = []
            exp_names = []

            for exp in experiments:
                if metric in exp["metrics"]:
                    values.append(exp["metrics"][metric])
                    # Укорачиваем имя эксперимента
                    short_name = (
                        exp["name"][:30] + "..."
                        if len(exp["name"]) > 30
                        else exp["name"]
                    )
                    exp_names.append(short_name)

            if values:
                # Создаем барный график
                plt.figure(figsize=(12, 6))
                bars = plt.bar(
                    range(len(values)),
                    values,
                    color=sns.color_palette("husl", len(values)),
                )

                plt.title(
                    f"Сравнение экспериментов по метрике: {metric}",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xlabel("Эксперименты")
                plt.ylabel(metric)
                plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha="right")

                # Добавляем значения на бары
                for _, (bar, value) in enumerate(zip(bars, values, strict=False)):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                plt.tight_layout()

                # Логируем график в ClearML
                self.dashboard_logger.report_matplotlib_figure(
                    title="Сравнение метрик",
                    series=metric,
                    figure=plt.gcf(),
                    iteration=0,
                )

                plt.close()

                # Также логируем как гистограмму
                self.dashboard_logger.report_histogram(
                    title="Распределение метрик",
                    series=metric,
                    values=values,
                    labels=exp_names,
                    iteration=0,
                )

        logger.info("Графики сравнения метрик добавлены в дашборд")

    def create_runtime_analysis(self, experiments: list[dict[str, Any]]):
        """
        Создает анализ времени выполнения экспериментов.

        Args:
            experiments: Список данных экспериментов
        """
        runtimes = [
            e["runtime_minutes"] for e in experiments if e["runtime_minutes"] > 0
        ]
        exp_names = [
            e["name"][:20] + "..." if len(e["name"]) > 20 else e["name"]
            for e in experiments
            if e["runtime_minutes"] > 0
        ]

        if not runtimes:
            return

        # График времени выполнения по экспериментам
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(runtimes)), runtimes, color="skyblue", alpha=0.7)

        plt.title("Время выполнения экспериментов", fontsize=14, fontweight="bold")
        plt.xlabel("Эксперименты")
        plt.ylabel("Время (минуты)")
        plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha="right")

        # Добавляем значения на бары
        for bar, runtime in zip(bars, runtimes, strict=False):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{runtime:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Добавляем среднюю линию
        mean_runtime = np.mean(runtimes)
        plt.axhline(
            y=mean_runtime,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Среднее: {mean_runtime:.1f} мин",
        )
        plt.legend()

        plt.tight_layout()

        # Логируем в ClearML
        self.dashboard_logger.report_matplotlib_figure(
            title="Анализ времени выполнения",
            series="runtime_by_experiment",
            figure=plt.gcf(),
            iteration=0,
        )

        plt.close()

        # Гистограмма распределения времени
        plt.figure(figsize=(10, 6))
        plt.hist(
            runtimes,
            bins=min(10, len(runtimes)),
            color="lightgreen",
            alpha=0.7,
            edgecolor="black",
        )

        plt.title("Распределение времени выполнения", fontsize=14, fontweight="bold")
        plt.xlabel("Время (минуты)")
        plt.ylabel("Количество экспериментов")

        # Добавляем статистические линии
        mean_runtime = np.mean(runtimes)
        median_runtime = np.median(runtimes)

        plt.axvline(
            mean_runtime,
            color="red",
            linestyle="--",
            label=f"Среднее: {mean_runtime:.1f}",
        )
        plt.axvline(
            median_runtime,
            color="orange",
            linestyle="--",
            label=f"Медиана: {median_runtime:.1f}",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Логируем в ClearML
        self.dashboard_logger.report_matplotlib_figure(
            title="Анализ времени выполнения",
            series="runtime_distribution",
            figure=plt.gcf(),
            iteration=0,
        )

        plt.close()

        # Логируем статистику времени выполнения
        self.dashboard_logger.report_scalar(
            title="Статистика времени",
            series="Среднее время (мин)",
            value=mean_runtime,
            iteration=0,
        )

        self.dashboard_logger.report_scalar(
            title="Статистика времени",
            series="Медиана времени (мин)",
            value=median_runtime,
            iteration=0,
        )

        logger.info("Анализ времени выполнения добавлен в дашборд")

    def create_performance_matrix(self, experiments: list[dict[str, Any]]):
        """
        Создает матрицу производительности экспериментов.

        Args:
            experiments: Список данных экспериментов
        """
        if not experiments:
            return

        # Подготавливаем данные для матрицы
        performance_metrics = []
        exp_names = []

        # Определяем метрики производительности
        perf_metric_keywords = ["accuracy", "f1", "precision", "recall"]

        for exp in experiments:
            exp_perf = {}
            exp_name = (
                exp["name"][:25] + "..." if len(exp["name"]) > 25 else exp["name"]
            )

            # Собираем метрики производительности для этого эксперимента
            for metric_name, value in exp["metrics"].items():
                if any(
                    keyword in metric_name.lower() for keyword in perf_metric_keywords
                ):
                    clean_name = metric_name.replace("Test Metrics_", "").replace(
                        "test_", ""
                    )
                    exp_perf[clean_name] = value

            if exp_perf:  # Если есть хотя бы одна метрика производительности
                performance_metrics.append(exp_perf)
                exp_names.append(exp_name)

        if not performance_metrics:
            return

        # Создаем DataFrame для матрицы
        df = pd.DataFrame(performance_metrics, index=exp_names)
        df = df.fillna(0)  # Заполняем отсутствующие значения нулями

        if df.empty:
            return

        # Создаем тепловую карту
        plt.figure(figsize=(10, max(6, len(exp_names) * 0.4)))

        # Нормализация для лучшей визуализации
        df_normalized = (df - df.min()) / (df.max() - df.min())
        df_normalized = df_normalized.fillna(0)

        # Создаем heatmap
        sns.heatmap(
            df_normalized,
            annot=df,  # Показываем реальные значения
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Нормализованная производительность"},
            linewidths=0.5,
        )

        plt.title(
            "Матрица производительности экспериментов", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Метрики производительности")
        plt.ylabel("Эксперименты")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Логируем в ClearML
        self.dashboard_logger.report_matplotlib_figure(
            title="Матрица производительности",
            series="performance_heatmap",
            figure=plt.gcf(),
            iteration=0,
        )

        plt.close()

        logger.info("Матрица производительности добавлена в дашборд")

    def create_experiments_timeline(self, experiments: list[dict[str, Any]]):
        """
        Создает временную шкалу экспериментов.

        Args:
            experiments: Список данных экспериментов
        """
        # Фильтруем эксперименты с датами
        dated_experiments = [e for e in experiments if e["created"]]

        if not dated_experiments:
            return

        # Сортируем по дате
        dated_experiments.sort(key=lambda x: x["created"])

        # Группируем по дням
        daily_counts = {}
        status_counts = {"completed": 0, "failed": 0}

        for exp in dated_experiments:
            date = exp["created"].date()
            daily_counts[date] = daily_counts.get(date, 0) + 1

            if exp["status"] in status_counts:
                status_counts[exp["status"]] += 1

        # График активности по дням
        dates = list(daily_counts.keys())
        counts = list(daily_counts.values())

        if dates:
            plt.figure(figsize=(12, 6))
            plt.plot(dates, counts, marker="o", linewidth=2, markersize=6)
            plt.title(
                "Активность экспериментов по дням", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Дата")
            plt.ylabel("Количество экспериментов")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # Логируем в ClearML
            self.dashboard_logger.report_matplotlib_figure(
                title="Временная шкала",
                series="daily_activity",
                figure=plt.gcf(),
                iteration=0,
            )

            plt.close()

        # Круговая диаграмма статусов
        if any(status_counts.values()):
            plt.figure(figsize=(8, 8))

            statuses = []
            values = []
            colors = []

            for status, count in status_counts.items():
                if count > 0:
                    statuses.append(f"{status}\n({count})")
                    values.append(count)
                    colors.append(
                        "lightgreen" if status == "completed" else "lightcoral"
                    )

            plt.pie(
                values, labels=statuses, colors=colors, autopct="%1.1f%%", startangle=90
            )
            plt.title(
                "Распределение статусов экспериментов", fontsize=14, fontweight="bold"
            )

            # Логируем в ClearML
            self.dashboard_logger.report_matplotlib_figure(
                title="Статусы экспериментов",
                series="status_distribution",
                figure=plt.gcf(),
                iteration=0,
            )

            plt.close()

        logger.info("Временная шкала добавлена в дашборд")

    def create_summary_table(self, experiments: list[dict[str, Any]]):
        """
        Создает сводную таблицу экспериментов.

        Args:
            experiments: Список данных экспериментов
        """
        if not experiments:
            return

        # Подготавливаем данные для таблицы
        table_data = []

        for exp in experiments:
            row = {
                "Experiment": exp["name"][:40] + "..."
                if len(exp["name"]) > 40
                else exp["name"],
                "Status": exp["status"],
                "Runtime (min)": round(exp["runtime_minutes"], 2)
                if exp["runtime_minutes"] > 0
                else "N/A",
                "Created": exp["created"].strftime("%Y-%m-%d %H:%M")
                if exp["created"]
                else "N/A",
            }

            # Добавляем основные метрики
            for metric_name, value in exp["metrics"].items():
                if "accuracy" in metric_name.lower():
                    row["Accuracy"] = round(value, 4)
                    break

            for metric_name, value in exp["metrics"].items():
                if "f1" in metric_name.lower():
                    row["F1 Score"] = round(value, 4)
                    break

            table_data.append(row)

        # Создаем DataFrame
        df = pd.DataFrame(table_data)

        # Сортируем по точности если есть
        if "Accuracy" in df.columns:
            df = df.sort_values("Accuracy", ascending=False, na_last=True)

        # Логируем таблицу в ClearML
        self.dashboard_logger.report_table(
            title="Сводная таблица экспериментов",
            series="experiments_summary",
            table_plot=df,
            iteration=0,
        )

        logger.info("Сводная таблица добавлена в дашборд")

    def generate_dashboard(self) -> str:
        """
        Генерирует полный дашборд экспериментов в ClearML.

        Returns:
            ID задачи дашборда
        """
        try:
            # Создаем задачу дашборда
            self.create_dashboard_task()

            # Собираем данные экспериментов
            logger.info("Сбор данных экспериментов...")
            experiments = self.collect_experiments_data()

            if not experiments:
                logger.warning("Нет экспериментов для анализа")
                return self.dashboard_task.id

            logger.info(f"Создание дашборда для {len(experiments)} экспериментов...")

            # Генерируем все компоненты дашборда
            self.generate_overview_metrics(experiments)
            self.create_metrics_comparison_charts(experiments)
            self.create_runtime_analysis(experiments)
            self.create_performance_matrix(experiments)
            self.create_experiments_timeline(experiments)
            self.create_summary_table(experiments)

            # Добавляем метаданные
            dashboard_metadata = {
                "project_name": self.project_name,
                "experiments_analyzed": len(experiments),
                "generation_time": datetime.now().isoformat(),
                "dashboard_version": "1.0.0",
            }

            self.dashboard_task.connect(dashboard_metadata, name="dashboard_metadata")

            logger.info(f"Дашборд создан успешно! Task ID: {self.dashboard_task.id}")
            logger.info(
                "Откройте веб-интерфейс ClearML для просмотра: http://localhost:8080"
            )

            return self.dashboard_task.id

        except Exception as e:
            logger.error(f"Ошибка создания дашборда: {e}")
            if self.dashboard_task:
                self.dashboard_task.mark_failed(status_reason=str(e))
            raise


def main():
    """Демонстрация создания дашборда ClearML."""
    try:
        logger.info("Создание встроенного дашборда ClearML...")

        # Создаем дашборд
        dashboard = ClearMLDashboard("ResearchHub")
        task_id = dashboard.generate_dashboard()

        print("\n" + "=" * 80)
        print("✅ ДАШБОРД CLEARML СОЗДАН УСПЕШНО!")
        print("=" * 80)
        print(f"🆔 Task ID: {task_id}")
        print("🌐 Откройте веб-интерфейс ClearML: http://localhost:8080")
        print(
            f"📊 Найдите задачу: 'Experiments Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}'"
        )
        print("📈 Все графики и таблицы доступны во вкладках:")
        print("   - Scalars: метрики и обзорная информация")
        print("   - Plots: графики сравнения и анализ")
        print("   - Debug Samples: таблицы данных")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Ошибка в демонстрации дашборда: {e}")


if __name__ == "__main__":
    main()
