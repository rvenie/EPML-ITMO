#!/usr/bin/env python3
"""
Система сравнения экспериментов ClearML с визуализацией
Автоматическое сравнение и анализ результатов множественных экспериментов
"""

import json
import logging
from datetime import datetime
from pathlib import Path
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
plt.style.use("default")
sns.set_palette("husl")


class ExperimentComparison:
    """Система сравнения экспериментов ClearML."""

    def __init__(self, project_name: str = "ResearchHub"):
        """
        Инициализация системы сравнения.

        Args:
            project_name: Имя проекта в ClearML
        """
        self.project_name = project_name
        self.experiments_data = []
        logger.info(
            f"Инициализация сравнения экспериментов для проекта: {project_name}"
        )

    def collect_experiments_data(
        self,
        task_ids: list[str] | None = None,
        experiment_names: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Собирает данные экспериментов из ClearML.

        Args:
            task_ids: Список ID задач (если None, берет последние эксперименты)
            experiment_names: Фильтр по именам экспериментов
            limit: Максимальное количество экспериментов

        Returns:
            Список данных экспериментов
        """
        try:
            experiments = []

            if task_ids:
                # Получаем конкретные задачи по ID
                for task_id in task_ids:
                    try:
                        task = Task.get_task(task_id=task_id)
                        exp_data = self._extract_experiment_data(task)
                        if exp_data:
                            experiments.append(exp_data)
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить задачу {task_id}: {e}")
            else:
                # Получаем последние эксперименты проекта
                tasks = Task.get_tasks(
                    project_name=self.project_name,
                    task_filter={"status": ["completed"]},
                    order_by=["-created"],
                )[:limit]

                for task in tasks:
                    # Фильтруем по именам если указано
                    if experiment_names and not any(
                        name in task.name for name in experiment_names
                    ):
                        continue

                    exp_data = self._extract_experiment_data(task)
                    if exp_data:
                        experiments.append(exp_data)

            self.experiments_data = experiments
            logger.info(f"Собрано данных по {len(experiments)} экспериментам")

            return experiments

        except Exception as e:
            logger.error(f"Ошибка сбора данных экспериментов: {e}")
            return []

    def _extract_experiment_data(self, task: Task) -> dict[str, Any] | None:
        """
        Извлекает данные из одного эксперимента.

        Args:
            task: ClearML задача

        Returns:
            Словарь с данными эксперимента
        """
        try:
            # Основная информация
            experiment_data = {
                "task_id": task.id,
                "name": task.name,
                "created": task.data.created,
                "completed": task.data.completed,
                "status": task.get_status(),
                "runtime": None,
                "metrics": {},
                "parameters": {},
                "artifacts": [],
            }

            # Время выполнения
            if task.data.created and task.data.completed:
                runtime = task.data.completed - task.data.created
                experiment_data["runtime"] = runtime.total_seconds()

            # Извлекаем параметры
            try:
                params = task.get_parameters()
                experiment_data["parameters"] = params
            except Exception:
                pass

            # Извлекаем метрики
            try:
                scalars = task.get_reported_scalars()
                metrics = {}

                # Извлекаем финальные значения метрик
                for title, series_dict in scalars.items():
                    for series, data in series_dict.items():
                        if data.get("y"):
                            # Берем последнее значение
                            final_value = data["y"][-1]
                            metric_key = (
                                f"{title}_{series}" if title != series else series
                            )
                            metrics[metric_key] = float(final_value)

                experiment_data["metrics"] = metrics
            except Exception:
                pass

            # Список артефактов
            try:
                artifacts = task.artifacts
                experiment_data["artifacts"] = (
                    list(artifacts.keys()) if artifacts else []
                )
            except Exception:
                pass

            return experiment_data

        except Exception as e:
            logger.error(f"Ошибка извлечения данных из задачи {task.id}: {e}")
            return None

    def create_metrics_comparison_table(self) -> pd.DataFrame:
        """
        Создает таблицу сравнения метрик экспериментов.

        Returns:
            DataFrame с сравнением метрик
        """
        if not self.experiments_data:
            logger.warning("Нет данных для сравнения")
            return pd.DataFrame()

        try:
            # Подготавливаем данные для таблицы
            rows = []
            for exp in self.experiments_data:
                row = {
                    "Experiment": exp["name"][:50] + "..."
                    if len(exp["name"]) > 50
                    else exp["name"],
                    "Task_ID": exp["task_id"][:8] + "...",
                    "Status": exp["status"],
                    "Runtime_min": round(exp["runtime"] / 60, 2)
                    if exp["runtime"]
                    else None,
                    "Created": exp["created"].strftime("%Y-%m-%d %H:%M")
                    if exp["created"]
                    else None,
                }

                # Добавляем метрики
                for metric_name, value in exp["metrics"].items():
                    if isinstance(value, int | float):
                        row[metric_name] = round(float(value), 4)

                rows.append(row)

            df = pd.DataFrame(rows)

            # Сортируем по лучшей метрике (если есть accuracy)
            if "test_accuracy" in df.columns:
                df = df.sort_values("test_accuracy", ascending=False)
            elif "Test Metrics_accuracy" in df.columns:
                df = df.sort_values("Test Metrics_accuracy", ascending=False)
            elif any("accuracy" in col for col in df.columns):
                accuracy_col = next(
                    col for col in df.columns if "accuracy" in col.lower()
                )
                df = df.sort_values(accuracy_col, ascending=False)

            logger.info(f"Создана таблица сравнения с {len(df)} экспериментами")
            return df

        except Exception as e:
            logger.error(f"Ошибка создания таблицы сравнения: {e}")
            return pd.DataFrame()

    def plot_metrics_comparison(
        self, metrics_to_compare: list[str] = None, save_path: str = None
    ) -> str:
        """
        Создает визуализацию сравнения метрик.

        Args:
            metrics_to_compare: Список метрик для сравнения
            save_path: Путь для сохранения графика

        Returns:
            Путь к сохраненному графику
        """
        if not self.experiments_data:
            logger.warning("Нет данных для визуализации")
            return ""

        try:
            df = self.create_metrics_comparison_table()
            if df.empty:
                return ""

            # Определяем метрики для сравнения
            if metrics_to_compare is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                metrics_to_compare = [
                    col
                    for col in numeric_columns
                    if col not in ["Runtime_min"] and "accuracy" in col.lower()
                ][:4]

            if not metrics_to_compare:
                metrics_to_compare = df.select_dtypes(include=[np.number]).columns[:4]

            # Создаем subplot для каждой метрики
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            for i, metric in enumerate(metrics_to_compare):
                if i >= 4:  # Максимум 4 графика
                    break

                if metric in df.columns:
                    ax = axes[i]

                    # Барный график для сравнения
                    data_to_plot = df.nlargest(10, metric) if len(df) > 10 else df

                    bars = ax.bar(
                        range(len(data_to_plot)),
                        data_to_plot[metric],
                        color=sns.color_palette("husl", len(data_to_plot)),
                    )

                    ax.set_title(
                        f"Сравнение по метрике: {metric}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.set_xlabel("Эксперименты")
                    ax.set_ylabel(metric)
                    ax.tick_params(axis="x", rotation=45)

                    # Подписи значений на барах
                    for _, bar in enumerate(bars):
                        height = bar.get_height()
                        if not pd.isna(height):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height,
                                f"{height:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                            )

                    # Подписи экспериментов
                    exp_names = [
                        name[:15] + "..." if len(name) > 15 else name
                        for name in data_to_plot["Experiment"]
                    ]
                    ax.set_xticks(range(len(exp_names)))
                    ax.set_xticklabels(exp_names, rotation=45, ha="right")

            # Удаляем пустые subplot'ы
            for i in range(len(metrics_to_compare), 4):
                fig.delaxes(axes[i])

            plt.tight_layout()

            # Сохраняем график
            if save_path is None:
                save_path = (
                    f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"График сравнения сохранен: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Ошибка создания визуализации: {e}")
            return ""

    def plot_runtime_analysis(self, save_path: str = None) -> str:
        """
        Создает анализ времени выполнения экспериментов.

        Args:
            save_path: Путь для сохранения графика

        Returns:
            Путь к сохраненному графику
        """
        try:
            df = self.create_metrics_comparison_table()
            if df.empty or "Runtime_min" not in df.columns:
                logger.warning("Нет данных о времени выполнения")
                return ""

            # Убираем эксперименты без данных о времени
            df_runtime = df.dropna(subset=["Runtime_min"])
            if df_runtime.empty:
                logger.warning("Нет валидных данных о времени выполнения")
                return ""

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # График времени выполнения
            bars = ax1.bar(
                range(len(df_runtime)),
                df_runtime["Runtime_min"],
                color=sns.color_palette("viridis", len(df_runtime)),
            )
            ax1.set_title("Время выполнения экспериментов (мин)", fontweight="bold")
            ax1.set_xlabel("Эксперименты")
            ax1.set_ylabel("Время (минуты)")

            # Подписи
            exp_names = [
                name[:10] + "..." if len(name) > 10 else name
                for name in df_runtime["Experiment"]
            ]
            ax1.set_xticks(range(len(exp_names)))
            ax1.set_xticklabels(exp_names, rotation=45, ha="right")

            # Значения на барах
            for _, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Гистограмма распределения времени
            ax2.hist(
                df_runtime["Runtime_min"],
                bins=min(10, len(df_runtime)),
                color="skyblue",
                alpha=0.7,
                edgecolor="black",
            )
            ax2.set_title("Распределение времени выполнения", fontweight="bold")
            ax2.set_xlabel("Время (минуты)")
            ax2.set_ylabel("Количество экспериментов")
            ax2.grid(True, alpha=0.3)

            # Добавляем статистику
            mean_time = df_runtime["Runtime_min"].mean()
            median_time = df_runtime["Runtime_min"].median()
            ax2.axvline(
                mean_time,
                color="red",
                linestyle="--",
                label=f"Среднее: {mean_time:.1f} мин",
            )
            ax2.axvline(
                median_time,
                color="orange",
                linestyle="--",
                label=f"Медиана: {median_time:.1f} мин",
            )
            ax2.legend()

            plt.tight_layout()

            # Сохраняем график
            if save_path is None:
                save_path = (
                    f"runtime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Анализ времени выполнения сохранен: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Ошибка анализа времени выполнения: {e}")
            return ""

    def create_performance_matrix(self, save_path: str = None) -> str:
        """
        Создает матрицу производительности экспериментов.

        Args:
            save_path: Путь для сохранения графика

        Returns:
            Путь к сохраненному графику
        """
        try:
            df = self.create_metrics_comparison_table()
            if df.empty:
                return ""

            # Выбираем только числовые метрики производительности
            performance_cols = [
                col
                for col in df.columns
                if any(
                    keyword in col.lower()
                    for keyword in ["accuracy", "f1", "precision", "recall"]
                )
            ]

            if not performance_cols:
                logger.warning("Не найдено метрик производительности")
                return ""

            # Подготавливаем данные для тепловой карты
            heatmap_data = df[["Experiment"] + performance_cols].set_index("Experiment")
            heatmap_data = heatmap_data.select_dtypes(include=[np.number])

            if heatmap_data.empty:
                logger.warning("Нет числовых данных для тепловой карты")
                return ""

            # Создаем тепловую карту
            plt.figure(figsize=(12, 8))

            # Нормализуем данные для лучшей визуализации
            heatmap_normalized = (heatmap_data - heatmap_data.min()) / (
                heatmap_data.max() - heatmap_data.min()
            )

            sns.heatmap(
                heatmap_normalized,
                annot=heatmap_data,  # Показываем реальные значения
                fmt=".3f",
                cmap="RdYlGn",
                cbar_kws={"label": "Нормализованная производительность"},
                linewidths=0.5,
            )

            plt.title(
                "Матрица производительности экспериментов",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Метрики производительности")
            plt.ylabel("Эксперименты")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

            plt.tight_layout()

            # Сохраняем график
            if save_path is None:
                save_path = (
                    f"performance_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Матрица производительности сохранена: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Ошибка создания матрицы производительности: {e}")
            return ""

    def generate_comparison_report(self, output_dir: str = "reports") -> str:
        """
        Генерирует полный отчет сравнения экспериментов.

        Args:
            output_dir: Директория для сохранения отчета

        Returns:
            Путь к HTML отчету
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Создаем все визуализации
            metrics_plot = self.plot_metrics_comparison(
                save_path=f"{output_dir}/metrics_comparison.png"
            )
            runtime_plot = self.plot_runtime_analysis(
                save_path=f"{output_dir}/runtime_analysis.png"
            )
            performance_matrix = self.create_performance_matrix(
                save_path=f"{output_dir}/performance_matrix.png"
            )

            # Получаем таблицу данных
            df = self.create_metrics_comparison_table()

            # Статистический анализ
            stats = self._calculate_comparison_stats()

            # Генерируем HTML отчет
            html_content = self._generate_html_report(
                df, stats, metrics_plot, runtime_plot, performance_matrix
            )

            # Сохраняем отчет
            report_path = f"{output_dir}/experiment_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # Сохраняем данные в JSON
            json_path = f"{output_dir}/comparison_data.json"
            comparison_data = {
                "experiments": self.experiments_data,
                "statistics": stats,
                "generated_at": datetime.now().isoformat(),
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Отчет сравнения сохранен: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return ""

    def _calculate_comparison_stats(self) -> dict[str, Any]:
        """Вычисляет статистику сравнения экспериментов."""
        if not self.experiments_data:
            return {}

        stats = {
            "total_experiments": len(self.experiments_data),
            "completed_experiments": len(
                [e for e in self.experiments_data if e["status"] == "completed"]
            ),
            "average_runtime_min": 0,
            "best_experiment": None,
            "metrics_summary": {},
        }

        # Статистика времени выполнения
        runtimes = [e["runtime"] for e in self.experiments_data if e["runtime"]]
        if runtimes:
            stats["average_runtime_min"] = sum(runtimes) / len(runtimes) / 60

        # Анализ метрик
        all_metrics = {}
        for exp in self.experiments_data:
            for metric, value in exp["metrics"].items():
                if isinstance(value, int | float):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

        # Сводка по метрикам
        for metric, values in all_metrics.items():
            if values:
                stats["metrics_summary"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }

        # Лучший эксперимент (по accuracy если есть)
        accuracy_metrics = [m for m in all_metrics.keys() if "accuracy" in m.lower()]
        if accuracy_metrics:
            best_metric = accuracy_metrics[0]
            best_exp = max(
                self.experiments_data, key=lambda x: x["metrics"].get(best_metric, 0)
            )
            stats["best_experiment"] = {
                "name": best_exp["name"],
                "task_id": best_exp["task_id"],
                "best_metric": best_metric,
                "best_value": best_exp["metrics"].get(best_metric, 0),
            }

        return stats

    def _generate_html_report(
        self,
        df: pd.DataFrame,
        stats: dict[str, Any],
        metrics_plot: str,
        runtime_plot: str,
        performance_matrix: str,
    ) -> str:
        """Генерирует HTML отчет."""
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Отчет сравнения экспериментов - {self.project_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1, h2 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 2em; font-weight: bold; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                .image-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .best-experiment {{ background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #4CAF50; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 Отчет сравнения экспериментов</h1>
                <p><strong>Проект:</strong> {self.project_name}</p>
                <p><strong>Дата генерации:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <h2>📊 Общая статистика</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{stats.get('total_experiments', 0)}</div>
                        <div class="stat-label">Всего экспериментов</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats.get('completed_experiments', 0)}</div>
                        <div class="stat-label">Завершено успешно</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats.get('average_runtime_min', 0):.1f}</div>
                        <div class="stat-label">Среднее время (мин)</div>
                    </div>
                </div>
        """

        # Добавляем информацию о лучшем эксперименте
        if stats.get("best_experiment"):
            best = stats["best_experiment"]
            html_template += f"""
                <div class="best-experiment">
                    <h3>🏆 Лучший эксперимент</h3>
                    <p><strong>Название:</strong> {best['name']}</p>
                    <p><strong>Метрика:</strong> {best['best_metric']} = {best['best_value']:.4f}</p>
                    <p><strong>Task ID:</strong> {best['task_id']}</p>
                </div>
            """

        # Добавляем визуализации
        if metrics_plot:
            html_template += f"""
                <h2>📈 Сравнение метрик</h2>
                <div class="image-container">
                    <img src="{Path(metrics_plot).name}" alt="Сравнение метрик">
                </div>
            """

        if runtime_plot:
            html_template += f"""
                <h2>⏱️ Анализ времени выполнения</h2>
                <div class="image-container">
                    <img src="{Path(runtime_plot).name}" alt="Анализ времени выполнения">
                </div>
            """

        if performance_matrix:
            html_template += f"""
                <h2>🎯 Матрица производительности</h2>
                <div class="image-container">
                    <img src="{Path(performance_matrix).name}" alt="Матрица производительности">
                </div>
            """

        # Добавляем таблицу данных
        if not df.empty:
            html_template += f"""
                <h2>📋 Детальные результаты</h2>
                {df.to_html(classes='table', escape=False, index=False)}
            """

        html_template += """
            </div>
        </body>
        </html>
        """

        return html_template


def main():
    """Демонстрация системы сравнения экспериментов."""
    try:
        logger.info("Демонстрация системы сравнения экспериментов ClearML")

        # Создаем экземпляр системы сравнения
        comparison = ExperimentComparison("ResearchHub")

        # Собираем данные последних экспериментов
        experiments = comparison.collect_experiments_data(limit=10)

        if not experiments:
            logger.warning("Не найдено экспериментов для сравнения")
            logger.info("Сначала запустите несколько экспериментов:")
            logger.info("python clearml/experiments/experiment_runner.py")
            return

        # Создаем таблицу сравнения
        df = comparison.create_metrics_comparison_table()
        print("\n" + "=" * 80)
        print("ТАБЛИЦА СРАВНЕНИЯ ЭКСПЕРИМЕНТОВ")
        print("=" * 80)
        print(df.to_string(index=False))

        # Создаем визуализации
        print("\n" + "=" * 80)
        print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("=" * 80)

        metrics_plot = comparison.plot_metrics_comparison()
        if metrics_plot:
            print(f"✓ График сравнения метрик: {metrics_plot}")

        runtime_plot = comparison.plot_runtime_analysis()
        if runtime_plot:
            print(f"✓ Анализ времени выполнения: {runtime_plot}")

    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {e}")


if __name__ == "__main__":
    main()
