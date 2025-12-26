# Эксперименты

Руководство по проведению, отслеживанию и сравнению ML экспериментов.

## Обзор системы экспериментов

ResearchHub предоставляет автоматизированную систему для:

- Проведения серий экспериментов
- Отслеживания параметров и метрик
- Сравнения результатов
- Выбора лучших моделей

## Запуск экспериментов

### Автоматическая серия

```bash
# Запуск всех 17 экспериментов
poetry run python scripts/run_experiments.py
```

### Типы экспериментов

| Группа | Эксперименты | Описание |
|--------|--------------|----------|
| **Random Forest** | RF_baseline, RF_more_trees, RF_deeper, RF_conservative, RF_more_features | Вариации RF |
| **SVM** | SVM_baseline, SVM_linear, SVM_high_C, SVM_low_C, SVM_poly | Вариации SVM |
| **Logistic Regression** | LR_baseline, LR_l1_penalty, LR_high_reg, LR_low_reg, LR_lbfgs | Вариации LR |
| **Feature Engineering** | RF_unigrams_only, LR_extended_ngrams | Эксперименты с признаками |

### Конфигурации экспериментов

```python
EXPERIMENTS = {
    "RF_baseline": {
        "algorithm": "RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "tfidf_params": {
            "max_features": 5000,
            "ngram_range": (1, 2)
        }
    },
    "SVM_linear": {
        "algorithm": "SVC",
        "params": {
            "kernel": "linear",
            "C": 1.0,
            "probability": True
        }
    },
    # ...
}
```

## MLflow Tracking

### Веб-интерфейс

```bash
# Запуск UI
poetry run mlflow server \
    --host 127.0.0.1 \
    --port 3000 \
    --backend-store-uri file:./mlruns

# Открыть http://localhost:3000
```

### Поиск экспериментов

```python
from researchhub.mlflow_utils import MLflowExperimentManager

manager = MLflowExperimentManager("file:./mlruns")

# Все запуски эксперимента
runs = manager.get_experiment_runs(
    experiment_name="research_publications_classification"
)

# С фильтрацией
runs = manager.get_experiment_runs(
    experiment_name="research_publications_classification",
    filter_string="params.algorithm = 'RandomForestClassifier'",
    order_by=["metrics.test_accuracy DESC"]
)
```

### Лучшие результаты

```python
# Лучший запуск по метрике
best_run = manager.get_best_run(
    experiment_name="research_publications_classification",
    metric_name="test_accuracy",
    maximize=True
)

print(f"Best run: {best_run.info.run_id}")
print(f"Accuracy: {best_run.data.metrics['test_accuracy']}")
```

### Leaderboard

```python
from researchhub.mlflow_utils import get_experiment_leaderboard

leaderboard = get_experiment_leaderboard(
    experiment_name="research_publications_classification",
    metric="test_accuracy",
    top_n=5
)

print(leaderboard[['run_name', 'accuracy', 'f1_score', 'algorithm']])
```

## Сравнение экспериментов

### В коде

```python
# Сравнение конкретных запусков
comparison = manager.compare_runs(
    run_ids=["run_id_1", "run_id_2", "run_id_3"],
    metrics=["accuracy", "f1_score", "precision", "recall"]
)

print(comparison)
```

### Сравнение алгоритмов

```python
from researchhub.mlflow_utils import quick_compare_algorithms

comparison = quick_compare_algorithms(
    experiment_name="research_publications_classification",
    algorithms=["RandomForestClassifier", "SVC", "LogisticRegression"]
)
```

### В MLflow UI

1. Открыть эксперимент
2. Выбрать несколько запусков (чекбоксы)
3. Нажать "Compare"
4. Анализировать графики и таблицы

## Экспорт результатов

### CSV/JSON/Excel

```python
manager.export_experiment_results(
    experiment_name="research_publications_classification",
    output_file="results.csv",
    format="csv"  # или "json", "excel"
)
```

### HTML отчёт

```python
from researchhub.mlflow_utils import create_experiment_summary_report

create_experiment_summary_report(
    experiment_name="research_publications_classification",
    output_file="experiment_report.html"
)
```

## Результаты проведённых экспериментов

### Лучшие результаты по алгоритмам

| Алгоритм | Лучший эксперимент | Accuracy | F1-score |
|----------|-------------------|----------|----------|
| Random Forest | RF_more_features | 0.350 | 0.205 |
| SVM | SVM_linear | 0.250 | 0.180 |
| Logistic Regression | LR_baseline | 0.200 | 0.150 |

### Статистика

```
📊 СТАТИСТИКА ЭКСПЕРИМЕНТОВ:
- Всего проведено: 17 экспериментов
- Успешных: 17 (100%)
- Время выполнения: ~2-3 минуты
```

## ClearML интеграция

### Запуск с ClearML

```bash
# Запуск сервера
make clearml-server

# Тестовый эксперимент
make clearml-test
```

### Веб-интерфейс

- URL: http://localhost:8080
- Projects → ResearchHub → Experiments

### Сравнение в ClearML

1. Выбрать эксперименты
2. Compare → Scalars
3. Parallel Coordinates для анализа

## Воспроизводимость

### Полное воспроизведение

```bash
# 1. Клонирование
git clone <repository>
cd research-agets-hub

# 2. Установка
poetry install

# 3. Получение данных
dvc pull

# 4. Запуск всех экспериментов
poetry run python scripts/run_experiments.py
```

### Фиксация случайности

```python
# Все эксперименты используют
RANDOM_STATE = 42
```

## Советы

!!! tip "Рекомендации"

    1. **Начинайте с baseline** — простая модель для сравнения
    2. **Меняйте один параметр** — для понимания влияния
    3. **Используйте тегирование** — для организации
    4. **Документируйте гипотезы** — в описании запуска

!!! info "Фильтры MLflow"

    ```
    # Поиск по точности
    metrics.test_accuracy >= 0.3
    
    # Поиск по алгоритму
    params.algorithm = "RandomForestClassifier"
    
    # Комбинация
    metrics.test_accuracy >= 0.3 AND params.algorithm = "SVC"
    ```

## Следующие шаги

- [API MLflow Utils](../api/mlflow-utils.md)
- [API Декораторы](../api/decorators.md)
- [Отчёты об экспериментах](../reports/experiments.md)
