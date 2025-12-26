# Обучение моделей

Руководство по обучению и оценке ML моделей в ResearchHub.

## Поддерживаемые алгоритмы

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

### Support Vector Machine

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',
    C=1.0,
    probability=True
)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs'
)
```

## Запуск обучения

### Через DVC

```bash
# Запуск этапа обучения
dvc repro train

# Принудительный перезапуск
dvc repro train --force
```

### Через скрипт

```bash
poetry run python scripts/train_model.py \
    --input data/processed/publications_processed.csv \
    --model-output models/classifier.pkl
```

## Интеграция с MLflow

```python
from researchhub.decorators import mlflow_track

@mlflow_track(
    experiment_name="training",
    auto_log=True
)
def train_model(X, y, **params):
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model
```

## Оценка модели

### Метрики

| Метрика | Описание |
|---------|----------|
| **Accuracy** | Общая точность |
| **Precision** | Точность |
| **Recall** | Полнота |
| **F1-score** | Гармоническое среднее |

### Кросс-валидация

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f}")
```

## Следующие шаги

- [Эксперименты](experiments.md)
- [MLflow Utils API](../api/mlflow-utils.md)
