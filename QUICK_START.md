# Quick Start Guide

Быстрый старт для воспроизведения результатов ML проекта с DVC и MLflow.

## Предварительные требования

- Python 3.11+
- Git
- Docker (опционально)

## 🚀 Быстрый запуск (5 минут)

### 1. Установка зависимостей

```bash
# Клонируем репозиторий
git clone <repository-url>
cd research_agets_hub

# Создаем виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Устанавливаем зависимости
pip install -r requirements.txt
```

### 2. Инициализация DVC

```bash
# DVC уже инициализирован, настраиваем remote storage
dvc remote add -d local_storage ../dvc-storage
mkdir -p ../dvc-storage

# Загружаем данные (если они есть в remote)
dvc pull  # Может выдать ошибку, если remote пустой - это нормально
```

### 3. Запуск MLflow

```bash
# Запускаем MLflow сервер в фоне
nohup mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri file:./mlruns > mlflow.log 2>&1 &

# Проверяем, что сервер запустился
curl http://127.0.0.1:5000/
```

### 4. Обучение модели

```bash
# Предобработка данных
python scripts/preprocess_data.py

# Обучение модели с логированием в MLflow
python scripts/train_model.py \
    --input data/processed/publications_processed.csv \
    --model-output models/classifier.pkl \
    --metrics metrics.json
```

### 5. Просмотр результатов

```bash
# Открываем MLflow UI
echo "MLflow UI: http://127.0.0.1:5000"

# Проверяем DVC статус
dvc status

# Просматриваем метрики
cat metrics.json
```

## 🐳 Docker запуск (альтернативный способ)

```bash
# Запуск через Docker Compose
docker-compose up -d mlflow-server

# Обучение модели в контейнере
docker-compose run --rm ml-app train

# Доступ к Jupyter (для разработки)
docker-compose --profile development up -d jupyter-dev
```

## 📊 Ожидаемые результаты

После успешного выполнения вы должны получить:

✅ **Data Pipeline**:
- `data/processed/publications_processed.csv` (51 записей, 21 признак)
- DVC отслеживает все версии данных

✅ **Model Performance**:
- Cross-validation accuracy: ~77.7%
- Test accuracy: ~90.9%
- F1-score: ~86.8%

✅ **MLflow Tracking**:
- Зарегистрированная модель в Model Registry
- 15+ параметров и 6+ метрик в эксперименте
- Артефакты модели доступны через UI

✅ **Версионирование**:
- Все данные и модели отслеживаются DVC
- Полная воспроизводимость через Docker

## 🔍 Верификация

```bash
# Проверяем данные
python -c "
import pandas as pd
df = pd.read_csv('data/processed/publications_processed.csv')
print(f'✓ Data shape: {df.shape}')
assert df.shape == (51, 21), 'Wrong data shape'
"

# Проверяем модель
python -c "
import pickle
with open('models/classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)
print('✓ Model loaded successfully')
print(f'✓ Model type: {type(model_data[\"model\"]).__name__}')
"

# Проверяем MLflow
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
print(f'✓ MLflow experiments: {len(experiments)}')
assert len(experiments) >= 1, 'No experiments found'
"
```

## 🛠️ Полезные команды

```bash
# DVC команды
dvc add data/raw/publications.csv    # Добавить файл в DVC
dvc push                             # Отправить в remote storage
dvc pull                             # Загрузить из remote storage
dvc status                           # Проверить статус
dvc dag                              # Показать граф зависимостей

# MLflow команды  
mlflow ui                            # Запустить UI
mlflow models serve -m "models:/research_publications_classification_model/1" # Деплой модели

# Docker команды
docker-compose up mlflow-server      # Только MLflow сервер
docker-compose --profile development up  # Режим разработки
docker-compose run --rm ml-app bash # Интерактивная сессия
```

## ❗ Решение проблем

**MLflow не запускается**:
```bash
pkill -f mlflow  # Остановить существующие процессы
rm -rf mlruns    # Очистить данные (осторожно!)
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri file:./mlruns
```

**DVC ошибки**:
```bash
dvc doctor       # Диагностика DVC
dvc remote list  # Проверить remote storage
dvc cache dir    # Проверить кэш
```

**Docker проблемы**:
```bash
docker-compose down  # Остановить все сервисы
docker system prune  # Очистить неиспользуемые ресурсы
```

## 📚 Дополнительная документация

- `REPRODUCIBILITY.md` - Подробные инструкции по воспроизводимости
- `REPORT_hw2_versioning.md` - Отчет о проделанной работе
- `params.yaml` - Все конфигурационные параметры
- `docker-compose.yml` - Конфигурация сервисов

## 🎯 Что дальше?

1. **Эксперименты**: Измените параметры в `params.yaml` и переобучите модель
2. **Данные**: Добавьте новые публикации в `data/raw/publications.csv`
3. **Развертывание**: Используйте MLflow для деплоя модели в production
4. **Мониторинг**: Настройте автоматический retraining при изменении данных

---

**Время выполнения**: ~5-10 минут  
**Поддержка**: Проверьте `REPRODUCIBILITY.md` для детального troubleshooting