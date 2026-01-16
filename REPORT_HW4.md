# ДЗ 4: Автоматизация ML пайплайнов

## 🎯 Выбранные инструменты

**Оркестрация пайплайнов:** DVC Pipelines  
**Управление конфигурациями:** Pydantic

---

## 📋 Что реализовано

### ✅ Настройка DVC Pipelines 

#### 1. Структура пайплайна
Пайплайн состоит из **3 этапов**:
```
fetch_data → preprocess → ┬→ train_rf (Random Forest)
                          ├→ train_svm (SVM)
                          └→ train_lr (Logistic Regression)
```

#### 2. Параллельное выполнение
**Обучение трех моделей** выполняются на одном уровне:
- `train_rf`, `train_svm`, `train_lr` зависят только от `preprocess`
- Имеют разные выходные файлы (нет конфликтов)
- Параллельный этап

#### 3. Кэширование
DVC автоматически:
- Отслеживает изменения в коде, данных и параметрах
- Пропускает неизменившиеся этапы при повторном запуске
- Восстанавливает результаты из кэша за секунды

#### 4. Зависимости
Все зависимости описаны в `dvc.yaml`:
- `deps` - входные файлы и скрипты
- `params` - параметры из `params.yaml`
- `outs` - выходные файлы (модели, метаданные)
- `metrics` - метрики для отслеживания

---

### ✅ Настройка Pydantic

#### 1. Модели валидации конфигураций
В `config/pipeline_config.py` определены Pydantic модели:
- `DataConfig` - валидация параметров получения данных
- `FeatureEngineeringConfig` - параметры обработки признаков
- `TrainingConfig` - параметры обучения
- `RandomForestConfig` - специфичные параметры RF
- `MLflowConfig` - настройки MLflow
- `EvaluationConfig` - параметры оценки
- `PipelineConfig` - главная модель, объединяющая все

#### 2. Автоматическая валидация
Pydantic автоматически проверяет:
- **Типы данных:** `max_results` должен быть `int` от 1 до 2000
- **Диапазоны значений:** `test_size` должен быть от 0 до 1
- **Допустимые значения:** `algorithm` только из списка поддерживаемых
- **Структуру конфигурации:** все обязательные поля присутствуют

#### 3. Использование во всех скриптах
**`scripts/train_model.py`** - ✅ Использует Pydantic:
```python
from config.pipeline_config import PipelineConfig, load_config

def load_params(params_file: str) -> PipelineConfig:
    config = load_config(params_file)
    logger.info(f"✅ Loaded and validated parameters")
    return config
```

**`scripts/preprocess_data.py`** - ✅ Использует Pydantic:
```python
from config.pipeline_config import PipelineConfig, load_config

def load_params(params_file: str) -> PipelineConfig:
    config = load_config(params_file)
    logger.info(f"✅ Loaded and validated preprocessing parameters")
    return config
```

**`config/simple_composer.py`** - ✅ Использует Pydantic для валидации:
```python
def validate_config(config):
    try:
        PipelineConfig(**config)
        return True, "Конфигурация валидна (проверено Pydantic)"
    except Exception as e:
        return False, f"Ошибка валидации: {str(e)}"
```

#### 4. Композиция конфигураций
В `config/simple_composer.py` реализована система композиции:
- Базовая конфигурация из `params.yaml`
- Специфичные настройки для каждого алгоритма
- Настройки для разных размеров данных (small/medium/large)
- **Валидация через Pydantic** при создании конфигураций

#### 5. Готовые конфигурации
Автоматически генерируются 6 конфигураций:
- `randomforestclassifier_small_config.yaml`
- `randomforestclassifier_medium_config.yaml`
- `svm_small_config.yaml`
- `svm_medium_config.yaml`
- `logisticregression_small_config.yaml`
- `logisticregression_medium_config.yaml`

---

### ✅ Интеграция и тестирование

- ✅ Интеграция DVC + Pydantic + MLflow
- ✅ Встроенный мониторинг через DVC и MLflow UI
- ✅ Детальное логирование в консоль и файлы (training.log, preprocessing.log)
- ✅ Уведомления о результатах через логи
- ✅ Полная воспроизводимость результатов (DVC кэширование + фиксированный random_state)

---

## 🚀 Быстрый старт

### 1. Запуск всего пайплайна
```bash
# Первый запуск - выполнятся все этапы
dvc repro

# Принудительный запуск
dvc repro --force
```
![DVC_Repro](/pics/dvc_repro.png)


### 2. Проверка структуры пайплайна
```bash
# Визуализация графа зависимостей
dvc dag
```
![DVC_dag](/pics/dvc_dag.png)

---

## 🔧 Команды DVC

### 1. Просмотр статуса пайплайна
```bash
dvc status

# Покажет какие этапы нужно перезапустить
```

### 2. Просмотр метрик всех моделей
```bash
dvc metrics show

# Покажет метрики из:
# - models/rf_metrics.json
# - models/svm_metrics.json
# - models/lr_metrics.json
```
![DVC_metrics](/pics/dvc_metrics_show.png)

### 3. Сравнение результатов моделей
```bash
dvc metrics diff

# Сравнит метрики между версиями
```

### 4. Принудительный запуск конкретного этапа
```bash
# Только Random Forest
dvc repro train_rf --force

# Только SVM
dvc repro train_svm --force

# Только Logistic Regression
dvc repro train_lr --force
```

### 5. Эксперименты с параметрами
```bash
# Изменить параметр и запустить
dvc exp run -S train.random_forest.n_estimators=200

# Просмотреть все эксперименты
dvc exp show

# Сравнить эксперименты
dvc exp diff
```
![DVC_exp_show](/pics/dvc_exp_show.png)
![DVC_exp_diff](/pics/dvc_exp_diff.png)

### 6. Проверка валидации Pydantic
```bash
# Генерация конфигураций с валидацией
python config/simple_composer.py

# Pydantic автоматически проверит:
# - типы данных
# - диапазоны значений
# - обязательные поля
```

---

## 📊 Результаты и артефакты

После выполнения созданы:
- `reports/pipeline_execution_report.yaml` - детальный отчет выполнения
- `reports/notifications.log` - уведомления о результатах  
- `pipeline.log` - полные логи с временными метками
- `models/rf_metrics.json` - метрики Random Forest
- `models/svm_metrics.json` - метрики SVM
- `models/lr_metrics.json` - метрики Logistic Regression
- `models/rf_classifier.pkl` - обученная модель Random Forest
- `models/svm_classifier.pkl` - обученная модель SVM
- `models/lr_classifier.pkl` - обученная модель Logistic Regression
- `data/processed/processing_metadata.yaml` - метаданные с параметрами Pydantic

---

## 🏗️ Структура автоматизации

### DVC Pipeline (`dvc.yaml`)
```yaml
stages:
  fetch_data:    # Получение данных из ArXiv
    deps:
      - scripts/fetch_arxiv_data.py
    outs:
      - data/raw/arxiv_publications.csv
      
  preprocess:    # Предобработка данных с Pydantic валидацией
    deps:
      - scripts/preprocess_data.py
      - data/raw/arxiv_publications.csv
    outs:
      - data/processed/publications_processed.csv
      
  train_rf:      # Обучение Random Forest (параллельно)
    deps:
      - data/processed/publications_processed.csv
    outs:
      - models/rf_classifier.pkl
      
  train_svm:     # Обучение SVM (параллельно)
    deps:
      - data/processed/publications_processed.csv
    outs:
      - models/svm_classifier.pkl
      
  train_lr:      # Обучение Logistic Regression (параллельно)
    deps:
      - data/processed/publications_processed.csv
    outs:
      - models/lr_classifier.pkl
```

### Pydantic модели (`config/pipeline_config.py`)
- `PipelineConfig` - главная модель конфигурации
- `DataConfig` - валидация параметров данных
- `FeatureEngineeringConfig` - параметры признаков
- `TrainingConfig` - параметры обучения
- `RandomForestConfig` - специфичные параметры RF
- `MLflowConfig` - настройки MLflow
- `EvaluationConfig` - параметры оценки



### Скрипты обучения (`scripts/train_model.py`)
- ✅ Загрузка параметров через Pydantic
- ✅ Валидация гиперпараметров
- ✅ Поддержка всех трех алгоритмов
- ✅ Параметр `--algorithm` для переопределения
- ✅ Интеграция с MLflow
- ✅ Сохранение метрик и моделей

### Скрипт предобработки (`scripts/preprocess_data.py`)
- ✅ Загрузка параметров через Pydantic
- ✅ Валидация параметров feature engineering
- ✅ Сохранение параметров в метаданные
- ✅ Логирование конфигурации

---

## 📈 Мониторинг и логирование

Система создает подробные логи:
- **Консольные логи**: Прогресс с временными метками
- **Файл логов**: `training.log`, `preprocessing.log`
- **YAML отчет**: `reports/pipeline_execution_report.yaml`
- **Метаданные**: Параметры Pydantic в `processing_metadata.yaml`
- **MLflow**: Все параметры и метрики логируются в MLflow

---

