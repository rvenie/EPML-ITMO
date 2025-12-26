# ДЗ 5: ClearML для MLOps

## 📋 Содержание
1. [Быстрый старт](#-быстрый-старт)
2. [Настройка ClearML](#-1-настройка-clearml)
3. [Трекинг экспериментов](#-2-трекинг-экспериментов)
4. [Управление моделями](#-3-управление-моделями)
5. [Пайплайны](#-4-пайплайны)

---

## 🚀 Быстрый старт

```bash
# 1. Установить зависимости
poetry install

# 2. Запустить ClearML сервер
make clearml-server

# 3. Подождать 1-2 минуты, открыть веб-интерфейс
open http://localhost:8080

# 4. Запустить ML эксперимент
make clearml-test

# 5. Посмотреть результаты в веб-интерфейсе
# Projects -> ResearchHub -> выбрать эксперимент
```

---

## 🔧 1. Настройка ClearML 

### 1.1 ClearML Server - Docker инфраструктура

**Файл конфигурации:** `clearml/config/docker-compose-clearml.yml`

Инфраструктура ClearML развернута через Docker Compose и включает:

| Сервис | Порт | Описание |
|--------|------|----------|
| `webserver` | 8080 | Веб-интерфейс ClearML |
| `apiserver` | 8008 | REST API для SDK |
| `fileserver` | 8081 | Хранение артефактов и моделей |
| `elasticsearch` | 9200 | Поиск и индексирование |
| `mongo` | 27017 | Основная база данных |
| `redis` | 6379 | Кэширование и очереди |

**Команды управления:**
```bash
# Запуск сервера
make clearml-server

# Остановка сервера
make clearml-stop

# Просмотр логов (в директории clearml/config)
cd clearml/config && docker-compose -f docker-compose-clearml.yml logs
```

### 1.2 База данных и хранилище

- **MongoDB** - хранение метаданных экспериментов, задач и моделей
- **Elasticsearch** - полнотекстовый поиск и индексирование
- **Redis** - кэширование и управление очередями
- **FileServer** - хранение артефактов, моделей и логов

Данные сохраняются в volumes:
```
clearml/config/clearml_data/
├── elastic/     # Elasticsearch данные
├── mongo/       # MongoDB данные
├── redis/       # Redis данные
└── fileserver/  # Артефакты и модели
```

### 1.3 Создание проекта

Проект "ResearchHub" создается автоматически при первом эксперименте:

```python
from clearml import Task

task = Task.init(
    project_name="ResearchHub",
    task_name="My Experiment",
    task_type=Task.TaskTypes.training
)
```

### 1.4 Аутентификация

**Конфигурация:** `clearml/config/clearml.conf` и `~/clearml.conf`

**Настройка учетных данных:**
1. Открыть http://localhost:8080
2. Создать аккаунт при первом входе

---

## 📊 2. Трекинг экспериментов 

### 2.1 Автоматическое логирование

**Файл:** `clearml/pipelines/ml_pipeline.py`

```python
class ClearMLExperimentTracker:
    def init_experiment(self, experiment_params):
        self.task = Task.init(
            project_name=self.project_name,
            task_name=f"ML Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=Task.TaskTypes.training,
            auto_connect_frameworks=True,  # Автоматическое логирование sklearn, etc.
        )
        
        # Логирование параметров
        for param_name, param_value in experiment_params.items():
            self.task.set_parameter(param_name, param_value)
        
        return self.task
```


### 2.2 Система сравнения экспериментов

В веб-интерфейсе ClearML (http://localhost:8080):
- **Projects → ResearchHub → Experiments** - список всех экспериментов
- Выбрать несколько экспериментов → Compare - сравнение метрик
- Scalars - графики метрик по итерациям
- Parallel Coordinates - анализ влияния параметров

### 2.3 Логирование метрик

```python
def log_training_metrics(self, metrics, epoch=0):
    logger = self.task.get_logger()
    for metric_name, metric_value in metrics.items():
        logger.report_scalar("Training Metrics", metric_name, metric_value, epoch)
```

Логируемые метрики:
- `accuracy` - точность классификации
- `f1_score` - F1-мера
- `train_samples` - размер обучающей выборки
- `training_time` - время обучения

### 2.4 Дашборды

Встроенные дашборды в ClearML:
- **Scalars** - графики метрик
- **Plots** - пользовательские визуализации
- **Debug Samples** - примеры данных
- **Artifacts** - загруженные файлы
- **Console** - логи выполнения в реальном времени

---

## 🤖 3. Управление моделями 

### 3.1 Регистрация и версионирование

**Файл:** `clearml/pipelines/ml_pipeline.py`

```python
from clearml import Model

def register_model(self, model_data, model_path, model_metadata):
    # Сохранение модели
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    # Регистрация в ClearML Model Registry
    self.model = Model(
        name=f"research_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        project=self.project_name,
        task=self.task,
        framework="scikit-learn"
    )
    
    # Загрузка весов модели
    self.model.update_weights(weights_filename=model_path)
    
    # Публикация
    self.model.publish()
```

### 3.2 Метаданные моделей

```python
model_metadata = {
    "accuracy": test_metrics["accuracy"],
    "f1_score": test_metrics["f1_score"],
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "features": X.shape[1],
    "training_date": datetime.now().isoformat(),
}

# Сохранение метаданных в ClearML
for key, value in model_metadata.items():
    self.task.set_parameter(f"model_metadata/{key}", value)
```

### 3.3 Model Registry

Просмотр зарегистрированных моделей:
- **Web UI:** http://localhost:8080 → Models
- Фильтрация по проекту, тегам, метрикам
- История версий и сравнение моделей

### 3.4 Сравнение моделей

```python
def compare_with_baseline(self, current_metrics, baseline_model_id):
    baseline_model = Model(model_id=baseline_model_id)
    baseline_task = Task.get_task(task_id=baseline_model.task)
    baseline_accuracy = baseline_task.get_parameter("model_metadata/accuracy")
    
    improvement = current_metrics["accuracy"] - float(baseline_accuracy)
    self.task.get_logger().report_scalar(
        "Model Comparison", "accuracy_improvement", improvement
    )
```

---

## ⚙️ 4. Пайплайны 

### 4.1 ML Workflow Pipeline

**Файл:** `clearml/pipelines/ml_pipeline.py`

```python
class MLPipeline:
    def run_training_experiment(self, input_file):
        # 1. Загрузка данных
        df = self._load_or_create_data(input_file)
        
        # 2. Предобработка
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(X_text).toarray()
        
        # 3. Обучение
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # 4. Оценка
        metrics = {"accuracy": accuracy_score(y_test, y_pred)}
        self.experiment_tracker.log_training_metrics(metrics)
        
        # 5. Регистрация модели
        self.experiment_tracker.register_model(model_data, model_path, metadata)
        
        return {"task_id": task.id, "model_id": model.id, "metrics": metrics}
```

### 4.2 Автоматический запуск (Scheduler)

**Файл:** `clearml/pipelines/pipeline_scheduler.py`

```python
class PipelineScheduler:
    def should_start_pipeline(self):
        interval = timedelta(hours=self.config["schedule"]["interval_hours"])
        return datetime.now() - last_run >= interval
    
    def start_pipeline(self):
        pipeline = MLPipeline(self.project_name)
        result = pipeline.run_training_experiment()
        return result["task_id"]
```

### 4.3 Мониторинг

**Файл:** `clearml/pipelines/pipeline_monitor.py`

```python
class ClearMLMonitor:
    def check_servers_health(self):
        for server_name, server_url in self.servers.items():
            response = requests.get(f"{server_url}/debug.ping", timeout=10)
            health_status[server_name] = response.status_code == 200
        return health_status
    
    def get_pipeline_statistics(self):
        tasks = Task.get_tasks(project_name=self.project_name)
        # Подсчет completed, failed, running
        return stats
```

### 4.4 Уведомления

Логирование событий в `pipeline_scheduler.py`:

```python
def _send_notification(self, title, message):
    log_message = f"{title}: {message}"
    logger.info(log_message)
    # Можно расширить: email, Slack, Telegram
```

Типы уведомлений:
- Запуск нового пайплайна
- Успешное завершение
- Ошибки выполнения
- Превышение таймаута

---


## 📁 Структура файлов

```
clearml/
├── config/
│   ├── docker-compose-clearml.yml  # Docker конфигурация
│   ├── clearml.conf                # Конфигурация SDK
│   └── scheduler_config.json       # Конфигурация планировщика
├── experiments/
│   ├── experiment_runner.py        # Запуск экспериментов
│   ├── experiment_comparison.py    # Сравнение экспериментов
│   └── clearml_dashboard.py        # Генерация дашбордов
├── models/
│   └── model_manager.py            # Управление моделями
└── pipelines/
    ├── ml_pipeline.py              # ML пайплайн с трекингом
    ├── pipeline_scheduler.py       # Планировщик запусков
    ├── pipeline_monitor.py         # Мониторинг системы
    └── run_system.py               # Управление инфраструктурой
```

---

## 🎯 Основные команды

```bash
make clearml-server   # Запуск ClearML сервера
make clearml-stop     # Остановка сервера
make clearml-test     # Запуск ML эксперимента
```
