# Features API

Документация модуля для работы с признаками.

## Обзор

Модуль `researchhub.features` предоставляет функции для:

- Извлечения признаков из текста
- TF-IDF векторизации
- Обработки категориальных данных
- Feature engineering

## Текстовые признаки

### TF-IDF векторизация

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(
    texts: list[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Создаёт TF-IDF признаки из текстов.
    
    Args:
        texts: Список текстов
        max_features: Максимальное количество признаков
        ngram_range: Диапазон n-грамм
        min_df: Минимальная частота документа
        max_df: Максимальная частота документа
    
    Returns:
        Матрица признаков и обученный векторизатор
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        min_df=min_df,
        max_df=max_df
    )
    
    X = vectorizer.fit_transform(texts)
    return X.toarray(), vectorizer
```

### Параметры TF-IDF

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `max_features` | `int` | `5000` | Размер словаря |
| `ngram_range` | `tuple` | `(1, 2)` | Диапазон n-грамм |
| `min_df` | `int` | `2` | Мин. документов с термином |
| `max_df` | `float` | `0.95` | Макс. доля документов |
| `stop_words` | `str` | `'english'` | Стоп-слова |

### Примеры конфигураций

```python
# Базовая конфигурация (униграммы)
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 1)
)

# Расширенная конфигурация (до триграмм)
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.9
)

# Консервативная конфигурация
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)
```

## Предобработка текста

### Очистка текста

```python
import re

def clean_text(text: str) -> str:
    """
    Очищает текст для ML обработки.
    
    Args:
        text: Исходный текст
    
    Returns:
        Очищенный текст
    """
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', '', text)
    
    # Удаление LaTeX формул
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление спецсимволов (кроме пробелов)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Удаление множественных пробелов
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
```

### Нормализация

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def normalize_text(text: str, lemmatize: bool = True) -> str:
    """
    Нормализует текст.
    
    Args:
        text: Очищенный текст
        lemmatize: Применять лемматизацию
    
    Returns:
        Нормализованный текст
    """
    tokens = word_tokenize(text)
    
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)
```

## Категориальные признаки

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

def encode_categories(
    categories: list[str]
) -> tuple[np.ndarray, LabelEncoder]:
    """
    Кодирует категории в числовые метки.
    
    Args:
        categories: Список категорий
    
    Returns:
        Закодированные метки и энкодер
    """
    encoder = LabelEncoder()
    labels = encoder.fit_transform(categories)
    return labels, encoder
```

### Извлечение основной категории

```python
def extract_primary_category(categories_str: str) -> str:
    """
    Извлекает основную категорию из строки категорий ArXiv.
    
    Args:
        categories_str: Строка категорий (напр. "cs.CV cs.LG")
    
    Returns:
        Основная категория (первая в списке)
    """
    categories = categories_str.split()
    return categories[0] if categories else "unknown"
```

## Дополнительные признаки

### Метаданные публикации

```python
def extract_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает дополнительные признаки из метаданных.
    
    Args:
        df: DataFrame с публикациями
    
    Returns:
        DataFrame с дополнительными признаками
    """
    features = pd.DataFrame()
    
    # Длина заголовка
    features['title_length'] = df['title'].str.len()
    features['title_word_count'] = df['title'].str.split().str.len()
    
    # Длина аннотации
    features['abstract_length'] = df['abstract'].str.len()
    features['abstract_word_count'] = df['abstract'].str.split().str.len()
    
    # Количество авторов
    features['author_count'] = df['authors'].str.split(',').str.len()
    
    # Количество категорий
    features['category_count'] = df['categories'].str.split().str.len()
    
    # Временные признаки
    if 'published' in df.columns:
        df['published'] = pd.to_datetime(df['published'])
        features['publication_year'] = df['published'].dt.year
        features['publication_month'] = df['published'].dt.month
        features['publication_dow'] = df['published'].dt.dayofweek
    
    return features
```

## Pipeline обработки

### Полный pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_feature_pipeline(
    text_column: str = 'abstract',
    max_features: int = 5000,
    ngram_range: tuple = (1, 2)
) -> Pipeline:
    """
    Создаёт полный pipeline обработки признаков.
    
    Args:
        text_column: Колонка с текстом
        max_features: Макс. количество TF-IDF признаков
        ngram_range: Диапазон n-грамм
    
    Returns:
        Sklearn Pipeline
    """
    from sklearn.preprocessing import StandardScaler
    
    # Текстовые признаки
    text_transformer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )
    
    # Числовые признаки
    numeric_transformer = StandardScaler()
    
    # Комбинированный transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, text_column),
            ('numeric', numeric_transformer, ['title_length', 'author_count'])
        ]
    )
    
    return Pipeline([
        ('preprocessor', preprocessor)
    ])
```

### Использование

```python
# Создание pipeline
pipeline = create_feature_pipeline(
    text_column='abstract',
    max_features=5000,
    ngram_range=(1, 2)
)

# Обучение
X = pipeline.fit_transform(df)

# Применение к новым данным
X_new = pipeline.transform(new_df)
```

## Сохранение и загрузка

### Сохранение vectorizer

```python
import pickle

def save_vectorizer(vectorizer, path: str):
    """Сохраняет векторизатор."""
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(path: str):
    """Загружает векторизатор."""
    with open(path, 'rb') as f:
        return pickle.load(f)
```

### Сохранение с DVC

```bash
# Добавление в DVC
dvc add models/vectorizer.pkl

# Коммит
git add models/vectorizer.pkl.dvc
git commit -m "Add TF-IDF vectorizer v1"
```

## Конфигурации экспериментов

### Примеры из проекта

| Эксперимент | max_features | ngram_range | Особенности |
|-------------|--------------|-------------|-------------|
| RF_baseline | 5000 | (1, 2) | Базовая конфигурация |
| RF_unigrams_only | 5000 | (1, 1) | Только униграммы |
| RF_more_features | 10000 | (1, 3) | Расширенный словарь |
| LR_extended_ngrams | 5000 | (1, 4) | До 4-грамм |

## Советы

!!! tip "Рекомендации"

    1. Начинайте с небольшого словаря (1000-5000)
    2. Используйте min_df для фильтрации редких терминов
    3. Проверяйте важность признаков после обучения
    4. Сохраняйте векторизатор для воспроизводимости

!!! warning "Предупреждения"

    - Большой словарь увеличивает время обучения
    - N-граммы выше 3 редко улучшают качество
    - Не забывайте про stop_words
