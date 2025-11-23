

## Введение
Рабочее место Data Scientist настроено по актуальным стандартам: Cookiecutter, Poetry, Docker, линтеры и pre-commit.

## Структура проекта
Сгенерировано с помощью:
```
pip install cookiecutter-data-science
ccds
```

## Управление зависимостями
Использовал Poetry:
```
poetry init --name research-agents-hub -n
poetry add pandas numpy scikit-learn
poetry add --group dev ruff mypy bandit pre-commit
poetry self add poetry-plugin-export
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Проверка качества кода
Инициализация pre-commit хуков:
```
poetry run pre-commit install
poetry run pre-commit run --all-files
```
## Git workflow
```
git init
git add .
git commit -m "Initial commit: Project structure setup"
```

## Docker
Сборка контейнера:
```
docker build -t research-agets-hub .
```
Пример команды запуска:
```
docker run --rm research-agets-hub
```

## Итог
Все инструменты настроены, код контролируется линтерами, процесс полностью автоматизирован.
