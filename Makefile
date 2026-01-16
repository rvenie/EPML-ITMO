#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = research_agets_hub
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Poetry dependencies
.PHONY: install
install:
	poetry install

## Install dependencies and pre-commit hooks
.PHONY: setup
setup:
	poetry install
	poetry run pre-commit install

## Update Poetry dependencies
.PHONY: update
update:
	poetry update

## Export dependencies to requirements.txt (for CI/CD compatibility)
.PHONY: export
export:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	poetry run ruff format --check
	poetry run ruff check
	poetry run mypy .

## Format source code with ruff
.PHONY: format
format:
	poetry run ruff check --fix
	poetry run ruff format

## Run security analysis with bandit
.PHONY: security
security:
	poetry run bandit -r researchhub/

## Run all quality checks
.PHONY: check
check: lint security

## Run pre-commit hooks on all files
.PHONY: pre-commit
pre-commit:
	poetry run pre-commit run --all-files

## Set up Python interpreter environment with Poetry
.PHONY: create_environment
create_environment:
	poetry env use $(PYTHON_VERSION)
	@echo ">>> Poetry environment created. Activate with:"
	@echo "poetry shell"
	@echo ">>> Or run commands with:"
	@echo "poetry run <command>"

## Show Poetry environment info
.PHONY: env-info
env-info:
	poetry env info

## Activate Poetry shell
.PHONY: shell
shell:
	poetry shell

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: install
	poetry run python researchhub/dataset.py

## Fetch ArXiv data (with ClearML Dataset upload, совпадает с dvc repro fetch_data)
.PHONY: fetch-data
fetch-data: install
	@echo "📥 Fetching ArXiv data (with ClearML integration)..."
	poetry run python scripts/fetch_arxiv_data.py --query "${data.query}" --max-results ${data.max_results} --output-dir data/raw

## Preprocess data (with ClearML Dataset upload, совпадает с dvc repro preprocess)
.PHONY: preprocess
preprocess: install
	@echo "🔧 Preprocessing data (with ClearML integration)..."
	poetry run python scripts/preprocess_data.py --input data/raw/arxiv_publications.csv --output data/processed/publications_processed.csv --params params.yaml

## Train single model (RandomForest, совпадает с dvc repro train_rf)
.PHONY: train
train: install
	@echo "🚀 Training model (with ClearML tracking)..."
	poetry run dvc repro train_rf

## Train all models (совпадает с dvc repro train_rf train_svm train_lr)
.PHONY: train-all
train-all: install
	@echo "🚀 Training all models (with ClearML tracking)..."
	poetry run dvc repro train_rf train_svm train_lr

## Run full DVC pipeline (fetch → preprocess → train all, с ClearML интеграцией)
## Эквивалентно: dvc repro --force
## Все скрипты автоматически логируют в ClearML если сервер запущен
.PHONY: pipeline
pipeline: install
	@echo "🔄 Running full DVC pipeline (fetch → preprocess → train all)..."
	@echo "   ClearML integration is automatic if server is running"
	poetry run dvc repro --force

## Run full DVC pipeline with force (то же что pipeline, явно указываем --force)
## Эквивалентно: make pipeline
.PHONY: pipeline-force
pipeline-force: pipeline

## Start MLflow UI (local)
.PHONY: mlflow-ui
mlflow-ui:
	poetry run mlflow ui

## Start MLflow Server (Docker)
.PHONY: mlflow-server-up
mlflow-server-up:
	docker-compose up -d mlflow-server
	@echo "✅ MLflow Server starting..."
	@echo "   Web UI: http://localhost:3000"

## Start all services (MLflow + ClearML)
.PHONY: services-up
services-up: install
	@echo "🚀 Starting all services (MLflow + ClearML)..."
	docker-compose up -d mlflow-server
	docker-compose up -d clearml-apiserver clearml-webserver clearml-fileserver clearml-elasticsearch clearml-mongo clearml-redis
	@echo "✅ Services starting..."
	@echo "   MLflow UI: http://localhost:3000"
	@echo "   ClearML UI: http://localhost:8090"
	@echo ""
	@echo "⏳ Wait 1-2 minutes for all services to be healthy"

## Run Jupyter notebook
.PHONY: notebook
notebook: install
	poetry run jupyter notebook

#################################################################################
# CLEARML COMMANDS                                                              #
#################################################################################

## Start ClearML Server
.PHONY: clearml-server-up
clearml-server-up:
	docker-compose up -d clearml-apiserver clearml-webserver clearml-fileserver clearml-elasticsearch clearml-mongo clearml-redis
	@echo "✅ ClearML Server starting..."
	@echo "   Web UI: http://localhost:8090"
	@echo "   API: http://localhost:8008"
	@echo ""
	@echo "⏳ Wait 1-2 minutes for all services to be healthy"
	@echo "   Check status: make clearml-status"

## Stop ClearML Server
.PHONY: clearml-server-down
clearml-server-down:
	docker-compose stop clearml-apiserver clearml-webserver clearml-fileserver clearml-elasticsearch clearml-mongo clearml-redis clearml-agent
	@echo "✅ ClearML Server stopped"

## Check ClearML Server status
.PHONY: clearml-status
clearml-status:
	@echo "=== ClearML Server Status ==="
	@docker-compose ps | grep clearml || echo "No ClearML containers running"
	@echo ""
	@echo "=== Health Checks ==="
	@docker ps --filter "name=clearml" --format "table {{.Names}}\t{{.Status}}" | grep clearml || echo "No ClearML containers running"

## Upload processed dataset to ClearML Dataset (manual)
.PHONY: clearml-upload-dataset
clearml-upload-dataset: install
	@echo "📦 Uploading processed dataset to ClearML..."
	poetry run python scripts/upload_dataset.py

## Full pipeline: fetch → preprocess → upload to ClearML
.PHONY: clearml-data-pipeline
clearml-data-pipeline: install
	@echo "🔄 Running data pipeline with ClearML..."
	@echo "1. Fetching ArXiv data..."
	poetry run python scripts/fetch_arxiv_data.py
	@echo "2. Preprocessing data..."
	poetry run python scripts/preprocess_data.py
	@echo "✅ Data pipeline completed! Check ClearML DATASETS section."

## Run ClearML Pipeline (7-step DAG with Pydantic validation)
.PHONY: clearml-pipeline
clearml-pipeline: install
	@echo "🚀 Running ClearML Pipeline..."
	poetry run python scripts/clearml_pipeline_simple.py

## Setup ClearML credentials (interactive)
.PHONY: clearml-init
clearml-init:
	@echo "📝 Setting up ClearML credentials..."
	@echo ""
	@echo "1. Open http://localhost:8090"
	@echo "2. Login with: admin / admin"
	@echo "3. Go to Settings → Workspace → Create new credentials"
	@echo "4. Run: poetry run clearml-init"
	@echo ""
	poetry run clearml-init

## Clean ClearML data (WARNING: deletes all experiments!)
.PHONY: clearml-clean
clearml-clean:
	@echo "⚠️  WARNING: This will delete all ClearML data!"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ]
	docker-compose stop clearml-apiserver clearml-webserver clearml-fileserver clearml-elasticsearch clearml-mongo clearml-redis clearml-agent
	docker volume rm $$(docker volume ls -q | grep clearml) 2>/dev/null || true
	@echo "✅ ClearML data cleaned"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
