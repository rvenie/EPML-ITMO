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

## Train baseline model
.PHONY: train
train: install
	poetry run dvc repro train

## Run all experiments with different algorithms
.PHONY: experiments
experiments: install
	poetry run dvc repro run_experiments

## Run full DVC pipeline (fetch, preprocess, train, experiments)
.PHONY: pipeline
pipeline: install
	poetry run dvc repro

## Run full pipeline with experiments
.PHONY: pipeline-full
pipeline-full: install
	poetry run dvc repro run_experiments

## Start MLflow UI
.PHONY: mlflow-ui
mlflow-ui:
	poetry run mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 3000

## Start MLflow server in background
.PHONY: mlflow-server
mlflow-server:
	poetry run mlflow server --backend-store-uri file:./mlruns --host 127.0.0.1 --port 3000 &

## Run Jupyter notebook
.PHONY: notebook
notebook: install
	poetry run jupyter notebook

## Fetch and preprocess data only
.PHONY: data-pipeline
data-pipeline: install
	poetry run dvc repro preprocess

## View experiment results summary
.PHONY: results
results:
	@if [ -f experiments_summary.json ]; then \
		cat experiments_summary.json | python -m json.tool | head -50; \
	else \
		echo "No experiments summary found. Run 'make experiments' first."; \
	fi

## Clean all experiment artifacts
.PHONY: clean-experiments
clean-experiments:
	rm -rf experiments/
	rm -rf mlruns/
	rm -f experiments_summary.json
	@echo "Cleaned all experiment artifacts"

## Show MLflow tracking URI
.PHONY: mlflow-info
mlflow-info:
	@echo "MLflow Tracking URI: file:./mlruns"
	@echo "MLflow UI URL: http://127.0.0.1:3000"
	@echo ""
	@echo "To start MLflow UI, run: make mlflow-ui"

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
