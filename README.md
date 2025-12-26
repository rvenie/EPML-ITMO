# ResearchHub

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

мультиагентная система, которая автоматизирует мониторинг и анализ научных публикаций в области цифровой патологии и анализа WSI (Whole Slide Imaging) данных. Система работает как умный исследовательский ассистент, который непрерывно отслеживает новые разработки в области анализа гистопатологических изображений.

--------

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Dockerfile
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Poetry configuration with dependencies, dev tools (ruff, mypy, bandit)
│                         and project metadata. Use `poetry install` to set up environment.
├── poetry.lock        <- Lock file with exact versions for reproducible builds
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│

│
└── researchhub   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes researchhub a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
