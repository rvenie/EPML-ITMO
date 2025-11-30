# Reproducibility Guide

This document provides comprehensive instructions for reproducing all results from the ML project with DVC and MLflow integration.

## Table of Contents
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Data Versioning with DVC](#data-versioning-with-dvc)
- [Model Versioning with MLflow](#model-versioning-with-mlflow)
- [Step-by-Step Reproduction](#step-by-step-reproduction)
- [Docker Reproduction](#docker-reproduction)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10/11
- **Python**: 3.11 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for initial setup and package downloads

### Recommended Tools
- **Git**: 2.30 or higher
- **Docker**: 20.10 or higher (for containerized reproduction)
- **Docker Compose**: 1.29 or higher

## Environment Setup

### Method 1: Local Python Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd research_agets_hub
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import dvc, mlflow, sklearn, pandas; print('All packages installed successfully')"
   ```

### Method 2: Using Poetry (Recommended)

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install project dependencies**
   ```bash
   poetry install
   poetry shell
   ```

## Data Versioning with DVC

### Initial DVC Setup

1. **Initialize DVC** (if not already done)
   ```bash
   dvc init
   ```

2. **Add remote storage**
   ```bash
   # For local storage (development)
   dvc remote add -d local_storage ../dvc-storage
   
   # For S3 storage (production)
   # dvc remote add -d s3_storage s3://your-bucket/dvc-storage
   # dvc remote modify s3_storage access_key_id YOUR_ACCESS_KEY
   # dvc remote modify s3_storage secret_access_key YOUR_SECRET_KEY
   ```

3. **Configure DVC**
   ```bash
   dvc config core.autostage true
   ```

### Data Pipeline Reproduction

1. **Pull data from DVC**
   ```bash
   dvc pull
   ```

2. **If data files are missing, recreate them**
   ```bash
   # The raw data is already committed, but if you need to recreate:
   cp data/raw/publications.csv.example data/raw/publications.csv
   dvc add data/raw/publications.csv
   ```

3. **Verify data integrity**
   ```bash
   dvc status
   dvc dag  # View pipeline DAG
   ```

## Model Versioning with MLflow

### MLflow Setup

1. **Start MLflow server** (in background)
   ```bash
   mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri file:./mlruns &
   ```

2. **Verify MLflow is running**
   ```bash
   curl http://127.0.0.1:5000/
   ```
   
3. **Access MLflow UI**
   Open http://127.0.0.1:5000 in your browser

### MLflow Configuration

The project is configured with the following MLflow settings in `params.yaml`:
- **Experiment name**: `research_publications_classification`
- **Tracking URI**: `file:./mlruns`
- **Model registry**: Enabled for model versioning

## Step-by-Step Reproduction

### Step 1: Data Preprocessing

```bash
# Run data preprocessing
python scripts/preprocess_data.py

# Verify outputs
ls -la data/processed/
cat data/processed/processing_metadata.yaml
```

**Expected outputs**:
- `data/processed/publications_processed.csv` (51 records, 21 columns)
- `data/processed/processing_metadata.yaml` (processing metadata)

### Step 2: Model Training

```bash
# Train model with MLflow logging
python scripts/train_model.py \
    --input data/processed/publications_processed.csv \
    --model-output models/classifier.pkl \
    --metrics metrics.json

# Check outputs
ls -la models/
cat metrics.json
```

**Expected outputs**:
- `models/classifier.pkl` (trained RandomForest model)
- `models/classifier_metadata.yaml` (model metadata)
- `metrics.json` (performance metrics)
- MLflow run logged with ID

### Step 3: Verify DVC Tracking

```bash
# Check DVC status
dvc status

# Add new model version to DVC
dvc add models/classifier.pkl

# Push to remote storage
dvc push
```

### Step 4: Version Control

```bash
# Commit changes to Git
git add .
git commit -m "Add trained model and update data processing"

# Tag the version
git tag -a v1.0.0 -m "First stable version with DVC and MLflow integration"
```

## Docker Reproduction

### Method 1: Using Docker Compose (Recommended)

1. **Build and start services**
   ```bash
   # Development mode with Jupyter
   docker-compose --profile development up -d
   
   # Access Jupyter at http://localhost:8888
   # Access MLflow at http://localhost:5000
   ```

2. **Run training pipeline**
   ```bash
   # Run preprocessing
   docker-compose --profile preprocessing up data-preprocessing
   
   # Run training
   docker-compose --profile training up model-training
   ```

3. **Production deployment**
   ```bash
   # Start MLflow server only
   docker-compose up -d mlflow-server
   
   # Run training in production mode
   docker-compose run --rm ml-app train
   ```

### Method 2: Manual Docker Build

1. **Build Docker image**
   ```bash
   docker build -t ml-project:latest .
   ```

2. **Run preprocessing**
   ```bash
   docker run --rm -v $(pwd)/data:/app/data ml-project:latest preprocess
   ```

3. **Run training**
   ```bash
   docker run --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/mlruns:/app/mlruns \
     ml-project:latest train
   ```

## Verification

### Expected Results

1. **Data Processing Metrics**:
   - Original dataset: 51 publications
   - Processed dataset: 51 records, 21 features
   - No missing data in critical fields

2. **Model Performance** (approximate values):
   - Cross-validation accuracy: ~0.77 (±0.05)
   - Test accuracy: ~0.90
   - F1-score: ~0.87
   - Model type: RandomForestClassifier

3. **MLflow Artifacts**:
   - Registered model: `research_publications_classification_model`
   - Model version: 1
   - Logged parameters: 15+ parameters
   - Logged metrics: 6+ metrics

### Verification Commands

```bash
# Check data integrity
python -c "
import pandas as pd
df = pd.read_csv('data/processed/publications_processed.csv')
print(f'Shape: {df.shape}')
print(f'Missing values: {df.isnull().sum().sum()}')
assert df.shape[0] == 51, 'Wrong number of records'
assert df.shape[1] == 21, 'Wrong number of features'
print('✓ Data verification passed')
"

# Check model file
python -c "
import pickle
with open('models/classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)
assert 'model' in model_data, 'Model not found in pickle file'
assert 'tfidf_vectorizer' in model_data, 'Vectorizer not found'
print('✓ Model file verification passed')
"

# Check MLflow tracking
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
assert len(experiments) >= 1, 'No MLflow experiments found'
runs = client.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])
assert len(runs) >= 1, 'No MLflow runs found'
print('✓ MLflow verification passed')
"
```

### Performance Benchmarks

| Metric | Expected Value | Acceptable Range |
|--------|----------------|------------------|
| Training Time | ~5-15 seconds | < 60 seconds |
| Model Size | ~10-50 KB | < 1 MB |
| Memory Usage | ~100-300 MB | < 1 GB |
| CV Accuracy | ~0.77 | 0.70-0.85 |
| Test Accuracy | ~0.90 | 0.80-0.95 |

## Troubleshooting

### Common Issues and Solutions

1. **DVC Remote Access Issues**
   ```bash
   # Check remote configuration
   dvc remote list
   
   # Test remote connectivity
   dvc remote --test -v local_storage
   
   # Reconfigure remote if needed
   dvc remote remove local_storage
   dvc remote add -d local_storage ../dvc-storage
   ```

2. **MLflow Server Not Starting**
   ```bash
   # Kill existing MLflow processes
   pkill -f mlflow
   
   # Clear MLflow cache
   rm -rf ~/.mlflow
   
   # Restart with verbose logging
   mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri file:./mlruns --verbose
   ```

3. **Memory Issues During Training**
   ```bash
   # Reduce model complexity in params.yaml
   # - Decrease n_estimators from 100 to 50
   # - Decrease tfidf_max_features from 5000 to 1000
   ```

4. **Docker Permission Issues**
   ```bash
   # Fix Docker permissions (Linux/macOS)
   sudo chown -R $USER:$USER .
   
   # Windows: Run Docker Desktop as Administrator
   ```

5. **Package Version Conflicts**
   ```bash
   # Create clean environment
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install -r requirements.txt
   ```

### Environment-Specific Issues

**macOS Apple Silicon**:
```bash
# Install compatible versions
pip install --upgrade tensorflow-macos tensorflow-metal
```

**Windows**:
```bash
# Use long path support
git config --system core.longpaths true
```

**Linux**:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

### Data Issues

**Missing Data Files**:
```bash
# Regenerate data if source files are missing
python scripts/generate_sample_data.py
dvc add data/raw/publications.csv
```

**Corrupted DVC Cache**:
```bash
# Clear and rebuild DVC cache
rm -rf .dvc/cache
dvc fetch
dvc checkout
```

## Support and Contact

If you encounter issues not covered in this guide:

1. **Check the logs**: Look at `preprocessing.log` and `training.log`
2. **Verify environment**: Run the verification commands above
3. **Check versions**: Ensure Python 3.11+ and all package versions match `requirements.txt`
4. **Clean installation**: Try a fresh virtual environment

For additional support, please check the project's issue tracker or contact the development team.

---

**Last Updated**: November 30, 2024  
**Version**: 1.0.0  
**Tested On**: Python 3.11, Ubuntu 20.04, macOS 13, Windows 11