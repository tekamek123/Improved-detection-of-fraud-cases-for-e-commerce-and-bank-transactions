# Fraud Detection System

## Overview

This project focuses on improving fraud detection for e-commerce and bank credit transactions. The system uses advanced machine learning models and detailed data analysis to identify fraudulent activities accurately, balancing security and user experience.

## Business Need

Adey Innovations Inc. requires accurate and robust fraud detection models that handle the unique challenges of both e-commerce transaction data and bank credit transaction data. The system uses geolocation analysis and transaction pattern recognition to improve detection capabilities.

## Key Challenges

- **Class Imbalance**: Both datasets are highly imbalanced, with far fewer fraudulent transactions than legitimate ones
- **Trade-off Management**: Balancing false positives (incorrectly flagging legitimate transactions) vs false negatives (missing actual fraud)
- **Real-time Performance**: Models must be efficient for real-time monitoring and reporting

## Project Structure

```
fraud-detection/
├── .vscode/              # VS Code settings
├── .github/              # GitHub workflows
├── data/                 # Data files (gitignored)
│   ├── raw/             # Original datasets (place your CSV files here)
│   └── processed/       # Cleaned and feature-engineered data (generated)
├── notebooks/           # Jupyter notebooks for analysis
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── src/                 # Source code modules
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── preprocessing.py
├── tests/               # Unit tests (see tests/README.md)
│   └── __init__.py
├── models/              # Saved model artifacts (see models/README.md)
│   ├── best_model_*.pkl
│   ├── *_model.pkl
│   ├── scaler.pkl
│   └── model_comparison_results.csv
├── reports/             # Generated reports (see reports/README.md)
│   └── interim_report_task1_*.pdf
└── scripts/             # Utility scripts
    ├── generate_interim_report.py
    └── run_modeling_notebook.py
```

### Directory Descriptions

- **`data/raw/`**: Place your input datasets here (Fraud_Data.csv, creditcard.csv, IpAddress_to_Country.csv)
- **`data/processed/`**: Generated processed datasets (created by feature-engineering notebook)
- **`models/`**: Trained model artifacts and preprocessing objects (created by modeling notebook)
- **`reports/`**: Generated PDF reports (created by report generation scripts)
- **`tests/`**: Unit tests for source code modules
- **`notebooks/`**: Jupyter notebooks for data analysis and modeling
- **`src/`**: Reusable Python modules for data processing
- **`scripts/`**: Utility scripts for report generation and automation

## Datasets

### 1. Fraud_Data.csv
E-commerce transaction data with the following features:
- `user_id`: Unique identifier for the user
- `signup_time`: Timestamp when user signed up
- `purchase_time`: Timestamp when purchase was made
- `purchase_value`: Purchase value in dollars
- `device_id`: Unique identifier for the device
- `source`: Source through which user came to site (SEO, Ads, etc.)
- `browser`: Browser used for transaction
- `sex`: Gender of user (M/F)
- `age`: Age of user
- `ip_address`: IP address of transaction
- `class`: Target variable (1 = fraud, 0 = legitimate)

### 2. IpAddress_to_Country.csv
Maps IP addresses to countries:
- `lower_bound_ip_address`: Lower bound of IP address range
- `upper_bound_ip_address`: Upper bound of IP address range
- `country`: Country corresponding to IP address range

### 3. creditcard.csv
Bank transaction data with:
- `Time`: Seconds elapsed since first transaction
- `V1-V28`: Anonymized PCA-transformed features
- `Amount`: Transaction amount in dollars
- `Class`: Target variable (1 = fraud, 0 = legitimate)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fraud-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```
   
   **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost, lightgbm; print('All packages installed successfully!')"
   ```

## Usage

### Step 1: Prepare Data
Place your datasets in the `data/raw/` directory:
- `Fraud_Data.csv`: E-commerce transaction data
- `IpAddress_to_Country.csv`: IP to country mapping
- `creditcard.csv`: Bank credit card transaction data

### Step 2: Run Analysis Pipeline

Execute notebooks in the following order:

1. **Exploratory Data Analysis**
   ```bash
   # Open in Jupyter
   python -m jupyter notebook notebooks/eda-fraud-data.ipynb
   python -m jupyter notebook notebooks/eda-creditcard.ipynb
   ```
   - Performs univariate and bivariate analysis
   - Visualizes class distribution and feature relationships
   - Generates insights for feature engineering

2. **Feature Engineering**
   ```bash
   python -m jupyter notebook notebooks/feature-engineering.ipynb
   ```
   - Cleans and preprocesses data
   - Engineers time-based and transaction frequency features
   - Maps IP addresses to countries
   - Applies SMOTE for class imbalance
   - **Output**: Processed datasets saved to `data/processed/`

3. **Model Training**
   ```bash
   python -m jupyter notebook notebooks/modeling.ipynb
   ```
   - Trains baseline (Logistic Regression) and ensemble models
   - Performs hyperparameter tuning
   - Runs cross-validation
   - Compares and selects best model
   - **Output**: Trained models saved to `models/`

4. **Model Explainability**
   ```bash
   python -m jupyter notebook notebooks/shap-explainability.ipynb
   ```
   - Generates SHAP values for model interpretation
   - Creates feature importance visualizations
   - Provides individual prediction explanations

### Step 3: Generate Reports

Generate interim and final reports:
```bash
python scripts/generate_interim_report.py
```
**Output**: PDF reports saved to `reports/`

## Output Locations

### Models
- **Location**: `models/` directory
- **Contents**: 
  - `best_model_*.pkl`: Selected best model for deployment
  - `*_model.pkl`: All trained models (Logistic Regression, Random Forest, XGBoost, LightGBM)
  - `scaler.pkl`, `label_encoders.pkl`: Preprocessing objects
  - `model_comparison_results.csv`: Model performance comparison

### Reports
- **Location**: `reports/` directory
- **Contents**:
  - `interim_report_task1_*.pdf`: Task 1 analysis reports
  - Future: Model comparison reports, final project report, SHAP explainability reports

### Processed Data
- **Location**: `data/processed/` directory
- **Contents**:
  - `X_train_processed.csv`, `y_train_processed.csv`: Training data
  - `X_test_processed.csv`, `y_test_processed.csv`: Test data

### Notebooks
- **Location**: `notebooks/` directory
- **Contents**: All analysis and modeling notebooks with executed outputs

## Evaluation Metrics

Given the class imbalance, models will be evaluated on:
- Precision and Recall
- F1-Score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix
- Cost-sensitive metrics considering false positive vs false negative costs

## Model Explainability

The project uses SHAP (SHapley Additive exPlanations) to interpret model decisions and understand feature importance.

## Testing

Run tests with:
```bash
pytest tests/
```

## License

Proprietary - Adey Innovations Inc.
