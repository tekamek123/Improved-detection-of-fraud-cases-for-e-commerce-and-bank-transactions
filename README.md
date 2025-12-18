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
│   ├── raw/             # Original datasets
│   └── processed/       # Cleaned and feature-engineered data
├── notebooks/           # Jupyter notebooks for analysis
├── src/                 # Source code modules
├── tests/               # Unit tests
├── models/              # Saved model artifacts
└── scripts/             # Utility scripts
```

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

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your datasets in `data/raw/` directory
2. Run notebooks in order:
   - `notebooks/eda-fraud-data.ipynb`: EDA for e-commerce data
   - `notebooks/eda-creditcard.ipynb`: EDA for credit card data
   - `notebooks/feature-engineering.ipynb`: Feature engineering
   - `notebooks/modeling.ipynb`: Model training and evaluation
   - `notebooks/shap-explainability.ipynb`: Model explainability analysis

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

