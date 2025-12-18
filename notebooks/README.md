# Notebooks Directory

This directory contains Jupyter notebooks for the fraud detection project.

## Notebooks

1. **eda-fraud-data.ipynb**: Exploratory Data Analysis for e-commerce transaction data (Fraud_Data.csv)
   - Data loading and initial inspection
   - Missing value analysis
   - Statistical summaries
   - Distribution analysis
   - Class imbalance visualization
   - Feature correlation analysis

2. **eda-creditcard.ipynb**: Exploratory Data Analysis for bank credit card transaction data (creditcard.csv)
   - Data loading and initial inspection
   - Missing value analysis
   - Statistical summaries
   - Distribution analysis
   - Class imbalance visualization
   - PCA feature analysis

3. **feature-engineering.ipynb**: Feature engineering and preprocessing
   - IP address to country mapping
   - Time-based feature extraction
   - Transaction pattern features
   - Geolocation features
   - Feature scaling and transformation
   - Handling class imbalance (SMOTE, undersampling, etc.)

4. **modeling.ipynb**: Model training and evaluation
   - Model selection (Logistic Regression, Random Forest, XGBoost, LightGBM)
   - Hyperparameter tuning
   - Cross-validation
   - Model evaluation with appropriate metrics
   - Model comparison and selection
   - Model persistence

5. **shap-explainability.ipynb**: Model explainability using SHAP
   - SHAP value calculation
   - Feature importance visualization
   - Individual prediction explanations
   - Global model interpretation
   - Feature interaction analysis

## Usage

Run notebooks in the order listed above for a complete analysis pipeline.

