# Models Directory

This directory stores trained model artifacts and related files.

## Contents

### Trained Models
- `best_model_*.pkl`: The selected best model for deployment
- `logistic_regression.pkl`: Baseline Logistic Regression model
- `random_forest.pkl`: Random Forest ensemble model
- `xgboost.pkl`: XGBoost gradient boosting model
- `xgboost_tuned.pkl`: Hyperparameter-tuned XGBoost model
- `lightgbm.pkl`: LightGBM gradient boosting model

### Preprocessing Objects
- `scaler.pkl`: StandardScaler used for feature scaling
- `label_encoders.pkl`: Label encoders for categorical features

### Model Comparison Results
- `model_comparison_results.csv`: Detailed comparison of all models with metrics

## Usage

To load a saved model:
```python
import joblib

# Load the best model
model = joblib.load('models/best_model_random_forest.pkl')

# Load preprocessing objects
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
```

## Model Selection

The best model is selected based on:
- **Primary Metric**: PR-AUC (Precision-Recall Area Under Curve)
- **Secondary Metrics**: F1-Score, Recall, Precision
- **Cross-Validation Stability**: Low standard deviation across folds

See `model_comparison_results.csv` for detailed performance metrics.

## Notes

- Models are saved using `joblib` for efficient serialization
- All models are trained on the processed training data
- Models are evaluated on the held-out test set
- Preprocessing objects must be loaded alongside models for inference

