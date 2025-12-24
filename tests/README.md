# Tests Directory

This directory contains unit tests and integration tests for the fraud detection system.

## Structure

```
tests/
├── __init__.py
├── test_data_cleaning.py      # Tests for data cleaning functions
├── test_feature_engineering.py # Tests for feature engineering functions
├── test_preprocessing.py       # Tests for preprocessing functions
└── test_models.py              # Tests for model training and evaluation
```

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run tests with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Coverage

Tests ensure:
- Data cleaning functions handle missing values and duplicates correctly
- Feature engineering produces expected features
- Preprocessing (scaling, encoding) works correctly
- Models can be trained and make predictions
- Evaluation metrics are calculated correctly

