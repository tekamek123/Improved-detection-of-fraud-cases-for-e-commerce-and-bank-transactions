"""
Preprocessing utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, List, Dict
import joblib
from pathlib import Path


def prepare_features(df: pd.DataFrame, 
                     numerical_features: List[str],
                     categorical_features: List[str],
                     target_col: str = 'class') -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Prepare features for modeling: scaling and encoding.
    
    Args:
        df: Input dataframe
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        target_col: Name of target column
        
    Returns:
        Tuple of (transformed features, target, preprocessing objects dict)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Keep only features that exist
    numerical_features = [f for f in numerical_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    X_transformed = X.copy()
    preprocessing_objects = {}
    
    # Scale numerical features
    if numerical_features:
        scaler = StandardScaler()
        X_transformed[numerical_features] = scaler.fit_transform(X[numerical_features])
        preprocessing_objects['scaler'] = scaler
    
    # Encode categorical features
    low_cardinality = []
    high_cardinality = []
    
    for feature in categorical_features:
        if feature in X.columns:
            unique_count = X[feature].nunique()
            if unique_count <= 20:
                low_cardinality.append(feature)
            else:
                high_cardinality.append(feature)
    
    # One-hot encode low cardinality
    if low_cardinality:
        X_encoded = pd.get_dummies(X_transformed[low_cardinality], 
                                  prefix=low_cardinality, drop_first=True)
        X_transformed = pd.concat([X_transformed.drop(low_cardinality, axis=1), 
                                  X_encoded], axis=1)
    
    # Label encode high cardinality
    label_encoders = {}
    if high_cardinality:
        for feature in high_cardinality:
            le = LabelEncoder()
            X_transformed[feature] = le.fit_transform(X[feature].astype(str))
            label_encoders[feature] = le
        preprocessing_objects['label_encoders'] = label_encoders
    
    # Drop time columns if present
    time_cols = ['signup_time', 'purchase_time', 'prev_purchase_time']
    time_cols = [c for c in time_cols if c in X_transformed.columns]
    if time_cols:
        X_transformed = X_transformed.drop(time_cols, axis=1)
    
    return X_transformed, y, preprocessing_objects


def split_and_resample(X: pd.DataFrame, 
                      y: pd.Series,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      use_smote: bool = True,
                      smote_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                             pd.Series, pd.Series]:
    """
    Split data and apply SMOTE to training set.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        use_smote: Whether to apply SMOTE
        smote_strategy: SMOTE sampling strategy
        
    Returns:
        Tuple of (X_train_resampled, X_test, y_train_resampled, y_test)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply SMOTE to training data only
    if use_smote:
        try:
            smote = SMOTE(random_state=random_state, sampling_strategy=smote_strategy)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"Warning: SMOTE failed with default parameters: {e}")
            print("Trying with adjusted parameters...")
            smote = SMOTE(random_state=random_state, k_neighbors=3, 
                         sampling_strategy=smote_strategy)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    return X_train_resampled, X_test, y_train_resampled, y_test

