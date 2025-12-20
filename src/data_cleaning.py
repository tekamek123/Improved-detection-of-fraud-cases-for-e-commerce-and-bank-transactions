"""
Data cleaning utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def clean_fraud_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean e-commerce fraud data (Fraud_Data.csv).
    
    Args:
        df: Raw fraud data dataframe
        
    Returns:
        Tuple of (cleaned dataframe, cleaning report dictionary)
    """
    df_clean = df.copy()
    report = {
        'original_shape': df.shape,
        'missing_values': {},
        'duplicates_removed': 0,
        'data_type_changes': []
    }
    
    # Check for missing values
    missing = df_clean.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # Remove duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        report['duplicates_removed'] = duplicates
    
    # Convert time columns to datetime
    if 'signup_time' in df_clean.columns:
        df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
        report['data_type_changes'].append('signup_time -> datetime')
    
    if 'purchase_time' in df_clean.columns:
        df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
        report['data_type_changes'].append('purchase_time -> datetime')
    
    # Convert IP address to integer
    if 'ip_address' in df_clean.columns:
        df_clean['ip_address'] = df_clean['ip_address'].astype('int64')
        report['data_type_changes'].append('ip_address -> int64')
    
    report['final_shape'] = df_clean.shape
    
    return df_clean, report


def clean_creditcard_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean credit card fraud data (creditcard.csv).
    
    Args:
        df: Raw credit card data dataframe
        
    Returns:
        Tuple of (cleaned dataframe, cleaning report dictionary)
    """
    df_clean = df.copy()
    report = {
        'original_shape': df.shape,
        'missing_values': {},
        'duplicates_removed': 0,
        'infinite_values_replaced': 0,
        'outliers_detected': 0
    }
    
    # Check for missing values
    missing = df_clean.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # Remove duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        report['duplicates_removed'] = duplicates
    
    # Check and replace infinite values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df_clean[numeric_cols]).sum().sum()
    if inf_count > 0:
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        report['infinite_values_replaced'] = inf_count
    
    # Detect outliers in Amount (IQR method)
    if 'Amount' in df_clean.columns:
        Q1 = df_clean['Amount'].quantile(0.25)
        Q3 = df_clean['Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df_clean['Amount'] < lower_bound) | 
                    (df_clean['Amount'] > upper_bound)).sum()
        report['outliers_detected'] = outliers
    
    report['final_shape'] = df_clean.shape
    
    return df_clean, report

