"""
Feature engineering utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


def map_ip_to_country(ip_address: int, ip_country_df: pd.DataFrame) -> str:
    """
    Map IP address to country using range-based lookup.
    
    Args:
        ip_address: IP address as integer
        ip_country_df: DataFrame with lower_bound_ip_address, upper_bound_ip_address, country
        
    Returns:
        Country name or 'Unknown'
    """
    mask = (ip_country_df['lower_bound_ip_address'] <= ip_address) & \
           (ip_country_df['upper_bound_ip_address'] >= ip_address)
    matches = ip_country_df[mask]
    
    if len(matches) > 0:
        return matches.iloc[0]['country']
    else:
        return 'Unknown'


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to fraud data.
    
    Args:
        df: DataFrame with signup_time and purchase_time columns
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Extract time features from purchase_time
    if 'purchase_time' in df.columns:
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['day_of_month'] = df['purchase_time'].dt.day
        df['month'] = df['purchase_time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & 
                                   (df['hour_of_day'] < 17)).astype(int)
    
    # Time since signup
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['time_since_signup'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds() / 3600  # in hours
    
    return df


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transaction frequency and velocity features.
    
    Args:
        df: DataFrame sorted by user_id and purchase_time
        
    Returns:
        DataFrame with added transaction features
    """
    df = df.copy()
    
    # Sort by user_id and purchase_time
    df = df.sort_values(['user_id', 'purchase_time']).reset_index(drop=True)
    
    # Transaction count per user
    user_transaction_count = df.groupby('user_id').size().reset_index(name='transaction_count')
    df = df.merge(user_transaction_count, on='user_id', how='left')
    
    # Time since last transaction
    df['prev_purchase_time'] = df.groupby('user_id')['purchase_time'].shift(1)
    df['time_since_last_transaction'] = (
        df['purchase_time'] - df['prev_purchase_time']
    ).dt.total_seconds() / 3600  # in hours
    df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(999999)
    
    # Initialize velocity features
    df['transactions_last_24h'] = 0
    df['transactions_last_7d'] = 0
    df['transactions_last_30d'] = 0
    
    # Calculate velocity (this is computationally expensive, consider optimization)
    for idx, row in df.iterrows():
        user_id = row['user_id']
        purchase_time = row['purchase_time']
        
        user_transactions = df[
            (df['user_id'] == user_id) & 
            (df['purchase_time'] < purchase_time)
        ]
        
        df.loc[idx, 'transactions_last_24h'] = len(
            user_transactions[user_transactions['purchase_time'] >= purchase_time - pd.Timedelta(hours=24)]
        )
        df.loc[idx, 'transactions_last_7d'] = len(
            user_transactions[user_transactions['purchase_time'] >= purchase_time - pd.Timedelta(days=7)]
        )
        df.loc[idx, 'transactions_last_30d'] = len(
            user_transactions[user_transactions['purchase_time'] >= purchase_time - pd.Timedelta(days=30)]
        )
    
    return df

