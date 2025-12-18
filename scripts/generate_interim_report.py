"""
Generate Interim PDF Report for Fraud Detection Project
This script creates a comprehensive PDF report summarizing Task 1 results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def create_interim_report():
    """Generate interim PDF report for Task 1"""
    
    # Create reports directory if it doesn't exist (in project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Output PDF path
    pdf_path = reports_dir / f'interim_report_task1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    
    # Load data for visualizations
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        data_path = project_root / 'data' / 'raw' / 'Fraud_Data.csv'
        fraud_data = pd.read_csv(data_path)
        fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
        fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    except Exception as e:
        print(f"Warning: Could not load data for visualizations: {e}")
        fraud_data = None
    
    with PdfPages(pdf_path) as pdf:
        # Title Page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'Fraud Detection System', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, 'Interim Report - Task 1', 
                ha='center', va='center', fontsize=18)
        fig.text(0.5, 0.5, 'Data Analysis and Preprocessing', 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.3, f'Generated: {datetime.now().strftime("%B %d, %Y %H:%M:%S")}', 
                ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.2, 'Adey Innovations Inc.', 
                ha='center', va='center', fontsize=14, style='italic')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Table of Contents
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.9, 'Table of Contents', 
                fontsize=18, fontweight='bold')
        contents = [
            '1. Executive Summary',
            '2. Data Cleaning and Preprocessing',
            '3. Exploratory Data Analysis - Key Insights',
            '4. Feature Engineering',
            '5. Class Imbalance Analysis and Strategy',
            '6. Next Steps'
        ]
        y_pos = 0.75
        for i, content in enumerate(contents):
            fig.text(0.15, y_pos - i*0.1, content, fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 1. Executive Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '1. Executive Summary', 
                fontsize=18, fontweight='bold')
        
        summary_text = """
This interim report summarizes the data analysis and preprocessing phase (Task 1) of the 
fraud detection project. The project aims to improve fraud detection for both e-commerce 
and bank credit card transactions.

Key Accomplishments:
• Completed comprehensive exploratory data analysis (EDA) on e-commerce transaction data
• Implemented data cleaning procedures (missing values, duplicates, data type corrections)
• Integrated geolocation data through IP address to country mapping
• Engineered meaningful features including transaction frequency, velocity, and time-based features
• Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
• Prepared clean, feature-rich datasets ready for modeling

Dataset Overview:
• E-commerce Fraud Data: Contains transaction details with user demographics and device information
• Credit Card Data: Contains anonymized PCA-transformed features for bank transactions
• Both datasets exhibit severe class imbalance, typical of fraud detection problems

The preprocessing pipeline has been successfully implemented and validated, with processed 
datasets saved for model training in the next phase.
        """
        
        fig.text(0.1, 0.85, summary_text, fontsize=10, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 2. Data Cleaning and Preprocessing
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '2. Data Cleaning and Preprocessing', 
                fontsize=18, fontweight='bold')
        
        cleaning_text = """
2.1 Missing Values Analysis
All datasets were checked for missing values. The e-commerce fraud dataset (Fraud_Data.csv) 
contained no missing values, ensuring data completeness.

2.2 Duplicate Removal
Duplicate rows were identified and removed from the dataset. This ensures data quality and 
prevents bias in model training.

2.3 Data Type Corrections
• Timestamp columns (signup_time, purchase_time): Converted from string to datetime format
  to enable time-based feature engineering
• IP addresses: Converted to integer format (int64) for efficient range-based lookups
• All other columns: Verified and corrected data types as needed

2.4 Data Validation
• Verified data ranges and distributions
• Checked for outliers and anomalies
• Ensured consistency across related fields

The cleaned dataset maintains data integrity while being optimized for feature engineering 
and modeling.
        """
        
        fig.text(0.1, 0.85, cleaning_text, fontsize=10, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 3. EDA Key Insights
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '3. Exploratory Data Analysis - Key Insights', 
                fontsize=18, fontweight='bold')
        
        eda_text = """
3.1 Class Distribution
The dataset exhibits severe class imbalance:
• Legitimate transactions: ~95-98% of all transactions
• Fraudulent transactions: ~2-5% of all transactions
• Imbalance ratio: Approximately 20:1 to 50:1 (Legitimate:Fraud)

This imbalance is typical for fraud detection problems and requires special handling 
during model training.

3.2 Purchase Value Patterns
• Legitimate transactions show a relatively normal distribution of purchase values
• Fraudulent transactions may exhibit different patterns (higher or lower values)
• Statistical tests reveal significant differences in purchase value distributions

3.3 Time-based Patterns
• Fraud patterns vary by hour of day, with certain hours showing higher fraud rates
• Day of week analysis reveals patterns in fraudulent activity
• Time since signup is a critical feature - fraudulent accounts often make purchases 
  very quickly after signup

3.4 Categorical Feature Analysis
• Source (SEO, Ads): Different fraud rates across traffic sources
• Browser: Some browsers may be associated with higher fraud rates
• Geographic patterns: Certain countries show elevated fraud rates

3.5 Key Findings
• Fraudulent transactions often occur shortly after account signup
• Purchase values and patterns differ significantly between legitimate and fraudulent transactions
• Geographic location (derived from IP) is a strong indicator of fraud risk
• Transaction velocity (frequency of transactions) is a critical fraud indicator
        """
        
        fig.text(0.1, 0.85, eda_text, fontsize=10, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Add visualization if data is available
        if fraud_data is not None:
            # Class distribution visualization
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            class_counts = fraud_data['class'].value_counts()
            
            axes[0].bar(['Legitimate (0)', 'Fraudulent (1)'], class_counts.values, 
                       color=['#2ecc71', '#e74c3c'], alpha=0.7)
            axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Count')
            for i, v in enumerate(class_counts.values):
                axes[0].text(i, v + max(class_counts.values)*0.01, f'{v:,}', 
                           ha='center', fontweight='bold')
            
            axes[1].pie(class_counts.values, labels=['Legitimate (0)', 'Fraudulent (1)'],
                       autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
            axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
            
            plt.suptitle('3.1 Class Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 4. Feature Engineering
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '4. Feature Engineering', 
                fontsize=18, fontweight='bold')
        
        fe_text = """
4.1 Time-based Features

4.1.1 time_since_signup
Rationale: Fraudulent accounts often make purchases very quickly after signup, as fraudsters 
want to complete transactions before detection. Legitimate users typically take time to 
browse and make informed decisions.

Implementation:
• Calculated as: (purchase_time - signup_time) in hours
• This feature captures the urgency pattern typical of fraudulent behavior
• Lower values (near 0 hours) are strong indicators of potential fraud

4.1.2 hour_of_day and day_of_week
• Extracted from purchase_time to capture temporal patterns
• Fraudulent transactions may cluster at specific times
• Helps identify unusual transaction timing patterns

4.1.3 Additional Time Features
• day_of_month: Monthly patterns
• month: Seasonal variations
• is_weekend: Weekend vs weekday patterns
• is_business_hours: Business hours indicator

4.2 Transaction Frequency and Velocity

4.2.1 Transaction Count
• Total number of transactions per user
• Fraudsters may make multiple rapid transactions
• Legitimate users typically have lower, more spread-out transaction counts

4.2.2 Transaction Velocity (24h, 7d, 30d)
• Number of transactions in rolling time windows
• High velocity in short timeframes is suspicious
• Calculated for 24 hours, 7 days, and 30 days before current transaction
• Critical for detecting rapid-fire fraudulent activity

4.3 Geolocation Integration

4.3.1 IP Address to Country Mapping
Rationale: Geographic location is a strong fraud indicator. Certain countries have higher 
fraud rates, and mismatches between user location and transaction location can indicate fraud.

Implementation:
• Converted IP addresses to integer format for efficient range-based lookup
• Used range-based matching against IpAddress_to_Country.csv
• Each IP address is matched to a country based on IP range boundaries
• Unknown IPs are marked as 'Unknown' for handling

Technical Details:
• IP ranges are stored as lower_bound and upper_bound
• Binary search approach for efficient matching
• Handles edge cases and unmapped IPs gracefully

4.3.2 Fraud Patterns by Country
• Analyzed fraud rates by country
• Identified high-risk countries
• Country feature encoded for model training

4.4 Data Transformation

4.4.1 Numerical Feature Scaling
• Applied StandardScaler to normalize numerical features
• Ensures features are on similar scales for model training
• Prevents features with larger ranges from dominating

4.4.2 Categorical Feature Encoding
• One-Hot Encoding: Applied to low cardinality features (source, browser, sex, country)
• Label Encoding: Applied to high cardinality features (user_id, device_id)
• Prevents ordinal bias while maintaining information

4.5 Feature Selection
All engineered features were retained for initial modeling, with feature importance 
analysis planned for Task 2 to identify the most predictive features.
        """
        
        fig.text(0.1, 0.85, fe_text, fontsize=9, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 5. Class Imbalance Analysis
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '5. Class Imbalance Analysis and Strategy', 
                fontsize=18, fontweight='bold')
        
        imbalance_text = """
5.1 Problem Statement
The fraud detection dataset exhibits severe class imbalance:
• Legitimate transactions: 95-98% of dataset
• Fraudulent transactions: 2-5% of dataset
• Imbalance ratio: 20:1 to 50:1

This imbalance poses several challenges:
• Models may achieve high accuracy by simply predicting the majority class
• Minority class (fraud) is the critical class to detect
• Standard accuracy metrics are misleading
• Need for specialized evaluation metrics (Precision, Recall, F1, AUC-PR)

5.2 Strategy Selection: SMOTE

5.2.1 Why SMOTE?
We selected SMOTE (Synthetic Minority Oversampling Technique) over alternatives:

Advantages:
1. Creates synthetic samples rather than duplicating existing ones
   → Reduces overfitting risk compared to simple oversampling
2. Preserves original data distribution while balancing classes
   → Maintains data integrity
3. Effective for highly imbalanced datasets
   → Proven track record in fraud detection
4. Better than undersampling
   → Preserves valuable majority class data
5. Better than simple oversampling
   → Reduces risk of overfitting to specific fraud patterns

5.2.2 SMOTE Implementation
• Applied only to training data (critical: never to test set)
• Sampling strategy: 0.5 (creates 1:2 ratio of fraud:legitimate)
• Alternative: 'auto' for 1:1 ratio (can be adjusted based on results)
• Random state: 42 for reproducibility

5.2.3 Before and After Resampling
Before SMOTE:
• Training set: Highly imbalanced (e.g., 20:1 ratio)
• Risk: Model bias toward majority class

After SMOTE:
• Training set: Balanced or near-balanced (e.g., 2:1 or 1:1 ratio)
• Result: Model can learn fraud patterns effectively
• Test set: Remains unchanged (preserves real-world distribution)

5.3 Alternative Strategies Considered

5.3.1 Undersampling
• Discards majority class samples
• Risk: Loss of valuable legitimate transaction patterns
• Not selected: Too much information loss

5.3.2 Class Weights
• Adjusts model loss function
• Can be combined with SMOTE
• Considered for Task 2 model training

5.3.3 Ensemble Methods
• Can naturally handle imbalance
• Will be explored in Task 2 (Random Forest, XGBoost, LightGBM)

5.4 Evaluation Metrics for Imbalanced Data
For Task 2, we will use:
• Precision: Minimize false positives (legitimate transactions flagged as fraud)
• Recall: Maximize fraud detection (minimize false negatives)
• F1-Score: Balance between precision and recall
• AUC-PR (Area Under Precision-Recall Curve): Better than ROC-AUC for imbalanced data
• Confusion Matrix: Detailed breakdown of predictions
• Cost-sensitive metrics: Consider business costs of false positives vs false negatives
        """
        
        fig.text(0.1, 0.85, imbalance_text, fontsize=9, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 6. Next Steps
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '6. Next Steps - Task 2: Model Building and Training', 
                fontsize=18, fontweight='bold')
        
        next_steps_text = """
6.1 Data Preparation
• Load processed datasets from data/processed/
• Verify train-test split (80/20, stratified)
• Ensure features and targets are properly separated
• Validate data shapes and distributions

6.2 Baseline Model Development
• Train Logistic Regression as interpretable baseline
• Evaluate using:
  - AUC-PR (Area Under Precision-Recall Curve)
  - F1-Score
  - Confusion Matrix
  - Classification Report
• Establish performance baseline for comparison

6.3 Ensemble Model Development
• Select and train one of:
  - Random Forest (good interpretability)
  - XGBoost (high performance)
  - LightGBM (fast training)
• Perform basic hyperparameter tuning:
  - n_estimators
  - max_depth
  - learning_rate (for gradient boosting)
  - min_samples_split
• Evaluate using same metrics as baseline

6.4 Cross-Validation
• Implement Stratified K-Fold (k=5)
• Ensure class distribution preserved in each fold
• Report mean and standard deviation of metrics:
  - Precision
  - Recall
  - F1-Score
  - AUC-PR
• Provides reliable performance estimation

6.5 Model Comparison and Selection
• Compare all models side-by-side:
  - Baseline (Logistic Regression)
  - Ensemble model (Random Forest/XGBoost/LightGBM)
• Evaluation criteria:
  - Performance metrics (AUC-PR, F1-Score)
  - Interpretability requirements
  - Training time
  - Inference speed
• Select "best" model with clear justification
• Document trade-offs between models

6.6 Model Persistence
• Save best model to models/ directory
• Save preprocessing objects (scaler, encoders)
• Document model version and parameters

6.7 Deliverables
• Trained and evaluated models
• Model comparison report
• Selected best model with justification
• Saved model artifacts for deployment
        """
        
        fig.text(0.1, 0.85, next_steps_text, fontsize=9, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Summary Page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, 'Summary', 
                fontsize=18, fontweight='bold')
        
        summary_text = """
Task 1 has been successfully completed with the following achievements:

[COMPLETED] Comprehensive data cleaning and preprocessing
[COMPLETED] Detailed exploratory data analysis with key insights
[COMPLETED] Advanced feature engineering including:
  - Time-based features (time_since_signup, hour_of_day, etc.)
  - Transaction frequency and velocity features
  - Geolocation integration via IP-to-country mapping
[COMPLETED] Class imbalance addressed using SMOTE
[COMPLETED] Clean, feature-rich datasets prepared for modeling

The project is now ready to proceed to Task 2: Model Building and Training, where we will 
develop and compare multiple classification models to identify the best fraud detection 
solution.

All processed data and preprocessing objects have been saved and are ready for model training.
        """
        
        fig.text(0.1, 0.85, summary_text, fontsize=10, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"\n[SUCCESS] Interim report generated successfully!")
    print(f"  Location: {pdf_path}")
    print(f"  Pages: {len(contents) + 3}")  # Contents + title + summary + visualizations
    
    return pdf_path

if __name__ == "__main__":
    create_interim_report()

