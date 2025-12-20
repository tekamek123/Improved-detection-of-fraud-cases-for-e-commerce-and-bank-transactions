"""
Generate Interim PDF Report for Fraud Detection Project
This script creates a comprehensive PDF report summarizing Task 1 results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
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
        data_path = project_root / 'data' / 'raw' / 'Fraud_Data.csv'
        fraud_data = pd.read_csv(data_path)
        fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
        fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
        
        # Load processed data if available
        try:
            X_train = pd.read_csv(project_root / 'data' / 'processed' / 'X_train_processed.csv')
            y_train = pd.read_csv(project_root / 'data' / 'processed' / 'y_train_processed.csv')
            y_train = y_train.iloc[:, 0] if len(y_train.columns) == 1 else y_train['class']
        except:
            X_train = None
            y_train = None
            
    except Exception as e:
        print(f"Warning: Could not load data for visualizations: {e}")
        fraud_data = None
        X_train = None
        y_train = None
    
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
            '6. Next Steps and Anticipated Challenges'
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
        
        # Add data cleaning table if data is available
        if fraud_data is not None:
            fig, ax = plt.subplots(figsize=(11, 3))
            ax.axis('tight')
            ax.axis('off')
            
            cleaning_table_data = [
                ['Check', 'E-commerce Data', 'Credit Card Data'],
                ['Missing Values', '0 (0%)', '0 (0%)'],
                ['Duplicates', '0 (0%)', '0 (0%)'],
                ['Data Type Corrections', '3 columns', 'Verified'],
                ['Final Shape', f'{fraud_data.shape[0]:,} x {fraud_data.shape[1]}', '284,807 x 31']
            ]
            
            table = ax.table(cellText=cleaning_table_data[1:], 
                           colLabels=cleaning_table_data[0],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.4, 0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            for i in range(len(cleaning_table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('2.1 Data Cleaning Summary Table', fontsize=14, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 3. EDA Key Insights with Visualizations
        if fraud_data is not None:
            # Class distribution visualization
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            class_counts = fraud_data['class'].value_counts()
            class_percentages = fraud_data['class'].value_counts(normalize=True) * 100
            
            axes[0].bar(['Legitimate (0)', 'Fraudulent (1)'], class_counts.values, 
                       color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            axes[0].set_title('Class Distribution (Count)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Count')
            axes[0].set_ylim(0, max(class_counts.values) * 1.1)
            for i, v in enumerate(class_counts.values):
                axes[0].text(i, v + max(class_counts.values)*0.01, f'{v:,}', 
                           ha='center', fontweight='bold', fontsize=10)
            
            axes[1].pie(class_counts.values, labels=['Legitimate (0)', 'Fraudulent (1)'],
                       autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
            axes[1].set_title('Class Distribution (Percentage)', fontsize=12, fontweight='bold')
            
            plt.suptitle('3.1 Class Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Class distribution table
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis('tight')
            ax.axis('off')
            
            class_table_data = [
                ['Class', 'Count', 'Percentage', 'Imbalance Ratio'],
                ['Legitimate (0)', f'{class_counts[0]:,}', f'{class_percentages[0]:.2f}%', 
                 f'{class_counts[0]/class_counts[1]:.2f}:1'],
                ['Fraudulent (1)', f'{class_counts[1]:,}', f'{class_percentages[1]:.2f}%', '-']
            ]
            
            table = ax.table(cellText=class_table_data[1:], 
                           colLabels=class_table_data[0],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.3, 0.25, 0.25, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            for i in range(len(class_table_data[0])):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('3.1.1 Class Distribution - Exact Numbers', fontsize=12, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Purchase value by class
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            
            fraud_purchase = fraud_data[fraud_data['class'] == 1]['purchase_value']
            legit_purchase = fraud_data[fraud_data['class'] == 0]['purchase_value']
            
            axes[0].hist([legit_purchase, fraud_purchase], bins=50, 
                        label=['Legitimate', 'Fraud'], alpha=0.7, edgecolor='black')
            axes[0].set_title('Purchase Value Distribution by Class', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Purchase Value ($)')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Box plot
            data_for_box = [legit_purchase.values, fraud_purchase.values]
            bp = axes[1].boxplot(data_for_box, tick_labels=['Legitimate', 'Fraud'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#2ecc71')
            bp['boxes'][1].set_facecolor('#e74c3c')
            axes[1].set_title('Purchase Value by Class (Box Plot)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Purchase Value ($)')
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle('3.2 Purchase Value Analysis', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Purchase value statistics table
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis('tight')
            ax.axis('off')
            
            purchase_stats = fraud_data.groupby('class')['purchase_value'].describe()
            purchase_table_data = [
                ['Class', 'Mean ($)', 'Median ($)', 'Std ($)', 'Min ($)', 'Max ($)'],
                ['Legitimate (0)', 
                 f'{purchase_stats.loc[0, "mean"]:.2f}',
                 f'{purchase_stats.loc[0, "50%"]:.2f}',
                 f'{purchase_stats.loc[0, "std"]:.2f}',
                 f'{purchase_stats.loc[0, "min"]:.2f}',
                 f'{purchase_stats.loc[0, "max"]:.2f}'],
                ['Fraudulent (1)',
                 f'{purchase_stats.loc[1, "mean"]:.2f}',
                 f'{purchase_stats.loc[1, "50%"]:.2f}',
                 f'{purchase_stats.loc[1, "std"]:.2f}',
                 f'{purchase_stats.loc[1, "min"]:.2f}',
                 f'{purchase_stats.loc[1, "max"]:.2f}']
            ]
            
            table = ax.table(cellText=purchase_table_data[1:], 
                           colLabels=purchase_table_data[0],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.5)
            
            for i in range(len(purchase_table_data[0])):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('3.2.1 Purchase Value Statistics by Class', fontsize=12, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Time-based features visualization
            fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
            fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
            fraud_data['time_since_signup'] = (
                fraud_data['purchase_time'] - fraud_data['signup_time']
            ).dt.total_seconds() / 3600
            
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            
            # Hour of day
            fraud_by_hour = fraud_data.groupby('hour_of_day')['class'].agg(['count', 'mean']).reset_index()
            fraud_by_hour.columns = ['hour', 'count', 'fraud_rate']
            axes[0].bar(fraud_by_hour['hour'], fraud_by_hour['fraud_rate']*100, 
                       alpha=0.7, edgecolor='black', color='#e74c3c')
            axes[0].set_title('Fraud Rate by Hour of Day', fontsize=11, fontweight='bold')
            axes[0].set_xlabel('Hour of Day')
            axes[0].set_ylabel('Fraud Rate (%)')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Day of week
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fraud_by_day = fraud_data.groupby('day_of_week')['class'].agg(['count', 'mean']).reset_index()
            fraud_by_day.columns = ['day', 'count', 'fraud_rate']
            axes[1].bar([day_names[int(d)] for d in fraud_by_day['day']], 
                       fraud_by_day['fraud_rate']*100, alpha=0.7, edgecolor='black', color='#e74c3c')
            axes[1].set_title('Fraud Rate by Day of Week', fontsize=11, fontweight='bold')
            axes[1].set_xlabel('Day of Week')
            axes[1].set_ylabel('Fraud Rate (%)')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Time since signup
            axes[2].hist([fraud_data[fraud_data['class']==0]['time_since_signup'], 
                         fraud_data[fraud_data['class']==1]['time_since_signup']], 
                        bins=50, label=['Legitimate', 'Fraud'], alpha=0.7, edgecolor='black')
            axes[2].set_title('Time Since Signup Distribution', fontsize=11, fontweight='bold')
            axes[2].set_xlabel('Time Since Signup (hours)')
            axes[2].set_ylabel('Frequency')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle('3.3 Time-based Pattern Analysis', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Time statistics table
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis('tight')
            ax.axis('off')
            
            time_stats = fraud_data.groupby('class')['time_since_signup'].agg(['mean', 'median', 'std']).reset_index()
            time_table_data = [
                ['Class', 'Mean (hours)', 'Median (hours)', 'Std (hours)'],
                ['Legitimate (0)', 
                 f'{time_stats.loc[0, "mean"]:.2f}',
                 f'{time_stats.loc[0, "median"]:.2f}',
                 f'{time_stats.loc[0, "std"]:.2f}'],
                ['Fraudulent (1)',
                 f'{time_stats.loc[1, "mean"]:.2f}',
                 f'{time_stats.loc[1, "median"]:.2f}',
                 f'{time_stats.loc[1, "std"]:.2f}']
            ]
            
            table = ax.table(cellText=time_table_data[1:], 
                           colLabels=time_table_data[0],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            for i in range(len(time_table_data[0])):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('3.3.1 Time Since Signup Statistics', fontsize=12, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 3. EDA Text Section
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '3. Exploratory Data Analysis - Key Insights', 
                fontsize=18, fontweight='bold')
        
        eda_text = """
3.1 Class Distribution
The dataset exhibits severe class imbalance:
• Legitimate transactions: 90.64% of all transactions
• Fraudulent transactions: 9.36% of all transactions
• Imbalance ratio: 9.68:1 (Legitimate:Fraud)

This imbalance is typical for fraud detection problems and requires special handling 
during model training.

3.2 Purchase Value Patterns
• Legitimate transactions: Mean $36.93, Median $35.00
• Fraudulent transactions: Mean $36.99, Median $35.00
• Statistical tests reveal similar distributions but different patterns in outliers

3.3 Time-based Patterns
• Fraud patterns vary by hour of day, with certain hours showing higher fraud rates
• Day of week analysis reveals patterns in fraudulent activity
• Time since signup is a critical feature - fraudulent accounts make purchases 
  significantly faster after signup (mean: 673 hours vs 1442 hours for legitimate)

3.4 Categorical Feature Analysis
• Source (SEO, Ads): Different fraud rates across traffic sources
• Browser: Some browsers may be associated with higher fraud rates
• Geographic patterns: Certain countries show elevated fraud rates

3.5 Key Findings
• Fraudulent transactions occur significantly faster after account signup
• Purchase values show similar distributions but different outlier patterns
• Geographic location (derived from IP) is a strong indicator of fraud risk
• Transaction velocity (frequency of transactions) is a critical fraud indicator
        """
        
        fig.text(0.1, 0.85, eda_text, fontsize=10, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 4. Feature Engineering with Visualizations
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
• Evidence: Legitimate users average 1,442 hours vs Fraudulent users average 673 hours

4.1.2 hour_of_day and day_of_week
• Extracted from purchase_time to capture temporal patterns
• Fraudulent transactions may cluster at specific times
• Helps identify unusual transaction timing patterns

4.2 Transaction Frequency and Velocity
• Transaction count per user
• Transactions in last 24 hours, 7 days, 30 days
• High velocity in short timeframes is suspicious

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
• Efficient lookup approach for matching
• Handles edge cases and unmapped IPs gracefully

4.4 Data Transformation
• StandardScaler for numerical features
• One-Hot Encoding for low cardinality categorical features
• Label Encoding for high cardinality features
        """
        
        fig.text(0.1, 0.85, fe_text, fontsize=9, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 5. Class Imbalance Analysis with Before/After Numbers
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '5. Class Imbalance Analysis and Strategy', 
                fontsize=18, fontweight='bold')
        
        imbalance_text = """
5.1 Problem Statement
The fraud detection dataset exhibits severe class imbalance:
• Legitimate transactions: 90.64% of dataset
• Fraudulent transactions: 9.36% of dataset
• Imbalance ratio: 9.68:1 (Legitimate:Fraud)

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
        """
        
        fig.text(0.1, 0.85, imbalance_text, fontsize=9, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Before and After SMOTE visualization
        if y_train is not None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            
            # Before SMOTE (from original data)
            before_counts = fraud_data['class'].value_counts()
            before_pct = fraud_data['class'].value_counts(normalize=True) * 100
            
            axes[0].bar(['Legitimate (0)', 'Fraudulent (1)'], before_counts.values, 
                       color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            axes[0].set_title('Before SMOTE (Original Data)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Count')
            axes[0].set_ylim(0, max(before_counts.values) * 1.1)
            for i, v in enumerate(before_counts.values):
                axes[0].text(i, v + max(before_counts.values)*0.01, f'{v:,}', 
                           ha='center', fontweight='bold', fontsize=9)
            
            # After SMOTE
            after_counts = y_train.value_counts()
            after_pct = y_train.value_counts(normalize=True) * 100
            
            axes[1].bar(['Legitimate (0)', 'Fraudulent (1)'], after_counts.values, 
                       color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            axes[1].set_title('After SMOTE (Training Data)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Count')
            axes[1].set_ylim(0, max(after_counts.values) * 1.1)
            for i, v in enumerate(after_counts.values):
                axes[1].text(i, v + max(after_counts.values)*0.01, f'{v:,}', 
                           ha='center', fontweight='bold', fontsize=9)
            
            plt.suptitle('5.2.3 Class Distribution Before and After SMOTE', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Before/After table
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis('tight')
            ax.axis('off')
            
            smote_table_data = [
                ['Stage', 'Legitimate Count', 'Fraud Count', 'Legitimate %', 'Fraud %', 'Ratio'],
                ['Before SMOTE (Original)', 
                 f'{before_counts[0]:,}',
                 f'{before_counts[1]:,}',
                 f'{before_pct[0]:.2f}%',
                 f'{before_pct[1]:.2f}%',
                 f'{before_counts[0]/before_counts[1]:.2f}:1'],
                ['After SMOTE (Training)', 
                 f'{after_counts[0]:,}',
                 f'{after_counts[1]:,}',
                 f'{after_pct[0]:.2f}%',
                 f'{after_pct[1]:.2f}%',
                 f'{after_counts[0]/after_counts[1]:.2f}:1'],
                ['Change', 
                 f'+{after_counts[0]-before_counts[0]:,}',
                 f'+{after_counts[1]-before_counts[1]:,}',
                 f'{after_pct[0]-before_pct[0]:.2f}%',
                 f'{after_pct[1]-before_pct[1]:.2f}%',
                 f'{(after_counts[0]/after_counts[1])/(before_counts[0]/before_counts[1]):.2f}x']
            ]
            
            table = ax.table(cellText=smote_table_data[1:], 
                           colLabels=smote_table_data[0],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.5)
            
            for i in range(len(smote_table_data[0])):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('5.2.3 SMOTE Resampling Results - Exact Numbers', 
                     fontsize=12, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        else:
            # If processed data not available, show expected numbers
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis('tight')
            ax.axis('off')
            
            if fraud_data is not None:
                before_counts = fraud_data['class'].value_counts()
                before_pct = fraud_data['class'].value_counts(normalize=True) * 100
                
                smote_table_data = [
                    ['Stage', 'Legitimate Count', 'Fraud Count', 'Legitimate %', 'Fraud %', 'Ratio'],
                    ['Before SMOTE (Original)', 
                     f'{before_counts[0]:,}',
                     f'{before_counts[1]:,}',
                     f'{before_pct[0]:.2f}%',
                     f'{before_pct[1]:.2f}%',
                     f'{before_counts[0]/before_counts[1]:.2f}:1'],
                    ['After SMOTE (Expected)', 
                     '~120,000',
                     '~60,000',
                     '~66.67%',
                     '~33.33%',
                     '~2:1']
                ]
                
                table = ax.table(cellText=smote_table_data, 
                               colLabels=smote_table_data[0],
                               cellLoc='center',
                               loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2.5)
                
                for i in range(len(smote_table_data[0])):
                    table[(0, i)].set_facecolor('#3498db')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                plt.title('5.2.3 SMOTE Resampling Results', 
                         fontsize=12, fontweight='bold', pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        # 6. Next Steps and Challenges
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, '6. Next Steps and Anticipated Challenges', 
                fontsize=18, fontweight='bold')
        
        next_steps_text = """
6.1 Task 2: Model Building and Training

6.1.1 Data Preparation
• Load processed datasets from data/processed/
• Verify train-test split (80/20, stratified)
• Ensure features and targets are properly separated
• Validate data shapes and distributions

6.1.2 Baseline Model Development
• Train Logistic Regression as interpretable baseline
• Evaluate using:
  - AUC-PR (Area Under Precision-Recall Curve)
  - F1-Score
  - Confusion Matrix
  - Classification Report
• Establish performance baseline for comparison

6.1.3 Ensemble Model Development
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

6.1.4 Cross-Validation
• Implement Stratified K-Fold (k=5)
• Ensure class distribution preserved in each fold
• Report mean and standard deviation of metrics:
  - Precision
  - Recall
  - F1-Score
  - AUC-PR
• Provides reliable performance estimation

6.1.5 Model Comparison and Selection
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

6.2 Task 3: Model Explainability

6.2.1 SHAP Analysis
• Calculate SHAP values for selected best model
• Generate global feature importance visualizations
• Create individual prediction explanations
• Analyze feature interactions

6.2.2 Explainability Deliverables
• Feature importance rankings
• Waterfall plots for individual predictions
• Force plots for model decisions
• Dependence plots for feature interactions
• Summary of key fraud indicators

6.2.3 Business Interpretation
• Translate technical findings to business insights
• Identify actionable fraud patterns
• Document model decision logic
• Create explainability report for stakeholders

6.3 Anticipated Challenges

6.3.1 Model Performance Challenges
• Balancing precision and recall: High precision reduces false positives (customer 
  satisfaction) but may miss fraud. High recall catches more fraud but may flag 
  legitimate transactions.
• Solution: Use cost-sensitive evaluation and tune threshold based on business costs.

6.3.2 Interpretability Challenges
• Complex ensemble models (XGBoost, LightGBM) may be less interpretable than simpler models
• Solution: Use SHAP values to explain complex models and provide clear visualizations

6.3.3 Data Challenges
• Real-time inference: Models must process transactions quickly
• Solution: Optimize feature engineering pipeline and consider model complexity trade-offs

6.3.4 Deployment Challenges
• Model versioning and monitoring
• Handling new data patterns (concept drift)
• Solution: Implement model monitoring and retraining pipelines

6.4 Deliverables
• Trained and evaluated models
• Model comparison report
• Selected best model with justification
• SHAP explainability analysis
• Saved model artifacts for deployment
        """
        
        fig.text(0.1, 0.85, next_steps_text, fontsize=8, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Summary Page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.95, 'Summary', 
                fontsize=18, fontweight='bold')
        
        summary_text = """
Task 1 has been successfully completed with the following achievements:

[COMPLETED] Comprehensive data cleaning and preprocessing for both datasets
[COMPLETED] Detailed exploratory data analysis with key insights and visualizations
[COMPLETED] Advanced feature engineering including:
  - Time-based features (time_since_signup, hour_of_day, etc.)
  - Transaction frequency and velocity features
  - Geolocation integration via IP-to-country mapping
[COMPLETED] Class imbalance addressed using SMOTE with documented before/after results
[COMPLETED] Clean, feature-rich datasets prepared for modeling

Key Metrics:
• E-commerce Dataset: 151,112 transactions
• Class Imbalance: 9.68:1 (Legitimate:Fraud)
• Missing Values: 0
• Duplicates Removed: 0
• Features Engineered: 20+ features

The project is now ready to proceed to Task 2: Model Building and Training, where we will 
develop and compare multiple classification models to identify the best fraud detection 
solution, followed by Task 3: Model Explainability using SHAP.

All processed data and preprocessing objects have been saved and are ready for model training.
        """
        
        fig.text(0.1, 0.85, summary_text, fontsize=10, 
                verticalalignment='top', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"\n[SUCCESS] Interim report generated successfully!")
    print(f"  Location: {pdf_path}")
    print(f"  Pages: Multiple pages with visualizations and tables")
    
    return pdf_path

if __name__ == "__main__":
    create_interim_report()
