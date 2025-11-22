"""
Data Preprocessing Module for Credit Risk Assessment

This module provides comprehensive data preprocessing functionality including:
- Missing value handling
- Feature encoding
- Feature scaling
- Feature engineering
- Data splitting

Author: Lazy Loaders Team
Date: October 31, 2025
Project: Credit Risk Assessment with Uncertainty-Aware Decision Making
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class CreditDataPreprocessor:
    """
    Handles all data preprocessing steps for credit risk assessment.
    
    This class provides methods for cleaning, transforming, and preparing
    credit data for machine learning models.
    
    Attributes:
        numerical_imputer (SimpleImputer): Imputer for numerical features
        categorical_imputer (SimpleImputer): Imputer for categorical features
        scaler (StandardScaler): Scaler for numerical features
        label_encoders (dict): Dictionary of label encoders for categorical features
        feature_names (list): List of feature names after preprocessing
    """
    
    def __init__(self):
        """Initialize the preprocessor with default transformers."""
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.numerical_cols = []
        self.categorical_cols = []
        
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values
                - 'auto': Use median for numerical, mode for categorical
                - 'drop': Drop rows with missing values
                - 'indicator': Create missing value indicators
        
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_processed = df.copy()
        
        # Identify numerical and categorical columns
        self.numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID and target-like columns
        exclude_cols = ['id', 'Default', 'title', 'desc']
        self.numerical_cols = [col for col in self.numerical_cols if col not in exclude_cols]
        self.categorical_cols = [col for col in self.categorical_cols if col not in exclude_cols]
        
        if strategy == 'drop':
            df_processed = df_processed.dropna()
            print(f"Dropped rows with missing values. New shape: {df_processed.shape}")
            
        elif strategy == 'indicator':
            # Create missing indicators for features with >5% missing
            for col in df_processed.columns:
                missing_pct = df_processed[col].isnull().sum() / len(df_processed)
                if missing_pct > 0.05:
                    df_processed[f'{col}_missing'] = df_processed[col].isnull().astype(int)
            
            # Then impute
            if self.numerical_cols:
                df_processed[self.numerical_cols] = self.numerical_imputer.fit_transform(
                    df_processed[self.numerical_cols]
                )
            if self.categorical_cols:
                df_processed[self.categorical_cols] = self.categorical_imputer.fit_transform(
                    df_processed[self.categorical_cols]
                )
                
        else:  # auto strategy
            if self.numerical_cols:
                df_processed[self.numerical_cols] = self.numerical_imputer.fit_transform(
                    df_processed[self.numerical_cols]
                )
            if self.categorical_cols:
                df_processed[self.categorical_cols] = self.categorical_imputer.fit_transform(
                    df_processed[self.categorical_cols]
                )
        
        print(f"Missing values handled using '{strategy}' strategy")
        return df_processed
    
    def encode_categorical(self, df, method='label'):
        """
        Encode categorical variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Encoding method
                - 'label': Label encoding (for tree-based models)
                - 'onehot': One-hot encoding (for linear models)
        
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_processed = df.copy()
        
        if method == 'label':
            for col in self.categorical_cols:
                if col in df_processed.columns:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
            print(f"Label encoded {len(self.categorical_cols)} categorical features")
            
        elif method == 'onehot':
            df_processed = pd.get_dummies(df_processed, columns=self.categorical_cols, 
                                         drop_first=True, dtype=int)
            print(f"One-hot encoded {len(self.categorical_cols)} categorical features")
        
        return df_processed
    
    def scale_features(self, df, columns=None):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to scale. If None, scales all numerical columns
        
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = self.numerical_cols
        
        if columns:
            df_processed[columns] = self.scaler.fit_transform(df_processed[columns])
            print(f"Scaled {len(columns)} numerical features")
        
        return df_processed
    
    def create_interaction_features(self, df):
        """
        Create interaction and engineered features.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with additional engineered features
        """
        df_processed = df.copy()
        
        # Loan to income ratio
        if 'loan_amnt' in df_processed.columns and 'revenue' in df_processed.columns:
            df_processed['loan_to_income_ratio'] = df_processed['loan_amnt'] / (df_processed['revenue'] + 1)
        
        # FICO score bins
        if 'fico_n' in df_processed.columns:
            df_processed['fico_category'] = pd.cut(
                df_processed['fico_n'],
                bins=[0, 580, 670, 740, 850],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
        
        # DTI risk indicator
        if 'dti_n' in df_processed.columns:
            df_processed['high_dti'] = (df_processed['dti_n'] > 36).astype(int)
        
        # Employment length as numeric
        if 'emp_length' in df_processed.columns:
            emp_length_map = {
                '< 1 year': 0,
                '1 year': 1,
                '2 years': 2,
                '3 years': 3,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '7 years': 7,
                '8 years': 8,
                '9 years': 9,
                '10+ years': 10
            }
            df_processed['emp_length_numeric'] = df_processed['emp_length'].map(
                emp_length_map
            ).fillna(0)
        
        # Credit quality score (composite)
        if 'fico_n' in df_processed.columns and 'dti_n' in df_processed.columns:
            # Normalize and combine
            fico_norm = (df_processed['fico_n'] - 300) / (850 - 300)
            dti_norm = 1 - (df_processed['dti_n'] / 100)  # Invert so higher is better
            df_processed['credit_quality_score'] = (fico_norm + dti_norm) / 2
        
        # Don't add title and desc columns during transform - they should only be in training data
        # Remove them if they were added
        if 'title' in df_processed.columns and 'title' not in df.columns:
            df_processed = df_processed.drop(columns=['title'])
        if 'desc' in df_processed.columns and 'desc' not in df.columns:
            df_processed = df_processed.drop(columns=['desc'])
        
        print(f"Created interaction features. New shape: {df_processed.shape}")
        return df_processed
    
    def split_data(self, df, target_col='Default', test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets with stratification.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of target column
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set (from remaining data)
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {X_train.shape[0]} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Validation set: {X_val.shape[0]} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test set: {X_test.shape[0]} samples ({len(X_test)/len(df)*100:.1f}%)")
        print(f"\n  Class distribution:")
        print(f"    Train - Default: {y_train.mean()*100:.2f}%")
        print(f"    Val   - Default: {y_val.mean()*100:.2f}%")
        print(f"    Test  - Default: {y_test.mean()*100:.2f}%")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform(self, df, target_col='Default', encoding='label', 
                      scaling=True, create_features=True):
        """
        Complete preprocessing pipeline: fit and transform.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of target column
            encoding (str): Encoding method for categorical variables
            scaling (bool): Whether to scale numerical features
            create_features (bool): Whether to create engineered features
        
        Returns:
            pd.DataFrame: Fully preprocessed dataframe
        """
        print("="*60)
        print("Starting Preprocessing Pipeline")
        print("="*60)
        
        df_processed = df.copy()
        
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df_processed, strategy='auto')
        
        # Step 2: Create engineered features
        if create_features:
            df_processed = self.create_interaction_features(df_processed)
        
        # Step 3: Encode categorical variables
        df_processed = self.encode_categorical(df_processed, method=encoding)
        
        # Step 4: Scale numerical features
        if scaling:
            df_processed = self.scale_features(df_processed)
        
        # Store feature names
        self.feature_names = [col for col in df_processed.columns if col != target_col]
        
        print("="*60)
        print(f"Preprocessing Complete! Final shape: {df_processed.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print("="*60)
        
        return df_processed
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        df_processed = df.copy()
        
        # Add missing columns with default values if they don't exist
        if 'id' not in df_processed.columns:
            df_processed['id'] = 0
        if 'issue_d' not in df_processed.columns:
            df_processed['issue_d'] = 'Unknown'
        
        # Add title and desc as they're expected by the model during training, but we'll remove them later
        if 'title' not in df_processed.columns:
            df_processed['title'] = 'Unknown'
        if 'desc' not in df_processed.columns:
            df_processed['desc'] = 'Unknown'
        
        # Create engineered features (same as in fit_transform)
        df_processed = self.create_interaction_features(df_processed)
        
        # Apply imputation
        if self.numerical_cols:
            # Only transform columns that exist
            existing_num_cols = [col for col in self.numerical_cols if col in df_processed.columns]
            if existing_num_cols:
                df_processed[existing_num_cols] = self.numerical_imputer.transform(
                    df_processed[existing_num_cols]
                )
        
        if self.categorical_cols:
            # Apply label encoding
            for col in self.categorical_cols:
                if col in df_processed.columns and col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unknown categories
                    df_processed[col] = df_processed[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        # Apply scaling
        if self.numerical_cols:
            existing_num_cols = [col for col in self.numerical_cols if col in df_processed.columns]
            if existing_num_cols:
                df_processed[existing_num_cols] = self.scaler.transform(
                    df_processed[existing_num_cols]
                )
        
        # Ensure all expected features are present in the correct order
        # But exclude 'title' and 'desc' as they're not used by the model
        features_to_use = [f for f in self.feature_names if f not in ['title', 'desc']]
        
        for feature in features_to_use:
            if feature not in df_processed.columns:
                df_processed[feature] = 0  # Add missing features with default value
        
        # Return only the features in the correct order (excluding title and desc)
        return df_processed[features_to_use]


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir='../data/splits'):
    """
    Save train/val/test splits to CSV files.
    
    Args:
        X_train, X_val, X_test: Feature dataframes
        y_train, y_val, y_test: Target series
        output_dir (str): Directory to save splits
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_val.to_csv(f'{output_dir}/X_val.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    
    # Save targets
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_val.to_csv(f'{output_dir}/y_val.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print(f"Splits saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("Credit Data Preprocessor Module")
    print("="*60)
    print("This module provides preprocessing functionality for credit risk data.")
    print("\nExample usage:")
    print("""
    from src.data_preprocessing import CreditDataPreprocessor
    
    # Initialize preprocessor
    preprocessor = CreditDataPreprocessor()
    
    # Load data
    df = pd.read_csv('data/raw/LC_loans_granting_model_dataset.csv')
    
    # Preprocess
    df_processed = preprocessor.fit_transform(df, target_col='Default')
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df_processed, target_col='Default'
    )
    """)
