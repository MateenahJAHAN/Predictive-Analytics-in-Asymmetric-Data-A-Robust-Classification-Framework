"""
Data Preprocessing Module for LoanTap Loan Default Prediction
Author: Vidyasagar — Data Scientist

This module handles data loading, cleaning, feature engineering,
and preparation for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load the LoanTap dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    df = df.copy()

    # Drop high-cardinality / redundant columns
    cols_to_drop = ['emp_title', 'title']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Fill emp_length with mode
    if 'emp_length' in df.columns:
        mode_val = df['emp_length'].mode()[0]
        df['emp_length'] = df['emp_length'].fillna(mode_val)

    # Fill revol_util with median
    if 'revol_util' in df.columns:
        df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())

    # Fill mort_acc with median
    if 'mort_acc' in df.columns:
        df['mort_acc'] = df['mort_acc'].fillna(df['mort_acc'].median())

    # Fill pub_rec_bankruptcies with 0
    if 'pub_rec_bankruptcies' in df.columns:
        df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)

    return df


def treat_outliers(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Cap outliers using IQR method."""
    df = df.copy()

    if columns is None:
        columns = ['annual_inc', 'revol_bal', 'open_acc', 'total_acc', 'dti']

    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features."""
    df = df.copy()

    # Flag variables
    if 'pub_rec' in df.columns:
        df['pub_rec_flag'] = (df['pub_rec'] > 0).astype(int)

    if 'mort_acc' in df.columns:
        df['mort_acc_flag'] = (df['mort_acc'] > 0).astype(int)

    if 'pub_rec_bankruptcies' in df.columns:
        df['pub_rec_bankruptcies_flag'] = (df['pub_rec_bankruptcies'] > 0).astype(int)

    # Extract state from address
    if 'address' in df.columns:
        df['state'] = df['address'].apply(
            lambda x: str(x).split()[-2] if pd.notna(x) else 'Unknown'
        )

    # Convert term to numeric
    if 'term' in df.columns:
        df['term_numeric'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)

    # Convert emp_length to numeric
    if 'emp_length' in df.columns:
        emp_length_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years': 8, '9 years': 9, '10+ years': 10
        }
        df['emp_length_numeric'] = df['emp_length'].astype(str).map(emp_length_map)
        df['emp_length_numeric'] = df['emp_length_numeric'].fillna(5.0)

    # Encode target variable
    if 'loan_status' in df.columns:
        df['target'] = (df['loan_status'] == 'Charged Off').astype(int)

    # Log transformations
    if 'annual_inc' in df.columns:
        df['log_annual_inc'] = np.log1p(df['annual_inc'])

    if 'revol_bal' in df.columns:
        df['log_revol_bal'] = np.log1p(df['revol_bal'])

    return df


def prepare_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataset for model training."""
    df = df.copy()

    # Drop columns not needed for modeling
    drop_cols = [
        'loan_status', 'address', 'issue_d', 'earliest_cr_line', 'term',
        'emp_length', 'sub_grade', 'annual_inc', 'revol_bal', 'pub_rec',
        'mort_acc', 'pub_rec_bankruptcies'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # One-hot encoding
    cat_features = df.select_dtypes(include=['category', 'object']).columns.tolist()
    if cat_features:
        df = pd.get_dummies(df, columns=cat_features, drop_first=True, dtype=int)

    # Fill remaining NaN
    df = df.fillna(0)

    return df


def preprocess_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    filepath : str
        Path to the LoanTapData.csv file
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test
    """
    # Load data
    df = load_data(filepath)

    # Handle missing values
    df = handle_missing_values(df)

    # Treat outliers
    df = treat_outliers(df)

    # Engineer features
    df = engineer_features(df)

    # Prepare for modeling
    df = prepare_for_modeling(df)

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Testing set: {X_test_scaled.shape[0]} samples")
    print(f"Default rate: {y.mean()*100:.2f}%")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data('../data/LoanTapData.csv')
    print(f"\nPreprocessing complete!")
    print(f"Feature dimensions: {X_train.shape[1]}")
