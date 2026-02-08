"""
Model Training Module for LoanTap Loan Default Prediction
Author: Vidyasagar — Data Scientist

This module handles Logistic Regression model training,
cross-validation, and coefficient analysis.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import statsmodels.api as sm


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    max_iter: int = 5000,
    class_weight: str = 'balanced',
    random_state: int = 42
) -> LogisticRegression:
    """
    Train a Logistic Regression model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Scaled training features
    y_train : pd.Series
        Training target variable
    C : float
        Regularization parameter (inverse of regularization strength)
    max_iter : int
        Maximum iterations for solver convergence
    class_weight : str
        Class weight strategy ('balanced' for imbalanced data)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    LogisticRegression
        Trained model
    """
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        solver='lbfgs',
        random_state=random_state,
        C=C
    )

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

    return model


def cross_validate(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """
    Perform stratified cross-validation.

    Returns
    -------
    dict
        Cross-validation scores for accuracy, ROC AUC, and F1
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = {
        'accuracy': cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'),
        'roc_auc': cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc'),
        'f1': cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    }

    print(f"\n{n_splits}-Fold Cross-Validation Results:")
    for metric, scores in cv_results.items():
        print(f"  {metric:>10}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return cv_results


def get_coefficients(
    model: LogisticRegression,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract and format model coefficients.

    Returns
    -------
    pd.DataFrame
        Sorted coefficient table with feature names, coefficients,
        absolute coefficients, and odds ratios
    """
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0]),
        'Odds_Ratio': np.exp(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)

    return coef_df


def train_statsmodels(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 5000
):
    """
    Train logistic regression using statsmodels for statistical inference.

    Returns
    -------
    statsmodels result object with p-values, confidence intervals, etc.
    """
    X_train_sm = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_train_sm)
    result = logit_model.fit(maxiter=max_iter, disp=0)

    print("Statsmodels Logistic Regression fitted successfully.")
    print(f"Pseudo R-squared: {result.prsquared:.4f}")
    print(f"Log-Likelihood: {result.llf:.2f}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")

    return result


if __name__ == "__main__":
    from data_preprocessing import preprocess_data

    X_train, X_test, y_train, y_test = preprocess_data('../data/LoanTapData.csv')

    # Train model
    model = train_model(X_train, y_train)

    # Cross-validate
    cv_results = cross_validate(model, X_train, y_train)

    # Get coefficients
    coef_df = get_coefficients(model, X_train.columns.tolist())
    print(f"\nTop 10 Features:")
    print(coef_df.head(10).to_string(index=False))
