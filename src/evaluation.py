"""
Model Evaluation Module for LoanTap Loan Default Prediction
Author: Vidyasagar — Data Scientist

This module handles model evaluation including classification reports,
ROC AUC curves, precision-recall curves, and threshold analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)


def evaluate_model(model, X_test, y_test, save_dir: str = None):
    """
    Comprehensive model evaluation.

    Parameters
    ----------
    model : trained sklearn model
        Model with predict and predict_proba methods
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    save_dir : str, optional
        Directory to save figures
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification Report
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred,
                                target_names=['Fully Paid (0)', 'Charged Off (1)']))

    # Key Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Average Precision': average_precision_score(y_test, y_pred_proba)
    }

    print("\nKey Metrics:")
    for name, value in metrics.items():
        print(f"  {name:>20}: {value:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    print(f"  Specificity:     {tn/(tn+fp):.4f}")
    print(f"  Sensitivity:     {tp/(tp+fn):.4f}")

    if save_dir:
        _plot_confusion_matrix(cm, save_dir)
        _plot_roc_curve(y_test, y_pred_proba, save_dir)
        _plot_precision_recall(y_test, y_pred_proba, save_dir)
        print(f"\nFigures saved to {save_dir}")

    return metrics


def threshold_analysis(model, X_test, y_test):
    """
    Analyze model performance across different thresholds.

    Returns
    -------
    pd.DataFrame
        Performance metrics at different thresholds
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results = []
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred_t = (y_pred_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_t, zero_division=0)
        rec = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        acc = accuracy_score(y_test, y_pred_t)

        cm_t = confusion_matrix(y_test, y_pred_t)
        if cm_t.shape == (2, 2):
            fp_count = cm_t[0, 1]
            fn_count = cm_t[1, 0]
        else:
            fp_count = 0
            fn_count = (y_test == 1).sum()

        results.append({
            'Threshold': round(threshold, 2),
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'FP': fp_count,
            'FN': fn_count
        })

    return pd.DataFrame(results)


def _plot_confusion_matrix(cm, save_dir):
    """Plot and save confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Fully Paid', 'Charged Off'],
                yticklabels=['Fully Paid', 'Charged Off'])
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
                xticklabels=['Fully Paid', 'Charged Off'],
                yticklabels=['Fully Paid', 'Charged Off'])
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_roc_curve(y_test, y_pred_proba, save_dir):
    """Plot and save ROC AUC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#e74c3c', linewidth=2.5,
            label=f'Logistic Regression (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC AUC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_auc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_precision_recall(y_test, y_pred_proba, save_dir):
    """Plot and save precision-recall curve."""
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color='#3498db', linewidth=2.5,
            label=f'Logistic Regression (AP = {avg_precision:.4f})')
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color='#3498db')
    ax.axhline(y=y_test.mean(), color='red', linestyle='--',
               label=f'Baseline ({y_test.mean():.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/precision_recall_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from model_training import train_model

    X_train, X_test, y_train, y_test = preprocess_data('../data/LoanTapData.csv')
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, save_dir='../reports/figures')
    threshold_df = threshold_analysis(model, X_test, y_test)
    print("\nThreshold Analysis:")
    print(threshold_df.to_string(index=False))
