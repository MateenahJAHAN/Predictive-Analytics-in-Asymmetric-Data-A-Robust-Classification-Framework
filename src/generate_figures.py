import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(CURRENT_DIR)

from loantap_modeling_utils import apply_outlier_caps, build_preprocessor, compute_outlier_caps


def main() -> None:
    sns.set(style="whitegrid")

    df = pd.read_csv("data/raw/LoanTapData.csv")
    df["pub_rec_flag"] = (df["pub_rec"] > 0).astype(int)
    df["mort_acc_flag"] = (df["mort_acc"] > 0).astype(int)
    df["pub_rec_bankruptcies_flag"] = (df["pub_rec_bankruptcies"] > 0).astype(int)

    X = df.drop(columns=["loan_status"])
    y = (df["loan_status"].astype(str) == "Charged Off").astype(int)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.columns.difference(numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    caps = compute_outlier_caps(X_train, numeric_features)
    X_train = apply_outlier_caps(X_train, caps)
    X_test = apply_outlier_caps(X_test, caps)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    metrics = {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "accuracy": report["accuracy"],
        "precision_default": report["1"]["precision"],
        "recall_default": report["1"]["recall"],
        "f1_default": report["1"]["f1-score"],
        "support_default": report["1"]["support"],
        "default_rate": float(y.mean()),
        "confusion_matrix": conf_matrix.tolist(),
    }

    os.makedirs("reports/figures", exist_ok=True)
    with open("reports/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(5, 4))
    sns.countplot(x=df["loan_status"])
    plt.title("Target Distribution: Loan Status")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("reports/figures/target_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 7))
    corr = df[numeric_features].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP={avg_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/precision_recall_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=160)
    plt.close()

    feature_names = clf.named_steps["preprocess"].get_feature_names_out()
    coefs = clf.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top_coef = coef_df.sort_values("abs_coef", ascending=False).head(12)

    plt.figure(figsize=(7, 4))
    sns.barplot(x="coef", y="feature", data=top_coef, hue="feature", palette="viridis", legend=False)
    plt.title("Top Coefficients (Absolute Impact)")
    plt.tight_layout()
    plt.savefig("reports/figures/top_coefficients.png", dpi=160)
    plt.close()

    print("Saved figures to reports/figures and metrics to reports/model_metrics.json")


if __name__ == "__main__":
    main()
