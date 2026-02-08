# Loan Default Prediction Using Logistic Regression
**Author:** Vidyasagar - Data Scientist  
**Organization:** LoanTap Analytics Lab  
**Date:** Feb 8, 2026

## Abstract
This study evaluates a structured credit dataset to predict loan default using logistic
regression. The workflow includes exploratory data analysis, feature engineering,
handling missing values and outliers, scaling, and rigorous evaluation using ROC AUC
and Precision-Recall curves. The model demonstrates clear risk drivers and provides
actionable underwriting guidance to balance revenue and risk.

## 1. Introduction
Loan default prediction is a critical capability for lenders operating in asymmetric
data environments, where default events are fewer yet financially significant. This
paper details a robust, interpretable logistic regression approach and highlights
the precision-recall tradeoff required to align decisions with risk appetite.

## 2. Data Description
The dataset contains borrower attributes (income, employment, home ownership),
loan terms (amount, interest rate, term length), and credit profile indicators
(DTI, revolving utilization, open accounts, public records). The target variable
`loan_status` identifies whether a loan was **Fully Paid** or **Charged Off**.

## 3. Methodology
### 3.1 Exploratory Data Analysis
Univariate distributions and bivariate relationships were examined through histograms,
count plots, box plots, and correlation heatmaps. This exposed key risk signals and
interaction patterns, especially among credit utilization and affordability metrics.

### 3.2 Feature Engineering
Binary flags were created for:
- `pub_rec`
- `mort_acc`
- `pub_rec_bankruptcies`

### 3.3 Preprocessing
Missing values were imputed (median for numeric, mode for categorical). Outliers were
capped at the 1st and 99th percentiles using training data only. Numerical features
were standardized, and categorical features were one-hot encoded.

### 3.4 Modeling
Logistic Regression was selected for interpretability and stability. Class weighting
was used to mitigate target imbalance.

## 4. Results & Evaluation
Model performance was assessed via:
- Confusion Matrix & Classification Report
- ROC AUC Curve
- Precision-Recall Curve

The Precision-Recall curve provides a more reliable view under class imbalance and
supports the threshold selection process.

## 5. Business Interpretation
High interest rates, elevated DTI, high revolving utilization, longer loan terms,
and negative credit history features are the dominant default risk drivers. Borrowers
with strong income and favorable home ownership categories show lower risk.

## 6. Recommendations
- Apply stricter underwriting thresholds for risk-intensive terms and grades
- Introduce early-warning flags for borrowers with public records or bankruptcies
- Use probability thresholds tuned to NPA tolerance rather than a fixed 0.5 cutoff

## 7. Conclusion
Logistic regression provides an interpretable and scalable baseline for loan default
prediction. Its coefficients highlight clear policy levers and allow decision-makers
to balance detection of defaulters with profitable loan growth.
