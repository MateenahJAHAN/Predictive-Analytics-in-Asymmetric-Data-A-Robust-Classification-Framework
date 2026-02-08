# LoanTap Loan Default Modeling - Analysis Report

**Author:** Vidyasagar - Data Scientist  
**Dataset:** `data/raw/LoanTapData.csv`  

## 1) Problem Statement
Predict loan default (Charged Off) using borrower and credit attributes. The business
objective is to detect real defaulters early while limiting false positives that can
reduce profitable lending.

## 2) EDA Highlights
- **Shape:** 3,000 rows x 27 columns
- **Target distribution:** ~8.2% Charged Off, ~91.8% Fully Paid
- **Missing values:** Small gaps in `emp_title`, `emp_length`, `mort_acc`, and `revol_util`
- **High correlation:** `loan_amnt` and `installment` show a strong positive correlation
- **Default signals:** Higher `int_rate`, `dti`, `revol_util`, longer `term` (60 months),
  and negative credit history features are associated with higher default risk

## 3) Feature Engineering
Binary flags were created for:
- `pub_rec_flag`
- `mort_acc_flag`
- `pub_rec_bankruptcies_flag`

## 4) Data Preprocessing
- Missing values: median for numeric, mode for categorical
- Outliers: capped at 1st and 99th percentile (training data only)
- Scaling: StandardScaler for numeric variables
- Encoding: OneHotEncoder for categorical variables

## 5) Model Summary
Logistic Regression with class weighting for imbalance. Top risk-increasing factors:
`int_rate`, `dti`, `revol_util`, `term_60`, and negative credit history flags. Protective
signals include higher `annual_inc`, `home_ownership` = OWN/MORTGAGE, and lower grade risk.

## 6) Evaluation
- Classification report + confusion matrix
- ROC AUC curve
- Precision-Recall curve
- Threshold analysis for precision/recall tradeoffs

## 7) Actionable Recommendations
- Tighten underwriting for 60-month terms and high-risk grades
- Apply stricter DTI and revolving utilization caps for marginal applicants
- Introduce manual review for any public record or bankruptcy signal
- Use probability thresholds aligned to NPA tolerance rather than a fixed 0.5 cut
