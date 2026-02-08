# Predictive Analytics in Asymmetric Data: A Robust Classification Framework for Loan Default Prediction

## A Research Paper on Credit Risk Assessment Using Logistic Regression

---

**Author:** Vidyasagar, Data Scientist  
**Affiliation:** Independent Researcher — Applied Machine Learning & FinTech Analytics  
**Date:** February 2026  
**Keywords:** Logistic Regression, Credit Risk, Loan Default Prediction, Classification, Precision-Recall Tradeoff, Non-Performing Assets, Feature Engineering, Financial Machine Learning

---

## Abstract

This paper presents a comprehensive study on predicting loan defaults using Logistic Regression applied to the LoanTap lending dataset. We analyze 10,000 loan records with 27 features to build a robust classification framework that addresses the inherent class imbalance in credit risk data. Our methodology encompasses extensive exploratory data analysis, systematic feature engineering including flag variable creation and log transformations, and thorough model evaluation using ROC AUC, Precision-Recall curves, and threshold optimization techniques. The model demonstrates strong discriminative ability with stable cross-validated performance. We provide a detailed analysis of the precision-recall tradeoff from a banking perspective, proposing a tiered threshold strategy that balances between Non-Performing Asset (NPA) minimization and revenue optimization. Our findings identify interest rate, loan grade, loan term, and debt-to-income ratio as the strongest predictors of default, offering actionable insights for credit risk management in the digital lending ecosystem.

---

## 1. Introduction

### 1.1 Background

The digital lending industry has experienced unprecedented growth in the last decade, with platforms like LoanTap revolutionizing access to credit for millennials and underserved populations. However, this rapid expansion has brought significant challenges in credit risk assessment. The ability to accurately predict which borrowers will default on their loans is paramount to the sustainability and profitability of lending institutions.

Non-Performing Assets (NPAs) represent one of the most critical challenges facing the financial services industry globally. When a borrower defaults on a loan, the lender faces not only the loss of principal and interest but also the operational costs of collections, legal proceedings, and portfolio management. According to industry estimates, the cost of a loan default can be 5-10 times higher than the opportunity cost of rejecting a potentially good borrower.

### 1.2 Problem Statement

This research addresses the binary classification problem of predicting whether a borrower will:
- **Fully repay** their loan (Class 0)
- **Default** on their loan, resulting in a "Charged Off" status (Class 1)

The challenge is compounded by:
1. **Class imbalance** — approximately 80% of loans are fully paid, creating an asymmetric classification problem
2. **High dimensionality** — 27 features spanning numerical, categorical, and temporal data types
3. **Business constraints** — the cost matrix is asymmetric (false negatives are significantly more costly than false positives)
4. **Interpretability requirements** — regulatory and business stakeholders require transparent, explainable models

### 1.3 Objectives

1. Develop a Logistic Regression model with strong predictive performance on the LoanTap dataset
2. Conduct comprehensive exploratory data analysis to understand the data structure and feature relationships
3. Implement systematic feature engineering techniques including flag variables, transformations, and encodings
4. Evaluate the model using multiple metrics appropriate for imbalanced classification
5. Analyze the precision-recall tradeoff from a banking perspective
6. Provide actionable recommendations for credit risk management

### 1.4 Significance

While more complex models (Random Forests, Gradient Boosting, Neural Networks) often achieve higher raw accuracy, Logistic Regression remains the gold standard in regulated financial services due to its:
- **Interpretability** — coefficients directly represent log-odds ratios
- **Regulatory compliance** — transparent decision-making process
- **Statistical inference** — p-values, confidence intervals, and hypothesis testing
- **Robustness** — less prone to overfitting with proper regularization
- **Computational efficiency** — fast training and inference for real-time scoring

---

## 2. Literature Review

### 2.1 Credit Risk Modeling

Credit risk modeling has evolved from expert-based judgment systems to sophisticated machine learning approaches. The foundational work by Altman (1968) with the Z-score model established quantitative credit assessment. Logistic regression was introduced to credit scoring by Wiginton (1980) and has since become the industry standard.

### 2.2 Logistic Regression in Financial Applications

Logistic Regression models the probability of default as a function of borrower characteristics:

**P(default) = 1 / (1 + exp(-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)))**

Where:
- β₀ is the intercept
- βᵢ are the feature coefficients
- Xᵢ are the feature values

The odds ratio exp(βᵢ) provides a multiplicative interpretation: for each unit increase in Xᵢ, the odds of default multiply by exp(βᵢ).

### 2.3 Class Imbalance in Credit Data

Credit default datasets are inherently imbalanced, with defaults typically representing 5-20% of observations. Strategies to address this include:
- **Class weight balancing** — adjusting the loss function to penalize minority class errors more heavily
- **Resampling techniques** — SMOTE, random oversampling/undersampling
- **Threshold optimization** — moving the classification boundary from the default 0.5
- **Cost-sensitive learning** — incorporating asymmetric misclassification costs

### 2.4 Feature Engineering for Credit Risk

Effective feature engineering is critical for credit risk models. Common techniques include:
- **Flag variables** — binary indicators for presence/absence of risk factors
- **Ratio features** — combining raw features into interpretable ratios
- **Log transformations** — normalizing skewed financial distributions
- **Binning** — converting continuous variables into categorical risk bands

---

## 3. Dataset Description

### 3.1 Data Source

The dataset originates from LoanTap's lending platform, comprising 10,000 individual loan records with 27 features spanning borrower demographics, loan characteristics, credit history, and geographic information.

### 3.2 Feature Categories

**Loan Characteristics:**
- `loan_amnt` (continuous): Loan amount ranging from $1,000 to $40,000
- `term` (categorical): Repayment period — 36 or 60 months
- `int_rate` (continuous): Interest rate — 5% to 30%
- `installment` (continuous): Monthly payment amount
- `grade` / `sub_grade` (ordinal): Risk grade assigned by LoanTap (A-G)

**Borrower Demographics:**
- `emp_title` (categorical): Employment title — high cardinality
- `emp_length` (ordinal): Employment duration — 0 to 10+ years
- `home_ownership` (categorical): RENT, MORTGAGE, OWN, OTHER
- `annual_inc` (continuous): Self-reported annual income
- `dti` (continuous): Debt-to-income ratio

**Credit History:**
- `open_acc` (discrete): Number of open credit lines
- `pub_rec` (discrete): Number of derogatory public records
- `revol_bal` (continuous): Total revolving credit balance
- `revol_util` (continuous): Credit utilization rate
- `total_acc` (discrete): Total credit lines
- `mort_acc` (discrete): Number of mortgage accounts
- `pub_rec_bankruptcies` (discrete): Public record bankruptcies
- `earliest_cr_line` (temporal): Earliest credit line date

**Target Variable:**
- `loan_status` (binary): "Fully Paid" or "Charged Off"

### 3.3 Data Quality Assessment

| Issue | Columns Affected | Resolution |
|-------|-----------------|------------|
| Missing values | pub_rec_bankruptcies (~3.9%), mort_acc (~3%), title (~2%), revol_util (~1.5%) | Median/mode imputation, column dropping |
| High cardinality | emp_title (35+ unique values) | Dropped (limited predictive power) |
| Skewed distributions | annual_inc, revol_bal | Log transformation |
| Outliers | annual_inc, revol_bal, open_acc, total_acc, dti | IQR capping |
| Redundant features | title (redundant with purpose) | Dropped |

---

## 4. Methodology

### 4.1 Exploratory Data Analysis

#### 4.1.1 Univariate Analysis
We conducted thorough univariate analysis of all features:
- **Continuous variables**: Histograms with KDE overlays, mean/median markers
- **Categorical variables**: Count plots and bar charts
- **Target variable**: Distribution analysis revealing ~80% Fully Paid, ~20% Charged Off

#### 4.1.2 Bivariate Analysis
Key bivariate relationships examined:
- **Grade vs Default Rate**: Monotonically increasing default rate from Grade A to Grade G
- **Interest Rate vs Default**: Significantly higher interest rates among defaulters
- **Term vs Default**: 60-month loans showing higher default rates
- **Home Ownership vs Default**: Moderate influence on default rates
- **Purpose vs Default**: Small business loans showing elevated default rates

#### 4.1.3 Correlation Analysis
- **Strong positive correlation** between loan_amnt and installment (r > 0.9)
- **Moderate correlations** between open_acc and total_acc
- **Weak correlations** between most features and the target variable individually

### 4.2 Data Preprocessing

#### 4.2.1 Missing Value Treatment
| Column | Strategy | Justification |
|--------|----------|---------------|
| emp_title | Drop | High cardinality, limited predictive value |
| title | Drop | Redundant with purpose |
| emp_length | Mode imputation | Ordinal nature preserved |
| revol_util | Median imputation | Continuous, minimal missingness |
| mort_acc | Median imputation | Right-skewed distribution |
| pub_rec_bankruptcies | Fill with 0 | Zero is the mode (82%) |

#### 4.2.2 Outlier Treatment
We employed the IQR (Interquartile Range) capping method:
- Lower bound: Q1 - 1.5 * IQR
- Upper bound: Q3 + 1.5 * IQR
- Outliers capped (not removed) to preserve sample size

Applied to: annual_inc, revol_bal, open_acc, total_acc, dti

#### 4.2.3 Feature Engineering

**Flag Variables (Binary Indicators):**
1. `pub_rec_flag`: 1 if pub_rec > 0, else 0
2. `mort_acc_flag`: 1 if mort_acc > 0, else 0
3. `pub_rec_bankruptcies_flag`: 1 if pub_rec_bankruptcies > 0, else 0

**Numeric Conversions:**
- `term_numeric`: Extracted numeric months from term string
- `emp_length_numeric`: Mapped ordinal employment categories to integers (0-10)

**Transformations:**
- `log_annual_inc`: log(1 + annual_inc) — normalizing right-skewed income
- `log_revol_bal`: log(1 + revol_bal) — normalizing right-skewed balance

**Derived Features:**
- `state`: Extracted from address field for geographic analysis

**Encoding:**
- One-hot encoding for categorical variables with drop_first=True to avoid multicollinearity

### 4.3 Model Building

#### 4.3.1 Data Splitting
- **Train set**: 80% with stratified sampling to maintain class proportions
- **Test set**: 20% held out for final evaluation
- **Random state**: 42 for reproducibility

#### 4.3.2 Feature Scaling
StandardScaler applied to normalize features to zero mean and unit variance:

**X_scaled = (X - μ) / σ**

This ensures that features with larger scales don't dominate the logistic regression coefficients.

#### 4.3.3 Model Specification

```
Logistic Regression Configuration:
- solver: 'lbfgs' (L-BFGS quasi-Newton method)
- max_iter: 5000
- class_weight: 'balanced' (inversely proportional to class frequencies)
- C: 1.0 (regularization parameter)
- random_state: 42
```

The `class_weight='balanced'` parameter adjusts weights inversely proportional to class frequencies:

**wⱼ = n_samples / (n_classes * n_samplesⱼ)**

This effectively upweights the minority class (defaults) during training, addressing the class imbalance without data-level resampling.

#### 4.3.4 Statsmodels Implementation
A parallel implementation using statsmodels was performed to obtain:
- P-values for coefficient significance testing
- Confidence intervals for each parameter
- Log-likelihood and pseudo-R-squared statistics
- Wald test statistics

---

## 5. Results

### 5.1 Model Performance

The model demonstrates consistent performance between training and testing sets, indicating good generalization without significant overfitting.

### 5.2 Cross-Validation

5-fold stratified cross-validation was performed to assess model stability:
- **Accuracy**: Consistent across folds with low standard deviation
- **ROC AUC**: Strong discriminative ability maintained across folds
- **F1 Score**: Balanced performance across folds

The low variance across folds confirms model robustness.

### 5.3 Feature Importance

The model coefficients reveal the relative importance of features in predicting default:

**Features Increasing Default Risk (Positive Coefficients):**
- Interest rate — strongest positive predictor
- Lower grade indicators
- Longer loan term
- Higher DTI ratio
- Public record indicators

**Features Decreasing Default Risk (Negative Coefficients):**
- Higher annual income
- Home ownership (mortgage/own)
- Higher grade indicators
- Longer employment duration

### 5.4 ROC AUC Analysis

The ROC curve demonstrates the model's ability to discriminate between defaulters and non-defaulters across all classification thresholds. The optimal threshold, determined by Youden's J statistic (maximizing TPR - FPR), provides the best balance between sensitivity and specificity.

### 5.5 Precision-Recall Analysis

The Precision-Recall curve is particularly informative for our imbalanced dataset. The Average Precision score summarizes the curve as the weighted mean of precisions achieved at each threshold. The F1-optimal threshold balances precision and recall.

---

## 6. Discussion

### 6.1 Precision vs Recall Tradeoff — Banking Perspective

#### 6.1.1 Minimizing False Positives (Detecting Defaults with Fewer Errors)

**Scenario**: The bank wants to maximize lending opportunities while accurately flagging defaults.

- **Strategy**: Increase the classification threshold (e.g., 0.5-0.6)
- **Effect**: Higher precision — when the model flags a borrower as a defaulter, the prediction is more reliable
- **Cost**: Lower recall — some actual defaulters will be missed
- **Business impact**: More loans approved, higher revenue from interest, but some increase in NPA risk

#### 6.1.2 Minimizing NPAs (Playing Safe)

**Scenario**: The bank prioritizes safety and wants to minimize Non-Performing Assets.

- **Strategy**: Lower the classification threshold (e.g., 0.2-0.3)
- **Effect**: Higher recall — the model catches most actual defaulters
- **Cost**: Lower precision — more good borrowers incorrectly flagged as risky
- **Business impact**: Fewer NPAs, lower financial losses, but reduced lending volume

#### 6.1.3 Recommended Tiered Approach

We propose a risk-proportional threshold strategy:

| Loan Category | Amount Range | Recommended Threshold | Rationale |
|--------------|-------------|----------------------|-----------|
| Small | < $10,000 | 0.5 | Balanced — lower individual risk |
| Medium | $10,000 - $25,000 | 0.4 | Slightly conservative |
| Large | > $25,000 | 0.3 | Conservative — high individual risk |

This approach maximizes lending volume for small loans (where individual NPA impact is manageable) while being more cautious with large loans (where defaults are more damaging).

### 6.2 Model Limitations

1. **Feature limitations**: The model uses self-reported data (income, employment) which may be unreliable
2. **Temporal dynamics**: The model is trained on historical data and may not capture emerging economic trends
3. **Geographic granularity**: State-level features provide limited geographic resolution
4. **Linear assumptions**: Logistic regression assumes linear relationships in log-odds space
5. **External factors**: Macroeconomic conditions, policy changes, and market dynamics are not captured

### 6.3 Comparison with Literature

Our findings align with established credit risk research:
- Interest rate as a top predictor is consistent with Merton (1974) credit risk theory
- The importance of DTI ratio aligns with regulatory guidelines (e.g., QM standards)
- Employment stability as a protective factor is well-documented in credit literature
- Grade as a strong predictor validates the internal rating system

---

## 7. Conclusions and Recommendations

### 7.1 Key Conclusions

1. **Logistic Regression provides a strong, interpretable baseline** for loan default prediction on the LoanTap dataset
2. **Interest rate, loan grade, term, and DTI** are the most powerful predictors of default
3. **Class imbalance can be effectively handled** using balanced class weights without data-level resampling
4. **Threshold optimization is critical** — the default 0.5 threshold may not be optimal for banking applications
5. **The precision-recall tradeoff** must be managed through business-specific strategies

### 7.2 Actionable Recommendations

1. **Implement tiered threshold strategy** based on loan amount
2. **Strengthen underwriting** for Grade D+ borrowers
3. **Encourage 36-month terms** through incentive structures
4. **Set DTI caps** at 35% for new applications
5. **Monitor revolving utilization** as an early warning indicator
6. **Retrain quarterly** with updated data to prevent model drift
7. **Implement A/B testing** for threshold optimization in production

### 7.3 Future Work

1. **Ensemble methods**: Compare with Random Forest, XGBoost, and LightGBM
2. **Feature enrichment**: Incorporate bureau data, transaction patterns, and social signals
3. **Deep learning**: Explore neural networks for capturing non-linear interactions
4. **Time series analysis**: Model temporal patterns in repayment behavior
5. **Causal inference**: Move beyond prediction to understanding causal mechanisms
6. **Fairness analysis**: Assess and mitigate potential bias in lending decisions

---

## References

1. Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589-609.
2. Merton, R. C. (1974). On the pricing of corporate debt: The risk structure of interest rates. *The Journal of Finance*, 29(2), 449-470.
3. Wiginton, J. C. (1980). A note on the comparison of logit and discriminant models of consumer credit behavior. *Journal of Financial and Quantitative Analysis*, 15(3), 757-770.
4. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression*. John Wiley & Sons.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
6. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
7. Thomas, L. C. (2009). *Consumer Credit Models: Pricing, Profit, and Portfolios*. Oxford University Press.
8. Scikit-learn documentation. https://scikit-learn.org/stable/
9. Statsmodels documentation. https://www.statsmodels.org/stable/

---

**Author:** Vidyasagar, Data Scientist  
**Contact:** [Available upon request]  
**Repository:** [GitHub — Loan Default Prediction Model](https://github.com/)

---

*This research paper was prepared as part of a comprehensive credit risk analysis project. All data is used for educational and research purposes.*
