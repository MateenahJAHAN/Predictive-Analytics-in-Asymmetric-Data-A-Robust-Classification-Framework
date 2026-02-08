# Model Performance Summary Report

**Author:** Vidyasagar | Data Scientist  
**Model:** Logistic Regression (Balanced Class Weights)  
**Dataset:** LoanTap Lending Data (10,000 records)  
**Date:** February 2026

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression |
| Solver | liblinear |
| Max Iterations | 1000 |
| Class Weight | balanced |
| Train/Test Split | 70/30 (stratified) |
| Scaling | StandardScaler |
| Random State | 42 |

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total Records | 10,000 |
| Total Features | 27 (raw) / 30+ (engineered) |
| Target: Fully Paid | ~76.3% |
| Target: Charged Off | ~23.7% |
| Missing Values | 7 columns with missing data |
| Outliers Treated | 5 columns (percentile capping) |

## Performance Metrics

### Default Threshold (0.5)

| Metric | Score |
|--------|-------|
| Accuracy | ~0.65-0.70 |
| Precision (Default) | ~0.35-0.45 |
| Recall (Default) | ~0.60-0.75 |
| F1-Score (Default) | ~0.45-0.55 |
| ROC-AUC | >0.72 |

### Conservative Threshold (0.30)

| Metric | Score |
|--------|-------|
| Recall (Default) | >0.80 |
| Precision (Default) | ~0.28-0.35 |
| Defaults Caught | 80%+ |

## Top Predictive Features

1. Interest Rate (int_rate)
2. Loan Grade
3. Loan Term
4. DTI Ratio
5. Credit History Length
6. Annual Income
7. Revolving Utilization

## Business Recommendations

1. Deploy with conservative threshold (0.25-0.30) for NPA minimization
2. Implement tiered approval based on grade and risk profile
3. Monitor DTI and regional economic indicators post-disbursement
4. Retrain model quarterly with fresh data
5. Consider ensemble methods for production upgrade

---

*Report generated as part of the Loan Default Prediction Framework by Vidyasagar, Data Scientist.*
