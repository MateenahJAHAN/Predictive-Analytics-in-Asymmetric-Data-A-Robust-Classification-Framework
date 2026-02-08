# Predictive Analytics in Asymmetric Data: Loan Default Prediction

**Author:** Mateenah Jahan  
**Under Supervision of:** Vidhya Sagar (Vidyasagar) Nesanuru  

## Supervisor Profile Reference
- LinkedIn: https://www.linkedin.com/in/vnesanuru/
- Name: Vidhya Sagar (Vidyasagar) Nesanuru
- Pronouns: He/Him
- Headline: AI | ML | Leadership | Freelance | Trainings & Consulting
- Location: Hyderabad, Telangana, India
- Experience: New Relic
- Education: IIM Bangalore
- Network: 500+ connections

## Executive Summary (CEO-Ready)
This repository demonstrates a **risk-first loan default framework** with business-grade
insights, explainable modeling, and clear governance. It is designed to show how leadership
can **reduce NPAs**, **protect revenue**, and **scale underwriting** with evidence-driven
decisioning.

## Results Snapshot (Baseline Logistic Regression)
Metrics generated from `reports/model_metrics.json` using a balanced logistic regression
pipeline on the current dataset.

- Default rate: **8.2%**
- ROC AUC: **0.574**
- Average Precision (PR AUC): **0.108**
- Accuracy: **0.783**
- Precision (default): **0.114**
- Recall (default): **0.245**
- F1 (default): **0.156**
- Confusion Matrix: **[[458, 93], [37, 12]]**

These baseline results are intentionally conservative and highlight why **threshold tuning**
and **policy overlays** are critical in risk-sensitive portfolios.

## Visual Evidence (Figures)
![Target Distribution](reports/figures/target_distribution.png)
![Correlation Heatmap](reports/figures/correlation_heatmap.png)
![ROC Curve](reports/figures/roc_curve.png)
![Precision-Recall Curve](reports/figures/precision_recall_curve.png)
![Confusion Matrix](reports/figures/confusion_matrix.png)
![Top Coefficients](reports/figures/top_coefficients.png)

## Project Structure
```
.
- data/
  - raw/LoanTapData.csv          # Source dataset (replace with original if available)
  - processed/
- files/
  - submission_text.md           # Recommendations + questionnaire answers
- notebooks/
  - LoanTap_Logistic_Regression.ipynb
- reports/
  - analysis_report.md
  - llm_future_predictions.md
  - model_metrics.json
  - figures/
- research_paper/
  - LoanTap_Research_Paper_Vidyasagar_DS.md
- src/
  - generate_loantap_dataset.py
  - generate_figures.py
  - loantap_modeling_utils.py
- requirements.txt
```

## Dataset
`data/raw/LoanTapData.csv` follows the provided data dictionary (loan amount, term,
interest rate, grades, employment data, credit history, etc.). The repository ships
with a generated dataset aligned to the dictionary. **If you have the original LoanTap
CSV, replace this file to reproduce the same analysis.**

## Quickstart
```bash
python3 -m pip install -r requirements.txt
python3 src/generate_loantap_dataset.py
python3 src/generate_figures.py
```
Open `notebooks/LoanTap_Logistic_Regression.ipynb` and run cells top-to-bottom.

## Requirements
- Python 3.10+ (tested on Python 3.12)
- See `requirements.txt` for full dependency list

## Code Highlights
**Modeling pipeline (simplified):**
```python
preprocessor = build_preprocessor(numeric_features, categorical_features)
model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
clf.fit(X_train, y_train)
```

**Figure generation:**
```bash
python3 src/generate_figures.py
```

## Business Focus
- Detect defaulters early without blocking credit-worthy applicants
- Quantify the precision-recall tradeoff and align it to NPA risk tolerance
- Provide actionable underwriting levers (pricing, term, DTI thresholds, etc.)

## AI Agents and LLM Roadmap (Future Predictions)
To extend this work, we plan to add an agentic layer that monitors portfolio drift,
summarizes risk drivers, and simulates future scenarios. This will be **LLM-assisted**
only and will not replace the statistical model.

**Super LLMs considered for future agent workflows:**
- GPT-4.1 / GPT-5
- Claude 3.5 Sonnet
- Gemini 2.0
- Llama 3.1 (70B)
- Mistral Large 2

## Research Paper Highlights (Expanded)
**Highlighted additions in the research paper:**
- **Full figures embedded** (target distribution, correlation heatmap, ROC, PR curve,
  confusion matrix, and top coefficients).
- **GLM and GLM-4 roadmap**: GLM is the current baseline, while GLM-4 (LLM) is planned
  for narrative insights and agentic decision support.
- **FHIR-inspired banking data**: Standardized data exchange to enable faster and more
  reliable eligibility validation and **prior authorization** decisions.
- **Eligibility AI agent (future)**: A governed agent workflow to decide who is eligible
  for a loan based on verified financial data, policy rules, and explainable reasoning.

## Deliverables
- **Notebook:** End-to-end EDA + modeling + evaluation
- **Research Paper:** Structured methodology and insights
- **Submission Text:** Recommendations + questionnaire answers (CEO-ready)