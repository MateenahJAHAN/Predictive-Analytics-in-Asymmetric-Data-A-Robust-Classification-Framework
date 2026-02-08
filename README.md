# Predictive Analytics in Asymmetric Data: Loan Default Prediction

**Author:** Vidyasagar - Data Scientist

## Profile Reference
Work completed under the following professional profile:
- LinkedIn: https://www.linkedin.com/in/vnesanuru/
- Name: Vidhya Sagar (Vidyasagar) Nesanuru
- Pronouns: He/Him
- Headline: AI | ML | Leadership | Freelance | Trainings & Consulting
- Location: Hyderabad, Telangana, India
- Experience: New Relic
- Education: IIM Bangalore
- Network: 500+ connections

## Executive Summary
This project delivers a production-ready, board-level view of loan default risk using
logistic regression. It includes a structured dataset, end-to-end EDA, feature
engineering, model evaluation (ROC AUC & Precision-Recall), and practical
recommendations geared toward high-stakes lending decisions.

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
- research_paper/
  - LoanTap_Research_Paper_Vidyasagar_DS.md
- src/
  - generate_loantap_dataset.py
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
```
Open `notebooks/LoanTap_Logistic_Regression.ipynb` and run cells top-to-bottom.

## Business Focus
- Detect defaulters early without blocking credit-worthy applicants
- Quantify the precision-recall tradeoff and align it to NPA risk tolerance
- Provide actionable underwriting levers (pricing, term, DTI thresholds, etc.)

## Future Predictions with AI Agents and LLMs
To extend this work, we plan to add an agentic layer that monitors portfolio drift,
summarizes risk drivers, and simulates future scenarios. This will be **LLM-assisted**
only and will not replace the statistical model.

**Super LLMs considered for future agent workflows:**
- GPT-4.1 / GPT-5
- Claude 3.5 Sonnet
- Gemini 2.0
- Llama 3.1 (70B)
- Mistral Large 2

## Deliverables
- **Notebook:** End-to-end EDA + modeling + evaluation
- **Research Paper:** Structured methodology and insights
- **Submission Text:** Recommendations + questionnaire answers (CEO-ready)