# Submission Text - Recommendations & Questionnaire

**Author:** Mateenah Jahan (Under supervision of Vidhya Sagar)  
**Notebook PDF Link (Google Drive):** _[Paste public link here after export]_

## Recommendations (Executive Summary)
- **Risk-focused decisioning:** Use probability thresholds (not a fixed 0.5) and tune
  for high recall of defaulters when NPA risk is the top concern.
- **Pricing discipline:** Tighten pricing/approval for high-interest and 60-month
  term loans; they consistently carry higher default risk.
- **Credit utilization guardrails:** Borrowers with high DTI and revolving utilization
  should be routed for manual review or offered smaller ticket sizes.
- **Early-warning flags:** Any public record or bankruptcy indicator should trigger
  enhanced verification and stricter limits.
- **Portfolio strategy:** Favor applicants with stable income and mortgage/own status
  where default risk is structurally lower.

---

## Tradeoff Questions
1. **How can we make sure that our model can detect real defaulters and there are less false positives?**  
   - Optimize the decision threshold using the Precision-Recall curve.  
   - Apply class weights or cost-sensitive learning to penalize missed defaulters.  
   - Use a two-tier process: approve low-risk, reject high-risk, and manually review the gray zone.  

2. **Since NPA is a real problem in this industry, it is important we play safe and should not disburse loans to anyone.**  
   - Prioritize **recall for defaulters**, lower the threshold, and tighten approvals for high-risk segments.  
   - Combine model scores with hard policy rules (e.g., DTI caps, recent bankruptcies) for safety-first lending.  

---

## Questionnaire
1. **What percentage of customers have fully paid their Loan Amount?**  
   **91.80%** of customers are fully paid.

2. **Comment about the correlation between Loan Amount and Installment features.**  
   Strong positive correlation (**~0.932**), indicating installment size rises
   proportionally with loan amount.

3. **The majority of people have home ownership as _______.**  
   **MORTGAGE**

4. **People with grades 'A' are more likely to fully pay their loan. (T/F)**  
   **True** - Grade A repayment rate is higher than other grades in this dataset.

5. **Name the top 2 afforded job titles.**  
   **Software Engineer** and **Operations Manager**

6. **Thinking from a bank's perspective, which metric should our primary focus be on?**  
   **Recall** (for defaulters) - to minimize false negatives and protect against NPA risk.

7. **How does the gap in precision and recall affect the bank?**  
   A **low recall** means missed defaulters (higher credit losses). A **low precision**
   means rejecting good borrowers (lost interest revenue). The bank should set a
   threshold that balances opportunity cost with loss avoidance based on risk appetite.

8. **Which were the features that heavily affected the outcome?**  
   Interest rate, DTI, revolving utilization, 60-month term, public record flags,
   and bankruptcy indicators were the most influential signals.

9. **Will the results be affected by geographical location? (Yes/No)**  
   **Yes** - regional economic conditions and local credit behavior can influence
   default risk, even if geography is a secondary feature.

---

**Discussion Forum Link:**  
https://www.scaler.com/academy/mentee-dashboard/discussion-forum/p/ask-me-anything-business-case-loantap/21146
