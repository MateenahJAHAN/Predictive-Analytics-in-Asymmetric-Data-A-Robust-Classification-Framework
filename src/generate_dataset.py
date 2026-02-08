"""
Generate synthetic LoanTapData.csv dataset for Loan Default Prediction
Author: Vidyasagar, Data Scientist
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 10000  # Number of records

# ---- Helper Functions ----
def random_date(start, end, n):
    start_u = start.timestamp()
    end_u = end.timestamp()
    return [datetime.fromtimestamp(random.uniform(start_u, end_u)).strftime('%b-%Y') for _ in range(n)]

# ---- Loan Amount ----
loan_amnt = np.random.choice(
    np.arange(1000, 40001, 500), size=N, 
    p=None
)

# ---- Term ----
term = np.random.choice([' 36 months', ' 60 months'], size=N, p=[0.72, 0.28])

# ---- Grade ----
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
grade_probs = [0.18, 0.28, 0.22, 0.15, 0.10, 0.05, 0.02]
grade = np.random.choice(grades, size=N, p=grade_probs)

# ---- Sub Grade ----
sub_grades = []
for g in grade:
    sub_num = np.random.randint(1, 6)
    sub_grades.append(f'{g}{sub_num}')
sub_grade = np.array(sub_grades)

# ---- Interest Rate (correlated with grade) ----
grade_int_rate = {'A': (5.0, 2.0), 'B': (8.0, 2.5), 'C': (12.0, 3.0), 
                  'D': (16.0, 3.5), 'E': (20.0, 3.0), 'F': (24.0, 3.0), 'G': (27.0, 2.5)}
int_rate = np.array([round(np.clip(np.random.normal(grade_int_rate[g][0], grade_int_rate[g][1]), 3.0, 31.0), 2) for g in grade])

# ---- Installment (correlated with loan amount and term) ----
term_months = np.array([36 if '36' in t else 60 for t in term])
monthly_rate = int_rate / 100 / 12
installment = np.round(loan_amnt * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1), 2)

# ---- Employment Title ----
emp_titles = ['Manager', 'Teacher', 'Engineer', 'Nurse', 'Driver', 'Sales', 'Analyst', 
              'Supervisor', 'Director', 'Technician', 'Accountant', 'Clerk', 'Owner',
              'Registered Nurse', 'Administrative Assistant', 'Project Manager',
              'Operations Manager', 'General Manager', 'Software Engineer', 'Data Analyst',
              'Marketing Manager', 'Financial Analyst', 'Business Analyst', 'IT Manager',
              'Human Resources', 'Customer Service', 'Security Officer', 'Electrician',
              'Mechanic', 'Attorney', 'Physician', 'Pharmacist', 'Police Officer',
              'Fire Fighter', 'Paramedic', 'Consultant', 'Vice President', 'President',
              'CEO', 'CFO', 'Office Manager', 'Warehouse Worker', 'Truck Driver',
              'Bus Driver', 'Chef', 'Cook', 'Bartender', 'Server', 'Retail']
emp_title = np.random.choice(emp_titles, size=N)

# ---- Employment Length ----
emp_lengths = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', 
               '6 years', '7 years', '8 years', '9 years', '10+ years']
emp_length = np.random.choice(emp_lengths, size=N, p=[0.06, 0.07, 0.08, 0.08, 0.07, 0.08, 0.06, 0.06, 0.06, 0.05, 0.33])

# ---- Home Ownership ----
home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], size=N, p=[0.40, 0.42, 0.15, 0.03])

# ---- Annual Income ----
annual_inc = np.round(np.random.lognormal(mean=11.0, sigma=0.6, size=N), 2)
annual_inc = np.clip(annual_inc, 10000, 500000)

# ---- Verification Status ----
verification_status = np.random.choice(['Verified', 'Source Verified', 'Not Verified'], size=N, p=[0.33, 0.33, 0.34])

# ---- Issue Date ----
issue_d = random_date(datetime(2015, 1, 1), datetime(2020, 12, 31), N)

# ---- Purpose ----
purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 
            'major_purchase', 'small_business', 'car', 'medical', 'moving',
            'vacation', 'house', 'wedding', 'renewable_energy', 'educational']
purpose_probs = [0.45, 0.15, 0.10, 0.08, 0.05, 0.04, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005]
purpose = np.random.choice(purposes, size=N, p=purpose_probs)

# ---- Title ----
title_map = {
    'debt_consolidation': 'Debt consolidation', 'credit_card': 'Credit card refinancing',
    'home_improvement': 'Home improvement', 'other': 'Other', 'major_purchase': 'Major purchase',
    'small_business': 'Business', 'car': 'Car financing', 'medical': 'Medical expenses',
    'moving': 'Moving and relocation', 'vacation': 'Vacation', 'house': 'Home buying',
    'wedding': 'Wedding', 'renewable_energy': 'Green loan', 'educational': 'Educational'
}
title = np.array([title_map.get(p, 'Other') for p in purpose])

# ---- DTI ----
dti = np.round(np.random.lognormal(mean=2.8, sigma=0.4, size=N), 2)
dti = np.clip(dti, 0, 60)

# ---- Earliest Credit Line ----
earliest_cr_line = random_date(datetime(1985, 1, 1), datetime(2015, 12, 31), N)

# ---- Open Accounts ----
open_acc = np.random.poisson(lam=11, size=N)
open_acc = np.clip(open_acc, 1, 50)

# ---- Public Records ----
pub_rec = np.random.choice(range(0, 6), size=N, p=[0.80, 0.12, 0.05, 0.02, 0.007, 0.003])

# ---- Revolving Balance ----
revol_bal = np.round(np.random.lognormal(mean=9.0, sigma=1.2, size=N), 2)
revol_bal = np.clip(revol_bal, 0, 200000)

# ---- Revolving Utilization ----
revol_util = np.round(np.random.beta(2, 3, size=N) * 100, 1)

# ---- Total Accounts ----
total_acc = open_acc + np.random.poisson(lam=12, size=N)
total_acc = np.clip(total_acc, 2, 100)

# ---- Initial List Status ----
initial_list_status = np.random.choice(['w', 'f'], size=N, p=[0.60, 0.40])

# ---- Application Type ----
application_type = np.random.choice(['Individual', 'Joint App'], size=N, p=[0.88, 0.12])

# ---- Mortgage Accounts ----
mort_acc = np.random.choice(range(0, 12), size=N, 
                             p=[0.25, 0.20, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.02, 0.015, 0.01, 0.005])

# ---- Public Record Bankruptcies ----
pub_rec_bankruptcies = np.random.choice(range(0, 5), size=N, p=[0.85, 0.10, 0.03, 0.015, 0.005])

# ---- Address ----
states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 
          'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
          'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
          'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'WV', 'ID', 'HI', 'NH',
          'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY', 'DC']
zip_codes = [str(np.random.randint(10000, 99999)) for _ in range(N)]
state = np.random.choice(states, size=N)
address = [f'{np.random.randint(1, 9999)} Main St\n{s} {z}' for s, z in zip(state, zip_codes)]

# ---- Loan Status (Target Variable) ----
# Higher grade, higher DTI, higher int_rate = more likely to default
default_prob = np.zeros(N)
grade_risk = {'A': 0.05, 'B': 0.12, 'C': 0.20, 'D': 0.30, 'E': 0.40, 'F': 0.50, 'G': 0.60}
for i in range(N):
    base = grade_risk[grade[i]]
    # DTI effect
    dti_effect = (dti[i] - 15) * 0.005
    # Income effect (higher income = less default)
    inc_effect = -0.00001 * (annual_inc[i] - 60000) / 1000
    # Home ownership effect
    home_effect = 0.02 if home_ownership[i] == 'RENT' else -0.02
    # Employment length effect
    emp_effect = -0.03 if '10+' in emp_length[i] else 0.01
    # Pub rec effect
    pub_effect = 0.05 * pub_rec[i]
    
    prob = np.clip(base + dti_effect + inc_effect + home_effect + emp_effect + pub_effect, 0.02, 0.95)
    default_prob[i] = prob

loan_status = np.array(['Charged Off' if np.random.random() < p else 'Fully Paid' for p in default_prob])

# ---- Create DataFrame ----
df = pd.DataFrame({
    'loan_amnt': loan_amnt,
    'term': term,
    'int_rate': int_rate,
    'installment': installment,
    'grade': grade,
    'sub_grade': sub_grade,
    'emp_title': emp_title,
    'emp_length': emp_length,
    'home_ownership': home_ownership,
    'annual_inc': annual_inc,
    'verification_status': verification_status,
    'issue_d': issue_d,
    'loan_status': loan_status,
    'purpose': purpose,
    'title': title,
    'dti': dti,
    'earliest_cr_line': earliest_cr_line,
    'open_acc': open_acc,
    'pub_rec': pub_rec,
    'revol_bal': revol_bal,
    'revol_util': revol_util,
    'total_acc': total_acc,
    'initial_list_status': initial_list_status,
    'application_type': application_type,
    'mort_acc': mort_acc,
    'pub_rec_bankruptcies': pub_rec_bankruptcies,
    'Address': address
})

# Introduce some missing values (realistic)
missing_indices_emp_title = np.random.choice(N, size=int(N * 0.04), replace=False)
missing_indices_emp_length = np.random.choice(N, size=int(N * 0.03), replace=False)
missing_indices_revol_util = np.random.choice(N, size=int(N * 0.005), replace=False)
missing_indices_mort_acc = np.random.choice(N, size=int(N * 0.08), replace=False)
missing_indices_pub_rec_bankruptcies = np.random.choice(N, size=int(N * 0.005), replace=False)
missing_indices_title = np.random.choice(N, size=int(N * 0.002), replace=False)
missing_indices_dti = np.random.choice(N, size=int(N * 0.003), replace=False)

df.loc[missing_indices_emp_title, 'emp_title'] = np.nan
df.loc[missing_indices_emp_length, 'emp_length'] = np.nan
df.loc[missing_indices_revol_util, 'revol_util'] = np.nan
df.loc[missing_indices_mort_acc, 'mort_acc'] = np.nan
df.loc[missing_indices_pub_rec_bankruptcies, 'pub_rec_bankruptcies'] = np.nan
df.loc[missing_indices_title, 'title'] = np.nan
df.loc[missing_indices_dti, 'dti'] = np.nan

# Save to CSV
df.to_csv('/workspace/data/LoanTapData.csv', index=False)
print(f"Dataset generated successfully with {len(df)} records.")
print(f"\nShape: {df.shape}")
print(f"\nTarget distribution:\n{df['loan_status'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nSample data:\n{df.head()}")
