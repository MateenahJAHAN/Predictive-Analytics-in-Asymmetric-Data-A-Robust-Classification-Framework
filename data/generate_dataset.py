"""
Generate a realistic LoanTapData.csv dataset for Loan Default Prediction
Author: Vidyasagar, Data Scientist
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n_samples = 10000

# Loan amounts
loan_amnt = np.random.choice([5000, 7500, 10000, 12000, 15000, 18000, 20000, 24000, 25000, 30000, 35000, 40000], size=n_samples, p=[0.1, 0.08, 0.15, 0.1, 0.12, 0.08, 0.1, 0.07, 0.06, 0.06, 0.04, 0.04])
loan_amnt = loan_amnt + np.random.randint(-2000, 2000, size=n_samples)
loan_amnt = np.clip(loan_amnt, 1000, 40000)

# Term
term = np.random.choice([' 36 months', ' 60 months'], size=n_samples, p=[0.7, 0.3])

# Interest rate
int_rate = np.round(np.random.uniform(5.0, 30.0, size=n_samples), 2)

# Installment
installment = np.round(loan_amnt * (int_rate/100/12) / (1 - (1 + int_rate/100/12)**(-np.where(term==' 36 months', 36, 60))), 2)

# Grade
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
grade = np.random.choice(grades, size=n_samples, p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.06, 0.04])

# Sub grade
sub_grades_map = {g: [f"{g}{i}" for i in range(1, 6)] for g in grades}
sub_grade = [np.random.choice(sub_grades_map[g]) for g in grade]

# Employment title
emp_titles = ['Teacher', 'Manager', 'Owner', 'Driver', 'Registered Nurse', 'Supervisor', 'Sales',
              'General Manager', 'Office Manager', 'Project Manager', 'Director', 'Analyst',
              'Engineer', 'Mechanic', 'RN', 'Accountant', 'Truck Driver', 'Vice President',
              'Operations Manager', 'Police Officer', 'Server', 'Assistant Manager', 'Cashier',
              'Store Manager', 'Administrative Assistant', 'Electrician', 'Nurse', 'Foreman',
              'Supervisor of Operations', 'Attorney', 'Software Engineer', 'Data Scientist',
              'Consultant', 'Marketing Manager', 'Financial Analyst', np.nan]
emp_title = np.random.choice(emp_titles, size=n_samples)

# Employment length
emp_lengths = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
               '6 years', '7 years', '8 years', '9 years', '10+ years', np.nan]
emp_length = np.random.choice(emp_lengths, size=n_samples,
                               p=[0.08, 0.06, 0.07, 0.07, 0.06, 0.07, 0.05, 0.05, 0.05, 0.04, 0.30, 0.10])

# Home ownership
home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], size=n_samples, p=[0.40, 0.42, 0.15, 0.03])

# Annual income
annual_inc = np.round(np.random.lognormal(mean=11.0, sigma=0.6, size=n_samples), 2)
annual_inc = np.clip(annual_inc, 10000, 500000)

# Verification status
verification_status = np.random.choice(['Verified', 'Source Verified', 'Not Verified'], size=n_samples, p=[0.35, 0.30, 0.35])

# Issue date
start_date = datetime(2015, 1, 1)
end_date = datetime(2020, 12, 31)
days_range = (end_date - start_date).days
issue_d = [(start_date + timedelta(days=np.random.randint(0, days_range))).strftime('%b-%Y') for _ in range(n_samples)]

# Loan status - Target Variable
# Higher interest rate, lower grade, lower income -> more likely to default
default_prob = 0.15 + 0.01 * (int_rate - 15) + 0.05 * np.array([grades.index(g) for g in grade]) / 6 - 0.00001 * (annual_inc - 60000) / 60000
default_prob = np.clip(default_prob, 0.05, 0.60)
loan_status = np.array(['Fully Paid' if np.random.random() > p else 'Charged Off' for p in default_prob])

# Purpose
purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase',
            'medical', 'small_business', 'car', 'vacation', 'moving', 'house', 'wedding',
            'renewable_energy', 'educational']
purpose = np.random.choice(purposes, size=n_samples,
                            p=[0.35, 0.20, 0.10, 0.08, 0.06, 0.04, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])

# Title
title_map = {
    'debt_consolidation': 'Debt consolidation',
    'credit_card': 'Credit card refinancing',
    'home_improvement': 'Home improvement',
    'other': 'Other',
    'major_purchase': 'Major purchase',
    'medical': 'Medical expenses',
    'small_business': 'Business',
    'car': 'Car financing',
    'vacation': 'Vacation',
    'moving': 'Moving and relocation',
    'house': 'Home buying',
    'wedding': 'Wedding',
    'renewable_energy': 'Green loan',
    'educational': 'Educational expenses'
}
title = [title_map.get(p, 'Other') for p in purpose]
# Add some NaN
for i in np.random.choice(range(n_samples), size=200, replace=False):
    title[i] = np.nan

# DTI
dti = np.round(np.random.uniform(0, 45, size=n_samples), 2)

# Earliest credit line
earliest_cr_line = [(start_date - timedelta(days=np.random.randint(365*2, 365*30))).strftime('%b-%Y') for _ in range(n_samples)]

# Open accounts
open_acc = np.random.randint(2, 30, size=n_samples)

# Public records
pub_rec = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.75, 0.15, 0.05, 0.03, 0.02])

# Revolving balance
revol_bal = np.round(np.random.lognormal(mean=9.5, sigma=1.0, size=n_samples), 2)
revol_bal = np.clip(revol_bal, 0, 200000)

# Revolving utilization
revol_util = np.round(np.random.uniform(0, 120, size=n_samples), 1)
# Add some NaN
revol_util_list = list(revol_util)
for i in np.random.choice(range(n_samples), size=150, replace=False):
    revol_util_list[i] = np.nan
revol_util = revol_util_list

# Total accounts
total_acc = open_acc + np.random.randint(1, 20, size=n_samples)

# Initial list status
initial_list_status = np.random.choice(['w', 'f'], size=n_samples, p=[0.55, 0.45])

# Application type
application_type = np.random.choice(['Individual', 'Joint App'], size=n_samples, p=[0.85, 0.15])

# Mortgage accounts
mort_acc = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n_samples,
                             p=[0.25, 0.20, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.02, 0.02, 0.01])
# Add some NaN
mort_acc_list = list(mort_acc.astype(float))
for i in np.random.choice(range(n_samples), size=300, replace=False):
    mort_acc_list[i] = np.nan
mort_acc = mort_acc_list

# Public record bankruptcies
pub_rec_bankruptcies = np.random.choice([0.0, 1.0, 2.0, 3.0, np.nan], size=n_samples,
                                         p=[0.82, 0.10, 0.03, 0.01, 0.04])

# Address
states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'NC', 'GA', 'NJ', 'VA', 'MI', 'WA',
          'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI', 'CO', 'MN', 'SC', 'AL', 'LA',
          'KY', 'OR', 'OK', 'CT', 'UT', 'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE',
          'WV', 'ID', 'HI', 'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY']
cities = ['Los Angeles', 'New York', 'Houston', 'Miami', 'Chicago', 'Philadelphia',
          'Columbus', 'Charlotte', 'Atlanta', 'Newark', 'Virginia Beach', 'Detroit',
          'Seattle', 'Phoenix', 'Boston', 'Nashville', 'Indianapolis', 'Kansas City',
          'Baltimore', 'Milwaukee', 'Denver', 'Minneapolis', 'Columbia', 'Birmingham',
          'New Orleans', 'Louisville', 'Portland', 'Oklahoma City', 'Hartford', 'Salt Lake City']
zip_codes = [str(np.random.randint(10000, 99999)) for _ in range(n_samples)]
address = [f"{np.random.randint(100, 9999)} {np.random.choice(['Main St', 'Oak Ave', 'Elm St', 'Maple Dr', 'Pine Rd', 'Cedar Ln', 'Broadway', 'Washington Blvd', 'Park Ave', 'River Rd'])}\n{np.random.choice(cities)}, {np.random.choice(states)} {z}" for z in zip_codes]

# Create DataFrame
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
    'address': address
})

df.to_csv('/workspace/data/LoanTapData.csv', index=False)
print(f"Dataset generated: {df.shape}")
print(f"\nLoan Status Distribution:")
print(df['loan_status'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nSample:")
print(df.head(3))
