import math
from datetime import datetime
import random

import numpy as np
import pandas as pd


def _random_month_year(start_year: int, end_year: int) -> str:
    month = random.randint(1, 12)
    year = random.randint(start_year, end_year)
    return datetime(year, month, 1).strftime("%b-%Y")


def _amortized_installment(loan_amount: float, annual_rate: float, term_months: int) -> float:
    if annual_rate <= 0:
        return loan_amount / term_months
    monthly_rate = annual_rate / 100 / 12
    factor = math.pow(1 + monthly_rate, term_months)
    return loan_amount * monthly_rate * factor / (factor - 1)


def generate_dataset(rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    grades = ["A", "B", "C", "D", "E", "F", "G"]
    grade_probs = [0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05]
    sub_grades = [f"{g}{i}" for g in grades for i in range(1, 6)]

    emp_titles = [
        "Software Engineer",
        "Teacher",
        "Nurse",
        "Sales Associate",
        "Accountant",
        "Project Manager",
        "Analyst",
        "Driver",
        "Consultant",
        "Operations Manager",
        "Data Scientist",
        "HR Specialist",
        "Customer Service Rep",
        "Electrician",
        "Mechanic",
        "Marketing Specialist",
        "Executive Assistant",
        "Business Analyst",
        "Product Manager",
        "Warehouse Associate",
    ]

    home_ownerships = ["RENT", "MORTGAGE", "OWN", "OTHER"]
    home_probs = [0.4, 0.4, 0.18, 0.02]

    verification_statuses = ["Verified", "Source Verified", "Not Verified"]
    verification_probs = [0.3, 0.3, 0.4]

    purposes = [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "major_purchase",
        "medical",
        "small_business",
        "vacation",
        "car",
        "moving",
        "house",
        "renewable_energy",
        "other",
    ]

    states = [
        "CA",
        "TX",
        "NY",
        "FL",
        "IL",
        "PA",
        "OH",
        "GA",
        "NC",
        "MI",
        "NJ",
        "VA",
        "WA",
        "AZ",
        "MA",
    ]
    cities = [
        "Austin",
        "San Jose",
        "Seattle",
        "Miami",
        "Chicago",
        "Phoenix",
        "Boston",
        "Raleigh",
        "Columbus",
        "Newark",
        "Atlanta",
        "Houston",
        "Dallas",
        "Tampa",
        "San Diego",
    ]

    grade_rate_map = {
        "A": (7, 11),
        "B": (9, 13),
        "C": (12, 16),
        "D": (15, 19),
        "E": (18, 22),
        "F": (21, 25),
        "G": (24, 28),
    }
    grade_score = {grade: idx * 0.2 for idx, grade in enumerate(grades)}

    data = []
    for _ in range(rows):
        grade = random.choices(grades, grade_probs, k=1)[0]
        sub_grade = random.choice([sg for sg in sub_grades if sg.startswith(grade)])
        term = random.choices([36, 60], [0.7, 0.3], k=1)[0]

        int_low, int_high = grade_rate_map[grade]
        int_rate = np.random.uniform(int_low, int_high)
        if term == 60:
            int_rate += np.random.uniform(0.2, 1.5)

        annual_inc = np.random.lognormal(mean=11.0, sigma=0.45)
        annual_inc = float(np.clip(annual_inc, 20000, 320000))

        loan_amnt = np.random.normal(15000 + (annual_inc - 60000) / 5, 6000)
        loan_amnt = float(np.clip(loan_amnt, 1000, 40000))

        installment = _amortized_installment(loan_amnt, int_rate, term)

        dti = np.random.normal(14, 7) + (70000 - annual_inc) / 20000
        dti = float(np.clip(dti, 0, 45))

        open_acc = int(np.clip(np.random.poisson(10), 1, 30))
        total_acc = int(np.clip(open_acc + np.random.poisson(8), 2, 60))

        pub_rec = int(np.clip(np.random.poisson(0.2), 0, 4))
        pub_rec_bankruptcies = int(np.clip(np.random.binomial(1, 0.08 + 0.05 * pub_rec), 0, 3))
        mort_acc = int(np.clip(np.random.poisson(1.8), 0, 10))

        revol_bal = max(0, loan_amnt * np.random.uniform(0.4, 1.4) + np.random.normal(0, 8000))
        revol_bal = float(np.clip(revol_bal, 0, 150000))

        revol_util = np.random.normal(45, 20) + dti / 2
        revol_util = float(np.clip(revol_util, 0, 120))

        home_ownership = random.choices(home_ownerships, home_probs, k=1)[0]
        verification_status = random.choices(verification_statuses, verification_probs, k=1)[0]
        purpose = random.choice(purposes)

        emp_title = random.choice(emp_titles)
        emp_length = int(np.clip(np.random.normal(5, 3), 0, 10))

        issue_d = _random_month_year(2014, 2020)
        earliest_cr_line = _random_month_year(1980, 2012)

        initial_list_status = random.choices(["W", "F"], [0.7, 0.3], k=1)[0]
        application_type = random.choices(["Individual", "Joint"], [0.9, 0.1], k=1)[0]

        address = f"{random.choice(cities)}, {random.choice(states)}"

        risk_score = (
            0.04 * int_rate
            + 0.035 * dti
            + 0.015 * revol_util
            + 0.45 * (term == 60)
            + 0.55 * pub_rec
            + 0.75 * pub_rec_bankruptcies
            + 0.55 * grade_score[grade]
            + 0.35 * (home_ownership == "RENT")
            - 0.000015 * annual_inc
            - 0.2 * (home_ownership == "MORTGAGE")
            - 0.3 * (home_ownership == "OWN")
            - 0.2 * (application_type == "Joint")
            + np.random.normal(0, 0.6)
        )
        default_prob = 1 / (1 + math.exp(-(risk_score - 4.3)))
        loan_status = "Charged Off" if random.random() < default_prob else "Fully Paid"

        data.append(
            {
                "loan_amnt": round(loan_amnt, 2),
                "term": term,
                "int_rate": round(float(int_rate), 2),
                "installment": round(float(installment), 2),
                "grade": grade,
                "sub_grade": sub_grade,
                "emp_title": emp_title,
                "emp_length": emp_length,
                "home_ownership": home_ownership,
                "annual_inc": round(annual_inc, 2),
                "verification_status": verification_status,
                "issue_d": issue_d,
                "loan_status": loan_status,
                "purpose": purpose,
                "title": purpose.replace("_", " ").title(),
                "dti": round(dti, 2),
                "earliest_cr_line": earliest_cr_line,
                "open_acc": open_acc,
                "pub_rec": pub_rec,
                "revol_bal": round(float(revol_bal), 2),
                "revol_util": round(float(revol_util), 2),
                "total_acc": total_acc,
                "initial_list_status": initial_list_status,
                "application_type": application_type,
                "mort_acc": mort_acc,
                "pub_rec_bankruptcies": pub_rec_bankruptcies,
                "Address": address,
            }
        )

    df = pd.DataFrame(data)

    for col, frac in [("emp_title", 0.05), ("emp_length", 0.03), ("revol_util", 0.04), ("mort_acc", 0.05)]:
        df.loc[df.sample(frac=frac, random_state=seed).index, col] = np.nan

    return df


def main() -> None:
    df = generate_dataset()
    output_path = "data/raw/LoanTapData.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path} with shape {df.shape}")


if __name__ == "__main__":
    main()
