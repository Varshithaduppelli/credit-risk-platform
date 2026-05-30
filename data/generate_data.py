import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

age = np.random.randint(21, 70, n)
income = np.random.randint(20000, 150000, n)
loan_amount = np.random.randint(5000, 100000, n)
credit_score = np.random.randint(300, 850, n)
employment_years = np.random.randint(0, 30, n)
num_credit_lines = np.random.randint(1, 15, n)
debt_to_income = np.round(np.random.uniform(0.1, 0.8, n), 2)
missed_payments = np.random.randint(0, 10, n)
loan_purpose = np.random.choice(['home', 'car', 'education', 'personal', 'business'], n)

# Risk logic
risk_score = (
    (credit_score < 580).astype(int) * 3 +
    (debt_to_income > 0.5).astype(int) * 2 +
    (missed_payments > 3).astype(int) * 3 +
    (income < 40000).astype(int) * 1 +
    (employment_years < 2).astype(int) * 1
)
default = (risk_score + np.random.randint(0, 3, n) >= 5).astype(int)

df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'employment_years': employment_years,
    'num_credit_lines': num_credit_lines,
    'debt_to_income': debt_to_income,
    'missed_payments': missed_payments,
    'loan_purpose': loan_purpose,
    'default': default
})

df.to_csv('/home/claude/credit_risk_platform/data/credit_data.csv', index=False)
print(f"Dataset created: {len(df)} rows, {df['default'].mean():.1%} default rate")
