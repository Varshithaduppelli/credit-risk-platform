import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path='data/cs-training.csv'):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.rename(columns={
        'SeriousDlqin2yrs': 'default',
        'RevolvingUtilizationOfUnsecuredLines': 'revolving_utilization',
        'NumberOfTime30-59DaysPastDueNotWorse': 'past_due_30_59',
        'DebtRatio': 'debt_ratio',
        'MonthlyIncome': 'monthly_income',
        'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines',
        'NumberOfTimes90DaysLate': 'times_90_days_late',
        'NumberRealEstateLoansOrLines': 'real_estate_loans',
        'NumberOfTime60-89DaysPastDueNotWorse': 'past_due_60_89',
        'NumberOfDependents': 'dependents'
    })
    # Fill missing values
    df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())
    df['dependents'] = df['dependents'].fillna(0)
    # Remove outliers
    df = df[df['age'] > 0]
    df = df[df['revolving_utilization'] <= 1.5]
    return df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def get_features_target(df):
    feature_cols = [
        'revolving_utilization', 'age', 'past_due_30_59',
        'debt_ratio', 'monthly_income', 'open_credit_lines',
        'times_90_days_late', 'real_estate_loans',
        'past_due_60_89', 'dependents'
    ]
    X = df[feature_cols]
    y = df['default']
    return X, y, feature_cols
