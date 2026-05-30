import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    # Total missed payments
    df['total_past_due'] = df['past_due_30_59'] + df['past_due_60_89'] + df['times_90_days_late']
    # Annual income
    df['annual_income'] = df['monthly_income'] * 12
    # High risk flag
    df['high_risk_flag'] = (
        (df['times_90_days_late'] > 0) & (df['revolving_utilization'] > 0.8)
    ).astype(int)
    # Income per dependent
    df['income_per_dependent'] = df['monthly_income'] / (df['dependents'] + 1)
    return df

def get_eda_stats(df):
    stats = {
        'total_records': len(df),
        'default_rate': df['default'].mean(),
        'avg_age': df['age'].mean(),
        'avg_monthly_income': df['monthly_income'].mean(),
        'avg_debt_ratio': df['debt_ratio'].mean(),
        'avg_utilization': df['revolving_utilization'].mean(),
    }
    return stats
