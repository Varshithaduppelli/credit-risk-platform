import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

from src.preprocess import load_data, scale_features, get_features_target
from src.features import engineer_features

def train_models():
    df = load_data()
    df = engineer_features(df)
    X, y, feature_cols = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    models = {
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_sc, y_train)
            y_pred  = model.predict(X_test_sc)
            y_prob  = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_prob  = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'model':            model,
            'accuracy':         accuracy_score(y_test, y_pred),
            'auc':              roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred':           y_pred,
            'y_prob':           y_prob,
            'report':           classification_report(y_test, y_pred, output_dict=True)
        }

    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature':    feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return results, X_test, y_test, feature_importance, scaler, feature_cols

def predict_single(input_data, model, scaler=None, use_scaled=False):
    df_input = pd.DataFrame([input_data])
    if use_scaled:
        prob = model.predict_proba(scaler.transform(df_input))[0][1]
    else:
        prob = model.predict_proba(df_input)[0][1]
    return prob, ('High Risk' if prob > 0.5 else 'Low Risk')

def shap_like_explanation(input_data, model, feature_cols, baseline_prob):
    """
    Manual SHAP-like feature contribution using individual feature perturbation.
    Shows how each feature pushes risk UP or DOWN from baseline.
    """
    df_base = pd.DataFrame([input_data])
    contributions = {}
    for col in feature_cols:
        df_perturbed = df_base.copy()
        # Set feature to median-like neutral value
        neutral_vals = {
            'revolving_utilization': 0.3,
            'age': 52,
            'past_due_30_59': 0,
            'debt_ratio': 0.35,
            'monthly_income': 6000,
            'open_credit_lines': 8,
            'times_90_days_late': 0,
            'real_estate_loans': 1,
            'past_due_60_89': 0,
            'dependents': 1
        }
        df_perturbed[col] = neutral_vals.get(col, 0)
        prob_without = model.predict_proba(df_perturbed)[0][1]
        contributions[col] = baseline_prob - prob_without  # positive = increases risk

    return pd.Series(contributions).sort_values()

def predict_batch(df_input, model, feature_cols):
    missing = [c for c in feature_cols if c not in df_input.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    X = df_input[feature_cols].fillna(df_input[feature_cols].median())
    probs = model.predict_proba(X)[:, 1]
    labels = ['High Risk' if p > 0.5 else 'Low Risk' for p in probs]
    tiers  = ['🔴 High' if p > 0.5 else '🟡 Medium' if p > 0.25 else '🟢 Low' for p in probs]
    result = df_input.copy()
    result['risk_probability'] = np.round(probs, 4)
    result['risk_label']       = labels
    result['risk_tier']        = tiers
    return result, None
