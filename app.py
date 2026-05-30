import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve
import io, warnings
warnings.filterwarnings('ignore')

from src.preprocess import load_data, get_features_target
from src.features import engineer_features, get_eda_stats
from src.model import train_models, predict_single, shap_like_explanation, predict_batch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk AI", page_icon="🏦", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.8rem}
.stTabs [data-baseweb="tab"]{font-size:15px;font-weight:500}
</style>""", unsafe_allow_html=True)

# ── Load & train (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_everything():
    df  = engineer_features(load_data())
    stats = get_eda_stats(df)
    results, X_test, y_test, feat_imp, scaler, feature_cols = train_models()
    return df, stats, results, X_test, y_test, feat_imp, scaler, feature_cols

with st.spinner("🤖 Training models on 1,49,399 real records… (first run ~2 min)"):
    df, stats, results, X_test, y_test, feat_imp, scaler, feature_cols = load_everything()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk AI")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 Data Explorer",
        "🤖 Model Performance",
        "🎯 Risk Predictor",
        "📦 Batch Prediction"
    ])
    st.markdown("---")
    st.markdown(f"**Records:** {stats['total_records']:,}")
    st.markdown(f"**Default Rate:** {stats['default_rate']:.2%}")
    st.markdown(f"**Source:** Kaggle — Give Me Some Credit")
    best = max(results, key=lambda k: results[k]['auc'])
    st.markdown(f"**Best Model:** {best}")
    st.markdown(f"**Best AUC:** {results[best]['auc']:.3f}")

# ── helpers ───────────────────────────────────────────────────────────────────
DARK = '#1e2130'
SPINE = '#2e3650'
def dark_fig(w=5, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK)
    ax.set_facecolor(DARK)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(SPINE)
    ax.spines['left'].set_color(SPINE)
    ax.tick_params(colors='white')
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Credit Risk Analytics Platform")
    st.markdown("Real-world delinquency risk prediction — **149,399 Kaggle records**, 3 ML models.")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records",      f"{stats['total_records']:,}")
    c2.metric("Default Rate",       f"{stats['default_rate']:.2%}")
    c3.metric("Avg Monthly Income", f"${stats['avg_monthly_income']:,.0f}")
    c4.metric("Best AUC",           f"{max(v['auc'] for v in results.values()):.3f}")
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Default Distribution")
        fig, ax = dark_fig()
        counts = df['default'].value_counts()
        bars = ax.bar(['Non-Default','Default'], counts.values,
                      color=['#2ecc71','#e74c3c'], width=0.5, edgecolor='none')
        for b,v in zip(bars, counts.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+400,
                    f"{v:,}", ha='center', color='white', fontsize=10)
        ax.set_ylabel('Count', color='white')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("Age Distribution by Default")
        fig, ax = dark_fig()
        for lbl, col, nm in [(0,'#2ecc71','Non-Default'),(1,'#e74c3c','Default')]:
            ax.hist(df[df['default']==lbl]['age'], bins=30, alpha=0.65,
                    color=col, label=nm, edgecolor='none')
        ax.set_xlabel('Age', color='white'); ax.set_ylabel('Count', color='white')
        ax.legend(facecolor=DARK, edgecolor=SPINE, labelcolor='white')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Feature Correlations with Default")
    num_cols = ['revolving_utilization','age','past_due_30_59','debt_ratio',
                'monthly_income','times_90_days_late','past_due_60_89','default']
    corr = df[num_cols].corr()['default'].drop('default').sort_values()
    fig, ax = dark_fig(10, 3)
    ax.barh(corr.index, corr.values,
            color=['#e74c3c' if v>0 else '#2ecc71' for v in corr.values], edgecolor='none')
    ax.axvline(0, color=SPINE, lw=1)
    ax.set_xlabel('Correlation with Default', color='white')
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")
    st.markdown("---")

    col1, col2 = st.columns([1,2])
    with col1:
        feat = st.selectbox("Select Feature", [
            'age','monthly_income','revolving_utilization','debt_ratio',
            'times_90_days_late','past_due_30_59','open_credit_lines','dependents'])
    with col2:
        filt = st.radio("Filter", ["All","Non-Default only","Default only"], horizontal=True)

    pdf = df if filt=="All" else df[df['default']==(0 if "Non" in filt else 1)]

    cl, cr = st.columns(2)
    with cl:
        st.subheader("Distribution")
        fig, ax = dark_fig()
        data = pdf[feat].clip(pdf[feat].quantile(0.01), pdf[feat].quantile(0.99))
        ax.hist(data, bins=40, color='#5b9bd5', edgecolor='none', alpha=0.85)
        ax.set_xlabel(feat, color='white'); ax.set_ylabel('Count', color='white')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with cr:
        st.subheader(f"Default Rate by {feat} Bucket")
        fig, ax = dark_fig()
        try:
            tmp = df.copy()
            tmp['bucket'] = pd.qcut(tmp[feat], q=5, duplicates='drop')
            bd = tmp.groupby('bucket', observed=True)['default'].mean()
            ax.bar(range(len(bd)), bd.values, color='#5b9bd5', edgecolor='none')
            ax.set_xticks(range(len(bd)))
            ax.set_xticklabels([str(b) for b in bd.index], rotation=20, ha='right', fontsize=8)
            ax.set_ylabel('Default Rate', color='white')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y:.1%}'))
            ax.tick_params(colors='white')
        except:
            ax.text(0.5,0.5,'Not enough variation',ha='center',va='center',color='white')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Raw Data (500 rows)")
    st.dataframe(df.head(500), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("---")

    for col_obj, name in zip(st.columns(3), results):
        col_obj.metric(name, f"AUC: {results[name]['auc']:.3f}",
                       f"Acc: {results[name]['accuracy']:.1%}")
    st.markdown("---")

    cl, cr = st.columns(2)
    with cl:
        st.subheader("ROC Curves")
        fig, ax = dark_fig()
        y_arr = np.array(y_test).ravel()
        for (nm, res), col in zip(results.items(), ['#e74c3c','#2ecc71','#5b9bd5']):
            fpr, tpr, _ = roc_curve(y_arr, np.array(res['y_prob']).ravel())
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{nm} ({res['auc']:.3f})")
        ax.plot([0,1],[0,1],'gray',ls='--',lw=1)
        ax.set_xlabel('FPR', color='white'); ax.set_ylabel('TPR', color='white')
        ax.legend(facecolor=DARK, edgecolor=SPINE, labelcolor='white', fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with cr:
        st.subheader("Feature Importance")
        fig, ax = dark_fig()
        colors_imp = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_imp)))
        ax.barh(feat_imp['feature'], feat_imp['importance'],
                color=colors_imp[::-1], edgecolor='none')
        ax.set_xlabel('Importance', color='white')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Confusion Matrix")
    sel = st.selectbox("Model", list(results.keys()))
    fig, ax = plt.subplots(figsize=(4,3.5), facecolor=DARK); ax.set_facecolor(DARK)
    sns.heatmap(results[sel]['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred No','Pred Yes'], yticklabels=['Actual No','Actual Yes'],
                linewidths=0.5, linecolor=DARK)
    ax.tick_params(colors='white')
    ax.xaxis.set_tick_params(labelcolor='white'); ax.yaxis.set_tick_params(labelcolor='white')
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK PREDICTOR  (with SHAP-like explanation + PDF/Excel download)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Risk Predictor":
    st.title("🎯 Individual Risk Predictor")
    st.markdown("Enter applicant details → get risk score + **SHAP-style explanation**.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        age                  = st.slider("Age", 18, 90, 45)
        monthly_income       = st.number_input("Monthly Income ($)", 0, 50000, 5000, step=500)
        revolving_utilization= st.slider("Revolving Utilization (0–1)", 0.0, 1.0, 0.3, 0.01)
    with c2:
        debt_ratio           = st.slider("Debt Ratio", 0.0, 2.0, 0.3, 0.01)
        open_credit_lines    = st.slider("Open Credit Lines", 0, 30, 8)
        real_estate_loans    = st.slider("Real Estate Loans", 0, 10, 1)
    with c3:
        past_due_30_59       = st.slider("30-59 Days Past Due", 0, 10, 0)
        past_due_60_89       = st.slider("60-89 Days Past Due", 0, 10, 0)
        times_90_days_late   = st.slider("90+ Days Late", 0, 10, 0)
        dependents           = st.slider("Dependents", 0, 10, 1)

    input_data = dict(
        revolving_utilization=revolving_utilization, age=age,
        past_due_30_59=past_due_30_59, debt_ratio=debt_ratio,
        monthly_income=monthly_income, open_credit_lines=open_credit_lines,
        times_90_days_late=times_90_days_late, real_estate_loans=real_estate_loans,
        past_due_60_89=past_due_60_89, dependents=dependents
    )

    if st.button("🔍 Predict & Explain", use_container_width=True):
        rf = results['Random Forest']['model']
        prob, label = predict_single(input_data, rf)
        shap_vals = shap_like_explanation(input_data, rf, feature_cols, prob)

        st.markdown("---")
        ca, cb, cc = st.columns(3)
        ca.metric("Risk Probability", f"{prob:.1%}")
        cb.metric("Risk Label",       label)
        cc.metric("Risk Tier",        "🔴 High" if prob>0.5 else "🟡 Medium" if prob>0.25 else "🟢 Low")

        bar_col = "#e74c3c" if prob>0.5 else "#f39c12" if prob>0.25 else "#2ecc71"
        st.markdown(f"""
        <div style="background:#1e2130;border-radius:8px;padding:4px;border:1px solid #2e3650;margin-bottom:1rem">
          <div style="background:{bar_col};width:{prob*100:.0f}%;height:26px;border-radius:6px;
               display:flex;align-items:center;padding-left:10px;min-width:50px">
            <span style="color:white;font-size:13px;font-weight:700">{prob:.1%}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── SHAP-like waterfall chart ─────────────────────────────────────
        st.subheader("🧠 SHAP-Style Feature Explanation")
        st.caption("Shows how much each feature INCREASES (🔴) or DECREASES (🟢) the risk from baseline.")

        fig, ax = dark_fig(9, 4)
        colors_shap = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_vals.values]
        bars = ax.barh(shap_vals.index, shap_vals.values, color=colors_shap, edgecolor='none', height=0.6)
        ax.axvline(0, color='white', lw=0.8, alpha=0.5)
        for bar, val in zip(bars, shap_vals.values):
            xpos = val + 0.001 if val >= 0 else val - 0.001
            ha   = 'left'      if val >= 0 else 'right'
            ax.text(xpos, bar.get_y()+bar.get_height()/2,
                    f"{val:+.3f}", va='center', ha=ha, color='white', fontsize=9)
        ax.set_xlabel("Feature Contribution to Risk (+ increases risk, − reduces risk)", color='white')
        red_p  = mpatches.Patch(color='#e74c3c', label='Increases Risk')
        green_p= mpatches.Patch(color='#2ecc71', label='Decreases Risk')
        ax.legend(handles=[red_p, green_p], facecolor=DARK, edgecolor=SPINE, labelcolor='white', fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        # ── Key risk factors text ─────────────────────────────────────────
        st.subheader("📋 Key Risk Factors")
        factors = []
        if revolving_utilization > 0.8: factors.append(("❌ Very high revolving utilization", "High impact"))
        if times_90_days_late    > 0:   factors.append(("❌ Has 90+ day late payments",       "High impact"))
        if past_due_30_59        > 1:   factors.append(("⚠️ Multiple 30-59 day late payments","Medium impact"))
        if debt_ratio            > 0.5: factors.append(("⚠️ High debt ratio",                "Medium impact"))
        if monthly_income        < 3000:factors.append(("⚠️ Low monthly income",             "Medium impact"))
        if revolving_utilization < 0.3: factors.append(("✅ Low revolving utilization",       "Positive"))
        if times_90_days_late == 0 and past_due_30_59 == 0:
            factors.append(("✅ No missed payments", "Positive"))
        if monthly_income        > 7000:factors.append(("✅ Good monthly income",             "Positive"))
        for f, impact in factors:
            st.markdown(f"- **{f}** — _{impact}_")

        # ── Download report ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📥 Download Report")
        dc1, dc2 = st.columns(2)

        # Excel report
        report_df = pd.DataFrame({
            'Feature':      list(input_data.keys()),
            'Value':        list(input_data.values()),
            'SHAP Contribution': [shap_vals.get(f, 0) for f in input_data.keys()]
        })
        summary_df = pd.DataFrame({
            'Metric': ['Risk Probability','Risk Label','Risk Tier'],
            'Value':  [f"{prob:.1%}", label,
                       "High" if prob>0.5 else "Medium" if prob>0.25 else "Low"]
        })
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary',  index=False)
            report_df.to_excel( writer, sheet_name='Features', index=False)
        excel_buf.seek(0)

        with dc1:
            st.download_button(
                "📊 Download Excel Report",
                data=excel_buf,
                file_name="credit_risk_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # CSV report
        full_report = pd.concat([
            summary_df.rename(columns={'Metric':'Field','Value':'Data'}),
            report_df.rename(columns={'Feature':'Field','Value':'Data'}).drop(columns=['SHAP Contribution'])
        ], ignore_index=True)
        with dc2:
            st.download_button(
                "📄 Download CSV Report",
                data=full_report.to_csv(index=False),
                file_name="credit_risk_report.csv",
                mime="text/csv",
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Prediction":
    st.title("📦 Batch Prediction")
    st.markdown("Upload a CSV with multiple applicants → download results with risk scores.")
    st.markdown("---")

    # Show required columns
    with st.expander("📋 Required CSV Columns"):
        st.code("\n".join(feature_cols))
        # Sample download
        sample = pd.DataFrame([{
            'revolving_utilization':0.3,'age':45,'past_due_30_59':0,
            'debt_ratio':0.35,'monthly_income':6000,'open_credit_lines':8,
            'times_90_days_late':0,'real_estate_loans':1,'past_due_60_89':0,'dependents':1
        },{
            'revolving_utilization':0.9,'age':35,'past_due_30_59':3,
            'debt_ratio':0.8,'monthly_income':2500,'open_credit_lines':4,
            'times_90_days_late':2,'real_estate_loans':0,'past_due_60_89':1,'dependents':3
        }])
        st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False),
                           "sample_applicants.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded:
        try:
            input_df = pd.read_csv(uploaded)
            st.success(f"✅ {len(input_df):,} records loaded!")
            st.dataframe(input_df.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                rf = results['Random Forest']['model']
                result_df, err = predict_batch(input_df, rf, feature_cols)

                if err:
                    st.error(f"❌ Error: {err}")
                else:
                    st.markdown("---")
                    st.subheader("📊 Batch Results")

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Total Applicants", f"{len(result_df):,}")
                    r2.metric("High Risk",  f"{(result_df['risk_label']=='High Risk').sum():,}")
                    r3.metric("Low Risk",   f"{(result_df['risk_label']=='Low Risk').sum():,}")

                    # Risk distribution pie
                    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor=DARK)
                    ax.set_facecolor(DARK)
                    tier_counts = result_df['risk_tier'].value_counts()
                    pie_colors  = {'🔴 High':'#e74c3c','🟡 Medium':'#f39c12','🟢 Low':'#2ecc71'}
                    clrs = [pie_colors.get(t, '#5b9bd5') for t in tier_counts.index]
                    ax.pie(tier_counts.values, labels=tier_counts.index, colors=clrs,
                           autopct='%1.1f%%', textprops={'color':'white'})
                    ax.set_title('Risk Distribution', color='white')
                    fig.tight_layout(); st.pyplot(fig); plt.close()

                    st.subheader("Results Preview")
                    st.dataframe(result_df[['risk_probability','risk_label','risk_tier'] +
                                            feature_cols[:4]].head(20), use_container_width=True)

                    # Download results
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.download_button("📄 Download CSV Results",
                            result_df.to_csv(index=False),
                            "batch_predictions.csv", "text/csv",
                            use_container_width=True)
                    with bc2:
                        excel_buf2 = io.BytesIO()
                        with pd.ExcelWriter(excel_buf2, engine='openpyxl') as writer:
                            result_df.to_excel(writer, sheet_name='Predictions', index=False)
                        excel_buf2.seek(0)
                        st.download_button("📊 Download Excel Results",
                            excel_buf2, "batch_predictions.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
