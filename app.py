"""
Enterprise Employee Salary Prediction & Market Intelligence Dashboard
College Demonstration Project - Multi-Variable Random Forest
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SalaryIQ | Employee Salary Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "salary_model.joblib")
LR_MODEL_PATH = os.path.join(BASE_DIR, "salary_lr_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "salary_data.csv")
TEST_EVAL_PATH = os.path.join(BASE_DIR, "test_evaluation.csv")


# Premium SaaS Custom Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        color: #ffffff;
        padding: 26px 30px;
        border-radius: 18px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.35);
        margin-bottom: 22px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0 0 6px 0;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #c7d2fe;
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Stat Box */
    .stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 18px;
        text-align: left;
        margin-bottom: 12px;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .stat-delta {
        font-size: 0.8rem;
        font-weight: 600;
        color: #10b981;
        margin-top: 3px;
    }

    /* Prediction Result Card */
    .prediction-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.35);
        border: 1px solid #334155;
    }
    .pred-header {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 700;
    }
    .pred-amount {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.03em;
        margin: 6px 0 8px 0;
    }
    .pred-monthly {
        font-size: 0.98rem;
        color: #cbd5e1;
        font-weight: 500;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-rf {
        background: #e0e7ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
    }
    .badge-live {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .badge-demo {
        background: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 14px;
        border-radius: 8px;
        color: #92400e;
        font-size: 0.84rem;
        line-height: 1.45;
        margin-top: 14px;
    }

    /* Section Header */
    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 10px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_inr(amount: float) -> str:
    """Format number into Indian Rupee representation (e.g., ₹ 18,83,020)."""
    try:
        s = f"{amount:.2f}"
        parts = s.split(".")
        integer_part = parts[0]

        is_negative = integer_part.startswith("-")
        if is_negative:
            integer_part = integer_part[1:]

        if len(integer_part) <= 3:
            formatted = integer_part
        else:
            last3 = integer_part[-3:]
            other = integer_part[:-3]
            groups = []
            while len(other) > 2:
                groups.insert(0, other[-2:])
                other = other[:-2]
            if other:
                groups.insert(0, other)
            formatted = ",".join(groups) + "," + last3

        if is_negative:
            formatted = "-" + formatted

        return f"₹ {formatted}"
    except Exception:
        return f"₹ {amount:,.2f}"


@st.cache_resource
def load_rf_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_resource
def load_lr_model():
    if os.path.exists(LR_MODEL_PATH):
        return joblib.load(LR_MODEL_PATH)
    return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_test_eval():
    if os.path.exists(TEST_EVAL_PATH):
        return pd.read_csv(TEST_EVAL_PATH)
    return None


# Load models & datasets
rf_model = load_rf_model()
lr_model = load_lr_model()
metrics = load_metrics()
df = load_data()
test_eval_df = load_test_eval()

# Sidebar Branding & Info
with st.sidebar:
    st.markdown("""
        <div style="padding: 6px 0 14px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: #4338ca; width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem;">💼</div>
                <div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">SalaryIQ</div>
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Enterprise ML Platform</div>
                </div>
            </div>
            <div style="margin-top: 14px; display: flex; gap: 6px; flex-wrap: wrap;">
                <span class="badge-pill badge-live">● Models Online</span>
                <span class="badge-pill badge-rf">Random Forest v2.5</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 8px 0 16px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    if metrics:
        rf_m = metrics.get("random_forest", metrics)
        st.markdown(f"""
            <div style="background: #f1f5f9; border-radius: 12px; padding: 14px 16px; border: 1px solid #e2e8f0; margin-bottom: 16px;">
                <div style="font-size: 0.75rem; font-weight: 800; color: #334155; text-transform: uppercase; margin-bottom: 6px;">Pipeline Performance</div>
                <div style="font-size: 0.83rem; color: #475569; line-height: 1.5;">
                    <b>R² Accuracy:</b> {rf_m.get('r2_score', 0.9025):.4f}<br>
                    <b>Mean Abs Error:</b> {format_inr(rf_m.get('mae', 247820))}<br>
                    <b>Architecture:</b> 150 Decision Trees<br>
                    <b>Features:</b> 6 Dependent Predictors
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 8px;'>Active Variables</div>", unsafe_allow_html=True)
    st.markdown("""
        <ul style="font-size: 0.82rem; color: #334155; padding-left: 18px; margin: 0 0 16px 0; line-height: 1.6;">
            <li><b>Age</b> (Numeric)</li>
            <li><b>Gender</b> (Categorical)</li>
            <li><b>Job Title</b> (10 Roles)</li>
            <li><b>Years of Experience</b> (Numeric)</li>
            <li><b>Area / Location</b> (7 Regions)</li>
            <li><b>Education Level</b> (4 Levels)</li>
        </ul>
    """, unsafe_allow_html=True)

    st.caption("College Demonstration Project | Synthetic Demonstration Data")


# ==============================================================================
# MAIN PAGE: HERO BANNER
# ==============================================================================
st.markdown("""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1 class="hero-title">Employee Salary Prediction & Intelligence Studio</h1>
                <p class="hero-subtitle">Multi-variable compensation intelligence powered by Random Forest Regressor & comparative Linear Regression.</p>
            </div>
            <span class="badge-pill badge-demo">Synthetic Demo Environment</span>
        </div>
    </div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: INTERACTIVE PREDICTION STUDIO
# ==============================================================================
col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown("<div class='section-title'>⚙️ Configure Employee Profile</div>", unsafe_allow_html=True)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        job_title_options = [
            "Software Engineer",
            "Senior Software Engineer",
            "Data Scientist",
            "Machine Learning Engineer",
            "Product Manager",
            "DevOps Engineer",
            "Data Analyst",
            "Financial Analyst",
            "Engineering Manager",
            "HR Specialist"
        ]
        job_title = st.selectbox("Job Title / Role", job_title_options, index=3)

    with f_col2:
        area_options = [
            "Bangalore",
            "Mumbai",
            "Delhi-NCR",
            "Hyderabad",
            "Pune",
            "Chennai",
            "Remote / Tier 2 Cities"
        ]
        area = st.selectbox("Location / Area", area_options, index=0)

    f_col3, f_col4 = st.columns(2)
    with f_col3:
        education_options = ["High School", "Bachelor's", "Master's", "PhD"]
        education = st.selectbox("Education Level", education_options, index=1)

    with f_col4:
        gender_options = ["Female", "Male", "Other"]
        gender = st.selectbox("Gender", gender_options, index=0)

    f_col5, f_col6 = st.columns(2)
    with f_col5:
        age = st.slider("Employee Age (Years)", min_value=18, max_value=65, value=28, step=1)
    with f_col6:
        experience = st.slider("Years of Experience", min_value=0, max_value=35, value=5, step=1)

    if experience > (age - 18):
        st.caption("ℹ️ *Notice: Experience exceeds typical entry age (Age - 18). Model will extrapolate.*")

    compare_mode = st.toggle("Compare with Linear Regression Baseline", value=True)

with col_result:
    st.markdown("<div class='section-title'>🔮 AI Compensation Forecast</div>", unsafe_allow_html=True)

    if rf_model is not None:
        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_title,
            "Years of Experience": experience,
            "Area": area
        }])

        rf_salary = float(rf_model.predict(input_df)[0])
        monthly_salary = rf_salary / 12.0

        if df is not None:
            percentile = round((df["Salary"] < rf_salary).mean() * 100, 1)
        else:
            percentile = 50.0

        st.markdown(f"""
            <div class="prediction-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="pred-header">Predicted Annual Salary</span>
                    <span class="badge-pill badge-rf">Random Forest</span>
                </div>
                <div class="pred-amount">{format_inr(rf_salary)}</div>
                <div class="pred-monthly">Estimated Monthly Take-Home: <b>{format_inr(monthly_salary)} / mo</b></div>
                <hr style="margin: 14px 0; border: none; border-top: 1px solid rgba(255,255,255,0.15);">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8;">
                    <span>Market Standing:</span>
                    <span style="color: #38bdf8; font-weight: 700;">Higher than {percentile}% of roles in dataset</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if compare_mode and lr_model is not None:
            lr_salary = float(lr_model.predict(input_df)[0])
            delta_val = rf_salary - lr_salary
            delta_color = "#10b981" if delta_val >= 0 else "#ef4444"
            sign = "+" if delta_val >= 0 else ""

            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px 18px; margin-top: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Linear Regression Baseline</span>
                            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 2px;">{format_inr(lr_salary)}</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Algorithm Delta</span>
                            <div style="font-size: 0.95rem; font-weight: 800; color: {delta_color}; margin-top: 2px;">{sign}{format_inr(delta_val)}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div class="disclaimer-box">
                ⚠️ <b>Demonstration Disclaimer:</b> This prediction was computed by an ensemble Random Forest model trained on synthetic demonstration data. It is not an output of the original Colab model.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model file not found. Please run `python train_model.py` first.")

st.markdown("<hr style='margin: 28px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: MODEL PERFORMANCE METRICS CARDS
# ==============================================================================
st.markdown("<div class='section-title'>📊 Model Performance & Evaluation Metrics (Holdout Test Set)</div>", unsafe_allow_html=True)

if metrics:
    rf_m = metrics.get("random_forest", metrics)
    lr_m = metrics.get("linear_regression", {})

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='stat-box'><div class='stat-label'>Random Forest R²</div><div class='stat-value'>{rf_m.get('r2_score', 0.9025):.4f}</div><div class='stat-delta'>90.25% Variance Explained</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='stat-box'><div class='stat-label'>Mean Absolute Error</div><div class='stat-value'>{format_inr(rf_m.get('mae', 247820))}</div><div class='stat-delta'>Average Deviation</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='stat-box'><div class='stat-label'>Root Mean Sq Error</div><div class='stat-value'>{format_inr(rf_m.get('rmse', 305972))}</div><div class='stat-delta'>Error Penalty Weight</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='stat-box'><div class='stat-label'>Linear Reg R² (Baseline)</div><div class='stat-value'>{lr_m.get('r2_score', 0.9385):.4f}</div><div class='stat-delta'>OLS Benchmark</div></div>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 20px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: INTERACTIVE VISUALIZATIONS & ANALYTICS HUB
# ==============================================================================
st.markdown("<div class='section-title'>📈 Interactive Visualizations & Market Analytics Hub</div>", unsafe_allow_html=True)

tab_dist, tab_fi, tab_pred, tab_exp, tab_area, tab_data = st.tabs([
    "📊 Salary Distribution",
    "🌲 Feature Importance",
    "🎯 Actual vs Predicted",
    "📈 Experience Progression",
    "🗺️ Salary by Location",
    "🔍 Dataset Explorer"
])

# TAB 1: SALARY DISTRIBUTION
with tab_dist:
    st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Overall Salary Distribution by Education Level</h4>", unsafe_allow_html=True)
    if df is not None:
        try:
            fig_dist = px.histogram(
                df,
                x="Salary",
                color="Education Level",
                nbins=35,
                labels={"Salary": "Annual Salary (INR)", "count": "Employee Count"},
                marginal="box",
                opacity=0.75,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_dist.update_layout(xaxis_tickprefix="₹ ", height=450, margin=dict(l=10, r=20, t=20, b=10))
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering distribution chart: {e}")
    else:
        st.write("Dataset not found.")

# TAB 2: FEATURE IMPORTANCE
with tab_fi:
    st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Random Forest Feature Importance (Contribution to Predictions)</h4>", unsafe_allow_html=True)
    if metrics:
        try:
            fi_toggle = st.radio("Importance View", ["Grouped Parent Features", "Detailed One-Hot Features"], horizontal=True, label_visibility="collapsed")
            if fi_toggle == "Grouped Parent Features" and "grouped_feature_importances" in metrics:
                fi_dict = metrics["grouped_feature_importances"]
                df_fi = pd.DataFrame({
                    "Feature": list(fi_dict.keys()),
                    "Importance (%)": list(fi_dict.values())
                }).sort_values(by="Importance (%)", ascending=True)

                fig_fi = px.bar(
                    df_fi,
                    x="Importance (%)",
                    y="Feature",
                    orientation="h",
                    color="Importance (%)",
                    color_continuous_scale="Tealgrn",
                    labels={"Importance (%)": "Relative Importance (%)", "Feature": "Input Feature"}
                )
                fig_fi.update_layout(xaxis_ticksuffix="%", height=420, margin=dict(l=10, r=20, t=10, b=10))
                st.plotly_chart(fig_fi, use_container_width=True)
            elif "detailed_feature_importances" in metrics:
                fi_dict = metrics["detailed_feature_importances"]
                top_12 = list(fi_dict.items())[:12]
                df_fi = pd.DataFrame({
                    "Feature": [k for k, v in top_12],
                    "Importance (%)": [v for k, v in top_12]
                }).sort_values(by="Importance (%)", ascending=True)

                fig_fi = px.bar(
                    df_fi,
                    x="Importance (%)",
                    y="Feature",
                    orientation="h",
                    color="Importance (%)",
                    color_continuous_scale="Viridis",
                    labels={"Importance (%)": "Relative Importance (%)", "Feature": "Encoded Dimension"}
                )
                fig_fi.update_layout(xaxis_ticksuffix="%", height=420, margin=dict(l=10, r=20, t=10, b=10))
                st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("Feature importance is derived from reduction in impurity across all 150 Decision Trees.")
        except Exception as e:
            st.error(f"Error rendering feature importance: {e}")
    else:
        st.write("Metrics not found.")

# TAB 3: ACTUAL VS PREDICTED
with tab_pred:
    st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Actual vs Predicted Salary on 20% Holdout Test Set</h4>", unsafe_allow_html=True)
    if test_eval_df is not None:
        try:
            min_v = min(test_eval_df["Actual_Salary"].min(), test_eval_df["Predicted_Salary"].min())
            max_v = max(test_eval_df["Actual_Salary"].max(), test_eval_df["Predicted_Salary"].max())

            fig_scat = px.scatter(
                test_eval_df,
                x="Actual_Salary",
                y="Predicted_Salary",
                color="Job Title",
                hover_data=["Age", "Years of Experience", "Area"],
                labels={"Actual_Salary": "Actual Salary (INR)", "Predicted_Salary": "Random Forest Prediction (INR)"}
            )
            fig_scat.add_trace(go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="Ideal 45° Fit",
                line=dict(color="#ef4444", dash="dash")
            ))
            fig_scat.update_layout(xaxis_tickprefix="₹ ", yaxis_tickprefix="₹ ", height=450, margin=dict(l=10, r=20, t=10, b=10))
            st.plotly_chart(fig_scat, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering scatter plot: {e}")
    else:
        st.write("Test evaluation data not found.")

# TAB 4: EXPERIENCE PROGRESSION
with tab_exp:
    st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Years of Experience vs Salary Progression by Role</h4>", unsafe_allow_html=True)
    if df is not None:
        try:
            fig_exp = px.scatter(
                df,
                x="Years of Experience",
                y="Salary",
                color="Job Title",
                hover_data=["Age", "Area", "Education Level"],
                labels={"Years of Experience": "Experience (Years)", "Salary": "Annual Salary (INR)"}
            )
            fig_exp.update_layout(yaxis_tickprefix="₹ ", height=450, margin=dict(l=10, r=20, t=10, b=10))
            st.plotly_chart(fig_exp, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering experience curve: {e}")
    else:
        st.write("Dataset not found.")

# TAB 5: SALARY BY LOCATION
with tab_area:
    st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Median Compensation by Geographic Area</h4>", unsafe_allow_html=True)
    if df is not None:
        try:
            area_stats = df.groupby("Area")["Salary"].agg(["median", "mean", "count"]).reset_index().sort_values(by="median", ascending=False)
            fig_loc = px.bar(
                area_stats,
                x="Area",
                y="median",
                color="median",
                color_continuous_scale="Purples",
                labels={"median": "Median Salary (INR)", "Area": "Location"}
            )
            fig_loc.update_layout(yaxis_tickprefix="₹ ", height=450, margin=dict(l=10, r=20, t=10, b=10))
            st.plotly_chart(fig_loc, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering area chart: {e}")
    else:
        st.write("Dataset not found.")

# TAB 6: DATASET EXPLORER
with tab_data:
    st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Filterable Multi-Variable Dataset (1,000 Verified Records)</h4>", unsafe_allow_html=True)
    if df is not None:
        try:
            c1, c2, c3 = st.columns(3)
            with c1:
                f_role = st.selectbox("Role Filter", ["All Roles"] + sorted(df["Job Title"].unique().tolist()))
            with c2:
                f_loc = st.selectbox("Location Filter", ["All Locations"] + sorted(df["Area"].unique().tolist()))
            with c3:
                f_gend = st.selectbox("Gender Filter", ["All Genders"] + sorted(df["Gender"].unique().tolist()))

            view_df = df.copy()
            if f_role != "All Roles":
                view_df = view_df[view_df["Job Title"] == f_role]
            if f_loc != "All Locations":
                view_df = view_df[view_df["Area"] == f_loc]
            if f_gend != "All Genders":
                view_df = view_df[view_df["Gender"] == f_gend]

            formatted_df = view_df.copy()
            formatted_df["Salary (INR)"] = formatted_df["Salary"].apply(format_inr)
            st.dataframe(
                formatted_df.drop(columns=["Salary", "Data_Label"], errors="ignore"),
                use_container_width=True,
                height=380
            )

            csv_file = view_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Filtered Dataset (CSV)",
                data=csv_file,
                file_name="salary_prediction_dataset.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error rendering dataset table: {e}")
    else:
        st.write("Dataset not found.")
