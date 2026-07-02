import streamlit as st
import pandas as pd
import joblib
import os

# ---------- Load model artifacts ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Model")

model = joblib.load(os.path.join(MODEL_DIR, "attrition_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

# ---------- Page config ----------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E2761;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0D9488;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.4rem;
    }
    div.stButton > button {
        background-color: #1E2761;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #0D9488;
        color: white;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .risk-high {
        background-color: #FDF2F0;
        border: 2px solid #C0392B;
    }
    .risk-low {
        background-color: #F0F9F4;
        border: 2px solid #1E8449;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<p class="main-header">📊 Employee Attrition Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter employee details to assess attrition risk in real time.</p>', unsafe_allow_html=True)

st.divider()

# ---------- Input form ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-title">👤 Personal & Role</p>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                          "Manufacturing Director", "Healthcare Representative", "Manager",
                                          "Sales Representative", "Research Director", "Human Resources"])
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing",
                                                          "Technical Degree", "Human Resources", "Other"])
    education = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)

with col2:
    st.markdown('<p class="section-title">💰 Compensation & Work</p>', unsafe_allow_html=True)
    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100)
    daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
    hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
    monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000)
    percent_salary_hike = st.slider("Percent Salary Hike", 10, 25, 15)
    stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
    overtime = st.selectbox("Works OverTime?", ["Yes", "No"])
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

with col3:
    st.markdown('<p class="section-title">📈 Tenure & Satisfaction</p>', unsafe_allow_html=True)
    total_working_years = st.slider("Total Working Years", 0, 40, 8)
    years_at_company = st.slider("Years At Company", 0, 40, 5)
    years_in_current_role = st.slider("Years In Current Role", 0, 20, 3)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    years_with_curr_manager = st.slider("Years With Current Manager", 0, 20, 3)
    num_companies_worked = st.slider("Num Companies Worked", 0, 10, 2)
    distance_from_home = st.slider("Distance From Home (km)", 1, 30, 10)
    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)

st.divider()

with st.expander("⚙️ Satisfaction & Performance Scores (optional — defaults are neutral)"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
    with c2:
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    with c3:
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
        performance_rating = st.slider("Performance Rating", 1, 4, 3)
    with c4:
        job_level = st.slider("Job Level", 1, 5, 2)

st.write("")
predict_clicked = st.button("🔍 Predict Attrition Risk")

# ---------- Prediction ----------
if predict_clicked:
    tenure_ratio = years_at_company / (total_working_years + 1)

    input_dict = {
        "Age": age, "DailyRate": daily_rate, "DistanceFromHome": distance_from_home,
        "Education": education, "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate, "JobInvolvement": job_involvement, "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction, "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike, "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction, "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years, "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role, "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager, "TenureRatio": tenure_ratio,
        "BusinessTravel_Travel_Frequently": 1 if business_travel == "Travel_Frequently" else 0,
        "BusinessTravel_Travel_Rarely": 1 if business_travel == "Travel_Rarely" else 0,
        "Department_Research & Development": 1 if department == "Research & Development" else 0,
        "Department_Sales": 1 if department == "Sales" else 0,
        "EducationField_Life Sciences": 1 if education_field == "Life Sciences" else 0,
        "EducationField_Marketing": 1 if education_field == "Marketing" else 0,
        "EducationField_Medical": 1 if education_field == "Medical" else 0,
        "EducationField_Other": 1 if education_field == "Other" else 0,
        "EducationField_Technical Degree": 1 if education_field == "Technical Degree" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "JobRole_Human Resources": 1 if job_role == "Human Resources" else 0,
        "JobRole_Laboratory Technician": 1 if job_role == "Laboratory Technician" else 0,
        "JobRole_Manager": 1 if job_role == "Manager" else 0,
        "JobRole_Manufacturing Director": 1 if job_role == "Manufacturing Director" else 0,
        "JobRole_Research Director": 1 if job_role == "Research Director" else 0,
        "JobRole_Research Scientist": 1 if job_role == "Research Scientist" else 0,
        "JobRole_Sales Executive": 1 if job_role == "Sales Executive" else 0,
        "JobRole_Sales Representative": 1 if job_role == "Sales Representative" else 0,
        "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
        "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
        "OverTime_Yes": 1 if overtime == "Yes" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.write("")
    result_col, gauge_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="risk-card risk-high">
                <h2 style="color:#C0392B; margin:0;">⚠️ High Attrition Risk</h2>
                <p style="font-size:1.3rem; margin:0.5rem 0 0 0;">Probability: <b>{probability:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-card risk-low">
                <h2 style="color:#1E8449; margin:0;">✅ Low Attrition Risk</h2>
                <p style="font-size:1.3rem; margin:0.5rem 0 0 0;">Probability: <b>{probability:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)

    with gauge_col:
        st.markdown("**Risk Meter**")
        st.progress(float(probability))
        st.caption(f"{probability:.1%} estimated probability of leaving")

st.write("")
st.divider()
st.caption("Employee Attrition Prediction System · Built with XGBoost & Streamlit · AIML Summer Internship 2026")