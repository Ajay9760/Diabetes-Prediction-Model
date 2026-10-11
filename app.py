import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Diabetes Risk Intelligence Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling metrics and callouts
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #1f77b4; font-weight: 700; }
    .sub-header { font-size: 1.2rem; color: #555555; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4; }
    </style>
""", unsafe_allow_html=True)

# 2. Load Model, Scaler, and Dataset (for visualizations)
@st.cache_resource
def load_assets():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('diabetes (1).csv')
    return model, scaler, df

try:
    model, scaler, df = load_assets()
except Exception as e:
    st.error(f"Error loading model assets. Make sure 'diabetes_model.pkl', 'scaler.pkl', and 'diabetes (1).csv' are in the directory. Details: {e}")
    st.stop()

# 3. Sidebar Configuration for Inputs
st.sidebar.markdown("## ⚙️ Patient Parameters")
st.sidebar.markdown("Adjust the medical values below to simulate a patient prediction:")

pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 20)
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 0, 846, 79)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 0.0, 67.1, 25.0, 0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, 0.01)
age = st.sidebar.slider("Age (Years)", 21, 81, 30)

# 4. Main App Layout & Tabs
st.markdown('<p class="main-header">🩺 Diabetes Prediction & Model Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive web application built with Streamlit and Scikit-Learn Logistic Regression.</p>', unsafe_allow_html=True)
st.divider()

tab1, tab2 = st.tabs(["🔮 Real-Time Prediction", "📊 Model Evaluation & Insights"])

with tab1:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📋 Selected Patient Profile Summary")
        input_df = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        st.dataframe(input_df, use_container_width=True)
        
        predict_btn = st.button("🚀 Run Risk Prediction", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎯 Prediction Output")
        if predict_btn:
            # Scale input data
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction[0] == 1:
                st.error("⚠️ **High Risk Status Detected**")
                st.markdown(f"The model predicts the patient is **at high risk for diabetes** with a confidence score of **{probability*100:.2f}%**.")
            else:
                st.success("✅ **Low Risk Status Detected**")
                st.markdown(f"The model predicts the patient is **unlikely to have diabetes** (Probability: **{probability*100:.2f}%**).")
        else:
            st.info("👈 Adjust parameters in the sidebar and click **Run Risk Prediction** to see results.")

with tab2:
    st.subheader("📈 Model Performance & Evaluation Metrics")
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric(label="Test Accuracy", value="73.38%")
    with mcol2:
        st.metric(label="ROC-AUC Score", value="0.79")
    with mcol3:
        st.metric(label="Precision (Class 1)", value="62%")
    with mcol4:
        st.metric(label="Recall (Class 1)", value="57%")
        
    st.divider()
    
    # Visualizations Row
    vcol1, vcol2 = st.columns(2)
    
    with vcol1:
        st.markdown("### 🔍 Feature Importance Coefficients")
        st.markdown("Shows how heavily each feature drives positive or negative prediction outcomes.")
        coef_df = pd.DataFrame({
            'Feature': input_df.columns,
            'Coefficient': model.coef_[0]
        }).sort_values(by='Coefficient', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=['skyblue' if c > 0 else 'salmon' for c in coef_df['Coefficient']])
        ax.set_xlabel("Coefficient Value")
        ax.set_title("Logistic Regression Coefficients")
        st.pyplot(fig)

    with vcol2:
        st.markdown("### 📉 Confusion Matrix Breakdown")
        st.markdown("Performance distribution showing correct vs incorrect classifications on test data.")
        cm = np.array([[83, 18], [23, 30]])
        fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        ax_cm.set_ylabel("Actual Label")
        ax_cm.set_title("Confusion Matrix Heatmap")
        st.pyplot(fig_cm)