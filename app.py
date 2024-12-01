import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(
    page_title="Mental Health Depression Predictor",
    page_icon="🧠",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'mental_health_classifier.joblib')
    return joblib.load(model_path)

def main():
    st.title("🧠 Mental Health Depression Predictor")
    st.write("This app predicts the likelihood of depression based on various factors.")
    
    # Personal Information
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=15, max_value=100, value=25)
        city = st.text_input("City")
    
    with col2:
        status = st.selectbox(
            "Working Professional or Student",
            ["Working Professional", "Student"]
        )
        profession = st.text_input("Profession (or Student if studying)")
        degree = st.selectbox("Degree", ["BBA", "BHM", "B.Pharm", "LLB"])

    # Academic/Work Pressures
    st.subheader("Academic/Work Information")
    col3, col4 = st.columns(2)
    
    with col3:
        academic_pressure = st.slider(
            "Academic Pressure (0-5)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            help="Rate your academic pressure (0 if working professional)"
        )
        work_pressure = st.slider(
            "Work Pressure (0-5)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            help="Rate your work pressure (0 if student)"
        )
        cgpa = st.number_input("CGPA/Performance Score", min_value=0.0, max_value=10.0, value=7.0)
        
    with col4:
        study_satisfaction = st.slider(
            "Study Satisfaction (1-5)",
            min_value=1.0,
            max_value=5.0,
            value=3.0
        )
        job_satisfaction = st.slider(
            "Job Satisfaction (1-5)",
            min_value=1.0,
            max_value=5.0,
            value=3.0
        )
        work_study_hours = st.number_input(
            "Daily Work/Study Hours",
            min_value=1.0,
            max_value=24.0,
            value=8.0
        )

    # Health and Lifestyle
    st.subheader("Health and Lifestyle")
    col5, col6 = st.columns(2)
    
    with col5:
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"]
        )
        dietary_habits = st.selectbox(
            "Dietary Habits",
            ["Healthy", "Moderate", "Unhealthy"]
        )
        
    with col6:
        financial_stress = st.slider(
            "Financial Stress Level (1-5)",
            min_value=1.0,
            max_value=5.0,
            value=1.0
        )
        family_history = st.selectbox(
            "Family History of Mental Illness",
            ["Yes", "No"]
        )
        suicidal_thoughts = st.selectbox(
            "Have you ever had suicidal thoughts?",
            ["Yes", "No"]
        )

    # Prediction button
    if st.button("Predict", type="primary"):
        # Create features dictionary
        features = {
            'Gender': gender,
            'Age': age,
            'Working Professional or Student': status,
            'Academic Pressure': academic_pressure,
            'Work Pressure': work_pressure,
            'CGPA': cgpa,
            'Study Satisfaction': study_satisfaction,
            'Job Satisfaction': job_satisfaction,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Degree': degree,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Work/Study Hours': work_study_hours,
            'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_history
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Load model and make prediction
        model = load_model()
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        # Show prediction
        st.divider()
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric(
                label="Depression Risk",
                value="High Risk" if prediction == 1 else "Low Risk"
            )
        
        with col_result2:
            st.metric(
                label="Risk Probability",
                value=f"{probability:.1%}"
            )
        
        # Additional information and warnings
        if prediction == 1:
            st.warning("⚠️ Based on the provided information, you may be at risk for depression.")
            st.info("🏥 Please consider consulting with a mental health professional for proper evaluation and support.")
        else:
            st.success("✅ Based on the provided information, you appear to be at lower risk for depression.")
            
        st.info("ℹ️ Disclaimer: This is a screening tool and not a medical diagnosis. Always consult with qualified healthcare professionals for proper evaluation and support.")

if __name__ == "__main__":
    main() 