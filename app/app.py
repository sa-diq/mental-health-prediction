import streamlit as st
import pandas as pd
import joblib
import os
import catboost

# Page config
st.set_page_config(
    page_title="Mental Health Depression Predictor",
    page_icon="🧠",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        # List all possible paths
        possible_paths = [
            os.path.join('models', 'mental_health_classifier_v1.joblib'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'mental_health_classifier_v1.joblib'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mental_health_classifier_v1.joblib'),
            '/mount/src/mental-health-prediction/models/mental_health_classifier_v1.joblib'
        ]
        
        print("Debugging information:")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir()}")
        print(f"Parent directory contents: {os.listdir('..')}")
        print("\nTrying these paths:")
        for path in possible_paths:
            print(f"Checking {path}: {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                return joblib.load(path)
                
        raise FileNotFoundError(f"Model file not found in any of the expected locations")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Error loading model: {str(e)}")
        raise

def main():
    st.title("🧠 Mental Health Depression Predictor")
    st.write("This app predicts whether someone might be experiencing depression based on various factors.")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=15, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        city = st.text_input("City")
        
    with col2:
        st.subheader("Professional Information")
        status = st.selectbox("Status", ["Working Professional", "Student"])
        profession = st.text_input("Profession (if Working Professional)")
        degree = st.selectbox("Degree", ["Undergraduate", "BSc", "MSc", "Ph.D", "Other"])
        work_study_hours = st.number_input("Work/Study Hours per day", min_value=0, max_value=24, value=8)

    st.subheader("Health & Lifestyle")
    col3, col4 = st.columns(2)
    
    with col3:
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["Less than 5 hours", "5-6 hours", "6-8 hours", "More than 8 hours"]
        )
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])

    with col4:
        if status == "Working Professional":
            work_pressure = st.slider("Work Pressure (low(1) to high(5))", 1, 5, 3)
            job_satisfaction = st.slider("Job Satisfaction (low(1) to high(5))", 1, 5, 3)
            academic_pressure = None
            study_satisfaction = None
            cgpa = None
        else:
            academic_pressure = st.slider("Academic Pressure (low(1) to high(5))", 1, 5, 3)
            study_satisfaction = st.slider("Study Satisfaction (low(1) to high(5))", 1, 5, 3)
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)
            work_pressure = None
            job_satisfaction = None
        
        financial_stress = st.slider("Financial Stress Level (low(1) to high(5))", 1, 5, 3)

    # Prediction button
    if st.button("Predict", type="primary"):
        # Create features dictionary
        features = {
            'Gender': 1 if gender == "Male" else 0,
            'Age': age,
            'City': hash(city) % 10000,  # Convert city to numeric hash
            'Working Professional or Student': 1 if status == "Working Professional" else 0,
            'Profession': hash(profession) % 10000 if profession else 0,  # Convert profession to numeric hash
            'Academic Pressure': float(academic_pressure) if academic_pressure is not None else 0.0,
            'Work Pressure': float(work_pressure) if work_pressure is not None else 0.0,
            'CGPA': float(cgpa) if cgpa is not None else 0.0,
            'Study Satisfaction': float(study_satisfaction) if study_satisfaction is not None else 0.0,
            'Job Satisfaction': float(job_satisfaction) if job_satisfaction is not None else 0.0,
            'Sleep Duration': {
                'Less than 5 hours': 0,
                '5-6 hours': 1,
                '6-8 hours': 2,
                'More than 8 hours': 3
            }[sleep_duration],
            'Dietary Habits': {
                'Healthy': 0,
                'Moderate': 1,
                'Unhealthy': 2
            }[dietary_habits],
            'Degree': {
                'Undergraduate': 0,
                'BSc': 1,
                'MSc': 2,
                'PhD': 3,
                'Other': 4
            }[degree],
            'Have you ever had suicidal thoughts ?': 1 if suicidal_thoughts == "Yes" else 0,
            'Work/Study Hours': float(work_study_hours),
            'Financial Stress': float(financial_stress),
            'Family History of Mental Illness': 1 if family_history == "Yes" else 0
        }
        
        # Create DataFrame and ensure all values are numeric
        df = pd.DataFrame([features])
        df = df.apply(pd.to_numeric, errors='coerce')

        # Fill any NaN values with 0
        df = df.fillna(0)
        
        # Load model and make prediction
        model = load_model()
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        # Show prediction
        st.divider()
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric(
                label="Prediction",
                value="Depression Indicated" if prediction == 1 else "No Depression Indicated"
            )
        
        with col_result2:
            st.metric(
                label="Probability",
                value=f"{probability:.1%}"
            )
        
        # Additional information
        if prediction == 1:
            st.warning("Based on the provided information, you may be experiencing symptoms of depression. Please consider consulting a mental health professional.")
        else:
            st.info("Based on the provided information, you may not be experiencing depression at this time. However, always consult with a healthcare professional if you have concerns.")
    
    # Add footer with creator info
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: grey;'>
        Created by: Sadiq Balogun ❤️<br>
        Contact: <a href='mailto:ballosadiq@gmail.com'>ballosadiq@gmail.com</a> | 
        <a href='https://github.com/sa-diq/mental-health-prediction'>GitHub</a> | 
        <a href='https://linkedin.com/in/sadiq-balogun'>LinkedIn</a> 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 