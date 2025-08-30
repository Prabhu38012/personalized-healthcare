import streamlit as st

class PatientInputForm:
    def __init__(self):
        self.patient_data = {}
    
    def render(self):
        """Render the patient input form"""
        
        # Basic Information
        st.markdown("### Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=45)
            sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        
        with col2:
            smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Smoker")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            family_history = st.selectbox("Family History of Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Clinical Measurements
        st.markdown("### Clinical Measurements")
        col3, col4 = st.columns(2)
        
        with col3:
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
            cholesterol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=500, value=200)
            max_heart_rate = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
        
        with col4:
            fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        # Categorical Clinical Data
        st.markdown("### Clinical Assessment")
        col5, col6 = st.columns(2)
        
        with col5:
            chest_pain_type = st.selectbox("Chest Pain Type", 
                                         ["typical", "atypical", "non_anginal", "asymptomatic"],
                                         format_func=lambda x: x.replace("_", " ").title())
            resting_ecg = st.selectbox("Resting ECG", 
                                     ["normal", "abnormal", "hypertrophy"],
                                     format_func=lambda x: x.title())
        
        with col6:
            slope = st.selectbox("ST Slope", 
                               ["upsloping", "flat", "downsloping"],
                               format_func=lambda x: x.title())
            ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", 
                              ["normal", "fixed", "reversible"],
                              format_func=lambda x: x.title())
        
        # Compile patient data
        self.patient_data = {
            "age": age,
            "sex": sex,
            "chest_pain_type": chest_pain_type,
            "resting_bp": resting_bp,
            "cholesterol": cholesterol,
            "fasting_blood_sugar": fasting_blood_sugar,
            "resting_ecg": resting_ecg,
            "max_heart_rate": max_heart_rate,
            "exercise_angina": exercise_angina,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "bmi": bmi,
            "smoking": smoking,
            "diabetes": diabetes,
            "family_history": family_history
        }
        
        return self.patient_data
    
    def get_sample_data(self):
        """Get sample patient data for testing"""
        return {
            "age": 45,
            "sex": "M",
            "chest_pain_type": "typical",
            "resting_bp": 130,
            "cholesterol": 200,
            "fasting_blood_sugar": 0,
            "resting_ecg": "normal",
            "max_heart_rate": 150,
            "exercise_angina": 0,
            "oldpeak": 1.0,
            "slope": "upsloping",
            "ca": 0,
            "thal": "normal",
            "bmi": 25.0,
            "smoking": 0,
            "diabetes": 0,
            "family_history": 0
        }
