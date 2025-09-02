import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.forms import PatientInputForm
from pages.dashboard import create_dashboard, calculate_risk_score
from utils.api_client import HealthcareAPI
import json
import time
import os
from datetime import datetime

def load_css():
    """Load custom CSS styles"""
    css_file = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline styles for essential components
        st.markdown("""
        <style>
            .card {
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            }
            .status-success { background-color: #f0fff4; color: #38a169; }
            .status-error { background-color: #fff5f5; color: #e53e3e; }
        </style>
        """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Initialize API client
api_client = HealthcareAPI()



def display_risk_assessment(prediction, patient_data=None):
    """Display risk assessment results with enhanced visualization"""
    risk_level = prediction.get('risk_level', 'Low')
    risk_prob = prediction.get('risk_probability', 0)
    
    # Use patient_data from session state if available
    if patient_data is None:
        patient_data = st.session_state.get('patient_data', {})
    
    # Create a container for the risk assessment
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Risk level display with icon and color coding
        if risk_level.lower() == "high":
            st.markdown("""
            <div style='background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%); 
                        border-left: 4px solid #e53e3e; 
                        padding: 1.5rem; 
                        border-radius: 8px; 
                        margin-bottom: 1.5rem;'>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <div style='background-color: #fc8181; width: 50px; height: 50px; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center; margin-right: 1rem;'>
                        <span style='font-size: 24px; color: white;'>⚠️</span>
                    </div>
                    <div>
                        <h3 style='margin: 0; color: #c53030;'>High Risk</h3>
                        <p style='margin: 0.25rem 0 0; color: #e53e3e;'>{:.1%} probability</p>
                    </div>
                </div>
                <p style='margin: 0; line-height: 1.6;'>
                    This patient has a <strong>high risk</strong> of cardiovascular disease. 
                    <strong>Immediate medical attention</strong> is strongly recommended. 
                    Please consult with a healthcare provider as soon as possible.
                </p>
            </div>
            """.format(risk_prob), unsafe_allow_html=True)
            
        elif risk_level.lower() == "medium":
            st.markdown("""
            <div style='background: linear-gradient(135deg, #fffaf0 0%, #fff5eb 100%); 
                        border-left: 4px solid #dd6b20; 
                        padding: 1.5rem; 
                        border-radius: 8px; 
                        margin-bottom: 1.5rem;'>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <div style='background-color: #f6ad55; width: 50px; height: 50px; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center; margin-right: 1rem;'>
                        <span style='font-size: 24px; color: white;'>🔍</span>
                    </div>
                    <div>
                        <h3 style='margin: 0; color: #9c4221;'>Moderate Risk</h3>
                        <p style='margin: 0.25rem 0 0; color: #dd6b20;'>{:.1%} probability</p>
                    </div>
                </div>
                <p style='margin: 0; line-height: 1.6;'>
                    This patient has a <strong>moderate risk</strong> of cardiovascular disease. 
                    <strong>Lifestyle changes</strong> and <strong>regular monitoring</strong> are recommended. 
                    Consider scheduling a follow-up appointment.
                </p>
            </div>
            """.format(risk_prob), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%); 
                        border-left: 4px solid #38a169; 
                        padding: 1.5rem; 
                        border-radius: 8px; 
                        margin-bottom: 1.5rem;'>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <div style='background-color: #68d391; width: 50px; height: 50px; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center; margin-right: 1rem;'>
                        <span style='font-size: 24px; color: white;'>✅</span>
                    </div>
                    <div>
                        <h3 style='margin: 0; color: #2f855a;'>Low Risk</h3>
                        <p style='margin: 0.25rem 0 0; color: #38a169;'>{:.1%} probability</p>
                    </div>
                </div>
                <p style='margin: 0; line-height: 1.6;'>
                    This patient has a <strong>low risk</strong> of cardiovascular disease. 
                    <strong>Maintain healthy habits</strong> and <strong>regular check-ups</strong> are recommended 
                    to sustain good cardiovascular health.
                </p>
            </div>
            """.format(risk_prob), unsafe_allow_html=True)
        
        # Key metrics in a grid
        st.markdown("### 📊 Health Metrics Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = patient_data.get('age', 'N/A')
            st.markdown("<div class='card' style='text-align: center; padding: 1rem;'>"
                       f"<div style='font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem;'>Age</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 600; color: #2d3748;'>{age}</div>"
                       "</div>", unsafe_allow_html=True)
        
        with col2:
            bp = patient_data.get('resting_bp', 0)
            bp_status = "Normal"
            bp_color = "#38a169"
            if bp >= 140:
                bp_status = "High"
                bp_color = "#e53e3e"
            elif bp >= 130:
                bp_status = "Elevated"
                bp_color = "#dd6b20"
                
            st.markdown(f"<div class='card' style='text-align: center; padding: 1rem;'>"
                       f"<div style='font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem;'>Blood Pressure</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 600; color: {bp_color};'>{bp} mmHg</div>"
                       f"<div style='font-size: 0.8rem; color: {bp_color};'>{bp_status}</div>"
                       "</div>", unsafe_allow_html=True)
        
        with col3:
            chol = patient_data.get('cholesterol', 0)
            chol_status = "Normal"
            chol_color = "#38a169"
            if chol >= 240:
                chol_status = "High"
                chol_color = "#e53e3e"
            elif chol >= 200:
                chol_status = "Borderline High"
                chol_color = "#dd6b20"
                
            st.markdown(f"<div class='card' style='text-align: center; padding: 1rem;'>"
                       f"<div style='font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem;'>Cholesterol</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 600; color: {chol_color};'>{chol} mg/dL</div>"
                       f"<div style='font-size: 0.8rem; color: {chol_color};'>{chol_status}</div>"
                       "</div>", unsafe_allow_html=True)
        
        # Additional metrics row
        col4, col5, col6 = st.columns(3)
        
        with col4:
            bmi = patient_data.get('bmi', 0)
            bmi_status = "Normal"
            bmi_color = "#38a169"
            if bmi >= 30:
                bmi_status = "Obese"
                bmi_color = "#e53e3e"
            elif bmi >= 25:
                bmi_status = "Overweight"
                bmi_color = "#dd6b20"
            elif bmi < 18.5:
                bmi_status = "Underweight"
                bmi_color = "#3182ce"
                
            st.markdown(f"<div class='card' style='text-align: center; padding: 1rem;'>"
                       f"<div style='font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem;'>BMI</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 600; color: {bmi_color};'>{bmi:.1f}</div>"
                       f"<div style='font-size: 0.8rem; color: {bmi_color};'>{bmi_status}</div>"
                       "</div>", unsafe_allow_html=True)
        
        with col5:
            # Use the original glucose value for display
            glucose = patient_data.get('fasting_glucose_value', patient_data.get('fasting_blood_sugar', 0))
            # If it's still 0 or 1, use a reasonable default
            if glucose in [0, 1]:
                glucose = 90 if glucose == 0 else 130
            glucose_status = "Normal"
            glucose_color = "#38a169"
            if glucose >= 126:
                glucose_status = "High"
                glucose_color = "#e53e3e"
            elif glucose >= 100:
                glucose_status = "Prediabetes"
                glucose_color = "#dd6b20"
                
            st.markdown(f"<div class='card' style='text-align: center; padding: 1rem;'>"
                       f"<div style='font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem;'>Fasting Glucose</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 600; color: {glucose_color};'>{glucose} mg/dL</div>"
                       f"<div style='font-size: 0.8rem; color: {glucose_color};'>{glucose_status}</div>"
                       "</div>", unsafe_allow_html=True)
        
        with col6:
            hdl = patient_data.get('hdl_cholesterol', 0)
            hdl_status = "Optimal"
            hdl_color = "#38a169"
            if hdl < 40:
                hdl_status = "Low"
                hdl_color = "#e53e3e"
            elif hdl < 60:
                hdl_status = "Borderline"
                hdl_color = "#dd6b20"
                
            st.markdown(f"<div class='card' style='text-align: center; padding: 1rem;'>"
                       f"<div style='font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem;'>HDL Cholesterol</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 600; color: {hdl_color};'>{hdl} mg/dL</div>"
                       f"<div style='font-size: 0.8rem; color: {hdl_color};'>{hdl_status}</div>"
                       "</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close card container

def display_recommendations(prediction):
    """Display personalized recommendations with enhanced UI"""
    with st.container():
        st.markdown("### 💡 Personalized Recommendations")
        st.markdown("<div class='card' style='padding: 1.5rem;'>", unsafe_allow_html=True)
        
        # Get recommendations from prediction
        recommendations = prediction.get('recommendations', [
            "No specific recommendations available. Please ensure all health metrics are provided for personalized advice."
        ])
        
        # Display each recommendation with a nice card and icon
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div style='background: #f8fafc; 
                        border-left: 4px solid #4299e1; 
                        padding: 1rem; 
                        margin-bottom: 0.75rem; 
                        border-radius: 0.5rem;
                        display: flex;
                        align-items: flex-start;'>
                <div style='background: #ebf8ff; 
                            width: 32px; 
                            height: 32px; 
                            border-radius: 50%; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            margin-right: 1rem; 
                            flex-shrink: 0;'>
                    <span style='color: #3182ce;'>🔍</span>
                </div>
                <div>
                    <div style='font-weight: 500; color: #2d3748; margin-bottom: 0.25rem;'>
                        Recommendation {i}
                    </div>
                    <div style='color: #4a5568; line-height: 1.5;'>{rec}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close card container

def display_risk_factors(prediction):
    """Display identified risk factors with enhanced UI"""
    with st.container():
        st.markdown("### ⚠️ Identified Risk Factors")
        st.markdown("<div class='card' style='padding: 1.5rem;'>", unsafe_allow_html=True)
        
        # Get risk factors from prediction
        risk_factors = prediction.get('risk_factors', [
            "No specific risk factors identified. This could be due to incomplete data."
        ])
        
        # Define risk levels and corresponding colors/icons
        risk_levels = {
            'high': {'color': '#e53e3e', 'icon': '⚠️'},
            'medium': {'color': '#dd6b20', 'icon': '🔍'},
            'low': {'color': '#3182ce', 'icon': 'ℹ️'}
        }
        
        # Display each risk factor with appropriate styling
        for i, factor in enumerate(risk_factors, 1):
            # Determine risk level (default to medium if not specified)
            risk_level = 'medium'
            if 'high' in factor.lower():
                risk_level = 'high'
            elif 'low' in factor.lower():
                risk_level = 'low'
                
            level = risk_levels[risk_level]
            
            st.markdown(f"""
            <div style='background: #fff5f5;
                        border-left: 4px solid {level['color']}; 
                        padding: 1rem; 
                        margin-bottom: 0.75rem; 
                        border-radius: 0.5rem;
                        display: flex;
                        align-items: flex-start;'>
                <div style='background: #fff5f5; 
                            color: {level['color']};
                            width: 32px; 
                            height: 32px; 
                            border-radius: 50%; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            margin-right: 1rem; 
                            flex-shrink: 0;'>
                    {level['icon']}
                </div>
                <div>
                    <div style='font-weight: 500; color: #2d3748; margin-bottom: 0.25rem;'>
                        Risk Factor {i}: <span style='color: {level['color']}; text-transform: capitalize;'>{risk_level} Risk</span>
                    </div>
                    <div style='color: #4a5568; line-height: 1.5;'>{factor}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close card container




def display_connection_status():
    """Display backend connection status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    try:
        if api_client.check_backend_health():
            st.sidebar.markdown(
                f"<div class='status-indicator status-success'>"
                f"<span style='margin-right: 8px;'><i class='fas fa-check-circle'></i></span>"
                "Backend Connected"
                "</div>",
                unsafe_allow_html=True
            )
            
            # Display model info if available
            model_info = api_client.get_model_info()
            if model_info:
                with st.sidebar.expander("Model Information", expanded=False):
                    st.markdown(f"**Model Type:** {model_info.get('model_type', 'N/A')}")
                    st.markdown(f"**Version:** {model_info.get('version', '1.0.0')}")
                    st.markdown(f"**Features:** {model_info.get('feature_count', 'N/A')}")
                    st.markdown(f"**Last Trained:** {model_info.get('last_trained', 'N/A')}")
                
        else:
            st.sidebar.markdown(
                f"<div class='status-indicator status-warning'>"
                f"<span style='margin-right: 8px;'><i class='fas fa-exclamation-triangle'></i></span>"
                "Backend Unavailable"
                "</div>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.sidebar.markdown(
            f"<div class='status-indicator status-error'>"
            f"<span style='margin-right: 8px;'><i class='fas fa-times-circle'></i></span>"
            "Connection Error"
            "</div>",
            unsafe_allow_html=True
        )
        st.sidebar.error(f"Error: {str(e)}")

def main():
    """Main application"""
    # Main header with logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    with col2:
        st.markdown(
            "<h1 class='main-header'>Personalized Healthcare<br>Recommendation System</h1>",
            unsafe_allow_html=True
        )
    
    # Check backend health
    if not api_client.check_backend_health():
        st.error("Backend service is not available. Please ensure the backend server is running.")
        
        # Add helpful instructions
        with st.expander("How to start the backend server", expanded=True):
            st.code("""# In a new terminal, run:
cd backend
uvicorn app:app --reload

# Or using Python:
python -m uvicorn app:app --reload""")
            
        if st.button("🔄 Retry Connection", type="primary"):
            st.rerun()
        return
    
    # Sidebar navigation with icons
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Risk Assessment", "📊 Dashboard", "ℹ️ About"],
        index=0
    )
    
    # Display connection status in sidebar
    display_connection_status()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.8rem; margin-top: 2rem;">
        <p>© 2023 Personalized Healthcare System</p>
        <p>v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    if "🏠 Risk Assessment" in page:
        st.header("👤 Patient Risk Assessment")
        
        # Create two columns
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            with st.container():
                st.markdown("### 📋 Patient Information")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                # Patient input form
                form = PatientInputForm()
                patient_data = form.render()
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                analyze_col1, analyze_col2 = st.columns([1, 2])
                with analyze_col1:
                    analyze_btn = st.button("🔍 Analyze Risk", 
                                          type="primary", 
                                          use_container_width=True,
                                          help="Analyze patient data and generate risk assessment")
                
                if analyze_btn and patient_data:
                    with st.spinner("Analyzing patient data and generating recommendations..."):
                        prediction = api_client.make_prediction(patient_data)
                        if prediction:
                            st.session_state['prediction'] = prediction
                            st.session_state['patient_data'] = patient_data
                            st.session_state['last_analyzed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.rerun()
        
        with col2:
            st.markdown("### 📊 Analysis Results")
            
            if 'prediction' in st.session_state and st.session_state['prediction']:
                prediction = st.session_state['prediction']
                
                # Display last analyzed time
                if 'last_analyzed' in st.session_state:
                    st.caption(f"Last analyzed: {st.session_state['last_analyzed']}")
                
                # Create tabs for better organization
                tab1, tab2, tab3 = st.tabs(["📈 Risk Assessment", "💡 Recommendations", "⚠️ Risk Factors"])
                
                with tab1:
                    display_risk_assessment(prediction, st.session_state.get('patient_data', {}))
                
                with tab2:
                    display_recommendations(prediction)
                
                with tab3:
                    display_risk_factors(prediction)
                
                # Download results section
                st.markdown("---")
                st.markdown("### 📥 Export Results")
                results_json = json.dumps({
                    "patient_data": st.session_state.get('patient_data', {}),
                    "prediction": prediction,
                    "analysis_date": st.session_state.get('last_analyzed', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                }, indent=2)
                
                col1_dl, col2_dl = st.columns([1, 1])
                with col1_dl:
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=results_json,
                        file_name=f"health_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2_dl:
                    if st.button("🔄 New Analysis", use_container_width=True):
                        if 'prediction' in st.session_state:
                            del st.session_state['prediction']
                        st.rerun()
            else:
                with st.container():
                    st.markdown("<div class='card' style='text-align: center; padding: 2rem;'>"
                               "<h3 style='color: #4a6fa5;'>No Analysis Results</h3>"
                               "<p>Please fill in the patient information and click 'Analyze Risk' to see results.</p>"
                               "<img src='https://img.icons8.com/color/200/000000/medical-doctor.png' width='150' style='opacity: 0.7; margin: 1rem 0;'/>"
                               "</div>", 
                               unsafe_allow_html=True)
    
    elif "📊 Dashboard" in page:
        st.header("📊 Healthcare Analytics Dashboard")
        # Pass patient data to dashboard if available
        patient_data = st.session_state.get('patient_data', None)
        create_dashboard(patient_data)
    
    elif "ℹ️ About" in page:
        st.header("About the System")
        
        st.markdown("""
## 🎯 Purpose
This Personalized Healthcare Recommendation System uses advanced machine learning 
to provide individualized health risk assessments and recommendations.

## 🔬 Technology Stack
- **Frontend**: Streamlit for interactive UI
- **Backend**: FastAPI for robust API services
- **ML Models**: Scikit-learn, Random Forest, Gradient Boosting
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Plotly for interactive charts

## 📊 Features
- **Risk Assessment**: AI-powered cardiovascular risk prediction
- **Personalized Recommendations**: Tailored health advice
- **Risk Factor Analysis**: Identification of key health risks
- **Interactive Dashboard**: Visual analytics and insights

## 🏥 Use Cases
- **Healthcare Providers**: Clinical decision support
- **Patients**: Personal health monitoring
- **Preventive Care**: Early risk identification
- **Population Health**: Community health insights

## 🔒 Privacy & Security
- Patient data is processed securely
- No personal information is stored permanently
- HIPAA-compliant design principles
""")
        
        # System metrics
        st.subheader("System Information")
        
        try:
            model_info = api_client.get_model_info()
            if model_info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Type", model_info.get('model_type', 'N/A'))
                
                with col2:
                    st.metric("Features", model_info.get('feature_count', 'N/A'))
                
                with col3:
                    st.metric("Status", "✅ Online")
            else:
                st.error("Cannot connect to backend API")
        except:
            st.error("Backend API is not available")

if __name__ == "__main__":
    main()