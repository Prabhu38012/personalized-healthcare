import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.forms import PatientInputForm
from pages.dashboard import create_dashboard
import json

# Page configuration
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

def make_prediction(patient_data):
    """Make API call to get prediction"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=patient_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Please ensure the backend server is running.")
        return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def display_risk_assessment(prediction):
    """Display risk assessment results"""
    risk_level = prediction['risk_level']
    risk_prob = prediction['risk_probability']
    
    # Risk level display
    if risk_level == "High":
        st.markdown(f'<div class="risk-high"><h3>⚠️ High Risk</h3><p>Risk Probability: {risk_prob:.2%}</p></div>', unsafe_allow_html=True)
    elif risk_level == "Medium":
        st.markdown(f'<div class="risk-medium"><h3>⚡ Medium Risk</h3><p>Risk Probability: {risk_prob:.2%}</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low"><h3>✅ Low Risk</h3><p>Risk Probability: {risk_prob:.2%}</p></div>', unsafe_allow_html=True)
    
    # Risk probability gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_recommendations(prediction):
    """Display personalized recommendations"""
    st.subheader("🎯 Personalized Recommendations")
    
    recommendations = prediction['recommendations']
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

def display_risk_factors(prediction):
    """Display identified risk factors"""
    st.subheader("⚠️ Identified Risk Factors")
    
    risk_factors = prediction['risk_factors']
    if risk_factors:
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.write("No significant risk factors identified.")

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🏥 Personalized Healthcare Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Risk Assessment", "Dashboard", "About"])
    
    if page == "Risk Assessment":
        st.header("Patient Risk Assessment")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Patient Information")
            
            # Patient input form
            form = PatientInputForm()
            patient_data = form.render()
            
            if st.button("Analyze Risk", type="primary"):
                if patient_data:
                    with st.spinner("Analyzing patient data..."):
                        prediction = make_prediction(patient_data)
                    
                    if prediction:
                        st.session_state['prediction'] = prediction
                        st.session_state['patient_data'] = patient_data
                        st.success("Analysis complete!")
        
        with col2:
            st.subheader("Risk Analysis Results")
            
            if 'prediction' in st.session_state:
                prediction = st.session_state['prediction']
                
                # Display risk assessment
                display_risk_assessment(prediction)
                
                # Display recommendations and risk factors
                col2a, col2b = st.columns(2)
                
                with col2a:
                    display_recommendations(prediction)
                
                with col2b:
                    display_risk_factors(prediction)
            else:
                st.info("Enter patient information and click 'Analyze Risk' to see results.")
    
    elif page == "Dashboard":
        st.header("Healthcare Analytics Dashboard")
        create_dashboard()
    
    elif page == "About":
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
            response = requests.get(f"{API_BASE_URL}/model-info")
            if response.status_code == 200:
                model_info = response.json()
                
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
