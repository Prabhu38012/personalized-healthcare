import streamlit as st

# Configure Streamlit page for professional appearance
st.set_page_config(
    page_title="MyVitals - AI-Powered Health Analytics",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/myvitals-system',
        'Report a bug': 'https://github.com/your-repo/myvitals-system/issues',
        'About': "# MyVitals\nAI-Powered Personalized Health Analytics Platform"
    }
)
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.forms import PatientInputForm
from components.lab_forms import LabPatientInputForm
from components.auth import (
    init_session_state, check_session_validity, render_login_page,
    render_user_info, require_auth, get_auth_headers, is_admin, is_doctor,
    show_development_credentials
)
from utils.api_client import HealthcareAPI
from utils.caching import cleanup_expired_cache, get_cache_stats
import json
import time
import os
from datetime import datetime, timedelta

def display_welcome_banner():
    """Display an engaging welcome banner with quick action guidance"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0891b2 0%, #10b981 100%);
                border-radius: 16px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px -5px rgba(8, 145, 178, 0.3);
                color: white;">
        <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1rem;">
            <div style="font-size: 3rem;">👨‍⚕️</div>
            <div>
                <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">Welcome to MyVitals</h1>
                <p style="margin: 0.5rem 0 0; opacity: 0.95; font-size: 1.1rem;">AI-Powered Personalized Health Analytics</p>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.15);
                    border-radius: 12px;
                    padding: 1.25rem;
                    backdrop-filter: blur(10px);">
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.6;">
                <strong>🎯 Quick Start:</strong> Select an option from the sidebar to begin. 
                Upload medical documents, enter health data manually, or access AI decision support for comprehensive health analysis.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_css():
    """Load modern custom CSS styles with shadcn-inspired design"""
    if 'css_loaded' not in st.session_state:
        st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* CSS Variables - Healthcare optimized color palette */
        :root {
            --primary: #0891b2;
            --primary-light: #06b6d4;
            --primary-dark: #0e7490;
            --secondary: #64748b;
            --accent: #10b981;
            --accent-light: #34d399;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
            --background: #f8fafc;
            --foreground: #0f172a;
            --card: #ffffff;
            --border: #e2e8f0;
            --shadow-color: rgba(0, 0, 0, 0.1);
        }
        
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Modern Card Component - Reduced spacing */
        .modern-card {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 3px 0 var(--shadow-color);
            border: 1px solid var(--border);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 0.75rem;
        }
        
        .modern-card:hover {
            box-shadow: 0 8px 16px -4px var(--shadow-color);
            transform: translateY(-2px);
        }
        
        /* Medical Header - Healthcare blue/teal gradient - Reduced spacing */
        .medical-header {
            background: linear-gradient(135deg, #0891b2 0%, #10b981 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 10px 25px -5px rgba(8, 145, 178, 0.3);
        }
        
        /* Stats Card */
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px 0 var(--shadow-color);
            border-left: 4px solid var(--primary);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            box-shadow: 0 4px 12px -2px var(--shadow-color);
            transform: translateX(4px);
        }
        
        /* Button Styles */
        .stButton>button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.2s;
            box-shadow: 0 2px 4px 0 rgba(8, 145, 178, 0.2);
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
            box-shadow: 0 4px 12px -2px rgba(8, 145, 178, 0.4);
            transform: translateY(-1px);
        }
        
        /* Input Fields */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            border-radius: 8px;
            border: 2px solid var(--border);
            padding: 10px 14px;
            transition: all 0.2s;
        }
        
        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.1);
        }
        
        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: 700;
            color: var(--primary);
        }
        
        /* Status Boxes with proper contrast */
        .success-box {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border-left: 4px solid var(--success);
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
            color: #065f46;
        }
        
        .error-box {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border-left: 4px solid var(--error);
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
            color: #991b1b;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left: 4px solid var(--warning);
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
            color: #92400e;
        }
        
        .info-box {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            border-left: 4px solid var(--info);
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
            color: #1e3a8a;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: white;
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
            font-weight: 600;
            border: 2px solid var(--border);
            color: var(--secondary);
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: white;
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 24px;
            transition: all 0.3s;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--primary);
            background: #f0f9ff;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: white;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-weight: 600;
            color: var(--foreground);
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.3s ease-out;
        }
        
        /* Loading Spinner */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-spinner {
            border: 4px solid #f3f4f6;
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        /* Tooltip Styles */
        .tooltip-icon {
            display: inline-block;
            background: var(--info);
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            text-align: center;
            font-size: 12px;
            font-weight: bold;
            cursor: help;
            margin-left: 6px;
        }
        
        /* Progress Bars */
        .progress-bar {
            background: #e5e7eb;
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin: 8px 0;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, var(--primary), var(--accent));
            height: 100%;
            border-radius: 8px;
            transition: width 0.3s ease;
        }
        
        /* Better form labels */
        .stTextInput label, .stNumberInput label, .stSelectbox label {
            font-weight: 600 !important;
            color: var(--foreground) !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Responsive improvements */
        @media (max-width: 768px) {
            .modern-card {
                padding: 0.75rem;
            }
            .medical-header {
                padding: 1rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.css_loaded = True

# Page configuration
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication
init_session_state()

# Load custom CSS
load_css()

# Authentication will be handled in main() function

# Initialize API client with auth headers (cached for performance)
if 'api_client' not in st.session_state:
    st.session_state.api_client = HealthcareAPI()
    
api_client = st.session_state.api_client
if hasattr(api_client, 'set_auth_headers'):
    api_client.set_auth_headers(get_auth_headers())



def display_llm_analysis(llm_analysis):
    """Display LLM analysis in an expandable section"""
    if not llm_analysis or not llm_analysis.get('analysis_available'):
        return
        
    with st.expander("🤖 AI-Powered Health Insights", expanded=True):
        st.markdown("### 🧠 AI Analysis Summary")
        st.markdown(f"{llm_analysis.get('summary', 'No analysis available')}")
        
        # Key risk factors
        if llm_analysis.get('key_risk_factors'):
            st.markdown("\n### 🔍 Key Risk Factors")
            for factor in llm_analysis['key_risk_factors']:
                st.markdown(f"- {factor}")
        
        # Health implications
        if llm_analysis.get('health_implications'):
            st.markdown("\n### ⚠️ Health Implications")
            st.markdown(llm_analysis['health_implications'])
        
        # Recommendations
        if llm_analysis.get('recommendations'):
            st.markdown("\n### 💡 Recommendations")
            for rec in llm_analysis['recommendations']:
                st.markdown(f"- {rec}")
        
        # Urgency level
        if llm_analysis.get('urgency_level'):
            urgency = llm_analysis['urgency_level'].lower()
            if urgency == 'high':
                st.error("🔴 **Urgency: High** - Immediate attention recommended")
            elif urgency == 'medium':
                st.warning("🟠 **Urgency: Medium** - Schedule follow-up soon")
            else:
                st.success("🟢 **Urgency: Low** - Routine monitoring recommended")

def display_risk_assessment(prediction, patient_data=None):
    """Display risk assessment results with enhanced visualization"""
    risk_level = prediction.get('risk_level', 'Low')
    risk_prob = prediction.get('risk_probability', 0)
    llm_analysis = prediction.get('llm_analysis')
    
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
                    Please consult with a doctor as soon as possible.
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
        
        # Display LLM analysis if available
        if 'llm_analysis' in st.session_state.get('prediction_result', {}):
            display_llm_analysis(st.session_state.prediction_result['llm_analysis'])

def display_prediction_results(result, analysis_type):
    """Display comprehensive prediction results"""
    st.markdown(f"### 📊 Health Risk Analysis Results ({analysis_type})")
    
    # Get patient data from result - try multiple sources
    patient_data = {}
    
    # Try different data sources in the result
    if 'data' in result and result['data']:
        patient_data.update(result['data'])
    
    # Also check for patient_data key
    if 'patient_data' in result and result['patient_data']:
        patient_data.update(result['patient_data'])
    
    # Check extraction_info for additional data
    if 'extraction_info' in result and 'extracted_data' in result['extraction_info']:
        extracted = result['extraction_info']['extracted_data']
        if extracted:
            patient_data.update(extracted)
    
    # Debug: Show what data we have
    if not patient_data:
        st.warning("⚠️ No patient data found in prediction result. Raw result keys: " + str(list(result.keys())))
    
    # Risk Level Display
    risk_level = result.get('risk_level', 'Unknown')
    risk_probability = result.get('risk_probability', 0)
    confidence = result.get('confidence', 0)
    
    # Color coding for risk levels
    risk_colors = {
        'Low': 'green',
        'Medium': 'orange', 
        'High': 'red'
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Level", risk_level, 
                 delta_color="inverse" if risk_level != "Low" else "normal")
    
    with col2:
        st.metric("Risk Probability", f"{risk_probability:.1%}")
    
    with col3:
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col4:
        # Extract BMI from the prediction result or calculate it
        bmi_value = 0.0
        
        # Try multiple sources for BMI data
        # 1. Direct BMI from result data
        if 'data' in result and result['data'] is not None and 'bmi' in result['data'] and result['data']['bmi'] is not None:
            bmi_value = float(result['data']['bmi'])
        # 2. BMI from patient_data in session state
        elif patient_data.get('bmi') is not None:
            bmi_value = float(patient_data['bmi'])
        # 3. Calculate from weight and height in patient_data
        elif patient_data.get('weight') and patient_data.get('height'):
            try:
                weight = float(patient_data.get('weight', 0))
                height = float(patient_data.get('height', 0))
                if weight > 0 and height > 0:
                    bmi_value = weight / ((height / 100) ** 2)
            except (ValueError, TypeError):
                pass
        # 4. Calculate from weight and height in result data
        elif 'data' in result and result['data'] is not None:
            data = result['data']
            if data.get('weight') and data.get('height'):
                try:
                    weight = float(data.get('weight', 0))
                    height = float(data.get('height', 0))
                    if weight > 0 and height > 0:
                        bmi_value = weight / ((height / 100) ** 2)
                except (ValueError, TypeError):
                    pass
        # 5. Try extraction info as fallback
        elif 'extraction_info' in result:
            extracted_data = result.get('extraction_info', {}).get('extracted_data', {})
            if 'bmi' in extracted_data and extracted_data['bmi'] is not None:
                bmi_value = float(extracted_data['bmi'])
            elif 'weight' in extracted_data and 'height' in extracted_data:
                try:
                    weight = float(extracted_data.get('weight', 0))
                    height = float(extracted_data.get('height', 0))
                    if weight > 0 and height > 0:
                        bmi_value = weight / ((height / 100) ** 2)
                except (ValueError, TypeError):
                    pass
        
        # BMI classification
        if bmi_value > 0:
            if bmi_value < 18.5:
                bmi_status = "Underweight"
            elif bmi_value < 25:
                bmi_status = "Normal"
            elif bmi_value < 30:
                bmi_status = "Overweight"
            else:
                bmi_status = "Obese"
        else:
            bmi_status = "Unknown"
        
        st.metric("BMI", f"{bmi_value:.1f}" if bmi_value > 0 else "N/A", bmi_status)
    
    # Vital Statistics Section
    st.markdown("### 📊 Vital Statistics")
    
    # Create columns for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Blood Pressure - try multiple field names
        systolic = patient_data.get('systolic_bp') or patient_data.get('resting_bp')
        diastolic = patient_data.get('diastolic_bp') or patient_data.get('diastolic')
        
        if systolic is not None and diastolic is not None:
            bp_status = "Normal"
            bp_color = "green"
            if systolic >= 140 or diastolic >= 90:
                bp_status = "High (Hypertension)"
                bp_color = "red"
            elif systolic >= 120 or diastolic >= 80:
                bp_status = "Elevated"
                bp_color = "orange"
            
            st.metric(
                "Blood Pressure", 
                f"{systolic}/{diastolic} mmHg",
                bp_status
            )
        else:
            # Show what BP data we have for debugging
            bp_fields = {k: v for k, v in patient_data.items() if 'bp' in k.lower() or 'pressure' in k.lower() or 'systolic' in k.lower() or 'diastolic' in k.lower()}
            if bp_fields:
                st.metric("Blood Pressure", "Available fields: " + str(bp_fields))
    
    with col2:
        # Cholesterol - try multiple field names
        total_chol = patient_data.get('total_cholesterol') or patient_data.get('cholesterol')
        if total_chol is not None:
            chol_status = "Normal"
            chol_color = "green"
            if total_chol >= 240:
                chol_status = "High"
                chol_color = "red"
            elif total_chol >= 200:
                chol_status = "Borderline High"
                chol_color = "orange"
            
            st.metric(
                "Total Cholesterol",
                f"{total_chol} mg/dL",
                chol_status
            )
        else:
            # Show available cholesterol fields for debugging
            chol_fields = {k: v for k, v in patient_data.items() if 'chol' in k.lower()}
            if chol_fields:
                st.metric("Cholesterol", "Available: " + str(chol_fields))
    
    with col3:
        # BMI (already calculated above)
        bmi_value = 0.0
        if 'bmi' in patient_data and patient_data['bmi'] is not None:
            bmi_value = float(patient_data['bmi'])
        elif 'weight' in patient_data and 'height' in patient_data:
            try:
                weight = float(patient_data.get('weight', 0))
                height = float(patient_data.get('height', 0))
                if weight > 0 and height > 0:
                    bmi_value = weight / ((height / 100) ** 2)
            except (ValueError, TypeError):
                pass
        
        if bmi_value > 0:
            if bmi_value < 18.5:
                bmi_status = "Underweight"
                bmi_color = "orange"
            elif bmi_value < 25:
                bmi_status = "Normal"
                bmi_color = "green"
            elif bmi_value < 30:
                bmi_status = "Overweight"
                bmi_color = "orange"
            else:
                bmi_status = "Obese"
                bmi_color = "red"
            
            st.metric(
                "BMI", 
                f"{bmi_value:.1f}",
                bmi_status
            )
    
    # Risk Level Indicator
    st.markdown("### 📈 Risk Assessment")
    if risk_level in risk_colors:
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 5px; background-color: {risk_colors[risk_level]}20; border-left: 4px solid {risk_colors[risk_level]};">
            <h4 style="margin: 0; color: {risk_colors[risk_level]};">{risk_level} Risk</h4>
            <p style="margin: 5px 0 0 0;">Probability: {risk_probability:.1%} • Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Factors
    risk_factors = result.get('risk_factors', [])
    if risk_factors:
        st.markdown("### ⚠️ Identified Risk Factors")
        for i, factor in enumerate(risk_factors, 1):
            st.write(f"{i}. {factor}")
    
    # Recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        st.markdown("### 💡 Health Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    # LLM Analysis (if available)
    llm_analysis = result.get('llm_analysis')
    if llm_analysis and llm_analysis.get('analysis_available'):
        st.markdown("### 🤖 AI-Powered Insights")
        
        # Summary
        summary = llm_analysis.get('summary')
        if summary:
            st.markdown("#### Summary")
            st.write(summary)
        
        # Key Risk Factors
        key_risk_factors = llm_analysis.get('key_risk_factors', [])
        if key_risk_factors:
            st.markdown("#### Key Risk Factors Identified by AI")
            for factor in key_risk_factors:
                st.write(f"• {factor}")
        
        # Health Implications
        health_implications = llm_analysis.get('health_implications')
        if health_implications:
            st.markdown("#### Health Implications")
            st.write(health_implications)
        
        # AI Recommendations
        ai_recommendations = llm_analysis.get('recommendations', [])
        if ai_recommendations:
            st.markdown("#### AI Recommendations")
            for rec in ai_recommendations:
                st.write(f"• {rec}")
        
        # Urgency Level
        urgency_level = llm_analysis.get('urgency_level')
        if urgency_level:
            urgency_colors = {
                'low': 'green',
                'medium': 'orange',
                'high': 'red'
            }
            color = urgency_colors.get(urgency_level.lower(), 'blue')
            st.markdown(f"""
            <div style="padding: 8px; border-radius: 4px; background-color: {color}20; border-left: 3px solid {color};">
                <strong>Urgency Level:</strong> {urgency_level.title()}
            </div>
            """, unsafe_allow_html=True)
    
    elif llm_analysis and not llm_analysis.get('analysis_available'):
        reason = llm_analysis.get('reason', 'Unknown reason')
        st.info(f"🤖 AI Analysis not available: {reason}")
    
    # Export Results
    st.markdown("### 📄 Export Results")
    if st.button("📋 Copy Results to Clipboard"):
        results_text = format_results_for_export(result, analysis_type)
        st.code(results_text, language="text")
        st.success("Results formatted for copying!")

def format_results_for_export(result, analysis_type):
    """Format results for export/copying"""
    risk_level = result.get('risk_level', 'Unknown')
    risk_probability = result.get('risk_probability', 0)
    confidence = result.get('confidence', 0)
    
    text = f"""HEALTH RISK ANALYSIS RESULTS ({analysis_type})
{'='*50}

RISK ASSESSMENT:
- Risk Level: {risk_level}
- Risk Probability: {risk_probability:.1%}
- Confidence: {confidence:.1%}

RISK FACTORS:
"""
    
    risk_factors = result.get('risk_factors', [])
    for i, factor in enumerate(risk_factors, 1):
        text += f"{i}. {factor}\n"
    
    text += "\nRECOMMENDATIONS:\n"
    recommendations = result.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        text += f"{i}. {rec}\n"
    
    # Add LLM analysis if available
    llm_analysis = result.get('llm_analysis')
    if llm_analysis and llm_analysis.get('analysis_available'):
        text += "\nAI INSIGHTS:\n"
        summary = llm_analysis.get('summary')
        if summary:
            text += f"Summary: {summary}\n"
        
        urgency = llm_analysis.get('urgency_level')
        if urgency:
            text += f"Urgency Level: {urgency}\n"
    
    text += f"\nGenerated by Personalized Healthcare AI System"
    return text



# display_connection_status function moved to avoid duplication - see line 584


def render_patient_management():
    """Render patient management interface for doctors"""
    st.header("👥 Patient Management")
    
    tab1, tab2 = st.tabs(["Patient Search", "Recent Analyses"])
    
    with tab1:
        st.subheader("Search Patients")
        
        search_col1, search_col2 = st.columns([2, 1])
        with search_col1:
            search_query = st.text_input("Search by name or ID", placeholder="Enter patient name or ID")
        with search_col2:
            search_btn = st.button("🔍 Search", use_container_width=True)
        
        if search_btn and search_query:
            st.info("Patient search functionality would be implemented here with proper database integration.")
            
        # Mock patient list for demonstration
        st.subheader("Recent Patients")
        mock_patients = [
            {"id": "P001", "name": "John Doe", "age": 45, "risk_level": "Medium", "last_visit": "2024-01-15"},
            {"id": "P002", "name": "Jane Smith", "age": 38, "risk_level": "Low", "last_visit": "2024-01-14"},
            {"id": "P003", "name": "Bob Wilson", "age": 62, "risk_level": "High", "last_visit": "2024-01-13"},
        ]
        
        for patient in mock_patients:
            with st.expander(f"{patient['name']} (ID: {patient['id']})"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Age", patient['age'])
                with col2:
                    risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[patient['risk_level']]
                    st.markdown(f"**Risk Level:** :{risk_color}[{patient['risk_level']}]")
                with col3:
                    st.metric("Last Visit", patient['last_visit'])
                with col4:
                    if st.button("View Details", key=f"view_{patient['id']}"):
                        st.info(f"Would open detailed view for {patient['name']}")
    
    with tab2:
        st.subheader("Recent Risk Analyses")
        st.info("This section would show recent analyses performed by you or your team.")
        
        # Mock analysis history
        analysis_data = {
            "Date": ["2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12"],
            "Patient": ["John Doe", "Jane Smith", "Bob Wilson", "Alice Brown"],
            "Risk Level": ["Medium", "Low", "High", "Low"],
            "Confidence": ["85%", "92%", "78%", "89%"]
        }
        
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, use_container_width=True)


def render_admin_panel():
    """Render admin panel interface"""
    st.header("🔧 Admin Panel")
    
    tab1, tab2, tab3 = st.tabs(["System Status", "Analytics", "Settings"])
    
    with tab1:
        st.subheader("System Health")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Users", "23", "2")
        with col2:
            st.metric("Predictions Today", "156", "12")
        with col3:
            st.metric("API Uptime", "99.8%")
        with col4:
            st.metric("Model Accuracy", "87.3%")
        
        st.markdown("---")
        
        # System logs (mock)
        st.subheader("Recent System Events")
        log_data = {
            "Timestamp": ["2024-01-15 10:30:45", "2024-01-15 10:25:12", "2024-01-15 10:20:33"],
            "Level": ["INFO", "WARNING", "INFO"],
            "Event": [
                "User login successful: doctor@healthcare.com",
                "High prediction load detected", 
                "Model retrained successfully"
            ]
        }
        
        log_df = pd.DataFrame(log_data)
        st.dataframe(log_df, use_container_width=True)
    
    with tab2:
        st.subheader("Usage Analytics")
        
        # Mock analytics data
        dates = pd.date_range('2024-01-01', '2024-01-15', freq='D')
        predictions = [20 + i*2 + (i%3)*5 for i in range(len(dates))]
        
        analytics_df = pd.DataFrame({
            'Date': dates,
            'Predictions': predictions
        })
        
        fig = px.line(analytics_df, x='Date', y='Predictions', title='Daily Predictions')
        st.plotly_chart(fig, use_container_width=True)
        
        # User role distribution
        role_data = {'Role': ['Doctors', 'Patients', 'Admins'], 'Count': [12, 8, 3]}
        role_df = pd.DataFrame(role_data)
        
        fig2 = px.pie(role_df, values='Count', names='Role', title='User Role Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("System Settings")
        
        with st.form("system_settings"):
            st.markdown("#### Model Configuration")
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
            auto_retrain = st.checkbox("Enable Auto-Retraining", True)
            
            st.markdown("#### Security Settings")
            session_timeout = st.selectbox("Session Timeout (hours)", [1, 2, 4, 8, 24], index=2)
            require_2fa = st.checkbox("Require Two-Factor Authentication", False)
            
            st.markdown("#### Notification Settings")
            email_alerts = st.checkbox("Email Alerts for High-Risk Predictions", True)
            daily_reports = st.checkbox("Daily Usage Reports", False)
            
            if st.form_submit_button("Save Settings", type="primary"):
                st.success("Settings saved successfully!")


def render_user_management():
    """Render user management interface for admins"""
    st.header("👤 User Management")
    
    tab1, tab2 = st.tabs(["User List", "Add New User"])
    
    with tab1:
        st.subheader("Current Users")
        
        # Get users from backend using API client
        try:
            users_response = api_client.get_users()
            if users_response:
                # Convert to DataFrame for display
                users_data = []
                for user in users_response:
                    users_data.append({
                        "ID": user.get("id", "N/A"),
                        "Name": user.get("full_name", "N/A"),
                        "Email": user.get("email", "N/A"),
                        "Role": user.get("role", "N/A").title(),
                        "Status": "Active" if user.get("is_active", False) else "Inactive",
                        "Created": user.get("created_at", "N/A")[:10] if user.get("created_at") else "N/A",
                        "Last Login": user.get("last_login", "Never")[:16] if user.get("last_login") else "Never"
                    })
                
                if users_data:
                    users_df = pd.DataFrame(users_data)
                else:
                    st.info("No users found in the system.")
                    return
            else:
                st.error("Failed to fetch users from backend.")
                return
        except Exception as e:
            st.error(f"Error fetching users: {str(e)}")
            return
        
        # Display users in a proper table format
        st.dataframe(users_df, use_container_width=True)
        
        # User management actions
        st.markdown("### User Actions")
        selected_user_id = st.selectbox("Select User for Actions", 
                                       options=[user["ID"] for user in users_data],
                                       format_func=lambda x: f"{x} - {next(u['Name'] for u in users_data if u['ID'] == x)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Users", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("✏️ Edit User", use_container_width=True):
                st.info(f"Edit functionality for user {selected_user_id} would be implemented here")
        
        with col3:
            if st.button("🗑️ Delete User", use_container_width=True, type="secondary"):
                if st.session_state.get(f"confirm_delete_{selected_user_id}"):
                    try:
                        result = api_client.delete_user(selected_user_id)
                        if result:
                            st.success("User deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete user")
                    except Exception as e:
                        st.error(f"Error deleting user: {str(e)}")
                else:
                    st.session_state[f"confirm_delete_{selected_user_id}"] = True
                    st.warning("Click again to confirm deletion")
        
        st.markdown("---")
        
    with tab2:
        st.subheader("Add New User")
        
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Full Name", placeholder="Enter full name")
                new_email = st.text_input("Email", placeholder="user@healthcare.com")
                new_password = st.text_input("Password", type="password", placeholder="Min 8 characters")
            
            with col2:
                new_role = st.selectbox("Role", ["Patient", "Doctor", "Admin"])
                is_active = st.checkbox("Active", value=True)
                send_welcome = st.checkbox("Send Welcome Email", value=True)
            
            if st.form_submit_button("Create User", type="primary"):
                if new_name and new_email and new_password:
                    try:
                        # Create user using API client
                        user_data = {
                            "email": new_email,
                            "password": new_password,
                            "full_name": new_name,
                            "role": new_role.lower(),
                            "is_active": is_active
                        }
                        
                        result = api_client.create_user(user_data)
                        if result:
                            st.success(f"User {new_name} created successfully!")
                            if send_welcome:
                                st.info("Welcome email would be sent in production environment.")
                            st.rerun()
                        else:
                            st.error("Failed to create user. Please check the details and try again.")
                    except Exception as e:
                        st.error(f"Error creating user: {str(e)}")
                else:
                    st.error("Please fill in all required fields.")


def render_health_log_page():
    """Render health log page with backend integration"""
    st.header("📊 Health Log")
    st.markdown("Track your vital signs and health metrics over time")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Health Metrics", "📝 Log New Data", "📋 Health Reports"])
    
    with tab1:
        render_health_metrics()
    
    with tab2:
        render_data_entry()
    
    with tab3:
        render_health_reports()

def render_health_metrics():
    """Render health metrics dashboard with backend integration"""
    st.markdown("### 📊 Your Health Metrics Overview")
    
    try:
        # Get health data from backend
        health_entries = api_client.get_health_data(limit=30)
        
        if not health_entries or len(health_entries) == 0:
            st.info("📝 No health data available. Please log your first health metrics in the 'Log New Data' tab.")
            return
        
        # Convert to DataFrame for easier handling
        health_data = pd.DataFrame(health_entries)
        latest_data = health_entries[0]  # Most recent entry
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            systolic = latest_data.get('systolic_bp', 0) or 0
            diastolic = latest_data.get('diastolic_bp', 0) or 0
            st.metric("🩸 Blood Pressure", f"{systolic}/{diastolic}")
        
        with col2:
            heart_rate = latest_data.get('heart_rate', 0) or 0
            st.metric("❤️ Heart Rate", f"{heart_rate} BPM")
        
        with col3:
            weight = latest_data.get('weight', 0) or 0
            st.metric("⚖️ Weight", f"{weight:.1f} kg")
        
        with col4:
            blood_sugar = latest_data.get('blood_sugar', 0) or 0
            st.metric("🩸 Blood Sugar", f"{blood_sugar:.1f} mg/dL")
        
    except Exception as e:
        st.error(f"Error loading health data: {str(e)}")
        return
    
    # Health trends chart
    st.markdown("### 📈 Health Trends (Last 30 Days)")
    
    st.markdown("""
    <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 0.75rem; 
                border-radius: 8px; margin-bottom: 1rem; color: #92400e; font-size: 0.9rem;">
        <strong>📊 Trend Analysis:</strong> Regular patterns indicate consistency. 
        Sudden spikes or drops may warrant medical attention.
    </div>
    """, unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Add Blood Pressure trace with better styling
    fig.add_trace(go.Scatter(
        x=health_data['date'],
        y=health_data['systolic_bp'],
        mode='lines+markers',
        name='Systolic BP',
        line=dict(color='#ef4444', width=3, shape='spline'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>Systolic BP</b><br>Date: %{x}<br>Value: %{y} mmHg<extra></extra>'
    ))
    
    # Add Heart Rate trace with better styling
    fig.add_trace(go.Scatter(
        x=health_data['date'],
        y=health_data['heart_rate'],
        mode='lines+markers',
        name='Heart Rate',
        yaxis='y2',
        line=dict(color='#10b981', width=3, shape='spline'),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>Heart Rate</b><br>Date: %{x}<br>Value: %{y} BPM<extra></extra>'
    ))
    
    # Add reference lines for healthy ranges
    fig.add_hline(y=120, line_dash="dash", line_color="orange", 
                  annotation_text="Elevated BP (120)", annotation_position="right")
    
    fig.update_layout(
        title="💓 Blood Pressure & Heart Rate Trends",
        xaxis_title="Date",
        yaxis=dict(
            title="Blood Pressure (mmHg)", 
            side="left",
            titlefont=dict(color='#ef4444'),
            tickfont=dict(color='#ef4444')
        ),
        yaxis2=dict(
            title="Heart Rate (BPM)", 
            side="right", 
            overlaying="y",
            titlefont=dict(color='#10b981'),
            tickfont=dict(color='#10b981')
        ),
        height=450,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)

def render_data_entry():
    """Render health data entry form"""
    st.markdown("### 📝 Log New Health Data")
    
    with st.form("health_data_entry", clear_on_submit=True):
        st.markdown("#### 🩺 Vital Signs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=120, value=80)
            heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=70)
        
        with col2:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6, step=0.1)
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=400, value=100)
        
        st.markdown("#### 📋 Additional Notes")
        notes = st.text_area("Health notes or symptoms", placeholder="Any symptoms, medication changes, or health observations...")
        
        submit_button = st.form_submit_button("💾 Save Health Data", type="primary")
        
        if submit_button:
            save_health_data(systolic_bp, diastolic_bp, heart_rate, weight, temperature, blood_sugar, notes)

def render_health_reports():
    """Render health reports and analytics"""
    st.markdown("### 📋 Health Reports & Analytics")
    
    if st.session_state.health_log_data.empty:
        st.info("📝 No health data available for reports. Please log some health metrics first.")
        return
    
    # Weekly summary
    st.markdown("#### 📊 Weekly Health Summary")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    weekly_data = st.session_state.health_log_data[
        st.session_state.health_log_data['date'] >= start_date
    ]
    
    if not weekly_data.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_systolic = weekly_data['systolic_bp'].mean()
            st.metric(
                "Avg Systolic BP", 
                f"{avg_systolic:.0f} mmHg",
                help="Average systolic blood pressure over the last 7 days"
            )
        
        with col2:
            avg_hr = weekly_data['heart_rate'].mean()
            st.metric(
                "Avg Heart Rate", 
                f"{avg_hr:.0f} BPM",
                help="Average resting heart rate over the last 7 days"
            )
        
        with col3:
            data_points = len(weekly_data)
            consistency = (data_points / 7) * 100
            st.metric(
                "Logging Consistency", 
                f"{consistency:.0f}%",
                help="Percentage of days with logged health data"
            )
            
        # Visual progress bar for consistency
        st.markdown("""
        <div style="margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; color: #0891b2;">🎯 Tracking Progress</span>
                <span style="color: #64748b;">{:.0f}% Complete</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {:.0f}%;"></div>
            </div>
        </div>
        """.format(consistency, consistency), unsafe_allow_html=True)

def save_health_data(systolic_bp, diastolic_bp, heart_rate, weight, temperature, blood_sugar, notes):
    """Save new health data entry"""
    try:
        new_entry = {
            'date': datetime.now(),
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'weight': weight,
            'temperature': temperature,
            'blood_sugar': blood_sugar,
            'notes': notes
        }
        
        # Add to health log data
        new_df = pd.DataFrame([new_entry])
        st.session_state.health_log_data = pd.concat([st.session_state.health_log_data, new_df], ignore_index=True)
        
        show_success_message("Health data saved successfully!")
        time.sleep(0.5)  # Brief pause for user to see success
        st.rerun()
    except Exception as e:
        show_error_message(f"Failed to save health data: {str(e)}")

def generate_sample_health_data():
    """Generate sample health data for demonstration"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    data = []
    for i, date in enumerate(dates):
        base_systolic = 120 + (i % 10) - 5
        base_diastolic = 80 + (i % 6) - 3
        base_hr = 70 + (i % 8) - 4
        base_weight = 70.0 + (i % 3) * 0.1
        base_temp = 98.6 + (i % 2) * 0.2
        
        data.append({
            'date': date,
            'systolic_bp': base_systolic,
            'diastolic_bp': base_diastolic,
            'heart_rate': base_hr,
            'weight': base_weight,
            'temperature': base_temp,
            'blood_sugar': 100 + (i % 5),
            'notes': f"Daily health check {i+1}"
        })
    
    return pd.DataFrame(data)

def display_connection_status():
    """Display backend connection status in sidebar - optimized with global caching"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Use the cached health check
    backend_healthy = api_client.check_backend_health()
    
    if backend_healthy:
        st.sidebar.markdown(
            f"<div class='status-indicator status-success'>"
            f"<span style='margin-right: 8px;'><i class='fas fa-check-circle'></i></span>"
            "Backend Connected"
            "</div>",
            unsafe_allow_html=True
        )

def show_loading_state(message="Processing..."):
    """Display a loading state with spinner"""
    return st.spinner(f"🔄 {message}")

def show_success_message(message, icon="✅"):
    """Display a success message with animation"""
    st.success(f"{icon} {message}")
    if "balloons" not in st.session_state or not st.session_state.balloons:
        st.balloons()
        st.session_state.balloons = True

def show_error_message(message, icon="❌", show_details=True):
    """Display an error message with helpful information"""
    st.error(f"{icon} {message}")
    if show_details:
        with st.expander("🛠️ Troubleshooting Tips"):
            st.markdown("""
            **Common Solutions:**
            - ✓ Check your internet connection
            - ✓ Ensure the backend server is running
            - ✓ Verify your login credentials
            - ✓ Try refreshing the page
            - ✓ Contact support if the issue persists
            """)

def show_info_message(message, icon="📌"):
    """Display an info message"""
    st.info(f"{icon} {message}")

def main():
    """Main application function with comprehensive error handling"""
    try:
        init_session_state()

        # Check if user is authenticated
        if not st.session_state.authenticated:
            render_login_page()
            return

        # Check session validity
        if not check_session_validity():
            render_login_page()
            return

        # Initialize page state if not exists
        if 'page' not in st.session_state:
            st.session_state.page = "🏠 Risk Assessment"

        # Handle page redirects from auth
        if hasattr(st.session_state, 'redirect_to'):
            st.session_state.page = st.session_state.redirect_to
            del st.session_state.redirect_to
            st.rerun()

        # Main header with MyVitals branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <h1 style="color: var(--primary-color); margin: 0; font-size: 2.5rem; font-weight: 700;">
                💚 MyVitals
            </h1>
            <p style="color: var(--text-secondary); margin: 0.25rem 0 0 0; font-size: 1.1rem;">
                Your Personal Health Companion
            </p>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Sidebar navigation with modern UI
        render_user_info()  # Display user info in sidebar
        
        # Try to import streamlit-option-menu, fallback to standard radio
        try:
            from streamlit_option_menu import option_menu
            use_modern_menu = True
        except ImportError:
            use_modern_menu = False
        
        # Build navigation options based on user role
        nav_options = ["Medical Consultation", "Document Analysis", "AI Decision Support", "Health Log"]
        nav_icons = ["mic", "file-text", "robot", "bar-chart"]
        
        # Add role-specific navigation
        if is_doctor() or is_admin():
            nav_options.append("Patient Management")
            nav_icons.append("people")
        
        # Admin-only navigation
        if is_admin():
            nav_options.extend(["Admin Panel", "User Management"])
            nav_icons.extend(["gear", "person-badge"])
        
        # Add About page
        nav_options.append("About")
        nav_icons.append("info-circle")
        
        # Modern horizontal navigation menu at the top
        if use_modern_menu:
            selected = option_menu(
                menu_title=None,  # No title for horizontal menu
                options=nav_options,
                icons=nav_icons,
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",  # Horizontal navbar
                styles={
                    "container": {
                        "padding": "0!important", 
                        "background": "linear-gradient(90deg, #0891b2 0%, #06b6d4 100%)",
                        "border-radius": "12px",
                        "margin-bottom": "1rem",
                        "box-shadow": "0 4px 6px rgba(0,0,0,0.1)"
                    },
                    "icon": {"color": "white", "font-size": "16px"}, 
                    "nav-link": {
                        "font-size": "14px",
                        "text-align": "center",
                        "margin": "0px",
                        "padding": "12px 16px",
                        "color": "white",
                        "border-radius": "8px",
                        "transition": "all 0.3s"
                    },
                    "nav-link-selected": {
                        "background-color": "rgba(255,255,255,0.25)", 
                        "color": "white",
                        "font-weight": "600"
                    },
                    "nav-link-hover": {
                        "background-color": "rgba(255,255,255,0.15)"
                    }
                }
            )
            page = selected
        else:
            # Fallback to standard navigation
            st.sidebar.title("Navigation")
            page = st.sidebar.radio(
                "Go to",
                [f"🏠 {nav_options[0]}", f"📊 {nav_options[1]}", f"🤖 {nav_options[2]}", f"📄 {nav_options[3]}"] + 
                ([f"👥 {o}" for o in nav_options[4:] if o in ["Patient Management", "Admin Panel", "User Management", "About"]]),
                key="page_selector"
            )
        
        # Update session state page
        st.session_state.page = page
        
        # Hide sidebar for cleaner look with horizontal navbar
        if use_modern_menu:
            st.markdown("""
            <style>
                /* Hide sidebar completely when using top navbar */
                section[data-testid="stSidebar"] {
                    display: none;
                }
                /* Adjust main content to full width */
                .main .block-container {
                    max-width: 100%;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Add user info and logout in top-right corner
            col1, col2, col3 = st.columns([6, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: right; padding: 0.5rem; color: #64748b;">
                    <small>👤 {st.session_state.get('user_name', 'User')}</small><br>
                    <small style="color: #94a3b8;">{st.session_state.get('user_role', 'Patient')}</small>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                if st.button("🚪 Logout", type="secondary", key="top_logout_btn"):
                    from components.auth import logout_user
                    logout_user()
        else:
            # Display connection status in sidebar
            display_connection_status()
            
            # Add logout button in footer
            if st.session_state.authenticated:
                if st.sidebar.button("🚪 Logout", use_container_width=True, type="primary", key="main_logout_btn"):
                    from components.auth import logout_user
                    logout_user()
            
            # Add professional footer
            st.sidebar.markdown("---")
            st.sidebar.markdown("""
                        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); color: white; border-radius: var(--border-radius-md); margin-top: 1rem;">
                            <p style="margin: 0; font-size: 0.9rem; font-weight: 500;">💚 MyVitals v2.1.0</p>
                            <p style="margin: 0; font-size: 0.8rem; opacity: 0.8;">Your Personal Health Companion</p>
                        </div>
        """, unsafe_allow_html=True)
        
        # Clean page name for routing (remove emojis if present)
        clean_page = page.replace('🏠 ', '').replace('📊 ', '').replace('🤖 ', '').replace('📄 ', '').replace('👥 ', '').strip()
        
        # Page routing with professional headers
        if clean_page == "Medical Consultation" or "Medical Consultation" in page:
            # Import and show medical consultation page
            try:
                from pages.medical_consultation import show_medical_consultation
                show_medical_consultation()
            except ImportError as e:
                st.error(f"Failed to load Medical Consultation module: {e}")
                st.info("Please ensure all consultation modules are properly installed.")
        
        elif clean_page == "Risk Assessment" or "Risk Assessment" in page:
            # Redirect to Medical Consultation
            st.info("Risk Assessment has been replaced with Medical Consultation")
            st.session_state.page = "Medical Consultation"
            st.rerun()
        
        elif clean_page == "Health Log" or "Health Log" in page:
            # Professional page header
            st.markdown("""
            <div style="background: linear-gradient(135deg, var(--background-primary) 0%, var(--secondary-color) 100%); 
                        padding: 2rem; border-radius: var(--border-radius-lg); margin-bottom: 2rem; 
                        border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
                <h1 style="color: var(--primary-color); margin: 0; font-size: 2.25rem; font-weight: 700;">
                    📊 Health Log & Analytics
                </h1>
                <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem; line-height: 1.5;">
                    Track your vital signs and health metrics with comprehensive analytics and insights
                </p>
            </div>
            """, unsafe_allow_html=True)
            render_health_log_page()
        
        elif clean_page == "AI Decision Support" or "AI Decision Support" in page:
            # Clean minimal interface
            try:
                from pages.ai_decision_support import AIDecisionDashboard
                dashboard = AIDecisionDashboard()
                dashboard.run()
            except ImportError as e:
                st.error(f"AI Decision Support module not available: {str(e)}")
                st.info("Please ensure the AI Decision Support module is properly installed.")
            except Exception as e:
                st.error(f"Error loading AI Decision Support: {str(e)}")
                with st.expander("Error Details"):
                    st.code(str(e))
        
        elif clean_page == "Document Analysis" or "Document Analysis" in page or "📄 Document Analysis" in page:
            # Add collapsible sidebar for cleaner view
            st.markdown("""
            <style>
                section[data-testid="stSidebar"] {
                    width: 80px !important;
                    min-width: 80px !important;
                }
                section[data-testid="stSidebar"]:hover {
                    width: 280px !important;
                    min-width: 280px !important;
                }
                section[data-testid="stSidebar"] > div {
                    opacity: 0;
                    transition: opacity 0.3s;
                }
                section[data-testid="stSidebar"]:hover > div {
                    opacity: 1;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Modern gradient header
            st.markdown("""
            <div class="medical-header">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; color: white;">
                    📄 AI-Powered Document Analysis
                </h1>
                <p style="margin: 0.75rem 0 0 0; font-size: 1.15rem; line-height: 1.6; color: rgba(255,255,255,0.95);">
                    Upload medical reports and prescriptions for instant AI analysis with clinical insights
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create modern tabs
            doc_tab1, doc_tab2 = st.tabs(["🏥 **Medical Reports**", "💊 **Prescriptions**"])
            
            with doc_tab1:
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.markdown("### 🏥 Medical Report Analysis")
                st.markdown("""
                <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem;">
                    Upload lab reports, diagnostic results, or any medical document for comprehensive AI analysis.
                    Our system extracts key findings, metrics, and provides actionable recommendations.
                </p>
                """, unsafe_allow_html=True)
                
                # Patient name input with modern styling
                patient_name = st.text_input(
                    "👤 Patient Name (Optional)", 
                    key="mr_patient_name",
                    placeholder="Enter patient name for personalized analysis"
                )
                
                # Modern file upload section
                st.markdown("---")
                uploaded_file = st.file_uploader(
                    "📤 Upload Medical Report", 
                    type=['pdf', 'png', 'jpg', 'jpeg', 'txt'], 
                    key="medical_report",
                    help="Supported: PDF, PNG, JPG, JPEG, TXT (Max 200MB)"
                )
                
                if uploaded_file:
                    # File info card
                    st.markdown("""
                    <div class="info-box" style="margin: 1rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>📄 {}</strong><br>
                                <span style="color: #64748b;">Size: {:.1f} KB</span>
                            </div>
                        </div>
                    </div>
                    """.format(uploaded_file.name, uploaded_file.size / 1024), unsafe_allow_html=True)
                    
                    # Analyze button
                    if st.button("🔍 Analyze Document", use_container_width=True, type="primary"):
                        with st.spinner("🤖 AI is analyzing your document..."):
                            try:
                                # Prepare file for upload
                                files = {
                                    'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                                }
                                data = {}
                                if patient_name:
                                    data['patient_name'] = patient_name
                                
                                # Call API
                                result = api_client._make_request(
                                    'POST',
                                    '/document/upload/medical-report',
                                    files=files,
                                    data=data
                                )
                                
                                if result:
                                    # Success message
                                    st.markdown("""
                                    <div class="success-box" style="margin: 1.5rem 0;">
                                        <h4 style="margin: 0; color: #10b981;">✅ Analysis Complete!</h4>
                                        <p style="margin: 0.5rem 0 0 0; color: #059669;">Your document has been analyzed successfully.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display AI analysis in organized tabs
                                    analysis = result.get('analysis', {})
                                    if analysis:
                                        # Create navigation tabs for better organization
                                        analysis_tabs = st.tabs([
                                            "🏥 Diagnosis", 
                                            "📊 Test Results", 
                                            "💡 What You Need To Do",
                                            "⚠️ Warnings",
                                            "🥗 Lifestyle & Diet Plan",
                                            "📄 Full Report"
                                        ])
                                        
                                        # Tab 1: Diagnosis
                                        with analysis_tabs[0]:
                                            # Diagnosis Card
                                            if analysis.get('diagnosis'):
                                                diagnosis = analysis['diagnosis']
                                                severity_colors = {
                                                    'mild': '#10b981',
                                                    'moderate': '#f59e0b', 
                                                    'severe': '#ef4444',
                                                    'critical': '#dc2626'
                                                }
                                                severity = diagnosis.get('severity', 'unknown').lower()
                                                severity_color = severity_colors.get(severity, '#64748b')
                                                
                                                st.markdown(f"""
                                                <div class="modern-card" style="background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%); 
                                                            border-left: 6px solid {severity_color}; margin: 0.75rem 0; padding: 1.25rem;">
                                                    <h2 style="margin: 0 0 0.75rem 0; color: #78350f; font-size: 1.5rem;">
                                                        🏥 Medical Condition Identified
                                                    </h2>
                                                    <div style="background: white; padding: 1.25rem; border-radius: 8px; margin-bottom: 0.75rem;">
                                                        <div style="font-size: 1.5rem; font-weight: 700; color: {severity_color}; margin-bottom: 0.5rem;">
                                                            {diagnosis.get('primary_condition', 'No specific condition identified')}
                                                        </div>
                                                        {f'<div style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">ICD-10: {diagnosis.get("icd10_code", "Not specified")}</div>' if diagnosis.get('icd10_code') else ''}
                                                        <div style="display: inline-block; background: {severity_color}; color: white; 
                                                                    padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; margin-right: 1rem;">
                                                            Severity: {severity.upper()}
                                                        </div>
                                                        <div style="display: inline-block; background: #e2e8f0; color: #334155; 
                                                                    padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600;">
                                                            Confidence: {diagnosis.get('confidence', 'medium').upper()}
                                                        </div>
                                                    </div>
                                                    {f'''<div style="background: white; padding: 1rem; border-radius: 8px;">
                                                        <strong style="color: #78350f;">Additional Conditions:</strong>
                                                        <ul style="margin: 0.5rem 0 0 1.5rem; color: #92400e;">
                                                            {"".join([f"<li>{cond}</li>" for cond in diagnosis.get('additional_conditions', [])])}
                                                        </ul>
                                                    </div>''' if diagnosis.get('additional_conditions') else ''}
                                                </div>
                                                """, unsafe_allow_html=True)
                                            
                                            # Clinical Interpretation
                                            if analysis.get('clinical_interpretation'):
                                                st.markdown(f"""
                                                <div class="info-box" style="margin: 0.75rem 0; padding: 1rem;">
                                                    <h4 style="margin: 0 0 0.5rem 0; color: #1e3a8a; font-size: 1.1rem;">🔬 Clinical Interpretation</h4>
                                                    <p style="margin: 0; font-size: 1rem; line-height: 1.6; color: #1e40af;">
                                                        {analysis['clinical_interpretation']}
                                                    </p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            
                                            # Executive Summary
                                            if analysis.get('summary'):
                                                st.markdown('<div class="modern-card" style="background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); padding: 1.25rem; margin: 0.75rem 0;">', unsafe_allow_html=True)
                                                st.markdown('<h3 style="margin: 0 0 0.75rem 0; color: #0369a1; font-size: 1.25rem;">📝 Executive Summary</h3>', unsafe_allow_html=True)
                                                # Use st.write for proper text rendering without encoding issues
                                                st.markdown('<div style="background: white; padding: 1rem; border-radius: 8px;">', unsafe_allow_html=True)
                                                st.write(analysis['summary'])
                                                st.markdown('</div></div>', unsafe_allow_html=True)
                                            
                                            # Disease Indicators
                                            if analysis.get('disease_indicators'):
                                                st.markdown('<div class="modern-card" style="margin: 0.75rem 0; padding: 1rem;">', unsafe_allow_html=True)
                                                st.markdown("<h4 style='margin: 0 0 0.75rem 0; font-size: 1.1rem;'>🧬 Disease Indicators & Biomarkers</h4>", unsafe_allow_html=True)
                                                for indicator in analysis['disease_indicators']:
                                                    st.markdown(f"""
                                                    <div style="padding: 0.6rem; margin: 0.4rem 0; background: #fef3c7; 
                                                                border-left: 4px solid #f59e0b; border-radius: 6px; font-size: 0.95rem;">
                                                        🔸 {indicator}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                st.markdown('</div>', unsafe_allow_html=True)
                                        
                                        # Tab 2: Test Results & Metrics
                                        with analysis_tabs[1]:
                                            st.markdown("### 📊 Your Test Results")
                                            
                                            # Key Findings
                                            if analysis.get('key_findings'):
                                                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                                                st.markdown("#### 🔍 Key Clinical Findings")
                                                for i, finding in enumerate(analysis['key_findings'], 1):
                                                    st.markdown(f"""
                                                    <div style="padding: 0.75rem; margin: 0.5rem 0; background: #f8fafc; 
                                                                border-left: 4px solid #0ea5e9; border-radius: 6px;">
                                                        <strong style="color: #0369a1;">{i}.</strong> {finding}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                st.markdown('</div>', unsafe_allow_html=True)
                                            
                                            # Metrics Grid
                                            if analysis.get('metrics'):
                                                st.markdown("#### 📈 Laboratory Values")
                                                cols = st.columns(min(3, len(analysis['metrics'])))
                                                for idx, (metric, value) in enumerate(analysis['metrics'].items()):
                                                    with cols[idx % 3]:
                                                        st.markdown("""
                                                        <div class="stat-card">
                                                            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                                                                {}
                                                            </div>
                                                            <div style="font-size: 1.25rem; font-weight: 700; color: #0ea5e9; margin-top: 0.5rem; line-height: 1.4;">
                                                                {}
                                                            </div>
                                                        </div>
                                                        """.format(metric.replace('_', ' ').title(), value), unsafe_allow_html=True)
                                        
                                        # Tab 3: What You Need To Do (Lifestyle & Actions)
                                        with analysis_tabs[2]:
                                            st.markdown("### 💡 Your Action Plan")
                                            st.markdown("""
                                            <div class="info-box">
                                                <p style="margin: 0; font-size: 0.95rem;">
                                                    Follow these recommendations to manage your condition and improve your health.
                                                </p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Categorize recommendations
                                            if analysis.get('recommendations'):
                                                recs = analysis['recommendations']
                                                
                                                # Lifestyle recommendations
                                                lifestyle_recs = [r for r in recs if any(keyword in r.lower() for keyword in ['diet', 'exercise', 'lifestyle', 'reduce', 'limit', 'avoid', 'increase', 'weight', 'activity', 'sleep', 'stress', 'smoking', 'alcohol'])]
                                                
                                                # Medical recommendations
                                                medical_recs = [r for r in recs if any(keyword in r.lower() for keyword in ['medication', 'therapy', 'initiate', 'start', 'drug', 'prescription', 'doctor', 'physician', 'specialist'])]
                                                
                                                # Follow-up recommendations
                                                followup_recs = [r for r in recs if any(keyword in r.lower() for keyword in ['follow-up', 'test', 'monitor', 'repeat', 'check', 'recheck', 'visit', 'appointment'])]
                                                
                                                # Other recommendations
                                                other_recs = [r for r in recs if r not in lifestyle_recs + medical_recs + followup_recs]
                                                
                                                # Display Lifestyle Changes
                                                if lifestyle_recs:
                                                    st.markdown('<div class="modern-card" style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);">', unsafe_allow_html=True)
                                                    st.markdown("#### 🥗 Lifestyle Changes")
                                                    st.markdown('<div style="background: white; padding: 1rem; border-radius: 8px;">', unsafe_allow_html=True)
                                                    for rec in lifestyle_recs:
                                                        st.markdown(f"""
                                                        <div style="padding: 0.75rem; margin: 0.5rem 0; background: #ecfdf5; 
                                                                    border-left: 4px solid #10b981; border-radius: 6px;">
                                                            <strong style="color: #059669;">✓</strong> {rec}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    st.markdown('</div></div>', unsafe_allow_html=True)
                                                
                                                # Display Medical Actions
                                                if medical_recs:
                                                    st.markdown('<div class="modern-card" style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);">', unsafe_allow_html=True)
                                                    st.markdown("#### 💊 Medical Treatment")
                                                    st.markdown('<div style="background: white; padding: 1rem; border-radius: 8px;">', unsafe_allow_html=True)
                                                    for rec in medical_recs:
                                                        st.markdown(f"""
                                                        <div style="padding: 0.75rem; margin: 0.5rem 0; background: #eff6ff; 
                                                                    border-left: 4px solid #3b82f6; border-radius: 6px;">
                                                            <strong style="color: #1e40af;">💊</strong> {rec}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    st.markdown('</div></div>', unsafe_allow_html=True)
                                                
                                                # Display Follow-up Actions
                                                if followup_recs:
                                                    st.markdown('<div class="modern-card" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);">', unsafe_allow_html=True)
                                                    st.markdown("#### 📅 Follow-up & Monitoring")
                                                    st.markdown('<div style="background: white; padding: 1rem; border-radius: 8px;">', unsafe_allow_html=True)
                                                    for rec in followup_recs:
                                                        st.markdown(f"""
                                                        <div style="padding: 0.75rem; margin: 0.5rem 0; background: #fefce8; 
                                                                    border-left: 4px solid #f59e0b; border-radius: 6px;">
                                                            <strong style="color: #d97706;">📅</strong> {rec}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    st.markdown('</div></div>', unsafe_allow_html=True)
                                                
                                                # Display Other Recommendations
                                                if other_recs:
                                                    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                                                    st.markdown("#### ℹ️ Additional Recommendations")
                                                    for rec in other_recs:
                                                        st.markdown(f"""
                                                        <div class="success-box" style="margin: 0.5rem 0;">
                                                            <strong style="color: #059669;">✓</strong> {rec}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    st.markdown('</div>', unsafe_allow_html=True)
                                            else:
                                                st.info("No specific recommendations provided in the analysis.")
                                        
                                        # Tab 4: Warnings & Concerns
                                        with analysis_tabs[3]:
                                            st.markdown("### ⚠️ Important Warnings")
                                            
                                            # Concerns/Alerts
                                            if analysis.get('concerns') or analysis.get('alerts'):
                                                concerns = analysis.get('concerns', []) + analysis.get('alerts', [])
                                                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                                                st.markdown("#### ⚠️ Areas Requiring Attention")
                                                for concern in concerns:
                                                    st.markdown(f"""
                                                    <div class="warning-box" style="margin: 0.75rem 0;">
                                                        <strong style="color: #d97706;">⚠️</strong> {concern}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                st.markdown('</div>', unsafe_allow_html=True)
                                            else:
                                                st.success("✅ No major concerns identified in the analysis.")
                                        
                                        # Tab 5: Lifestyle & Diet Plan
                                        with analysis_tabs[4]:
                                            lifestyle_plan = result.get('lifestyle_diet_plan')
                                            if lifestyle_plan:
                                                st.markdown("### 🥗 Personalized Lifestyle & Diet Plan")
                                                st.info("Based on your medical report analysis, here's a personalized health plan:")
                                                
                                                # Import display function from ai_decision_support page
                                                try:
                                                    from pages.ai_decision_support import AIDecisionDashboard
                                                    ai_page = AIDecisionDashboard()
                                                    ai_page.display_lifestyle_diet_plan(lifestyle_plan)
                                                except Exception as e:
                                                    st.error(f"Error displaying lifestyle plan: {e}")
                                                    
                                                    # Fallback display
                                                    if lifestyle_plan.get('diet_plan'):
                                                        with st.expander("🍽️ Diet Plan", expanded=True):
                                                            diet = lifestyle_plan['diet_plan']
                                                            st.write(f"**Approach:** {diet.get('approach')}")
                                                            st.write(f"**Daily Calories:** {diet.get('daily_calories')} kcal")
                                                            
                                                    if lifestyle_plan.get('exercise_plan'):
                                                        with st.expander("🏃 Exercise Plan", expanded=True):
                                                            exercise = lifestyle_plan['exercise_plan']
                                                            st.write(f"**Weekly Target:** {exercise.get('weekly_target')}")
                                                            
                                                    if lifestyle_plan.get('lifestyle_modifications'):
                                                        with st.expander("🌟 Lifestyle Changes", expanded=True):
                                                            for mod in lifestyle_plan['lifestyle_modifications'][:5]:
                                                                st.write(f"• {mod}")
                                            else:
                                                st.info("Lifestyle & Diet Plan not available for this report. Complete medical data required.")
                                        
                                        # Tab 6: Full Report
                                        with analysis_tabs[5]:
                                            st.markdown("### 📄 Complete Analysis Report")
                                            
                                            # Summary if available
                                            if analysis.get('summary'):
                                                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                                                st.markdown("#### Executive Summary")
                                                st.write(analysis['summary'])
                                                st.markdown('</div>', unsafe_allow_html=True)
                                            
                                            # Full JSON view
                                            with st.expander("🔍 View Complete JSON Data", expanded=False):
                                                st.json(analysis)
                                            
                                            # Extracted text
                                            with st.expander("📄 View Raw Extracted Text", expanded=False):
                                                st.text_area(
                                                    "Extracted Text", 
                                                    value=result.get('extracted_text', 'N/A'), 
                                                    height=300, 
                                                    key="extracted_mr", 
                                                    disabled=True
                                                )
                                    else:
                                        st.markdown("""
                                        <div class="warning-box">
                                            <strong>⚠️ Analysis Incomplete</strong><br>
                                            The document was processed but AI analysis data is incomplete.
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-box">
                                    <h4 style="margin: 0; color: #dc2626;">❌ Analysis Error</h4>
                                    <p style="margin: 0.5rem 0 0 0; color: #991b1b;">{str(e)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show previous analyses
                st.markdown("---")
                st.markdown("### 📚 Previous Analyses")
                if st.button("🔄 Load Previous Reports", key="load_mr"):
                    try:
                        results = api_client._make_request('GET', '/document/list')
                        if results:
                            medical_reports = [r for r in results if r.get('document_type') == 'medical_report']
                            if medical_reports:
                                for report in medical_reports[:5]:  # Show last 5
                                    with st.expander(f"📄 {report.get('filename', 'Unknown')} - {report.get('created_at', '')}"):
                                        st.write(f"**Patient:** {report.get('patient_name', 'N/A')}")
                                        if report.get('summary'):
                                            st.write(f"**Summary:** {report['summary']}")
                            else:
                                st.info("No previous medical reports found.")
                    except Exception as e:
                        st.error(f"Error loading reports: {str(e)}")
            
            with doc_tab2:
                st.markdown("### 💊 Prescription Analysis")
                st.markdown("Upload prescriptions for AI-powered drug information and interaction analysis.")
                
                # Patient name input
                patient_name_rx = st.text_input("Patient Name (Optional)", key="rx_patient_name")
                
                # File upload
                uploaded_rx = st.file_uploader(
                    "Upload Prescription", 
                    type=['pdf', 'png', 'jpg', 'jpeg', 'txt'], 
                    key="prescription",
                    help="Supported formats: PDF, PNG, JPG, JPEG, TXT"
                )
                
                if uploaded_rx:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 Selected: {uploaded_rx.name} ({uploaded_rx.size / 1024:.1f} KB)")
                    with col2:
                        analyze_rx_btn = st.button("🔍 Analyze", use_container_width=True, type="primary", key="analyze_rx")
                    
                    if analyze_rx_btn:
                        with st.spinner("🤖 Analyzing prescription..."):
                            try:
                                # Prepare file for upload
                                files = {
                                    'file': (uploaded_rx.name, uploaded_rx.getvalue(), uploaded_rx.type)
                                }
                                data = {}
                                if patient_name_rx:
                                    data['patient_name'] = patient_name_rx
                                
                                # Call API
                                result = api_client._make_request(
                                    'POST',
                                    '/document/upload/prescription',
                                    files=files,
                                    data=data
                                )
                                
                                if result:
                                    st.success("✅ Analysis Complete!")
                                    
                                    # Display analysis results with professional mini navbar
                                    st.markdown("---")
                                    
                                    # Display AI analysis with tab navigation
                                    analysis = result.get('analysis', {})
                                    if analysis:
                                        # Custom CSS for professional card design
                                        st.markdown("""
                                        <style>
                                        .medication-card {
                                            background: #f8f9fa;
                                            border: none;
                                            padding: 24px;
                                            margin-bottom: 20px;
                                            border-radius: 12px;
                                            transition: all 0.2s;
                                        }
                                        .medication-card:hover {
                                            background: #e9ecef;
                                        }
                                        .medication-header {
                                            color: #1a1a1a;
                                            font-size: 20px;
                                            font-weight: 600;
                                            margin-bottom: 16px;
                                            padding-bottom: 12px;
                                            border-bottom: 2px solid #dee2e6;
                                        }
                                        .info-row {
                                            display: flex;
                                            gap: 24px;
                                            margin-top: 12px;
                                        }
                                        .info-item {
                                            flex: 1;
                                        }
                                        .info-label {
                                            color: #6c757d;
                                            font-size: 13px;
                                            font-weight: 600;
                                            text-transform: uppercase;
                                            letter-spacing: 0.5px;
                                            margin-bottom: 4px;
                                        }
                                        .info-value {
                                            color: #212529;
                                            font-size: 16px;
                                            font-weight: 500;
                                        }
                                        .alert-box {
                                            padding: 16px 20px;
                                            border-radius: 8px;
                                            margin-bottom: 12px;
                                            border-left: 4px solid;
                                            font-size: 15px;
                                        }
                                        .alert-error {
                                            background: #fff5f5;
                                            border-left-color: #e53e3e;
                                            color: #742a2a;
                                        }
                                        .alert-info {
                                            background: #f0f9ff;
                                            border-left-color: #3b82f6;
                                            color: #1e3a8a;
                                        }
                                        .alert-success {
                                            background: #f0fdf4;
                                            border-left-color: #22c55e;
                                            color: #14532d;
                                        }
                                        </style>
                                        """, unsafe_allow_html=True)
                                        
                                        # Create tabs for each section
                                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                                            "💊 Medications", 
                                            "⚠️ Interactions", 
                                            "🛡️ Safety", 
                                            "📊 Dosage", 
                                            "💡 Recommendations", 
                                            "📄 Source Text"
                                        ])
                                        
                                        # Tab 1: Medications
                                        with tab1:
                                            if analysis.get('medications'):
                                                meds = analysis['medications']
                                                if meds and isinstance(meds[0], dict):
                                                    for i, med in enumerate(meds, 1):
                                                        st.markdown(f"""
                                                        <div class="medication-card">
                                                            <div class="medication-header">{i}. {med.get('name', 'Unknown Medication')}</div>
                                                            <div class="info-row">
                                                                <div class="info-item">
                                                                    <div class="info-label">Purpose</div>
                                                                    <div class="info-value">{med.get('indication', 'Not specified')}</div>
                                                                </div>
                                                                <div class="info-item">
                                                                    <div class="info-label">Dosage</div>
                                                                    <div class="info-value">{med.get('dosage', 'N/A')}</div>
                                                                </div>
                                                                <div class="info-item">
                                                                    <div class="info-label">Frequency</div>
                                                                    <div class="info-value">{med.get('frequency', 'N/A')}</div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                else:
                                                    for i, med in enumerate(meds, 1):
                                                        st.markdown(f"""
                                                        <div class="medication-card">
                                                            <strong>{i}.</strong> {med}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                            else:
                                                st.info("ℹ️ No medications identified in this prescription")
                                        
                                        # Tab 2: Drug Interactions
                                        with tab2:
                                            if analysis.get('interactions'):
                                                for interaction in analysis['interactions']:
                                                    st.markdown(f"""
                                                    <div class="alert-box alert-error">
                                                        <strong>⚠️ Warning:</strong> {interaction}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                            else:
                                                st.markdown("""
                                                <div class="alert-box alert-success">
                                                    <strong>✓ No significant drug interactions detected</strong>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        
                                        # Tab 3: Safety Information
                                        with tab3:
                                            if analysis.get('safety_info'):
                                                for info in analysis['safety_info']:
                                                    st.markdown(f"""
                                                    <div class="alert-box alert-info">
                                                        ℹ️ {info}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                            else:
                                                st.info("ℹ️ No specific safety warnings")
                                        
                                        # Tab 4: Dosage Information
                                        with tab4:
                                            if analysis.get('dosages'):
                                                for dosage in analysis['dosages']:
                                                    st.markdown(f"""
                                                    <div class="medication-card">
                                                        • {dosage}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                            else:
                                                st.info("ℹ️ Refer to individual medication dosages in the Medications tab")
                                        
                                        # Tab 5: Recommendations
                                        with tab5:
                                            if analysis.get('recommendations'):
                                                for rec in analysis['recommendations']:
                                                    st.markdown(f"""
                                                    <div class="alert-box alert-success">
                                                        ✓ {rec}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                            else:
                                                st.info("ℹ️ Follow your physician's instructions")
                                        
                                        # Tab 6: Extracted Text
                                        with tab6:
                                            st.text_area("Source Text from Prescription", value=result.get('extracted_text', 'No text extracted'), height=400, disabled=True)
                                            
                                            if analysis.get('details'):
                                                with st.expander("📋 Technical Details (JSON)", expanded=False):
                                                    st.json(analysis['details'])
                                    else:
                                        st.warning("No AI analysis available. The prescription was processed but analysis data is missing.")
                                    
                                    # Show extracted text only in collapsed section
                                    with st.expander("📄 View Extracted Text", expanded=False):
                                        st.text_area("Extracted Text", value=result.get('extracted_text', 'N/A'), height=200, key="extracted_rx", disabled=True)
                                
                            except Exception as e:
                                st.error(f"❌ Error analyzing prescription: {str(e)}")
                
                # Show previous analyses
                st.markdown("---")
                st.markdown("### 📚 Previous Analyses")
                if st.button("🔄 Load Previous Prescriptions", key="load_rx"):
                    try:
                        results = api_client._make_request('GET', '/document/list')
                        if results:
                            prescriptions = [r for r in results if r.get('document_type') == 'prescription']
                            if prescriptions:
                                for rx in prescriptions[:5]:  # Show last 5
                                    with st.expander(f"💊 {rx.get('filename', 'Unknown')} - {rx.get('created_at', '')}"):
                                        st.write(f"**Patient:** {rx.get('patient_name', 'N/A')}")
                                        if rx.get('summary'):
                                            st.write(f"**Summary:** {rx['summary']}")
                            else:
                                st.info("No previous prescriptions found.")
                    except Exception as e:
                        st.error(f"Error loading prescriptions: {str(e)}")
        
        elif clean_page == "Patient Management" or "Patient Management" in page:
            if is_doctor():
                # Professional page header
                st.markdown("""
                <div style="background: linear-gradient(135deg, var(--background-primary) 0%, var(--secondary-color) 100%); 
                            padding: 2rem; border-radius: var(--border-radius-lg); margin-bottom: 2rem; 
                            border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
                    <h1 style="color: var(--primary-color); margin: 0; font-size: 2.25rem; font-weight: 700;">
                        👥 Patient Management
                    </h1>
                    <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem; line-height: 1.5;">
                        Comprehensive patient records and healthcare management system
                    </p>
                </div>
                """, unsafe_allow_html=True)
                render_patient_management()
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); 
                            padding: 2rem; border-radius: var(--border-radius-lg); margin: 2rem 0; 
                            border: 1px solid #fecaca; text-align: center;">
                    <h2 style="color: var(--danger-color); margin: 0 0 1rem 0;">🚫 Access Restricted</h2>
                    <p style="color: #991b1b; margin: 0; font-size: 1.1rem;">
                        This section requires Doctor or Administrator privileges.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        elif clean_page == "Admin Panel" or "Admin Panel" in page:
            if is_admin():
                # Professional page header
                st.markdown("""
                <div style="background: linear-gradient(135deg, var(--background-primary) 0%, var(--secondary-color) 100%); 
                            padding: 2rem; border-radius: var(--border-radius-lg); margin-bottom: 2rem; 
                            border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
                    <h1 style="color: var(--primary-color); margin: 0; font-size: 2.25rem; font-weight: 700;">
                        🔧 System Administration
                    </h1>
                    <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem; line-height: 1.5;">
                        Advanced system configuration and administrative controls
                    </p>
                </div>
                """, unsafe_allow_html=True)
                render_admin_panel()
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); 
                            padding: 2rem; border-radius: var(--border-radius-lg); margin: 2rem 0; 
                            border: 1px solid #fecaca; text-align: center;">
                    <h2 style="color: var(--danger-color); margin: 0 0 1rem 0;">🚫 Administrator Access Required</h2>
                    <p style="color: #991b1b; margin: 0; font-size: 1.1rem;">
                        This section is restricted to system administrators only.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        elif clean_page == "User Management" or "User Management" in page:
            if is_admin():
                # Professional page header
                st.markdown("""
                <div style="background: linear-gradient(135deg, var(--background-primary) 0%, var(--secondary-color) 100%); 
                            padding: 2rem; border-radius: var(--border-radius-lg); margin-bottom: 2rem; 
                            border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
                    <h1 style="color: var(--primary-color); margin: 0; font-size: 2.25rem; font-weight: 700;">
                        👤 User Management
                    </h1>
                    <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem; line-height: 1.5;">
                        Comprehensive user account management and access control system
                    </p>
                </div>
                """, unsafe_allow_html=True)
                render_user_management()
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); 
                            padding: 2rem; border-radius: var(--border-radius-lg); margin: 2rem 0; 
                            border: 1px solid #fecaca; text-align: center;">
                    <h2 style="color: var(--danger-color); margin: 0 0 1rem 0;">🚫 Administrator Access Required</h2>
                    <p style="color: #991b1b; margin: 0; font-size: 1.1rem;">
                        User management is restricted to system administrators only.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        elif clean_page == "About" or "About" in page:
            # Professional page header
            st.markdown("""
            <div style="background: linear-gradient(135deg, var(--background-primary) 0%, var(--secondary-color) 100%); 
                        padding: 2rem; border-radius: var(--border-radius-lg); margin-bottom: 2rem; 
                        border: 1px solid var(--border-color); box-shadow: var(--shadow-sm);">
                <h1 style="color: var(--primary-color); margin: 0; font-size: 2.25rem; font-weight: 700;">
                    ℹ️ About MyVitals
                </h1>
                <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem; line-height: 1.5;">
                    Your personal AI-powered health analytics and wellness companion
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Professional about content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                            border: 1px solid var(--border-color); box-shadow: var(--shadow-sm); margin-bottom: 1.5rem;">
                    <h2 style="color: var(--primary-color); margin-top: 0;">🎯 Mission Statement</h2>
                    <p style="color: var(--text-secondary); line-height: 1.6; font-size: 1.05rem;">
                        MyVitals empowers individuals to take control of their health through advanced AI-powered analytics. 
                        We provide personalized health insights, risk assessments, and intelligent recommendations 
                        to help you make informed decisions about your wellness journey.
                    </p>
                </div>
                
                <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                            border: 1px solid var(--border-color); box-shadow: var(--shadow-sm); margin-bottom: 1.5rem;">
                    <h2 style="color: var(--primary-color); margin-top: 0;">🔬 Advanced Technology Stack</h2>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">🎨 Frontend Technologies</h4>
                            <ul style="color: var(--text-secondary); margin: 0; padding-left: 1.5rem;">
                                <li>Streamlit for interactive UI</li>
                                <li>Professional CSS framework</li>
                                <li>Responsive design system</li>
                                <li>Real-time data visualization</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">⚙️ Backend Infrastructure</h4>
                            <ul style="color: var(--text-secondary); margin: 0; padding-left: 1.5rem;">
                                <li>FastAPI for robust API services</li>
                                <li>SQLAlchemy ORM</li>
                                <li>JWT authentication</li>
                                <li>RESTful architecture</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">🤖 AI & Machine Learning</h4>
                            <ul style="color: var(--text-secondary); margin: 0; padding-left: 1.5rem;">
                                <li>OpenAI Whisper (Speech-to-Text)</li>
                                <li>Facebook BART (Summarization)</li>
                                <li>Random Forest & Gradient Boosting</li>
                                <li>Natural language processing</li>
                                <li>Predictive analytics</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">📊 Data & Analytics</h4>
                            <ul style="color: var(--text-secondary); margin: 0; padding-left: 1.5rem;">
                                <li>Pandas & NumPy processing</li>
                                <li>Plotly interactive charts</li>
                                <li>Statistical analysis</li>
                                <li>Healthcare data standards</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                            border: 1px solid var(--border-color); box-shadow: var(--shadow-sm); margin-bottom: 1.5rem;">
                    <h3 style="color: var(--primary-color); margin-top: 0;">🏆 Key Features</h3>
                    <div style="space-y: 1rem;">
                        <div style="margin-bottom: 1rem;">
                            <h4 style="color: var(--accent-color); margin: 0 0 0.25rem 0; font-size: 1rem;">🔬 Risk Assessment</h4>
                            <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">AI-powered health risk analysis</p>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <h4 style="color: var(--accent-color); margin: 0 0 0.25rem 0; font-size: 1rem;">📊 Health Analytics</h4>
                            <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">Comprehensive health tracking</p>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <h4 style="color: var(--accent-color); margin: 0 0 0.25rem 0; font-size: 1rem;">🏥 Professional Tools</h4>
                            <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">Medical professional features</p>
                        </div>
                    </div>
                </div>
                
                <div style="background: linear-gradient(135deg, var(--accent-color), #059669); padding: 2rem; 
                            border-radius: var(--border-radius-lg); color: white; text-align: center;">
                    <h3 style="margin: 0 0 1rem 0;">📈 Platform Statistics</h3>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 2rem; font-weight: 700;">99.9%</div>
                        <div style="font-size: 0.9rem; opacity: 0.9;">System Uptime</div>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 2rem; font-weight: 700;">HIPAA</div>
                        <div style="font-size: 0.9rem; opacity: 0.9;">Compliant Security</div>
                    </div>
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;">24/7</div>
                        <div style="font-size: 0.9rem; opacity: 0.9;">AI Assistance</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # System metrics and additional information
            st.markdown("""
            <div style="background: var(--background-primary); padding: 2rem; border-radius: var(--border-radius-lg); 
                        border: 1px solid var(--border-color); box-shadow: var(--shadow-sm); margin-top: 1.5rem;">
                <h2 style="color: var(--primary-color); margin-top: 0;">🔒 Privacy & Security</h2>
                <ul style="color: var(--text-secondary); line-height: 1.6;">
                    <li><strong>HIPAA Compliance:</strong> All patient data is handled according to healthcare privacy standards</li>
                    <li><strong>Secure Processing:</strong> Data is encrypted in transit and at rest</li>
                    <li><strong>No Permanent Storage:</strong> Personal health information is not stored permanently</li>
                    <li><strong>Access Controls:</strong> Role-based authentication and authorization</li>
                    <li><strong>Audit Trails:</strong> Comprehensive logging for security monitoring</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
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

    except Exception as e:
        st.error("🚨 Application Error")
        st.error(f"An unexpected error occurred: {str(e)}")

        # Provide recovery options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Page"):
                st.rerun()
        with col2:
            if st.button("🚪 Logout & Restart"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        # Show error details in expander for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()