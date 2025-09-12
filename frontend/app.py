import streamlit as st
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
from components.chatbot import render_chatbot_interface, render_chatbot_sidebar
from components.prescription_chatbot import render_prescription_chatbot
from pages.health_log import create_health_log_page
from utils.api_client import HealthcareAPI
from utils.caching import cleanup_expired_cache, get_cache_stats
import json
import time
import os
from datetime import datetime

def load_css():
    """Load custom CSS styles - cached for performance"""
    if 'css_loaded' not in st.session_state:
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

# Check authentication before proceeding
if not st.session_state.authenticated or not check_session_validity():
    render_login_page()
    # Show development credentials in sidebar during login (for development purposes)
    import os
    if os.getenv("SHOW_TEST_CREDENTIALS", "true").lower() == "true":
        show_development_credentials()
    st.stop()

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
        
        # Get users from backend (mock implementation)
        users_data = {
            "ID": ["U001", "U002", "U003", "U004"],
            "Name": ["Dr. Jane Smith", "John Doe", "Alice Johnson", "Bob Wilson"],
            "Email": ["doctor@healthcare.com", "patient@healthcare.com", "alice@healthcare.com", "bob@healthcare.com"],
            "Role": ["Doctor", "Patient", "Doctor", "Patient"],
            "Status": ["Active", "Active", "Active", "Inactive"],
            "Last Login": ["2024-01-15 10:30", "2024-01-14 15:20", "2024-01-13 09:15", "Never"]
        }
        
        users_df = pd.DataFrame(users_data)
        
        # Add action buttons
        for idx, row in users_df.iterrows():
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 2, 2, 1, 1, 1, 1])
            
            with col1:
                st.write(row['ID'])
            with col2:
                st.write(row['Name'])
            with col3:
                st.write(row['Email'])
            with col4:
                role_color = "blue" if row['Role'] == "Doctor" else "green"
                st.markdown(f":{role_color}[{row['Role']}]")
            with col5:
                status_color = "green" if row['Status'] == "Active" else "red"
                st.markdown(f":{status_color}[{row['Status']}]")
            with col6:
                if st.button("Edit", key=f"edit_{idx}"):
                    st.info(f"Edit user {row['Name']}")
            with col7:
                if st.button("Delete", key=f"delete_{idx}", type="secondary"):
                    st.warning(f"Delete user {row['Name']}?")
        
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
                    # Here you would integrate with the backend API
                    st.success(f"User {new_name} created successfully!")
                    st.info("In a real implementation, this would call the backend API to create the user.")
                else:
                    st.error("Please fill in all required fields.")


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
        
        # Display model info if available (cached)
        model_info = api_client.get_model_info()
        if model_info:
            with st.sidebar.expander("Model Information", expanded=False):
                st.markdown(f"**Model Type:** {model_info.get('model_type', 'N/A')}")
                st.markdown(f"**Version:** {model_info.get('version', '1.0.0')}")
                st.markdown(f"**Features:** {model_info.get('feature_count', 'N/A')}")
                st.markdown(f"**Last Trained:** {model_info.get('last_trained', 'N/A')}")
            
    else:
        st.sidebar.markdown(
            f"<div class='status-indicator status-error'>"
            f"<span style='margin-right: 8px;'><i class='fas fa-times-circle'></i></span>"
            "Connection Error"
            "</div>",
            unsafe_allow_html=True
        )

def main():
    """Main application"""
    # Initialize page state if not exists
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Risk Assessment"
    
    # Handle page redirects from auth
    if hasattr(st.session_state, 'redirect_to'):
        st.session_state.page = st.session_state.redirect_to
        del st.session_state.redirect_to
        st.rerun()
    
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
    
    # Sidebar navigation with authentication-aware options
    render_user_info()  # Display user info in sidebar
    
    st.sidebar.title("Navigation")
    
    # Build navigation options based on user role
    nav_options = ["🏠 Risk Assessment", "💊 Prescription Analysis", "🏥 Medical Report Analysis", "📊 Health Log", "🤖 AI Assistant"]
    
    # Add role-specific navigation
    if is_doctor() or is_admin():
        nav_options.extend(["👥 Patient Management"])
    
    # Admin-only navigation
    if is_admin():
        nav_options.extend(["🔧 Admin Panel", "👤 User Management"])
    
    # Add About page
    nav_options.append("ℹ️ About")
    
    # Get the current page from session state or default to first option
    default_index = nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0
    
    # Update page based on sidebar selection
    page = st.sidebar.radio(
        "Go to",
        nav_options,
        index=default_index,
        key="page_selector"
    )
    
    # Update session state page
    st.session_state.page = page
    
    # Display connection status in sidebar
    display_connection_status()
    
    # Add sidebar chatbot (only show if not on main chatbot page)
    if "🤖 AI Assistant" not in page:
        render_chatbot_sidebar()
    
    # Add logout button in footer
    if st.session_state.authenticated:
        if st.sidebar.button("🚪 Logout", use_container_width=True, type="primary", key="main_logout_btn"):
            from components.auth import logout_user
            logout_user()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.8rem; margin-top: 1rem;">
        <p> 2023 Personalized Healthcare System</p>
        <p>v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    if "🏠 Risk Assessment" in page:
        st.header("🔬 Advanced Health Risk Assessment")
        st.markdown("""
        **Comprehensive health risk assessment with laboratory data integration**
        
        Enhanced lab-based analysis for accurate health predictions using comprehensive laboratory parameters.
        """)
        
        # Create tabs for different analysis types
        lab_tab, results_tab = st.tabs(["🔬 Lab Analysis", "📊 Results"])
        
        with lab_tab:
            st.markdown("### 🔬 Enhanced Lab-Based Analysis")
            st.markdown("""
            **Advanced health risk assessment using comprehensive laboratory data**
            
            This enhanced analysis incorporates detailed hematology and blood chemistry values 
            to provide more accurate and comprehensive health risk predictions.
            """)
            
            # Initialize lab form
            lab_form = LabPatientInputForm()
            lab_patient_data = lab_form.render()
            
            # Analysis options
            st.markdown("### Analysis Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                lab_force_llm = st.checkbox("Enable AI Analysis", 
                                          value=True,
                                          key="lab_force_llm",
                                          help="Uses AI for comprehensive lab interpretation and clinical insights")
            
            with col2:
                lab_analysis_type = st.selectbox("Analysis Type", 
                                               ["🔬 Lab Analysis", "📊 Basic Assessment"],
                                               key="lab_analysis_type",
                                               help="Lab Analysis uses AI for complete medical interpretation")
            
            # Prediction button
            if st.button("🔍 Analyze Health Risk", type="primary", key="lab_analyze_btn"):
                if lab_patient_data:
                    with st.spinner("Analyzing your health data..."):
                        try:
                            if lab_analysis_type == "🔬 Lab Analysis":
                                result = api_client.make_lab_prediction(lab_patient_data, lab_force_llm)
                            else:
                                result = api_client.make_prediction(lab_patient_data, lab_force_llm)
                            
                            if result:
                                st.session_state['prediction'] = result
                                st.session_state['prediction_result'] = result
                                st.session_state['patient_data'] = lab_patient_data
                                st.session_state['last_analyzed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state['analysis_type'] = lab_analysis_type
                                st.success("✅ Analysis completed! Check the 'Results' tab.")
                            else:
                                st.error("❌ Analysis failed. Please check your input data and try again.")
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                else:
                    st.warning("⚠️ Please fill in the required basic information to proceed.")
        
        with results_tab:
            if 'prediction' in st.session_state and st.session_state['prediction']:
                display_prediction_results(st.session_state['prediction'], 
                                         st.session_state.get('analysis_type', 'Unknown'))
            else:
                st.info("🔍 No analysis results yet. Please run an analysis first.")
    
    elif "💊 Prescription Analysis" in page:
        st.header("💊 Prescription Analysis")
        render_prescription_chatbot()
    
    elif "🏥 Medical Report Analysis" in page:
        from pages.medical_report_analysis import MedicalReportAnalyzer
        analyzer = MedicalReportAnalyzer()
        analyzer.run()
    
    elif "📊 Health Log" in page:
        st.header("📊 Health Log")
        create_health_log_page()
    
    elif "🤖 AI Assistant" in page:
        render_chatbot_interface()
    
    
    elif "👥 Patient Management" in page:
        if is_doctor():
            render_patient_management()
        else:
            st.error("🙅‍♂️ Access denied. Doctor or Admin access required.")
    
    elif "🔧 Admin Panel" in page:
        if is_admin():
            render_admin_panel()
        else:
            st.error("🙅‍♂️ Access denied. Administrator access required.")
    
    elif "👤 User Management" in page:
        if is_admin():
            render_user_management()
        else:
            st.error("🙅‍♂️ Access denied. Administrator access required.")
    
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