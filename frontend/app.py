import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.forms import PatientInputForm
from components.auth import (
    init_session_state, check_session_validity, render_login_page,
    render_user_info, require_auth, get_auth_headers, is_admin, is_doctor,
    show_development_credentials
)
from components.chatbot import render_chatbot_interface, render_chatbot_sidebar
from pages.dashboard import create_dashboard, calculate_risk_score
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
                    if st.button(f"View Details", key=f"view_{patient['id']}"):
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
    nav_options = ["🏠 Risk Assessment", "🤖 AI Assistant"]
    
    if is_doctor():
        nav_options.extend(["📊 Dashboard", "👥 Patient Management"])
    
    if is_admin():
        nav_options.extend(["🔧 Admin Panel", "👤 User Management"])
    
    nav_options.append("ℹ️ About")
    
    page = st.sidebar.radio(
        "Go to",
        nav_options,
        index=0
    )
    
    # Display connection status in sidebar
    display_connection_status()
    
    # Add sidebar chatbot (only show if not on main chatbot page)
    if "🤖 AI Assistant" not in page:
        render_chatbot_sidebar()
    
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
                    if st.button("🤖 Ask AI about Results", use_container_width=True, help="Get AI explanation of your risk assessment"):
                        # Store the prediction for the chatbot to access
                        st.session_state['last_prediction'] = prediction
                        st.rerun()
            else:
                with st.container():
                    st.markdown("<div class='card' style='text-align: center; padding: 2rem;'>"
                               "<h3 style='color: #4a6fa5;'>No Analysis Results</h3>"
                               "<p>Please fill in the patient information and click 'Analyze Risk' to see results.</p>"
                               "<img src='https://img.icons8.com/color/200/000000/medical-doctor.png' width='150' style='opacity: 0.7; margin: 1rem 0;'/>"
                               "</div>", 
                               unsafe_allow_html=True)
    
    elif "🤖 AI Assistant" in page:
        render_chatbot_interface()
    
    elif "📊 Dashboard" in page:
        if is_doctor():
            st.header("📊 Healthcare Analytics Dashboard")
            # Pass patient data to dashboard if available
            patient_data = st.session_state.get('patient_data', None)
            create_dashboard(patient_data)
        else:
            st.error("🙅‍♂️ Access denied. Doctor or Admin access required.")
    
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