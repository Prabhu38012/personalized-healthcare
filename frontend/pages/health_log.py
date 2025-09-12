"""
Enhanced Health Log Page with Prescription Analysis
Professional interface for health tracking and prescription management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any, Optional
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from components.prescription_chatbot import PrescriptionChatbot

def create_health_log_page():
    """Create the enhanced health log page with prescription functionality"""
    
    # Professional header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; color: white;'>
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <div>
                <h1 style='margin: 0; font-size: 2.5rem; font-weight: 300;'>📊 Health Log</h1>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>Track your vital signs, medications, and health metrics</p>
            </div>
            <div style='text-align: right;'>
                <div style='font-size: 2rem; opacity: 0.8;'>🏥</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'health_log_data' not in st.session_state:
        st.session_state.health_log_data = generate_sample_health_data()
    
    if 'prescription_chatbot' not in st.session_state:
        st.session_state.prescription_chatbot = PrescriptionChatbot()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Health Metrics", 
        "💊 Prescription Analysis", 
        "📝 Log New Data", 
        "📋 Health Reports"
    ])
    
    with tab1:
        render_health_metrics()
    
    with tab2:
        render_prescription_section()
    
    with tab3:
        render_data_entry()
    
    with tab4:
        render_health_reports()

def render_health_metrics():
    """Render health metrics dashboard"""
    st.markdown("### 📊 Your Health Metrics Overview")
    
    # Get recent health data
    health_data = st.session_state.health_log_data
    latest_data = health_data.iloc[-1] if not health_data.empty else None
    
    if latest_data is not None:
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🩸 Blood Pressure", 
                f"{latest_data.get('systolic_bp', 0)}/{latest_data.get('diastolic_bp', 0)}", 
                delta=f"{latest_data.get('bp_change', 0):+.0f} mmHg"
            )
        
        with col2:
            st.metric(
                "❤️ Heart Rate", 
                f"{latest_data.get('heart_rate', 0)} BPM", 
                delta=f"{latest_data.get('hr_change', 0):+.0f} BPM"
            )
        
        with col3:
            st.metric(
                "⚖️ Weight", 
                f"{latest_data.get('weight', 0):.1f} kg", 
                delta=f"{latest_data.get('weight_change', 0):+.1f} kg"
            )
        
        with col4:
            st.metric(
                "🌡️ Temperature", 
                f"{latest_data.get('temperature', 0):.1f}°F", 
                delta=f"{latest_data.get('temp_change', 0):+.1f}°F"
            )
        
        # Health trends charts
        st.markdown("### 📈 Health Trends (Last 30 Days)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Blood pressure trend
            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(
                x=health_data['date'],
                y=health_data['systolic_bp'],
                mode='lines+markers',
                name='Systolic',
                line=dict(color='#FF6B6B', width=3)
            ))
            fig_bp.add_trace(go.Scatter(
                x=health_data['date'],
                y=health_data['diastolic_bp'],
                mode='lines+markers',
                name='Diastolic',
                line=dict(color='#4ECDC4', width=3)
            ))
            fig_bp.update_layout(
                title="Blood Pressure Trend",
                xaxis_title="Date",
                yaxis_title="mmHg",
                height=400
            )
            st.plotly_chart(fig_bp, use_container_width=True)
        
        with col2:
            # Heart rate and weight trend
            fig_multi = go.Figure()
            fig_multi.add_trace(go.Scatter(
                x=health_data['date'],
                y=health_data['heart_rate'],
                mode='lines+markers',
                name='Heart Rate (BPM)',
                yaxis='y',
                line=dict(color='#FF9F43', width=3)
            ))
            fig_multi.add_trace(go.Scatter(
                x=health_data['date'],
                y=health_data['weight'],
                mode='lines+markers',
                name='Weight (kg)',
                yaxis='y2',
                line=dict(color='#6C5CE7', width=3)
            ))
            fig_multi.update_layout(
                title="Heart Rate & Weight Trend",
                xaxis_title="Date",
                yaxis=dict(title="Heart Rate (BPM)", side="left"),
                yaxis2=dict(title="Weight (kg)", side="right", overlaying="y"),
                height=400
            )
            st.plotly_chart(fig_multi, use_container_width=True)
        
        # Health status indicators
        st.markdown("### 🎯 Health Status Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Blood pressure status
            systolic = latest_data.get('systolic_bp', 120)
            if systolic < 120:
                bp_status = "Normal"
                bp_color = "#4CAF50"
            elif systolic < 140:
                bp_status = "Elevated"
                bp_color = "#FF9800"
            else:
                bp_status = "High"
                bp_color = "#F44336"
            
            st.markdown(f"""
            <div style='background: {bp_color}20; padding: 1rem; border-radius: 10px; border-left: 5px solid {bp_color};'>
                <h4 style='color: {bp_color}; margin: 0;'>🩸 Blood Pressure</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;'>{bp_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Heart rate status
            hr = latest_data.get('heart_rate', 70)
            if 60 <= hr <= 100:
                hr_status = "Normal"
                hr_color = "#4CAF50"
            elif hr < 60:
                hr_status = "Low"
                hr_color = "#FF9800"
            else:
                hr_status = "High"
                hr_color = "#F44336"
            
            st.markdown(f"""
            <div style='background: {hr_color}20; padding: 1rem; border-radius: 10px; border-left: 5px solid {hr_color};'>
                <h4 style='color: {hr_color}; margin: 0;'>❤️ Heart Rate</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;'>{hr_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # BMI status
            weight = latest_data.get('weight', 70)
            height = 1.75  # Default height in meters
            bmi = weight / (height ** 2)
            
            if bmi < 18.5:
                bmi_status = "Underweight"
                bmi_color = "#FF9800"
            elif bmi < 25:
                bmi_status = "Normal"
                bmi_color = "#4CAF50"
            elif bmi < 30:
                bmi_status = "Overweight"
                bmi_color = "#FF9800"
            else:
                bmi_status = "Obese"
                bmi_color = "#F44336"
            
            st.markdown(f"""
            <div style='background: {bmi_color}20; padding: 1rem; border-radius: 10px; border-left: 5px solid {bmi_color};'>
                <h4 style='color: {bmi_color}; margin: 0;'>⚖️ BMI</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;'>{bmi_status} ({bmi:.1f})</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("📝 No health data available. Please log your first health metrics in the 'Log New Data' tab.")

def render_prescription_section():
    """Render prescription analysis section"""
    st.markdown("### 💊 Prescription Analysis & Management")
    
    # Prescription upload section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
        <h4 style='margin: 0 0 0.5rem 0;'>📋 Upload Your Prescription</h4>
        <p style='margin: 0; opacity: 0.9;'>Upload a clear image of your prescription for AI-powered analysis and health recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose prescription image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a clear, well-lit image of your prescription",
        key="health_log_prescription_upload"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Prescription Image")
            st.image(uploaded_file, caption="Uploaded Prescription", use_column_width=True)
        
        with col2:
            st.markdown("#### 🔍 Analysis")
            
            if st.button("🚀 Analyze Prescription", type="primary", use_container_width=True, key="analyze_prescription_health_log"):
                analyze_prescription_health_log(uploaded_file)
    
    # Display previous prescription analyses
    if 'prescription_analyses' in st.session_state and st.session_state.prescription_analyses:
        st.markdown("### 📋 Previous Prescription Analyses")
        
        for i, analysis in enumerate(reversed(st.session_state.prescription_analyses[-5:]), 1):
            with st.expander(f"📅 Analysis {i} - {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}", expanded=i==1):
                display_prescription_analysis_summary(analysis)
    
    # Medication reminders
    render_medication_reminders()

def analyze_prescription_health_log(uploaded_file):
    """Analyze prescription in health log context"""
    
    with st.spinner("🔍 Analyzing prescription and updating health log..."):
        uploaded_file.seek(0)
        result = st.session_state.prescription_chatbot.upload_prescription(uploaded_file)
        
        if result.get('success'):
            st.success("✅ Prescription analyzed successfully!")
            
            # Store analysis in session state
            if 'prescription_analyses' not in st.session_state:
                st.session_state.prescription_analyses = []
            
            st.session_state.prescription_analyses.append(result)
            
            # Display results
            display_prescription_analysis_summary(result)
            
            # Update health log with medication data
            update_health_log_with_prescription(result)
            
        else:
            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

def display_prescription_analysis_summary(analysis):
    """Display prescription analysis summary"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Analysis Summary:** {analysis['analysis_summary']}")
        
        if analysis.get('identified_medicines'):
            st.markdown("**Medicines Identified:**")
            for med in analysis['identified_medicines']:
                st.markdown(f"• {med['medicine_name']} ({med['category'].replace('_', ' ').title()})")
    
    with col2:
        st.metric("Confidence", f"{analysis.get('confidence_score', 0)*100:.1f}%")
        st.metric("Medicines Found", analysis.get('total_medicines_found', 0))
    
    # Professional advice
    if analysis.get('professional_advice'):
        with st.expander("👨‍⚕️ Professional Medical Advice"):
            st.markdown(analysis['professional_advice'])

def update_health_log_with_prescription(analysis):
    """Update health log with prescription data"""
    
    # Add medication entry to health log
    new_entry = {
        'date': datetime.now(),
        'type': 'prescription',
        'medicines': [med['medicine_name'] for med in analysis.get('identified_medicines', [])],
        'analysis_id': analysis.get('analysis_id'),
        'confidence': analysis.get('confidence_score', 0)
    }
    
    if 'medication_log' not in st.session_state:
        st.session_state.medication_log = []
    
    st.session_state.medication_log.append(new_entry)
    
    st.info("📝 Prescription data has been added to your health log for tracking.")

def render_medication_reminders():
    """Render medication reminders section"""
    
    st.markdown("### ⏰ Medication Reminders")
    
    if 'medication_log' in st.session_state and st.session_state.medication_log:
        st.markdown("#### 💊 Current Medications")
        
        for entry in st.session_state.medication_log[-3:]:  # Show last 3 entries
            medicines_text = ", ".join(entry['medicines'])
            st.markdown(f"""
            <div style='background: #f0f8ff; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2196F3;'>
                <strong>📅 {entry['date'].strftime('%Y-%m-%d')}</strong><br>
                <strong>💊 Medicines:</strong> {medicines_text}
            </div>
            """, unsafe_allow_html=True)
        
        # Reminder settings
        st.markdown("#### 🔔 Set Reminders")
        
        col1, col2 = st.columns(2)
        with col1:
            reminder_time = st.time_input("Daily reminder time", value=datetime.now().time())
        with col2:
            reminder_enabled = st.checkbox("Enable daily reminders", value=True)
        
        if st.button("💾 Save Reminder Settings"):
            st.success("✅ Reminder settings saved!")
    
    else:
        st.info("📝 Upload a prescription first to set up medication reminders.")

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
        
        # Prescription upload in data entry
        st.markdown("#### 💊 Prescription (Optional)")
        prescription_file = st.file_uploader(
            "Upload prescription image (optional)",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            key="data_entry_prescription"
        )
        
        submit_button = st.form_submit_button("💾 Save Health Data", type="primary")
        
        if submit_button:
            save_health_data(systolic_bp, diastolic_bp, heart_rate, weight, temperature, blood_sugar, notes, prescription_file)

def save_health_data(systolic_bp, diastolic_bp, heart_rate, weight, temperature, blood_sugar, notes, prescription_file):
    """Save new health data entry"""
    
    new_entry = {
        'date': datetime.now(),
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'weight': weight,
        'temperature': temperature,
        'blood_sugar': blood_sugar,
        'notes': notes,
        'bp_change': 0,  # Calculate based on previous entry
        'hr_change': 0,
        'weight_change': 0,
        'temp_change': 0
    }
    
    # Calculate changes from previous entry
    if not st.session_state.health_log_data.empty:
        last_entry = st.session_state.health_log_data.iloc[-1]
        new_entry['bp_change'] = systolic_bp - last_entry.get('systolic_bp', systolic_bp)
        new_entry['hr_change'] = heart_rate - last_entry.get('heart_rate', heart_rate)
        new_entry['weight_change'] = weight - last_entry.get('weight', weight)
        new_entry['temp_change'] = temperature - last_entry.get('temperature', temperature)
    
    # Add to health log data
    new_df = pd.DataFrame([new_entry])
    st.session_state.health_log_data = pd.concat([st.session_state.health_log_data, new_df], ignore_index=True)
    
    # Analyze prescription if uploaded
    if prescription_file is not None:
        with st.spinner("🔍 Analyzing prescription..."):
            prescription_file.seek(0)
            result = st.session_state.prescription_chatbot.upload_prescription(prescription_file)
            
            if result.get('success'):
                if 'prescription_analyses' not in st.session_state:
                    st.session_state.prescription_analyses = []
                st.session_state.prescription_analyses.append(result)
                update_health_log_with_prescription(result)
    
    st.success("✅ Health data saved successfully!")
    st.balloons()
    st.rerun()

def render_health_reports():
    """Render health reports and analytics"""
    st.markdown("### 📋 Health Reports & Analytics")
    
    if st.session_state.health_log_data.empty:
        st.info("📝 No health data available for reports. Please log some health metrics first.")
        return
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        ["Weekly Summary", "Monthly Trends", "Medication Analysis", "Health Goals Progress"]
    )
    
    if report_type == "Weekly Summary":
        render_weekly_summary()
    elif report_type == "Monthly Trends":
        render_monthly_trends()
    elif report_type == "Medication Analysis":
        render_medication_analysis()
    elif report_type == "Health Goals Progress":
        render_health_goals()

def render_weekly_summary():
    """Render weekly health summary"""
    st.markdown("#### 📊 Weekly Health Summary")
    
    # Get last 7 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    weekly_data = st.session_state.health_log_data[
        st.session_state.health_log_data['date'] >= start_date
    ]
    
    if not weekly_data.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_systolic = weekly_data['systolic_bp'].mean()
            st.metric("Avg Systolic BP", f"{avg_systolic:.0f} mmHg")
            
            avg_hr = weekly_data['heart_rate'].mean()
            st.metric("Avg Heart Rate", f"{avg_hr:.0f} BPM")
        
        with col2:
            avg_weight = weekly_data['weight'].mean()
            st.metric("Avg Weight", f"{avg_weight:.1f} kg")
            
            weight_change = weekly_data['weight'].iloc[-1] - weekly_data['weight'].iloc[0] if len(weekly_data) > 1 else 0
            st.metric("Weight Change", f"{weight_change:+.1f} kg")
        
        with col3:
            data_points = len(weekly_data)
            st.metric("Data Points", f"{data_points}")
            
            consistency = (data_points / 7) * 100
            st.metric("Logging Consistency", f"{consistency:.0f}%")
        
        # Weekly trend chart
        fig = px.line(weekly_data, x='date', y=['systolic_bp', 'heart_rate'], 
                     title="Weekly Health Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📝 No data available for the past week.")

def render_monthly_trends():
    """Render monthly health trends"""
    st.markdown("#### 📈 Monthly Health Trends")
    
    # Monthly aggregation
    monthly_data = st.session_state.health_log_data.copy()
    monthly_data['month'] = monthly_data['date'].dt.to_period('M')
    
    monthly_summary = monthly_data.groupby('month').agg({
        'systolic_bp': 'mean',
        'diastolic_bp': 'mean',
        'heart_rate': 'mean',
        'weight': 'mean',
        'temperature': 'mean'
    }).reset_index()
    
    if not monthly_summary.empty:
        # Monthly trends chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_summary['month'].astype(str),
            y=monthly_summary['systolic_bp'],
            mode='lines+markers',
            name='Systolic BP',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_summary['month'].astype(str),
            y=monthly_summary['heart_rate'],
            mode='lines+markers',
            name='Heart Rate',
            yaxis='y2',
            line=dict(color='#4ECDC4', width=3)
        ))
        
        fig.update_layout(
            title="Monthly Health Trends",
            xaxis_title="Month",
            yaxis=dict(title="Blood Pressure (mmHg)", side="left"),
            yaxis2=dict(title="Heart Rate (BPM)", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📝 Not enough data for monthly trends.")

def render_medication_analysis():
    """Render medication analysis"""
    st.markdown("#### 💊 Medication Analysis")
    
    if 'prescription_analyses' in st.session_state and st.session_state.prescription_analyses:
        # Medication categories pie chart
        all_medicines = []
        for analysis in st.session_state.prescription_analyses:
            for med in analysis.get('identified_medicines', []):
                all_medicines.append(med['category'].replace('_', ' ').title())
        
        if all_medicines:
            category_counts = pd.Series(all_medicines).value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Medication Categories Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Medication timeline
            st.markdown("#### 📅 Medication Timeline")
            for analysis in st.session_state.prescription_analyses:
                st.markdown(f"""
                **{analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}** - 
                {len(analysis.get('identified_medicines', []))} medicines identified
                """)
        
    else:
        st.info("📝 No prescription analyses available. Upload prescriptions to see medication analysis.")

def render_health_goals():
    """Render health goals progress"""
    st.markdown("#### 🎯 Health Goals Progress")
    
    # Sample health goals
    goals = [
        {"name": "Maintain Normal Blood Pressure", "target": "< 120/80 mmHg", "current": "Good"},
        {"name": "Regular Exercise", "target": "5 days/week", "current": "In Progress"},
        {"name": "Weight Management", "target": "Maintain ±2kg", "current": "Good"},
        {"name": "Medication Adherence", "target": "100%", "current": "Excellent"}
    ]
    
    for goal in goals:
        status_color = "#4CAF50" if goal["current"] == "Good" or goal["current"] == "Excellent" else "#FF9800"
        
        st.markdown(f"""
        <div style='background: {status_color}20; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {status_color};'>
            <strong>{goal['name']}</strong><br>
            <strong>Target:</strong> {goal['target']}<br>
            <strong>Status:</strong> <span style='color: {status_color}; font-weight: bold;'>{goal['current']}</span>
        </div>
        """, unsafe_allow_html=True)

def generate_sample_health_data():
    """Generate sample health data for demonstration"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    data = []
    for i, date in enumerate(dates):
        # Generate realistic health data with some variation
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
            'bp_change': 0,
            'hr_change': 0,
            'weight_change': 0,
            'temp_change': 0,
            'notes': f"Daily health check {i+1}"
        })
    
    return pd.DataFrame(data)
