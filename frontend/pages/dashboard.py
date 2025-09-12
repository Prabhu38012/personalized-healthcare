import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Optional import for metric cards styling
try:
    from streamlit_extras.metric_cards import style_metric_cards
    HAS_EXTRAS = True
except ImportError:
    HAS_EXTRAS = False
    def style_metric_cards(*args, **kwargs):
        """Fallback function when streamlit_extras is not available"""
        pass

def calculate_risk_score(patient_data):
    """Calculate a simple risk score based on patient data (0-1 scale)"""
    if not patient_data:
        return 0.0
    
    score = 0.0
    
    # Age (0-0.2)
    age = patient_data.get('age', 40)
    score += min(0.2, (age - 30) * 0.01)  # 1% per year over 30, max 20%
    
    # Blood Pressure (0-0.15) - Support both new and old field names
    bp = patient_data.get('systolic_bp', patient_data.get('resting_bp', 120))
    if bp >= 140:
        score += 0.15
    elif bp >= 130:
        score += 0.1
    elif bp >= 120:
        score += 0.05
    
    # Cholesterol (0-0.15) - Support both new and old field names
    chol = patient_data.get('total_cholesterol', patient_data.get('cholesterol', 200))
    if chol >= 240:
        score += 0.15
    elif chol >= 200:
        score += 0.07
    
    # BMI (0-0.1)
    bmi = patient_data.get('bmi', 25)
    if bmi >= 30:
        score += 0.1
    elif bmi >= 25:
        score += 0.05
    
    # Smoking (0-0.1)
    if patient_data.get('smoking', 0) == 2:  # Current smoker
        score += 0.1
    elif patient_data.get('smoking', 0) == 1:  # Former smoker
        score += 0.05
    
    # Diabetes (0-0.1)
    if patient_data.get('diabetes', 0) == 1:  # Diabetes
        score += 0.1
    elif patient_data.get('diabetes', 0) == 2:  # Pre-diabetes
        score += 0.05
    
    # Family History (0-0.05)
    if patient_data.get('family_history', 0) == 1:
        score += 0.05
    
    # Physical Activity (0-0.05)
    if patient_data.get('physical_activity', 0) < 150:  # Less than recommended
        score += 0.05
    
    # Stress (0-0.05)
    stress = patient_data.get('stress_level', 5)
    if stress >= 7:
        score += 0.05
    
    # Ensure score is between 0 and 1
    return min(max(score, 0.0), 1.0)

def create_dashboard(patient_data=None):
    """Create healthcare analytics dashboard with patient-specific insights"""
    
    # Key Metrics Row
    st.subheader("📊 Health Overview")
    
    if patient_data:
        # Patient-specific metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bmi_status = "Normal"
            bmi = patient_data.get('bmi', 0)
            if bmi >= 30:
                bmi_status = "Obese"
            elif bmi >= 25:
                bmi_status = "Overweight"
            elif bmi < 18.5:
                bmi_status = "Underweight"
                
            st.metric("BMI", f"{bmi:.1f}", bmi_status)
        
        with col2:
            # Use new field names with fallback to old ones
            bp = patient_data.get('systolic_bp', patient_data.get('resting_bp', 0))
            bp_status = "Normal"
            if bp >= 140:
                bp_status = "High"
            elif bp >= 120:
                bp_status = "Elevated"
            st.metric("Blood Pressure", f"{bp} mmHg", bp_status)
        
        with col3:
            # Use new field names with fallback to old ones
            chol = patient_data.get('total_cholesterol', patient_data.get('cholesterol', 0))
            chol_status = "Normal"
            if chol >= 240:
                chol_status = "High"
            elif chol >= 200:
                chol_status = "Borderline High"
            st.metric("Cholesterol", f"{chol} mg/dL", chol_status)
        
        with col4:
            risk_score = calculate_risk_score(patient_data)
            risk_level = "Low"
            if risk_score > 0.7:
                risk_level = "High"
            elif risk_score > 0.4:
                risk_level = "Medium"
            st.metric("Cardiac Risk", risk_level, f"{risk_score*100:.1f}%")
        
        if HAS_EXTRAS:
            style_metric_cards(background_color="#f0f2f6", border_left_color="#1f77b4")
    else:
        # Population-level metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assessments", "1,234", "↗️ 12%")
        
        with col2:
            st.metric("High Risk Patients", "156", "↘️ 8%")
        
        with col3:
            st.metric("Avg Risk Score", "0.34", "↗️ 2%")
        
        with col4:
            st.metric("Prevention Success", "78%", "↗️ 15%")
    
    st.divider()
    
    # Health Metrics Overview
    if patient_data:
        st.subheader("📈 Health Metrics Analysis")
        
        # Create tabs for different health aspects
        tab1, tab2, tab3, tab4 = st.tabs(["Cardiovascular", "Metabolic", "Lifestyle", "Risk Factors"])
        
        with tab1:
            # Cardiovascular Health
            col1, col2 = st.columns(2)
            
            with col1:
                # Blood Pressure Gauge
                bp_value = patient_data.get('systolic_bp', patient_data.get('resting_bp', 120))
                fig_bp = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = bp_value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Blood Pressure (mmHg)"},
                    gauge = {
                        'axis': {'range': [80, 200]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [80, 120], 'color': "#4CAF50"},
                            {'range': [120, 140], 'color': "#FFC107"},
                            {'range': [140, 200], 'color': "#F44336"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': bp_value
                        }
                    }
                ))
                st.plotly_chart(fig_bp, use_container_width=True)
            
            with col2:
                # Cholesterol Levels
                fig_chol = go.Figure()
                
                # Use new field names with fallback to old ones
                total_chol = patient_data.get('total_cholesterol', patient_data.get('cholesterol', 0))
                
                # Add bars for each cholesterol type
                fig_chol.add_trace(go.Bar(
                    x=['Total', 'LDL', 'HDL', 'Triglycerides'],
                    y=[
                        total_chol,
                        patient_data.get('ldl_cholesterol', 0),
                        patient_data.get('hdl_cholesterol', 0),
                        patient_data.get('triglycerides', 0)
                    ],
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                ))
                
                # Add reference lines
                fig_chol.add_hline(y=200, line_dash="dash", line_color="green", annotation_text="Desirable")
                fig_chol.add_hline(y=240, line_dash="dash", line_color="red", annotation_text="High")
                
                fig_chol.update_layout(
                    title="Cholesterol Profile (mg/dL)",
                    yaxis_title="mg/dL",
                    showlegend=False
                )
                st.plotly_chart(fig_chol, use_container_width=True)
        
        with tab2:
            # Metabolic Health
            col1, col2 = st.columns(2)
            
            with col1:
                # Blood Sugar Indicators
                fig_glucose = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = patient_data.get('fasting_blood_sugar', 0) * 100 if isinstance(patient_data.get('fasting_blood_sugar'), int) else 90,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fasting Glucose (mg/dL)"},
                    gauge = {
                        'axis': {'range': [70, 200]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [70, 100], 'color': "#4CAF50"},
                            {'range': [100, 126], 'color': "#FFC107"},
                            {'range': [126, 200], 'color': "#F44336"}
                        ]
                    }
                ))
                st.plotly_chart(fig_glucose, use_container_width=True)
                
                # HbA1c
                hba1c = patient_data.get('hba1c', 5.5)
                fig_hba1c = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = hba1c,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"HbA1c: {hba1c}%"},
                    gauge = {
                        'axis': {'range': [4, 12]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [4, 5.6], 'color': "#4CAF50"},
                            {'range': [5.7, 6.4], 'color': "#FFC107"},
                            {'range': [6.5, 12], 'color': "#F44336"}
                        ]
                    }
                ))
                st.plotly_chart(fig_hba1c, use_container_width=True)
            
            with col2:
                # BMI and Weight Management
                bmi = patient_data.get('bmi', 25)
                weight = patient_data.get('weight', 70)
                height = patient_data.get('height', 170) / 100  # Convert to meters
                
                # Calculate ideal weight range (BMI 18.5-24.9)
                ideal_min = 18.5 * (height ** 2)
                ideal_max = 24.9 * (height ** 2)
                
                fig_weight = go.Figure()
                
                # Add current weight
                fig_weight.add_trace(go.Indicator(
                    mode = "number",
                    value = weight,
                    title = "Current Weight (kg)",
                    domain = {'row': 0, 'column': 0}
                ))
                
                # Add BMI classification
                bmi_class = ""
                if bmi < 18.5:
                    bmi_class = "Underweight"
                elif bmi < 25:
                    bmi_class = "Normal"
                elif bmi < 30:
                    bmi_class = "Overweight"
                else:
                    bmi_class = "Obese"
                
                fig_weight.add_trace(go.Indicator(
                    mode = "number",
                    value = bmi,
                    title = f"BMI: {bmi_class}",
                    domain = {'row': 0, 'column': 1}
                ))
                
                # Add weight range
                fig_weight.add_trace(go.Indicator(
                    mode = "number",
                    value = ideal_min,
                    title = "Ideal Min Weight (kg)",
                    domain = {'row': 1, 'column': 0}
                ))
                
                fig_weight.add_trace(go.Indicator(
                    mode = "number",
                    value = ideal_max,
                    title = "Ideal Max Weight (kg)",
                    domain = {'row': 1, 'column': 1}
                ))
                
                fig_weight.update_layout(
                    grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
                    height=300
                )
                
                st.plotly_chart(fig_weight, use_container_width=True)
                
                # Weight management goal
                if bmi >= 25:
                    target_weight = ideal_max
                    weight_to_lose = weight - target_weight
                    st.info(f"💡 Goal: Lose {weight_to_lose:.1f} kg to reach a healthy BMI")
        
        with tab3:
            # Lifestyle Factors
            col1, col2 = st.columns(2)
            
            with col1:
                # Physical Activity
                activity = patient_data.get('physical_activity', 0)
                activity_goal = 150  # minutes per week
                
                fig_activity = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = activity,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Weekly Activity: {activity}/{activity_goal} min"},
                    gauge = {
                        'axis': {'range': [0, 300]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 150], 'color': "#FFC107"},
                            {'range': [150, 300], 'color': "#4CAF50"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': activity
                        }
                    }
                ))
                st.plotly_chart(fig_activity, use_container_width=True)
                
                # Sleep Quality
                sleep = patient_data.get('sleep_duration', 7.0)
                
                fig_sleep = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sleep,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Sleep Duration: {sleep} hours"},
                    gauge = {
                        'axis': {'range': [0, 12]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 6], 'color': "#F44336"},
                            {'range': [6, 7], 'color': "#FFC107"},
                            {'range': [7, 9], 'color': "#4CAF50"},
                            {'range': [9, 12], 'color': "#F44336"}
                        ]
                    }
                ))
                st.plotly_chart(fig_sleep, use_container_width=True)
            
            with col2:
                # Stress and Mental Health
                stress = patient_data.get('stress_level', 5)
                
                fig_stress = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = stress,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Stress Level: {stress}/10"},
                    gauge = {
                        'axis': {'range': [0, 10]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 3], 'color': "#4CAF50"},
                            {'range': [3, 7], 'color': "#FFC107"},
                            {'range': [7, 10], 'color': "#F44336"}
                        ]
                    }
                ))
                st.plotly_chart(fig_stress, use_container_width=True)
                
                # Alcohol Consumption
                drinks = patient_data.get('alcohol_consumption', 0)
                
                fig_alcohol = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = drinks,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Alcoholic Drinks per Week: {drinks}"},
                    gauge = {
                        'axis': {'range': [0, 20]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 7], 'color': "#4CAF50"},
                            {'range': [7, 14], 'color': "#FFC107"},
                            {'range': [14, 20], 'color': "#F44336"}
                        ]
                    }
                ))
                st.plotly_chart(fig_alcohol, use_container_width=True)
        
        with tab4:
            # Risk Factors
            st.subheader("🚨 Identified Risk Factors")
            
            # Calculate risk factors based on patient data
            risk_factors = []
            
            # Blood Pressure
            if patient_data.get('resting_bp', 0) >= 140:
                risk_factors.append({"Factor": "High Blood Pressure", "Severity": "High"})
            elif patient_data.get('resting_bp', 0) >= 130:
                risk_factors.append({"Factor": "Elevated Blood Pressure", "Severity": "Medium"})
            
            # Cholesterol
            if patient_data.get('cholesterol', 0) >= 240:
                risk_factors.append({"Factor": "High Total Cholesterol", "Severity": "High"})
            
            if patient_data.get('ldl_cholesterol', 0) >= 160:
                risk_factors.append({"Factor": "High LDL Cholesterol", "Severity": "High"})
            
            if patient_data.get('hdl_cholesterol', 0) < 40:
                risk_factors.append({"Factor": "Low HDL Cholesterol", "Severity": "Medium"})
            
            if patient_data.get('triglycerides', 0) >= 200:
                risk_factors.append({"Factor": "High Triglycerides", "Severity": "High"})
            
            # Blood Sugar
            if (isinstance(patient_data.get('fasting_blood_sugar'), int) and patient_data['fasting_blood_sugar'] > 0) or \
               patient_data.get('fasting_blood_sugar', 0) >= 126:
                risk_factors.append({"Factor": "High Fasting Blood Sugar", "Severity": "High"})
            
            if patient_data.get('hba1c', 0) >= 6.5:
                risk_factors.append({"Factor": "Elevated HbA1c (Diabetes)", "Severity": "High"})
            elif patient_data.get('hba1c', 0) >= 5.7:
                risk_factors.append({"Factor": "Elevated HbA1c (Pre-diabetes)", "Severity": "Medium"})
            
            # BMI
            bmi = patient_data.get('bmi', 0)
            if bmi >= 30:
                risk_factors.append({"Factor": "Obesity (BMI ≥30)", "Severity": "High"})
            elif bmi >= 25:
                risk_factors.append({"Factor": "Overweight (BMI 25-29.9)", "Severity": "Medium"})
            
            # Lifestyle
            if patient_data.get('smoking', 0) == 2:  # Current smoker
                risk_factors.append({"Factor": "Current Smoker", "Severity": "High"})
            elif patient_data.get('smoking', 0) == 1:  # Former smoker
                risk_factors.append({"Factor": "Former Smoker", "Severity": "Low"})
            
            if patient_data.get('physical_activity', 0) < 150:
                risk_factors.append({"Factor": "Insufficient Physical Activity", "Severity": "Medium"})
            
            if patient_data.get('sleep_duration', 0) < 6 or patient_data.get('sleep_duration', 0) > 9:
                risk_factors.append({"Factor": "Poor Sleep Duration", "Severity": "Medium"})
            
            if patient_data.get('stress_level', 0) >= 7:
                risk_factors.append({"Factor": "High Stress Level", "Severity": "Medium"})
            
            if patient_data.get('alcohol_consumption', 0) > 14:
                risk_factors.append({"Factor": "High Alcohol Consumption", "Severity": "High"})
            
            # Medical History
            if patient_data.get('diabetes', 0) == 1:
                risk_factors.append({"Factor": "Diabetes", "Severity": "High"})
            elif patient_data.get('diabetes', 0) == 2:
                risk_factors.append({"Factor": "Pre-diabetes", "Severity": "Medium"})
            
            if patient_data.get('hypertension', 0) == 1:
                risk_factors.append({"Factor": "Hypertension", "Severity": "High"})
            
            if patient_data.get('kidney_disease', 0) == 1:
                risk_factors.append({"Factor": "Chronic Kidney Disease", "Severity": "High"})
            
            if patient_data.get('family_history', 0) == 1:
                risk_factors.append({"Factor": "Family History of Heart Disease", "Severity": "Medium"})
            
            # Display risk factors
            if risk_factors:
                # Sort by severity (High > Medium > Low)
                severity_order = {"High": 0, "Medium": 1, "Low": 2}
                risk_factors.sort(key=lambda x: severity_order[x["Severity"]])
                
                # Create columns for each severity level
                high_col, med_col, low_col = st.columns(3)
                
                with high_col:
                    st.subheader("🚨 High Risk")
                    for factor in [f for f in risk_factors if f["Severity"] == "High"]:
                        st.error(f"• {factor['Factor']}")
                
                with med_col:
                    st.subheader("⚠️ Medium Risk")
                    for factor in [f for f in risk_factors if f["Severity"] == "Medium"]:
                        st.warning(f"• {factor['Factor']}")
                
                with low_col:
                    st.subheader("ℹ️ Low Risk")
                    for factor in [f for f in risk_factors if f["Severity"] == "Low"]:
                        st.info(f"• {factor['Factor']}")
            else:
                st.success("🎉 No significant risk factors identified!")
    else:
        # Population-level charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            risk_data = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Count': [650, 428, 156],
                'Percentage': [52.7, 34.7, 12.6]
            })
            
            fig_pie = px.pie(risk_data, values='Count', names='Risk Level',
                            color_discrete_map={'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'})
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Risk Trends Over Time")
            trend_data = generate_trend_data()
            
            fig_line = px.line(trend_data, x='Date', y=['Low Risk', 'Medium Risk', 'High Risk'],
                            color_discrete_map={'Low Risk': '#4CAF50', 'Medium Risk': '#FF9800', 'High Risk': '#F44336'})
            fig_line.update_layout(height=400, yaxis_title="Number of Patients")
            st.plotly_chart(fig_line, use_container_width=True)
    
    st.divider()
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Group Analysis")
        age_data = pd.DataFrame({
            'Age Group': ['18-30', '31-45', '46-60', '61-75', '75+'],
            'High Risk %': [5, 12, 28, 45, 62],
            'Total Patients': [180, 320, 410, 280, 44]
        })
        
        fig_bar = px.bar(age_data, x='Age Group', y='High Risk %',
                        color='High Risk %', color_continuous_scale='Reds')
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Risk Factors Prevalence")
        risk_factors_data = pd.DataFrame({
            'Risk Factor': ['High BP', 'High Cholesterol', 'Smoking', 'Diabetes', 'Obesity', 'Family History'],
            'Prevalence %': [34, 28, 22, 18, 31, 25]
        })
        
        fig_horizontal = px.bar(risk_factors_data, x='Prevalence %', y='Risk Factor',
                               orientation='h', color='Prevalence %', color_continuous_scale='Blues')
        fig_horizontal.update_layout(height=400)
        st.plotly_chart(fig_horizontal, use_container_width=True)
    
    st.divider()
    
    # Detailed Analytics
    st.subheader("📈 Detailed Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Patient Demographics", "Clinical Metrics", "Outcomes"])
    
    with tab1:
        # Gender distribution
        col1, col2 = st.columns(2)
        
        with col1:
            gender_data = pd.DataFrame({
                'Gender': ['Male', 'Female'],
                'Count': [678, 556],
                'High Risk %': [15.2, 9.7]
            })
            
            fig_gender = px.bar(gender_data, x='Gender', y=['Count'],
                               title="Patient Distribution by Gender")
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # BMI distribution
            bmi_data = generate_bmi_distribution()
            fig_hist = px.histogram(bmi_data, x='BMI', nbins=20,
                                   title="BMI Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        # Clinical metrics correlation
        st.subheader("Clinical Metrics Correlation")
        
        correlation_data = generate_correlation_data()
        fig_corr = px.imshow(correlation_data.corr(),
                            title="Clinical Metrics Correlation Matrix",
                            color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Blood pressure vs cholesterol scatter
        clinical_scatter = generate_clinical_scatter()
        fig_scatter = px.scatter(clinical_scatter, x='Blood Pressure', y='Cholesterol',
                               color='Risk Level', size='Age',
                               title="Blood Pressure vs Cholesterol by Risk Level")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        # Intervention outcomes
        st.subheader("Intervention Outcomes")
        
        outcomes_data = pd.DataFrame({
            'Intervention': ['Lifestyle Changes', 'Medication', 'Surgery', 'Monitoring'],
            'Success Rate %': [78, 85, 92, 68],
            'Patient Count': [450, 280, 45, 320]
        })
        
        fig_outcomes = px.scatter(outcomes_data, x='Patient Count', y='Success Rate %',
                                 size='Success Rate %', color='Intervention',
                                 title="Intervention Effectiveness")
        st.plotly_chart(fig_outcomes, use_container_width=True)

def generate_sample_analytics():
    """Generate sample data for analytics"""
    np.random.seed(42)
    return pd.DataFrame({
        'patient_id': range(1, 1235),
        'age': np.random.randint(18, 90, 1234),
        'risk_score': np.random.beta(2, 5, 1234),
        'risk_level': np.random.choice(['Low', 'Medium', 'High'], 1234, p=[0.527, 0.347, 0.126])
    })

def generate_trend_data():
    """Generate trend data for the last 30 days"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    trend_data = pd.DataFrame({
        'Date': dates,
        'Low Risk': np.random.poisson(20, len(dates)) + np.random.randint(15, 25, len(dates)),
        'Medium Risk': np.random.poisson(12, len(dates)) + np.random.randint(8, 16, len(dates)),
        'High Risk': np.random.poisson(4, len(dates)) + np.random.randint(2, 8, len(dates))
    })
    
    return trend_data

def generate_bmi_distribution():
    """Generate BMI distribution data"""
    np.random.seed(42)
    return pd.DataFrame({
        'BMI': np.random.normal(26.5, 4.2, 1000)
    })

def generate_correlation_data():
    """Generate clinical metrics correlation data"""
    np.random.seed(42)
    n = 500
    
    # Generate correlated clinical data
    age = np.random.randint(30, 80, n)
    bp = 90 + age * 0.8 + np.random.normal(0, 15, n)
    cholesterol = 150 + age * 1.2 + np.random.normal(0, 30, n)
    heart_rate = 220 - age + np.random.normal(0, 20, n)
    bmi = 22 + np.random.normal(0, 4, n)
    
    return pd.DataFrame({
        'Age': age,
        'Blood Pressure': bp,
        'Cholesterol': cholesterol,
        'Heart Rate': heart_rate,
        'BMI': bmi
    })

def generate_clinical_scatter():
    """Generate clinical scatter plot data"""
    np.random.seed(42)
    n = 300
    
    data = []
    for risk in ['Low', 'Medium', 'High']:
        if risk == 'Low':
            bp = np.random.normal(120, 15, n//3)
            chol = np.random.normal(180, 25, n//3)
            age = np.random.randint(25, 50, n//3)
        elif risk == 'Medium':
            bp = np.random.normal(140, 20, n//3)
            chol = np.random.normal(220, 30, n//3)
            age = np.random.randint(40, 65, n//3)
        else:
            bp = np.random.normal(160, 25, n//3)
            chol = np.random.normal(260, 35, n//3)
            age = np.random.randint(55, 80, n//3)
        
        for i in range(len(bp)):
            data.append({
                'Blood Pressure': bp[i],
                'Cholesterol': chol[i],
                'Age': age[i],
                'Risk Level': risk
            })
    
    return pd.DataFrame(data)
