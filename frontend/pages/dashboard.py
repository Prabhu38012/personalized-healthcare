import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def create_dashboard():
    """Create healthcare analytics dashboard"""
    
    # Generate sample analytics data
    sample_data = generate_sample_analytics()
    
    # Key Metrics Row
    st.subheader("📊 Key Health Metrics")
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
    
    # Charts Row 1
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
