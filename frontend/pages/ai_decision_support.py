"""
AI Decision Support System Dashboard - Frontend
Advanced AI/ML-powered predictions with explainability and personalized lifestyle recommendations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import json
from datetime import datetime

try:
    from frontend.utils.api_client import HealthcareAPI
    from frontend.components.auth import get_auth_headers
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.api_client import HealthcareAPI
    from components.auth import get_auth_headers


class AIDecisionDashboard:
    """AI Decision Support Dashboard"""
    
    def __init__(self):
        self.api_client = HealthcareAPI()
        self.api_client.set_auth_headers(get_auth_headers())
    
    def render_header(self):
        """Render dashboard header"""
        # Header removed - using cleaner minimal design
        pass
    
    def render_feature_cards(self):
        """Render feature overview cards"""
        # Feature cards removed per user preference
        pass
    
    def render_prediction_interface(self):
        """Render AI prediction interface"""
        st.markdown("""
        <div style="margin: 2rem 0 1.5rem 0;">
            <h2 style="color: #0891b2; font-size: 1.75rem; font-weight: 600; margin: 0;">
                🔮 AI-Powered Prediction with Explainability
            </h2>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">
                Enter patient health metrics for comprehensive AI analysis with explainable results
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("ai_prediction_form"):
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                    <h4 style="color: #0891b2; margin: 0 0 1rem 0;">📊 Basic Health Metrics</h4>
                </div>
                """, unsafe_allow_html=True)
                age = st.number_input("Age (years)", min_value=18, max_value=120, value=55, help="Patient's current age")
                blood_pressure = st.number_input("Blood Pressure - Systolic (mmHg)", min_value=80, max_value=200, value=130, help="Upper number in BP reading")
                cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=220, help="Total blood cholesterol level")
                bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=28.5, step=0.1, help="Weight(kg) / Height(m)²")
            
            with col2:
                st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                    <h4 style="color: #0891b2; margin: 0 0 1rem 0;">🏃 Lifestyle & Risk Factors</h4>
                </div>
                """, unsafe_allow_html=True)
                smoking = st.checkbox("Current Smoker", help="Currently smoking tobacco")
                st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                exercise = st.number_input("Exercise (minutes/week)", min_value=0, max_value=1000, value=120, help="Total weekly physical activity")
                st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                family_history = st.checkbox("Family History of Heart Disease", help="Close relatives with heart disease")
                st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                diabetes = st.checkbox("Diabetes", help="Diagnosed with diabetes")
            
            st.markdown("""
            <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0 1rem 0; border: 1px solid #bae6fd;">
                <h4 style="color: #0369a1; margin: 0 0 1rem 0;">AI Analysis Options</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                include_explanation = st.checkbox("Enable Explainability (SHAP)", value=True, help="Show feature importance and SHAP values")
            with col4:
                include_bias = st.checkbox("Lifestyle & Diet Plan", value=True, help="Get personalized diet and exercise recommendations")
            with col5:
                include_confidence = st.checkbox("Confidence Intervals", value=True, help="Display prediction confidence ranges")
            
            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("Run Comprehensive AI Analysis", use_container_width=True, type="primary")
            
            if submit:
                patient_data = {
                    'age': age,
                    'blood_pressure': blood_pressure,
                    'cholesterol': cholesterol,
                    'bmi': bmi,
                    'smoking': smoking,
                    'exercise': exercise,
                    'family_history': family_history,
                    'diabetes': diabetes
                }
                
                self.perform_ai_prediction(patient_data, include_explanation, include_bias, include_confidence)
    
    def perform_ai_prediction(self, patient_data: Dict, include_explanation: bool, 
                             include_bias: bool, include_confidence: bool):
        """Perform AI prediction and display results"""
        with st.spinner("🤖 AI is analyzing... This may take a moment"):
            try:
                # Call AI decision support API
                result = self.api_client._make_request(
                    'POST',
                    '/ai-decision/predict',
                    json={
                        'patient_data': patient_data,
                        'include_explanation': include_explanation,
                        'include_lifestyle_plan': include_bias,
                        'include_confidence_intervals': include_confidence
                    }
                )
                
                if result:
                    st.session_state['ai_prediction_result'] = result
                    self.display_prediction_results(result)
                else:
                    st.error("Prediction failed: No response from server")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    def display_prediction_results(self, result: Dict[str, Any]):
        """Display AI prediction results with visualizations"""
        st.success("✅ AI Analysis Complete!")
        
        # Main prediction card
        prediction = result['prediction']
        risk_score = prediction['risk_score']
        risk_category = prediction['risk_category']
        
        # Color based on risk
        if risk_category == 'HIGH':
            color = '#dc2626'
            emoji = '🔴'
            bg_color = '#fef2f2'
        elif risk_category == 'MODERATE':
            color = '#f59e0b'
            emoji = '🟡'
            bg_color = '#fffbeb'
        else:
            color = '#10b981'
            emoji = '🟢'
            bg_color = '#f0fdf4'
        
        st.markdown(f"""
        <div style="background: {bg_color}; 
                    border-left: 5px solid {color}; padding: 1.75rem; border-radius: 12px; margin: 1.5rem 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h2 style="margin: 0; color: {color}; font-size: 1.75rem; font-weight: 600;">{emoji} {risk_category} RISK</h2>
                    <p style="margin: 0.5rem 0 0; font-size: 1rem; color: #64748b;">
                        Risk Score: {risk_score:.2%} | Confidence: {result['confidence_level']}
                    </p>
                </div>
                <div style="font-size: 3rem;">{emoji}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧠 Explainability",
            "📊 Feature Importance",
            "🥗 Lifestyle & Diet Plan",
            "🎯 Decision Path",
            "💡 Recommendations"
        ])
        
        with tab1:
            self.display_explainability(result['explainability'])
        
        with tab2:
            self.display_feature_importance(result['explainability'])
        
        with tab3:
            if result.get('lifestyle_diet_plan'):
                self.display_lifestyle_diet_plan(result['lifestyle_diet_plan'])
            else:
                st.info("Lifestyle & Diet Plan not included in this prediction")
        
        with tab4:
            self.display_decision_path(result['explainability'])
        
        with tab5:
            self.display_recommendations(result['recommendations'])
    
    def display_explainability(self, explainability: Dict[str, Any]):
        """Display SHAP values and explainability"""
        st.subheader("🧠 Model Explainability (SHAP Values)")
        
        if explainability.get('shap_values'):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to the prediction.
            - **Positive values** (red) increase the predicted risk
            - **Negative values** (green) decrease the predicted risk
            """)
            
            # Create SHAP waterfall chart
            shap_values = explainability['shap_values']
            features = list(shap_values.keys())
            values = list(shap_values.values())
            
            # Create figure
            fig = go.Figure(go.Waterfall(
                name="SHAP Values",
                orientation="h",
                x=values,
                y=features,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#38a169"}},
                increasing={"marker": {"color": "#e53e3e"}},
                totals={"marker": {"color": "#667eea"}}
            ))
            
            fig.update_layout(
                title="Feature Contributions to Prediction (SHAP Values)",
                xaxis_title="SHAP Value (Impact on Risk)",
                yaxis_title="Features",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence information
        st.markdown("#### Prediction Confidence")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence Score", f"{explainability['confidence_score']:.2%}")
        
        with col2:
            if explainability.get('uncertainty_range'):
                lower = explainability['uncertainty_range']['lower_bound']
                upper = explainability['uncertainty_range']['upper_bound']
                st.metric("95% CI Lower", f"{lower:.2%}")
        
        with col3:
            if explainability.get('uncertainty_range'):
                st.metric("95% CI Upper", f"{upper:.2%}")
    
    def display_feature_importance(self, explainability: Dict[str, Any]):
        """Display feature importance visualization"""
        st.subheader("📊 Feature Importance Analysis")
        
        feature_importance = explainability['feature_importance']
        
        if feature_importance:
            # Create DataFrame
            df = pd.DataFrame([
                {'Feature': k, 'Importance': v}
                for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            ])
            
            # Bar chart
            fig = px.bar(
                df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance Rankings',
                color='Importance',
                color_continuous_scale='RdYlGn_r'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Importance table
            st.markdown("#### Detailed Feature Importance")
            st.dataframe(
                df.style.background_gradient(cmap='RdYlGn_r', subset=['Importance']),
                use_container_width=True
            )
    
    def display_lifestyle_diet_plan(self, plan: Dict[str, Any]):
        """Display personalized lifestyle and diet recommendations"""
        st.subheader("🥗 Personalized Lifestyle & Diet Plan")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🍽️ Diet Plan", 
            "🏃 Exercise Plan", 
            "🌟 Lifestyle Changes",
            "💊 Supplements & Routine"
        ])
        
        with tab1:
            st.markdown("### Diet Plan")
            diet = plan['diet_plan']
            
            # Overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Daily Calories", f"{diet['daily_calories']} kcal")
            with col2:
                st.metric("Sodium Limit", diet['sodium_limit'])
            with col3:
                st.metric("Fiber Target", diet['fiber_target'])
            
            st.info(f"**Recommended Approach**: {diet['approach']}")
            
            # Macros
            st.markdown("#### Macronutrient Distribution")
            for macro, value in diet['macros'].items():
                st.write(f"• **{macro.title()}**: {value}")
            
            # Foods to emphasize
            st.markdown("#### ✅ Foods to Emphasize")
            for food in diet['foods_to_emphasize']:
                st.markdown(f"• {food}")
            
            # Foods to avoid
            st.markdown("#### ❌ Foods to Limit/Avoid")
            for food in diet['foods_to_avoid']:
                st.markdown(f"• {food}")
            
            # Meal suggestions
            st.markdown("### 🍽️ Meal Suggestions")
            meals = plan['meal_suggestions']
            
            meal_tabs = st.tabs(["Breakfast", "Lunch", "Dinner", "Snacks"])
            for meal_tab, (meal_name, options) in zip(meal_tabs, meals.items()):
                with meal_tab:
                    for option in options:
                        st.markdown(f"• {option}")
        
        with tab2:
            st.markdown("### Exercise Plan")
            exercise = plan['exercise_plan']
            
            # Overview metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weekly Target", exercise['weekly_target'])
            with col2:
                st.metric("Current Level", exercise['current_level'])
            
            st.info(f"**Progression**: {exercise['progression']}")
            
            # Exercise components
            st.markdown("#### Exercise Components")
            st.write(f"• {exercise['cardio_focus']}")
            st.write(f"• {exercise['strength_training']}")
            st.write(f"• {exercise['flexibility']}")
            
            st.markdown(f"**💡 Intensity Guide**: {exercise['intensity_guide']}")
            
            # Weekly schedule
            st.markdown("#### 📅 Weekly Schedule")
            schedule_df = pd.DataFrame([
                {"Day": day, "Activity": activity}
                for day, activity in exercise['weekly_schedule'].items()
            ])
            st.table(schedule_df)
        
        with tab3:
            st.markdown("### Lifestyle Modifications")
            
            st.markdown("These lifestyle changes are tailored to your health profile and risk factors:")
            
            for i, modification in enumerate(plan['lifestyle_modifications'], 1):
                # Color code based on urgency
                if "URGENT" in modification:
                    st.error(f"**{i}.** {modification}")
                elif any(word in modification for word in ["Reduce", "Limit", "Weight management"]):
                    st.warning(f"**{i}.** {modification}")
                else:
                    st.info(f"**{i}.** {modification}")
        
        with tab4:
            st.markdown("### Daily Routine")
            routine = plan['daily_routine']
            
            for time_period, activities in routine.items():
                st.markdown(f"**{time_period.title()}**")
                st.write(f"• {activities}")
                st.markdown("---")
            
            st.markdown("### 💊 Recommended Supplements")
            st.markdown("*Consult with your doctor before starting any supplements*")
            
            for supplement in plan['supplements_recommended']:
                if "Consult" in supplement:
                    st.warning(f"⚠️ {supplement}")
                else:
                    st.write(f"• {supplement}")
    
    def display_decision_path(self, explainability: Dict[str, Any]):
        """Display decision tree path"""
        st.subheader("🎯 AI Decision Path")
        
        st.markdown("""
        This shows the step-by-step reasoning process the AI used to arrive at the prediction.
        Each step represents a decision point based on specific health factors.
        """)
        
        decision_path = explainability['decision_path']
        
        for i, step in enumerate(decision_path, 1):
            st.markdown(f"""
            <div style="background: #f8fafc; border-left: 4px solid #0891b2; 
                        padding: 1rem; margin: 0.5rem 0; border-radius: 6px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <strong style="color: #0891b2;">Step {i}:</strong> <span style="color: #334155;">{step}</span>
            </div>
            """, unsafe_allow_html=True)
    
    def display_recommendations(self, recommendations: List[str]):
        """Display AI-generated recommendations"""
        st.subheader("💡 Personalized Recommendations")
        
        st.markdown("""
        These recommendations are generated by analyzing your health data and risk factors.
        They are prioritized based on potential impact.
        """)
        
        for i, recommendation in enumerate(recommendations, 1):
            priority = "HIGH" if "🔴" in recommendation else "MODERATE" if "🟡" in recommendation else "NORMAL"
            
            if priority == "HIGH":
                st.error(f"**{i}.** {recommendation}")
            elif priority == "MODERATE":
                st.warning(f"**{i}.** {recommendation}")
            else:
                st.info(f"**{i}.** {recommendation}")
    
    def render_pattern_analysis(self):
        """Render pattern analysis interface"""
        st.subheader("📈 Health Pattern Recognition & Trends")
        
        st.info("🔄 Pattern analysis would analyze your historical health data to identify trends, anomalies, and predict future risks.")
        
        # Placeholder for pattern analysis
        st.markdown("""
        **Features:**
        - Trend detection in vital signs
        - Anomaly identification
        - Predictive risk forecasting
        - Seasonal pattern recognition
        """)
    
    def render_real_time_monitoring(self):
        """Render real-time monitoring interface"""
        st.subheader("⚡ Real-Time Health Monitoring")
        
        st.markdown("""
        Real-time monitoring provides instant alerts and immediate risk assessment
        based on current vital signs.
        """)
        
        with st.form("realtime_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
                bp_sys = st.number_input("BP Systolic", min_value=80, max_value=200, value=120)
            
            with col2:
                bp_dia = st.number_input("BP Diastolic", min_value=40, max_value=130, value=80)
                spo2 = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=98)
            
            with col3:
                temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6)
                respiratory_rate = st.number_input("Respiratory Rate", min_value=8, max_value=40, value=16)
            
            if st.form_submit_button("🔍 Analyze Vital Signs", use_container_width=True):
                self.perform_realtime_analysis({
                    'heart_rate': heart_rate,
                    'blood_pressure_systolic': bp_sys,
                    'blood_pressure_diastolic': bp_dia,
                    'oxygen_saturation': spo2,
                    'temperature': temperature,
                    'respiratory_rate': respiratory_rate
                })
    
    def perform_realtime_analysis(self, vital_signs: Dict):
        """Perform real-time vital signs analysis"""
        try:
            response = self.api_client._make_request(
                'POST',
                '/ai-decision/real-time-monitoring',
                json={
                    'patient_id': 'current_user',
                    'vital_signs': vital_signs,
                    'enable_alerts': True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display status
                if result['status'] == 'ALERT':
                    st.error(f"🚨 {result['risk_level']} RISK DETECTED")
                elif result['status'] == 'WARNING':
                    st.warning(f"⚠️ {result['risk_level']} Risk")
                else:
                    st.success(f"✅ {result['risk_level']} Status")
                
                # Display alerts
                if result['alerts']:
                    st.markdown("### ⚠️ Active Alerts")
                    for alert in result['alerts']:
                        st.error(f"**{alert['parameter'].upper()}**: {alert['message']}")
                
                # Immediate actions
                if result['immediate_actions']:
                    st.markdown("### 🎯 Recommended Actions")
                    for action in result['immediate_actions']:
                        st.info(action)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    def run(self):
        """Main dashboard runner"""
        # Clean interface without header and feature cards
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "🔮 AI Prediction",
            "📈 Pattern Analysis",
            "⚡ Real-Time Monitoring"
        ])
        
        with tab1:
            self.render_prediction_interface()
            
            # Show previous result if available
            if 'ai_prediction_result' in st.session_state:
                st.markdown("---")
                st.markdown("### 📋 Previous Analysis")
                if st.button("🔄 View Last Result"):
                    self.display_prediction_results(st.session_state['ai_prediction_result'])
        
        with tab2:
            self.render_pattern_analysis()
        
        with tab3:
            self.render_real_time_monitoring()


def main():
    """Main entry point"""
    dashboard = AIDecisionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
