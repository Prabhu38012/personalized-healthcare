import streamlit as st
import pandas as pd
from frontend.components.lab_forms import LabPatientInputForm
from frontend.utils.api_client import HealthcareAPI

def main():
    # Don't set page config here since it's already set in the main app
    st.title("🔬 Lab-Enhanced Health Risk Analysis")
    st.markdown("""
    **Advanced health risk assessment using comprehensive laboratory data**
    
    This enhanced analysis incorporates detailed hematology and blood chemistry values 
    to provide more accurate and comprehensive health risk predictions.
    """)
    
    # Initialize API client
    api_client = HealthcareAPI()
    
    # Check backend health
    if not api_client.check_backend_health():
        st.error("⚠️ Backend service is not available. Please ensure the backend server is running.")
        st.stop()
    
    # Create tabs for different functionalities
    input_tab, sample_tab, results_tab = st.tabs(["📝 Lab Data Input", "🧪 Sample Report", "📊 Analysis Results"])
    
    with input_tab:
        st.markdown("### Enter Your Lab Values")
        
        # Initialize form
        lab_form = LabPatientInputForm()
        patient_data = lab_form.render()
        
        # Analysis options
        st.markdown("### Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            force_llm = st.checkbox("Enable AI Analysis", 
                                  value=True,
                                  help="Uses AI for comprehensive lab interpretation and clinical insights")
        
        with col2:
            analysis_type = st.selectbox("Analysis Type", 
                                       ["🔬 Lab Analysis", "📊 Basic Assessment"],
                                       help="Lab Analysis uses AI for complete medical interpretation")
        
        # Prediction button
        if st.button("🔍 Analyze Health Risk", type="primary"):
            if patient_data:
                with st.spinner("Analyzing your health data..."):
                    try:
                        if analysis_type == "🔬 Lab Analysis":
                            result = api_client.make_lab_prediction(patient_data, force_llm)
                        else:
                            result = api_client.make_prediction(patient_data, force_llm)
                        
                        if result:
                            st.session_state['prediction_result'] = result
                            st.session_state['analysis_type'] = analysis_type
                            st.success("✅ Analysis completed! Check the 'Analysis Results' tab.")
                        else:
                            st.error("❌ Analysis failed. Please check your input data and try again.")
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
            else:
                st.warning("⚠️ Please fill in the required basic information to proceed.")
    
    with sample_tab:
        st.markdown("### 📋 Sample Report Analysis")
        st.markdown("*Try the system with pre-loaded sample lab data or upload your own report*")
        
        # Create tabs for sample data and file upload
        sample_data_tab, upload_report_tab = st.tabs(["📊 Sample Data", "📄 Upload Report"])
        
        with sample_data_tab:
            st.markdown("#### Sample Lab Report Data")
            st.markdown("*Based on the hematology report you provided*")
            
            # Display sample data
            sample_data = LabPatientInputForm().get_sample_lab_data()
            
            # Create a formatted display of the sample data
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Patient Demographics")
                st.write(f"**Age:** {sample_data['age']} years")
                st.write(f"**Sex:** {'Male' if sample_data['sex'] == 'M' else 'Female'}")
                st.write(f"**Weight:** {sample_data['weight']} kg")
                st.write(f"**Height:** {sample_data['height']} cm")
                
                st.markdown("#### Vital Signs")
                st.write(f"**Blood Pressure:** {sample_data['systolic_bp']}/{sample_data['diastolic_bp']} mmHg")
                
                st.markdown("#### Complete Blood Count")
                st.write(f"**Hemoglobin:** {sample_data['hemoglobin']} g/dL")
                st.write(f"**Total WBC:** {sample_data['total_leukocyte_count']} × 10³/μL")
                st.write(f"**RBC Count:** {sample_data['red_blood_cell_count']} × 10⁶/μL")
                st.write(f"**Hematocrit:** {sample_data['hematocrit']}%")
            
            with col2:
                st.markdown("#### RBC Indices")
                st.write(f"**MCV:** {sample_data['mean_corpuscular_volume']} fL")
                st.write(f"**MCH:** {sample_data['mean_corpuscular_hb']} pg")
                st.write(f"**MCHC:** {sample_data['mean_corpuscular_hb_conc']} g/dL")
                st.write(f"**RDW:** {sample_data['red_cell_distribution_width']}%")
                
                st.markdown("#### Differential Count")
                st.write(f"**Neutrophils:** {sample_data['neutrophils_percent']}%")
                st.write(f"**Lymphocytes:** {sample_data['lymphocytes_percent']}%")
                st.write(f"**Monocytes:** {sample_data['monocytes_percent']}%")
                st.write(f"**Eosinophils:** {sample_data['eosinophils_percent']}%")
                st.write(f"**Basophils:** {sample_data['basophils_percent']}%")
                
                st.markdown("#### Platelet Parameters")
                st.write(f"**Platelet Count:** {sample_data['platelet_count']} × 10³/μL")
                st.write(f"**ESR:** {sample_data['erythrocyte_sedimentation_rate']} mm/hr")
            
            # Button to load sample data
            if st.button("📋 Analyze Sample Data"):
                with st.spinner("Analyzing sample lab data..."):
                    try:
                        result = api_client.make_lab_prediction(sample_data, force_llm=True)
                        if result:
                            st.session_state['prediction_result'] = result
                            st.session_state['analysis_type'] = "🔬 Lab Analysis (Sample)"
                            st.success("✅ Analysis completed! Check the 'Analysis Results' tab.")
                        else:
                            st.error("❌ Analysis failed.")
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
        
        with upload_report_tab:
            st.markdown("#### 📤 Upload Lab Report")
            st.markdown("Upload an image (JPG, PNG) or PDF of your lab report for automatic analysis")
            
            uploaded_file = st.file_uploader(
                "Choose a lab report file",
                type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'pdf'],
                help="Supported formats: Images (JPEG, PNG, GIF, BMP, TIFF, WebP) and PDF files"
            )
            
            if uploaded_file is not None:
                # Display file info
                st.info(f"**File:** {uploaded_file.name}\n**Size:** {uploaded_file.size:,} bytes\n**Type:** {uploaded_file.type}")
                
                # Show preview for images
                if uploaded_file.type.startswith('image/'):
                    st.image(uploaded_file, caption="Uploaded Lab Report", use_column_width=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    analyze_btn = st.button("🔍 Analyze Uploaded Report", type="primary", key="analyze_upload_btn")
                    
                if analyze_btn:
                    with st.spinner("Extracting data from report and analyzing..."):
                        try:
                            # Read file content
                            file_content = uploaded_file.read()
                            
                            # Upload and analyze
                            result = api_client.upload_lab_report(
                                file_content, 
                                uploaded_file.name, 
                                uploaded_file.type
                            )
                            
                            if result and result.get("success", True):
                                st.success("✅ Report analyzed successfully!")
                                
                                # Show extraction info if available
                                if "extraction_info" in result:
                                    extraction_info = result["extraction_info"]
                                    with st.expander("📋 Extraction Details"):
                                        st.json(extraction_info.get("extraction_metadata", {}))
                                
                                # Check if fallback was used
                                if result.get("fallback_used"):
                                    st.warning("⚠️ " + result.get("message", "Using sample data"))
                                
                                # Display results
                                st.session_state['prediction_result'] = result
                                st.session_state['analysis_type'] = "🔬 Lab Analysis (Uploaded Report)"
                                
                                # Switch to results tab without page reload
                                st.session_state['active_tab'] = 'results'
                                st.success("✅ Analysis completed! Results shown below.")
                                
                            else:
                                error_msg = result.get("error", "Unknown error occurred") if result else "Failed to process report"
                                st.error(f"❌ {error_msg}")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing file: {str(e)}")
                
                with col2:
                    st.warning("**Tips for best results:**\n- Ensure text is clearly visible\n- Avoid blurry or rotated images\n- Include complete lab parameter names and values\n- PDF files should contain selectable text")
            
            else:
                st.markdown("**Supported file types:**")
                st.markdown("- **Images:** JPEG, PNG, GIF, BMP, TIFF, WebP")
                st.markdown("- **Documents:** PDF")
                st.markdown("- **Max file size:** 10MB")
    
    with results_tab:
        if 'prediction_result' in st.session_state:
            display_prediction_results(st.session_state['prediction_result'], 
                                     st.session_state.get('analysis_type', 'Unknown'))
        else:
            st.info("🔍 No analysis results yet. Please run an analysis first.")

def display_prediction_results(result, analysis_type):
    """Display comprehensive prediction results"""
    st.markdown(f"### 📊 Health Risk Analysis Results ({analysis_type})")
    
    # Get patient data from result
    patient_data = result.get('data', {})
    
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
        
        # First try to get BMI directly from the patient data
        if 'data' in result and 'bmi' in result['data'] and result['data']['bmi'] is not None:
            bmi_value = float(result['data']['bmi'])
        # Then try to get from extraction info
        elif 'extraction_info' in result:
            extracted_data = result.get('extraction_info', {}).get('extracted_data', {})
            if 'bmi' in extracted_data and extracted_data['bmi'] is not None:
                bmi_value = float(extracted_data['bmi'])
            # If no BMI but we have weight and height, calculate it
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
        # Blood Pressure
        systolic = patient_data.get('systolic_bp')
        diastolic = patient_data.get('diastolic_bp')
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
    
    with col2:
        # Cholesterol
        total_chol = patient_data.get('total_cholesterol')
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

if __name__ == "__main__":
    main()
