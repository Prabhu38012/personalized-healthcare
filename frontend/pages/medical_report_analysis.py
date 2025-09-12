"""
Medical Report Analysis Page
Upload and analyze medical reports with comprehensive results display
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
from typing import Dict, List, Any, Optional

# Import utilities
try:
    from frontend.utils.api_client import APIClient
    from frontend.utils.caching import cache_data
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.api_client import APIClient
    from utils.caching import cache_data

class MedicalReportAnalyzer:
    """Frontend class for medical report analysis"""
    
    def __init__(self):
        self.api_client = APIClient()
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="Medical Report Analysis",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def upload_and_analyze_report(self, uploaded_file, patient_name: str) -> Optional[Dict[str, Any]]:
        """Upload and analyze medical report"""
        try:
            # Prepare file for upload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"patient_name": patient_name}
            
            # Make API request
            response = requests.post(
                f"{self.api_client.base_url}/api/medical-report/upload",
                files=files,
                data=data,
                timeout=120  # 2 minute timeout for analysis
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Analysis failed: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Analysis timed out. Please try with a smaller file or try again later.")
            return None
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            return None
    
    def get_analysis_list(self, limit: int = 10, patient_name: str = None) -> Optional[Dict[str, Any]]:
        """Get list of previous analyses"""
        try:
            params = {"limit": limit}
            if patient_name:
                params["patient_name"] = patient_name
            
            response = requests.get(
                f"{self.api_client.base_url}/api/medical-report/list",
                params=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            st.error(f"Failed to fetch analysis list: {str(e)}")
            return None
    
    def get_analysis_details(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis results"""
        try:
            response = requests.get(
                f"{self.api_client.base_url}/api/medical-report/analysis/{analysis_id}"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            st.error(f"Failed to fetch analysis details: {str(e)}")
            return None
    
    def download_pdf_report(self, analysis_id: str, patient_name: str):
        """Download PDF report"""
        try:
            response = requests.get(
                f"{self.api_client.base_url}/api/medical-report/download/{analysis_id}"
            )
            
            if response.status_code == 200:
                # Create download button
                st.download_button(
                    label="📄 Download PDF Report",
                    data=response.content,
                    file_name=f"medical_report_{patient_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("Failed to generate PDF report")
                
        except Exception as e:
            st.error(f"PDF download failed: {str(e)}")
    
    def display_confidence_meter(self, confidence: float):
        """Display confidence score as a meter"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Analysis Confidence"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
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
    
    def display_summary_metrics(self, summary: Dict[str, Any]):
        """Display summary metrics in columns"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Conditions Found", summary.get('conditions_found', 0))
            st.metric("Lab Values", summary.get('lab_values_extracted', 0))
        
        with col2:
            st.metric("Medications", summary.get('medications_identified', 0))
            st.metric("Future Risks", summary.get('future_risks', 0))
        
        with col3:
            st.metric("Symptoms", summary.get('symptoms_noted', 0))
            st.metric("Recommendations", summary.get('recommendations', 0))
    
    def display_conditions_analysis(self, conditions: List[Dict[str, Any]]):
        """Display medical conditions analysis"""
        if not conditions:
            st.info("No medical conditions were identified in the report.")
            return
        
        st.subheader("🏥 Medical Conditions Identified")
        
        # Create DataFrame for better display
        conditions_df = pd.DataFrame([
            {
                'Condition': condition['text'],
                'Type': condition['type'].replace('CONDITION_', '').replace('_', ' ').title(),
                'Confidence': f"{condition['confidence']:.1%}",
                'Context Preview': condition['context'][:100] + "..." if len(condition['context']) > 100 else condition['context']
            }
            for condition in conditions
        ])
        
        st.dataframe(conditions_df, use_container_width=True)
        
        # Condition type distribution
        if len(conditions) > 1:
            type_counts = pd.DataFrame([
                condition['type'].replace('CONDITION_', '').replace('_', ' ').title() 
                for condition in conditions
            ], columns=['Type']).value_counts().reset_index()
            
            fig = px.pie(type_counts, values='count', names='Type', 
                        title="Distribution of Condition Types")
            st.plotly_chart(fig, use_container_width=True)
    
    def display_medications_analysis(self, medications: List[Dict[str, Any]]):
        """Display medications analysis"""
        if not medications:
            st.info("No medications were identified in the report.")
            return
        
        st.subheader("💊 Medications and Treatments")
        
        # Create DataFrame
        medications_df = pd.DataFrame([
            {
                'Medication/Treatment': med['text'],
                'Confidence': f"{med['confidence']:.1%}",
                'Context Preview': med['context'][:100] + "..." if len(med['context']) > 100 else med['context']
            }
            for med in medications
        ])
        
        st.dataframe(medications_df, use_container_width=True)
    
    def display_lab_values_analysis(self, lab_values: List[Dict[str, Any]]):
        """Display lab values analysis"""
        if not lab_values:
            st.info("No laboratory values were identified in the report.")
            return
        
        st.subheader("🔬 Laboratory Values")
        
        # Create DataFrame
        lab_df = pd.DataFrame([
            {
                'Test/Parameter': lab['test'],
                'Value': lab['value'],
                'Context Preview': lab['context'][:80] + "..." if len(lab['context']) > 80 else lab['context']
            }
            for lab in lab_values
        ])
        
        st.dataframe(lab_df, use_container_width=True)
    
    def display_risk_assessment(self, future_risks: List[str]):
        """Display future risk assessment"""
        if not future_risks:
            st.info("No specific future health risks were identified.")
            return
        
        st.subheader("⚠️ Future Health Risk Assessment")
        
        st.warning("""
        **Important:** The following risk assessments are based on statistical correlations and medical literature. 
        They are not definitive predictions and should be discussed with your healthcare provider.
        """)
        
        for i, risk in enumerate(future_risks, 1):
            st.write(f"**{i}.** {risk}")
    
    def display_recommendations(self, recommendations: List[str]):
        """Display health recommendations"""
        if not recommendations:
            st.info("No specific recommendations were generated.")
            return
        
        st.subheader("💡 Health Recommendations")
        
        st.success("""
        **Note:** These recommendations are based on identified conditions and established medical guidelines. 
        Please consult with your healthcare provider before making any changes to your treatment plan.
        """)
        
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"**{i}.** {recommendation}")
    
    def display_text_preview(self, text_preview: str):
        """Display extracted text preview"""
        with st.expander("📄 Extracted Text Preview"):
            st.text_area(
                "Original Text (Preview)",
                value=text_preview,
                height=200,
                disabled=True
            )
    
    def render_upload_section(self):
        """Render the file upload section"""
        st.header("🏥 Medical Report Analysis")
        st.markdown("""
        Upload your medical report (PDF or image) to get comprehensive analysis including:
        - **Medical conditions** identification and classification
        - **Medications and treatments** extraction
        - **Laboratory values** analysis
        - **Future health risks** prediction
        - **Personalized recommendations**
        - **Downloadable PDF report**
        """)
        
        # Upload form
        with st.form("upload_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose medical report file",
                    type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
                    help="Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP (Max 10MB)"
                )
            
            with col2:
                patient_name = st.text_input(
                    "Patient Name",
                    value="Patient",
                    help="Enter patient name for the report"
                )
            
            submit_button = st.form_submit_button(
                "🔍 Analyze Report",
                use_container_width=True
            )
            
            if submit_button and uploaded_file is not None:
                # Validate file size
                if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
                    st.error("File size too large. Maximum size is 10MB.")
                    return
                
                # Show progress
                with st.spinner("Analyzing medical report... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    
                    # Simulate progress
                    import time
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Perform analysis
                    result = self.upload_and_analyze_report(uploaded_file, patient_name)
                    
                    if result:
                        st.success("✅ Analysis completed successfully!")
                        
                        # Store result in session state for display
                        st.session_state['current_analysis'] = result
                        st.session_state['show_results'] = True
                        st.rerun()
    
    def render_results_section(self, analysis_data: Dict[str, Any]):
        """Render the analysis results section"""
        st.header("📊 Analysis Results")
        
        # Analysis metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Patient:** {analysis_data['patient_name']}")
        with col2:
            st.info(f"**Analysis ID:** {analysis_data['analysis_id'][:8]}...")
        with col3:
            analysis_date = datetime.fromisoformat(analysis_data['analysis_date'].replace('Z', '+00:00'))
            st.info(f"**Date:** {analysis_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Confidence score
        st.subheader("🎯 Analysis Confidence")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self.display_confidence_meter(analysis_data['confidence_score'])
        
        with col2:
            confidence_pct = analysis_data['confidence_score'] * 100
            if confidence_pct >= 80:
                st.success(f"**High Confidence ({confidence_pct:.1f}%)**\n\nThe analysis is based on clear, well-structured medical text with identifiable medical terminology.")
            elif confidence_pct >= 60:
                st.warning(f"**Moderate Confidence ({confidence_pct:.1f}%)**\n\nThe analysis found some medical information, but there may be ambiguity in the source text.")
            else:
                st.error(f"**Low Confidence ({confidence_pct:.1f}%)**\n\nLimited medical information was extracted. Consider uploading a clearer document.")
        
        # Summary metrics
        st.subheader("📈 Summary Overview")
        self.display_summary_metrics(analysis_data['summary'])
        
        # Detailed analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏥 Conditions", "💊 Medications", "🔬 Lab Values", 
            "⚠️ Risk Assessment", "💡 Recommendations", "📄 Text Preview"
        ])
        
        with tab1:
            self.display_conditions_analysis(analysis_data['conditions'])
        
        with tab2:
            self.display_medications_analysis(analysis_data['medications'])
        
        with tab3:
            self.display_lab_values_analysis(analysis_data['lab_values'])
        
        with tab4:
            self.display_risk_assessment(analysis_data['future_risks'])
        
        with tab5:
            self.display_recommendations(analysis_data['recommendations'])
        
        with tab6:
            self.display_text_preview(analysis_data['text_preview'])
        
        # Download PDF report
        st.subheader("📄 Download Report")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.download_pdf_report(analysis_data['analysis_id'], analysis_data['patient_name'])
        
        with col2:
            if st.button("🔄 Analyze Another Report", use_container_width=True):
                # Clear session state
                if 'current_analysis' in st.session_state:
                    del st.session_state['current_analysis']
                if 'show_results' in st.session_state:
                    del st.session_state['show_results']
                st.rerun()
    
    def render_history_section(self):
        """Render analysis history section"""
        st.header("📚 Analysis History")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_filter = st.text_input("Filter by Patient Name", placeholder="Enter patient name...")
        
        with col2:
            limit = st.selectbox("Number of Results", [5, 10, 20, 50], index=1)
        
        with col3:
            if st.button("🔍 Search", use_container_width=True):
                st.session_state['refresh_history'] = True
        
        # Get analysis list
        if st.session_state.get('refresh_history', True):
            analysis_list = self.get_analysis_list(limit, patient_filter if patient_filter else None)
            st.session_state['analysis_history'] = analysis_list
            st.session_state['refresh_history'] = False
        else:
            analysis_list = st.session_state.get('analysis_history')
        
        if analysis_list and analysis_list['reports']:
            # Display results
            st.subheader(f"Found {analysis_list['total_count']} analyses")
            
            for report in analysis_list['reports']:
                with st.expander(f"📋 {report['patient_name']} - {report['filename']} ({report['analysis_date'][:10]})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Confidence", f"{report['confidence_score']:.1%}")
                    
                    with col2:
                        st.metric("Conditions", report['conditions_count'])
                    
                    with col3:
                        st.metric("Medications", report['medications_count'])
                    
                    with col4:
                        file_size_mb = report['file_size'] / (1024 * 1024)
                        st.metric("File Size", f"{file_size_mb:.1f} MB")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"👁️ View Details", key=f"view_{report['analysis_id']}"):
                            details = self.get_analysis_details(report['analysis_id'])
                            if details:
                                st.session_state['current_analysis'] = details['analysis_data']
                                st.session_state['current_analysis']['analysis_id'] = report['analysis_id']
                                st.session_state['show_results'] = True
                                st.rerun()
                    
                    with col2:
                        self.download_pdf_report(report['analysis_id'], report['patient_name'])
                    
                    with col3:
                        if st.button(f"🗑️ Delete", key=f"delete_{report['analysis_id']}", type="secondary"):
                            if st.session_state.get(f"confirm_delete_{report['analysis_id']}", False):
                                # Perform deletion
                                try:
                                    response = requests.delete(
                                        f"{self.api_client.base_url}/api/medical-report/analysis/{report['analysis_id']}"
                                    )
                                    if response.status_code == 200:
                                        st.success("Analysis deleted successfully")
                                        st.session_state['refresh_history'] = True
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete analysis")
                                except Exception as e:
                                    st.error(f"Deletion failed: {str(e)}")
                            else:
                                st.session_state[f"confirm_delete_{report['analysis_id']}"] = True
                                st.warning("Click delete again to confirm")
        else:
            st.info("No previous analyses found.")
    
    def run(self):
        """Main application runner"""
        # Initialize session state
        if 'show_results' not in st.session_state:
            st.session_state['show_results'] = False
        
        # Sidebar navigation
        st.sidebar.title("🏥 Medical Report Analysis")
        
        page = st.sidebar.radio(
            "Navigation",
            ["📤 Upload & Analyze", "📊 View Results", "📚 Analysis History"],
            index=1 if st.session_state.get('show_results', False) else 0
        )
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### 🔍 Supported Analysis
        - Medical conditions identification
        - Medication extraction
        - Lab values analysis
        - Risk assessment
        - Health recommendations
        - PDF report generation
        
        ### 📁 Supported Formats
        - PDF documents
        - Image files (PNG, JPG, TIFF, BMP)
        - Maximum file size: 10MB
        """)
        
        # Main content based on navigation
        if page == "📤 Upload & Analyze":
            self.render_upload_section()
        
        elif page == "📊 View Results":
            if st.session_state.get('show_results', False) and 'current_analysis' in st.session_state:
                self.render_results_section(st.session_state['current_analysis'])
            else:
                st.info("No analysis results to display. Please upload and analyze a medical report first.")
                if st.button("📤 Go to Upload", use_container_width=True):
                    st.session_state['show_results'] = False
                    st.rerun()
        
        elif page == "📚 Analysis History":
            self.render_history_section()

def main():
    """Main function to run the medical report analysis app"""
    analyzer = MedicalReportAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
