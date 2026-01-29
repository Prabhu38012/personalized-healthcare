"""
Medical Consultation Page
Audio transcription, summarization, and prescription extraction
Replaces risk assessment functionality
"""

import streamlit as st
import requests
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.api_client import HealthcareAPI
    from components.auth import get_auth_headers
except ImportError:
    from frontend.utils.api_client import HealthcareAPI
    from frontend.components.auth import get_auth_headers


def show_medical_consultation():
    """Display medical consultation analysis interface"""
    
    # Initialize API client
    api_client = HealthcareAPI()
    api_client.set_auth_headers(get_auth_headers())
    
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4;">🎙️ Medical Consultation Analysis</h1>
            <p style="font-size: 1.1rem; color: #666;">
                AI-Powered Audio Transcription, Summarization & Prescription Extraction
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check service status
    try:
        status_response = api_client._make_request('GET', '/consultation/status')
        service_available = status_response.get('available', False) if status_response else False
    except:
        service_available = False
    
    if not service_available:
        st.error("""
        ⚠️ **Medical Consultation Service Unavailable**
        
        The AI models are not loaded. This could be because:
        - Models are still loading (first-time use takes 1-2 minutes)
        - Required packages are not installed
        - Insufficient system resources
        
        Please wait a moment and refresh the page, or contact support if the issue persists.
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 New Consultation", "📚 History", "ℹ️ About"])
    
    with tab1:
        show_upload_interface(api_client)
    
    with tab2:
        show_consultation_history(api_client)
    
    with tab3:
        show_about_consultation()


def show_upload_interface(api_client):
    """Show the audio upload and processing interface"""
    
    st.markdown("### 📤 Upload Medical Consultation Audio")
    
    # Info box
    st.info("""
    **Supported Formats:** WAV, MP3, M4A, FLAC, OGG  
    **Maximum Size:** 200 MB  
    **Best Results:** Clear audio with minimal background noise
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
        help="Upload a medical consultation recording"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        
        # Audio player
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Consultation", use_container_width=True, type="primary"):
                process_consultation_audio(uploaded_file, api_client)


def process_consultation_audio(uploaded_file, api_client):
    """Process the uploaded audio file"""
    
    with st.spinner("🔄 Processing consultation... This may take a few minutes..."):
        try:
            # Prepare file for upload
            files = {'audio_file': (uploaded_file.name, uploaded_file.getvalue(), f'audio/{uploaded_file.name.split(".")[-1]}')}
            
            # Estimate processing time (large-v2 on CPU is ~2x slower than realtime)
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            # Rough estimate: 1MB ≈ 1 minute of audio, processing takes 2-3x audio duration
            estimated_audio_minutes = file_size_mb
            estimated_processing_minutes = int(estimated_audio_minutes * 2.5)
            timeout = max(900, estimated_processing_minutes * 60 + 180)  # Min 15 min, or estimate + 3min buffer
            
            st.info(f"⏱️ Processing your {file_size_mb:.1f}MB file (~{int(estimated_audio_minutes)} min audio)...")
            st.info(f"🔄 Estimated processing time: **{estimated_processing_minutes} minutes**")
            st.warning("⚠️ Please keep this page open. Processing large files with AI models on CPU takes time.")
            
            # Make API request using the API client
            result = api_client._make_request(
                'POST',
                '/consultation/process',
                files=files,
                timeout=timeout
            )
            
            if result and result.get('success'):
                st.success("✅ Consultation processed successfully!")
                st.session_state.last_consultation_result = result['data']
                display_consultation_results(result['data'])
            elif result:
                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            else:
                st.error("❌ Failed to process consultation")
                
        except requests.exceptions.Timeout:
            st.error(f"⏱️ Request timed out after {timeout//60} minutes. Large files with large-v2 model on CPU take significant time. Consider:")
            st.info("💡 Tips:\n- Use shorter audio files (under 5 minutes)\n- Wait and try again\n- The model may still be processing in background")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def display_consultation_results(results):
    """Display the processing results"""
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Processing Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Audio Duration", f"{results.get('transcript', {}).get('audio_duration', 0):.1f}s")
    with col2:
        st.metric("Processing Time", f"{results.get('processing_time', {}).get('total', 0):.1f}s")
    with col3:
        st.metric("Language", results.get('transcript', {}).get('language', 'N/A').upper())
    with col4:
        prescription_count = len(results.get('prescriptions', []))
        st.metric("Prescriptions", prescription_count)
    
    # Errors/Warnings
    if results.get('errors'):
        with st.expander("⚠️ Warnings", expanded=False):
            for error in results['errors']:
                st.warning(error)
    
    # Transcript
    st.markdown("### 📝 Transcript")
    with st.expander("View Full Transcript", expanded=False):
        st.text_area(
            "",
            results.get('transcript', {}).get('text', 'No transcript available'),
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )
    
    # Medical Summary
    st.markdown("### 📋 Medical Summary")
    summary = results.get('summary', {})
    
    # General Summary
    st.markdown("**Overview:**")
    st.info(summary.get('general_summary', 'No summary available'))
    
    # Structured sections
    col1, col2 = st.columns(2)
    
    with col1:
        if summary.get('chief_complaint'):
            st.markdown("**Chief Complaint:**")
            st.success(summary['chief_complaint'])
        
        if summary.get('assessment'):
            st.markdown("**Assessment/Diagnosis:**")
            st.success(summary['assessment'])
    
    with col2:
        if summary.get('plan'):
            st.markdown("**Treatment Plan:**")
            st.success(summary['plan'])
    
    # Prescriptions
    st.markdown("### 💊 Prescriptions")
    
    prescriptions = results.get('prescriptions', [])
    
    if prescriptions:
        for idx, rx in enumerate(prescriptions, 1):
            with st.container():
                st.markdown(f"**Prescription #{idx}**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Medicine:**")
                    st.write(rx.get('medicine', 'N/A'))
                
                with col2:
                    st.markdown("**Dosage:**")
                    st.write(rx.get('dosage', 'Not specified'))
                
                with col3:
                    st.markdown("**Frequency:**")
                    st.write(rx.get('frequency', 'Not specified'))
                
                with col4:
                    st.markdown("**Duration:**")
                    st.write(rx.get('duration', 'Not specified'))
                
                if rx.get('instructions'):
                    st.markdown("**Additional Instructions:**")
                    for instruction in rx['instructions'][:2]:
                        st.write(f"• {instruction}")
                
                st.markdown("---")
        
        # Download prescriptions
        df = pd.DataFrame(prescriptions)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prescriptions (CSV)",
            data=csv,
            file_name=f"prescriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No prescriptions found in this consultation.")
    
    # Download complete report
    st.markdown("### 📥 Export Report")
    report_json = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="📥 Download Complete Report (JSON)",
        data=report_json,
        file_name=f"consultation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def show_consultation_history(api_client):
    """Show consultation history"""
    
    st.markdown("### 📚 Consultation History")
    
    try:
        data = api_client._make_request('GET', '/consultation/history', timeout=10)
        
        if data:
            consultations = data.get('consultations', [])
            
            if consultations:
                st.info(f"Found {len(consultations)} consultation record(s)")
                
                for filename in consultations:
                    with st.expander(f"📄 {filename}"):
                        if st.button(f"Load Report", key=f"load_{filename}"):
                            load_consultation_report(filename, api_client)
            else:
                st.info("No consultation history found. Process your first consultation above!")
        else:
            st.error("Failed to load consultation history")
            
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")


def load_consultation_report(filename, api_client):
    """Load and display a specific consultation report"""
    
    try:
        data = api_client._make_request(
            'GET',
            f'/consultation/report/{filename}',
            timeout=10
        )
        
        if data:
            report = data.get('report', {})
            # Display the report
            display_consultation_results(report)
        else:
            st.error("Failed to load report")
            
    except Exception as e:
        st.error(f"Error loading report: {str(e)}")


def show_about_consultation():
    """Show information about the consultation feature"""
    
    st.markdown("""
    ### 🏥 About Medical Consultation Analysis
    
    This AI-powered feature helps healthcare providers and researchers analyze medical consultation recordings.
    
    #### 🎯 Features
    
    - **🎙️ Speech-to-Text:** Convert consultation audio to accurate text transcripts using OpenAI Whisper
    - **📝 Smart Summarization:** Generate concise medical summaries with BART AI model
    - **💊 Prescription Extraction:** Automatically detect medicines, dosages, frequencies, and durations
    - **📊 Structured Output:** Save results in JSON/CSV formats for easy integration
    - **🔒 Secure:** All processing happens on your server with user authentication
    
    #### 🚀 How It Works
    
    1. **Upload** a medical consultation audio recording
    2. **AI Processing** converts speech to text and analyzes content
    3. **Extraction** identifies key medical information and prescriptions
    4. **Review** the transcript, summary, and prescriptions
    5. **Export** complete reports for medical records
    
    #### 📋 Supported Audio Formats
    
    - WAV (`.wav`)
    - MP3 (`.mp3`)
    - M4A (`.m4a`)
    - FLAC (`.flac`)
    - OGG (`.ogg`)
    
    #### 💡 Tips for Best Results
    
    - Use clear audio with minimal background noise
    - Ensure proper microphone quality
    - Speak clearly and at a moderate pace
    - Keep file size under 200 MB
    - For longer consultations, consider using larger AI models
    
    #### ⚙️ Technical Details
    
    - **Speech Recognition:** OpenAI Whisper (base model)
    - **Summarization:** Facebook BART-large-CNN
    - **Processing:** PyTorch with CPU/GPU support
    - **Security:** JWT token authentication
    
    #### ⚠️ Important Notes
    
    - This tool is for research and educational purposes
    - Always verify AI-generated information
    - Ensure proper patient consent for recordings
    - Follow HIPAA and local privacy regulations
    - Not a substitute for professional medical judgment
    
    ---
    
    **Need Help?** Contact your system administrator or refer to the documentation.
    """)


# Main entry point
if __name__ == "__main__":
    show_medical_consultation()
