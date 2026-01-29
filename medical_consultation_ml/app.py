"""
Streamlit Web Application
Medical Consultation Transcription & Summarization System
Provides user interface for uploading audio files and viewing results
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import modules
try:
    from config import (
        SPEECH_TO_TEXT_CONFIG,
        SUMMARIZATION_CONFIG,
        PRESCRIPTION_CONFIG,
        TEXT_PREPROCESSING,
        AUDIO_PREPROCESSING,
        OUTPUT_CONFIG,
        STREAMLIT_CONFIG,
        AUDIO_DIR,
        TRANSCRIPTS_DIR,
        SUMMARIES_DIR,
        PRESCRIPTIONS_DIR
    )
    from modules.audio_processor import AudioProcessor
    from modules.text_processor import TextProcessor
    from modules.summarizer import MedicalSummarizer
    from modules.prescription_extractor import PrescriptionExtractor
    from modules.data_handler import DataHandler
    from utils.helpers import (
        format_timestamp,
        validate_audio_file,
        get_file_size_mb,
        format_duration,
        sanitize_filename
    )
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all modules are properly installed.")
    st.stop()

# Configure page
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2ca02c;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
    }
    .warning-box {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff8c00;
    }
    .prescription-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False


@st.cache_resource
def load_models():
    """Load all ML models (cached for performance)"""
    with st.spinner("Loading AI models... This may take a moment..."):
        try:
            audio_processor = AudioProcessor(SPEECH_TO_TEXT_CONFIG)
            text_processor = TextProcessor(TEXT_PREPROCESSING)
            summarizer = MedicalSummarizer(SUMMARIZATION_CONFIG)
            prescription_extractor = PrescriptionExtractor(PRESCRIPTION_CONFIG)
            data_handler = DataHandler({
                **OUTPUT_CONFIG,
                "transcripts_dir": TRANSCRIPTS_DIR,
                "summaries_dir": SUMMARIES_DIR,
                "prescriptions_dir": PRESCRIPTIONS_DIR
            })
            
            return audio_processor, text_processor, summarizer, prescription_extractor, data_handler
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.info("Please ensure all required packages are installed: pip install -r requirements.txt")
            st.stop()


def process_audio_file(audio_file, models):
    """
    Process uploaded audio file through the complete pipeline
    
    Args:
        audio_file: Uploaded file object
        models: Tuple of (audio_processor, text_processor, summarizer, prescription_extractor, data_handler)
        
    Returns:
        Dictionary with all processed results
    """
    audio_processor, text_processor, summarizer, prescription_extractor, data_handler = models
    
    # Save uploaded file temporarily
    temp_audio_path = AUDIO_DIR / sanitize_filename(audio_file.name)
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_file.getbuffer())
    
    results = {
        "audio_file": audio_file.name,
        "file_size": get_file_size_mb(temp_audio_path),
        "processing_time": {},
        "errors": []
    }
    
    # Step 1: Transcription
    with st.spinner("🎙️ Converting speech to text..."):
        try:
            start_time = datetime.now()
            transcript_result = audio_processor.process_audio_file(temp_audio_path)
            results["transcript"] = transcript_result
            results["processing_time"]["transcription"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            results["errors"].append(f"Transcription failed: {str(e)}")
            return results
    
    # Step 2: Text Processing
    with st.spinner("📝 Cleaning transcript..."):
        try:
            start_time = datetime.now()
            cleaned_text = text_processor.clean_transcript(transcript_result["text"])
            results["cleaned_text"] = cleaned_text
            results["processing_time"]["text_processing"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            results["errors"].append(f"Text processing failed: {str(e)}")
            results["cleaned_text"] = transcript_result["text"]
    
    # Step 3: Summarization
    with st.spinner("🤖 Generating medical summary..."):
        try:
            start_time = datetime.now()
            summary = summarizer.generate_medical_summary(results["cleaned_text"])
            results["summary"] = summary
            results["processing_time"]["summarization"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            results["errors"].append(f"Summarization failed: {str(e)}")
            results["summary"] = {"general_summary": "Failed to generate summary"}
    
    # Step 4: Prescription Extraction
    with st.spinner("💊 Extracting prescription details..."):
        try:
            start_time = datetime.now()
            prescriptions = prescription_extractor.extract_all_prescriptions(results["cleaned_text"])
            results["prescriptions"] = prescriptions
            results["processing_time"]["prescription_extraction"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            results["errors"].append(f"Prescription extraction failed: {str(e)}")
            results["prescriptions"] = []
    
    # Step 5: Save Results
    with st.spinner("💾 Saving results..."):
        try:
            data_handler.save_complete_report(
                audio_file.name,
                results["transcript"],
                results["summary"],
                results["prescriptions"]
            )
        except Exception as e:
            results["errors"].append(f"Failed to save results: {str(e)}")
    
    # Calculate total time
    results["processing_time"]["total"] = sum(results["processing_time"].values())
    
    return results


def display_results(results):
    """Display processing results in the UI"""
    
    # Processing Summary
    st.markdown('<div class="section-header">📊 Processing Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Audio File", results["audio_file"])
    with col2:
        st.metric("File Size", f"{results['file_size']} MB")
    with col3:
        st.metric("Duration", f"{results['transcript'].get('audio_duration', 0):.1f}s")
    with col4:
        st.metric("Processing Time", f"{results['processing_time']['total']:.1f}s")
    
    # Errors (if any)
    if results["errors"]:
        st.markdown('<div class="section-header">⚠️ Warnings</div>', unsafe_allow_html=True)
        for error in results["errors"]:
            st.warning(error)
    
    # Transcript
    st.markdown('<div class="section-header">📝 Transcript</div>', unsafe_allow_html=True)
    with st.expander("View Full Transcript", expanded=False):
        st.text_area("", results["transcript"]["text"], height=200, disabled=True)
    
    # Summary
    st.markdown('<div class="section-header">📋 Medical Summary</div>', unsafe_allow_html=True)
    
    summary = results["summary"]
    
    # General Summary
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("**General Summary:**")
    st.write(summary.get("general_summary", "No summary available"))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Structured sections
    col1, col2 = st.columns(2)
    
    with col1:
        if summary.get("chief_complaint"):
            st.markdown("**Chief Complaint:**")
            st.info(summary["chief_complaint"])
        
        if summary.get("assessment"):
            st.markdown("**Assessment:**")
            st.info(summary["assessment"])
    
    with col2:
        if summary.get("plan"):
            st.markdown("**Treatment Plan:**")
            st.info(summary["plan"])
    
    # Prescriptions
    st.markdown('<div class="section-header">💊 Prescriptions</div>', unsafe_allow_html=True)
    
    prescriptions = results["prescriptions"]
    
    if prescriptions:
        for idx, rx in enumerate(prescriptions, 1):
            st.markdown(f'<div class="prescription-card">', unsafe_allow_html=True)
            st.markdown(f"**Prescription #{idx}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Medicine:**")
                st.write(rx.get("medicine", "N/A"))
            
            with col2:
                st.markdown("**Dosage:**")
                st.write(rx.get("dosage", "Not specified"))
            
            with col3:
                st.markdown("**Frequency:**")
                st.write(rx.get("frequency", "Not specified"))
            
            with col4:
                st.markdown("**Duration:**")
                st.write(rx.get("duration", "Not specified"))
            
            if rx.get("instructions"):
                st.markdown("**Additional Instructions:**")
                for instruction in rx["instructions"][:2]:
                    st.write(f"• {instruction}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download prescriptions as CSV
        df = pd.DataFrame(prescriptions)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prescriptions (CSV)",
            data=csv,
            file_name=f"prescriptions_{format_timestamp()}.csv",
            mime="text/csv"
        )
    else:
        st.info("No prescriptions found in the consultation.")
    
    # Download complete report
    st.markdown('<div class="section-header">📥 Download Report</div>', unsafe_allow_html=True)
    
    import json
    report_json = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="📥 Download Complete Report (JSON)",
        data=report_json,
        file_name=f"medical_report_{format_timestamp()}.json",
        mime="application/json"
    )


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Medical Consultation Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666;">AI-Powered Speech-to-Text, Summarization & Prescription Extraction</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📚 About")
        st.info("""
        This application uses AI to:
        - 🎙️ Convert medical consultations to text
        - 📝 Generate concise summaries
        - 💊 Extract prescription details
        - 💾 Store structured medical records
        """)
        
        st.header("⚙️ Settings")
        st.write(f"**Model:** {SPEECH_TO_TEXT_CONFIG['model_name']}")
        st.write(f"**Device:** {SPEECH_TO_TEXT_CONFIG['device']}")
        st.write(f"**Summarizer:** {SUMMARIZATION_CONFIG['model_name'].split('/')[-1]}")
        
        st.header("📊 Statistics")
        try:
            models = load_models()
            _, _, _, _, data_handler = models
            stats = data_handler.get_statistics()
            st.metric("Transcripts", stats["transcripts"])
            st.metric("Summaries", stats["summaries"])
            st.metric("Prescriptions", stats["prescriptions"])
        except:
            st.write("Stats unavailable")
    
    # Main content
    st.markdown('<div class="section-header">📤 Upload Audio File</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Supported formats:** WAV, MP3, M4A, FLAC, OGG  
    **Max file size:** 200 MB  
    **Recommended:** Clear audio with minimal background noise
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
        help="Upload a medical consultation audio recording"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name} ({get_file_size_mb(Path(uploaded_file.name)):.2f} MB)")
        
        # Audio player
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button("🚀 Process Audio File", use_container_width=True, type="primary")
        
        if process_button:
            # Load models
            models = load_models()
            
            # Process
            st.markdown("---")
            st.markdown('<div class="section-header">⚙️ Processing...</div>', unsafe_allow_html=True)
            
            results = process_audio_file(uploaded_file, models)
            
            st.session_state.processed_data = results
            st.session_state.processing_complete = True
            
            # Display results
            st.markdown("---")
            st.success("✅ Processing completed successfully!")
            display_results(results)
    
    # Display previous results if available
    elif st.session_state.processing_complete and st.session_state.processed_data:
        st.markdown("---")
        st.info("Showing previously processed results")
        display_results(st.session_state.processed_data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🏥 Medical Consultation ML System | Built with Streamlit, Whisper & Transformers</p>
    <p>⚠️ For research and educational purposes only. Not a substitute for professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
