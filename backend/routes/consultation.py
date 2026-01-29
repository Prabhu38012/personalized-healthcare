"""
Medical Consultation API Routes
Handles audio transcription, summarization, and prescription extraction
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os
from pathlib import Path
from datetime import datetime
import sys
import logging

# Add consultation modules to path
consultation_path = Path(__file__).parent.parent / "consultation"
sys.path.insert(0, str(consultation_path))

try:
    from audio_processor import AudioProcessor
    from text_processor import TextProcessor
    from summarizer import MedicalSummarizer
    from llm_summarizer import LLMSummarizer  # New LLM-based summarizer
    from prescription_extractor import PrescriptionExtractor
    from llm_prescription_extractor import LLMPrescriptionExtractor
    from medical_feature_extractor import MedicalFeatureExtractor  # NEW: Extract features for ML
    from diagnosis_extractor import DiagnosisExtractor  # NEW: Extract diagnoses for ML
    from data_handler import DataHandler
    import config as consultation_config
except ImportError as e:
    print(f"Warning: Could not import consultation modules: {e}")
    AudioProcessor = None

# Import authentication dependencies
try:
    from backend.auth.routes import get_current_user
    auth_available = True
except ImportError:
    try:
        from auth.routes import get_current_user
        auth_available = True
    except ImportError:
        auth_available = False
        def get_current_user():
            return None

router = APIRouter(tags=["consultation"])

# Initialize processors (eager loading at startup)
processors = {
    'audio': None,
    'text': None,
    'summarizer': None,
    'llm_summarizer': None,  # New LLM-based summarizer
    'prescription': None,
    'llm_prescription': None,  # New LLM-based extractor
    'feature_extractor': None,  # NEW: Extract medical features for ML
    'diagnosis_extractor': None,  # NEW: Extract diagnoses for ML
    'data': None
}

def initialize_processors():
    """Initialize all processors at startup"""
    global processors
    if AudioProcessor is not None:
        try:
            print("🔄 Initializing Medical Consultation models...")
            print("   This may take 5-10 minutes on first run (downloading models)")
            
            print("   📥 Loading Speech-to-Text model (large-v2)...")
            processors['audio'] = AudioProcessor(consultation_config.SPEECH_TO_TEXT_CONFIG)
            print("   ✅ Speech-to-Text model loaded")
            
            print("   📥 Loading Text Processor...")
            processors['text'] = TextProcessor(consultation_config.TEXT_PREPROCESSING)
            print("   ✅ Text Processor loaded")
            
            print("   📥 Loading Medical Summarizer...")
            processors['summarizer'] = MedicalSummarizer(consultation_config.SUMMARIZATION_CONFIG)
            print("   ✅ Medical Summarizer loaded")
            
            print("   📥 Loading LLM Medical Summarizer...")
            processors['llm_summarizer'] = LLMSummarizer()
            print("   ✅ LLM Medical Summarizer loaded")
            
            print("   📥 Loading Prescription Extractor...")
            processors['prescription'] = PrescriptionExtractor(consultation_config.PRESCRIPTION_CONFIG)
            print("   ✅ Prescription Extractor loaded")
            
            print("   📥 Loading LLM Prescription Extractor...")
            processors['llm_prescription'] = LLMPrescriptionExtractor()
            print("   ✅ LLM Prescription Extractor loaded")
            
            print("   📥 Loading Medical Feature Extractor...")
            processors['feature_extractor'] = MedicalFeatureExtractor()
            print("   ✅ Medical Feature Extractor loaded")
            
            print("   📥 Loading Diagnosis Extractor...")
            processors['diagnosis_extractor'] = DiagnosisExtractor()
            print("   ✅ Diagnosis Extractor loaded")
            
            processors['data'] = DataHandler({
                **consultation_config.OUTPUT_CONFIG,
                "transcripts_dir": consultation_config.TRANSCRIPTS_DIR,
                "summaries_dir": consultation_config.SUMMARIES_DIR,
                "prescriptions_dir": consultation_config.PRESCRIPTIONS_DIR
            })
            
            print("✅ All Medical Consultation models ready!")
            return True
        except Exception as e:
            print(f"❌ Error initializing processors: {e}")
            import traceback
            traceback.print_exc()
            return False
    return False

# Initialize processors at module load time
print("\n" + "="*60)
initialization_success = initialize_processors()
print("="*60 + "\n")

def get_processors():
    """Get the pre-initialized processors"""
    if processors['audio'] is None:
        return None
    return processors

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@router.post('/process')
async def process_consultation(
    audio_file: UploadFile = File(...),
    current_user = Depends(get_current_user) if auth_available else None
):
    """
    Process medical consultation audio file
    Requires authentication
    """
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not allowed_file(audio_file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Get processors
        procs = get_processors()
        if not procs or procs['audio'] is None:
            raise HTTPException(
                status_code=503,
                detail="Consultation processing service not available"
            )
        
        # Save uploaded file
        from werkzeug.utils import secure_filename
        filename = secure_filename(audio_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        
        upload_dir = Path('data/audio_files')
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / unique_filename
        
        # Save the uploaded file
        with open(file_path, 'wb') as f:
            content = await audio_file.read()
            f.write(content)
        
        # Process audio file
        results = {
            "audio_file": filename,
            "user_id": current_user.id if current_user else None,
            "user_email": current_user.email if current_user else None,
            "processing_time": {},
            "errors": []
        }
        
        # Step 1: Transcription
        try:
            start_time = datetime.now()
            transcript_result = procs['audio'].process_audio_file(file_path)
            results["transcript"] = transcript_result
            results["processing_time"]["transcription"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            results["errors"].append(f"Transcription failed: {str(e)}")
            raise HTTPException(status_code=500, detail=results)
        
        # Step 2: Text Processing
        try:
            start_time = datetime.now()
            cleaned_text = procs['text'].clean_transcript(transcript_result["text"])
            results["cleaned_text"] = cleaned_text
            results["processing_time"]["text_processing"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            results["errors"].append(f"Text processing failed: {str(e)}")
            results["cleaned_text"] = transcript_result["text"]
        
        # Step 3: Summarization (using LLM for accuracy)
        try:
            start_time = datetime.now()
            
            # Try LLM summarizer first (much more accurate)
            if procs['llm_summarizer']:
                summary = procs['llm_summarizer'].generate_medical_summary(results["cleaned_text"])
                logging.info("✓ Using LLM-based summarization")
            else:
                # Fallback to transformer-based summarizer
                summary = procs['summarizer'].generate_medical_summary(results["cleaned_text"])
                logging.info("✓ Using transformer-based summarization")
            
            results["summary"] = summary
            results["processing_time"]["summarization"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            results["errors"].append(f"Summarization failed: {str(e)}")
            results["summary"] = {
                "general_summary": "Failed to generate summary",
                "chief_complaint": "Unable to extract",
                "assessment": "Unable to extract",
                "plan": "Unable to extract"
            }
        
        # Step 4: Prescription Extraction (LLM-based for accuracy)
        try:
            start_time = datetime.now()
            
            # Try LLM extraction first (much more accurate)
            prescriptions = procs['llm_prescription'].extract_prescriptions(results["cleaned_text"])
            
            # If LLM didn't find anything, it returns empty list (not an error)
            results["prescriptions"] = prescriptions
            
            if prescriptions:
                logging.info(f"✓ Found {len(prescriptions)} prescriptions using LLM")
            else:
                logging.info("ℹ️  No prescriptions found in consultation")
            
            results["processing_time"]["prescription_extraction"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            logging.error(f"Prescription extraction failed: {str(e)}")
            results["errors"].append(f"Prescription extraction failed: {str(e)}")
            results["prescriptions"] = []
        
        # Step 5: NEW - Medical Feature Extraction (for AI/ML prediction)
        try:
            start_time = datetime.now()
            
            if procs['feature_extractor']:
                extracted_features = procs['feature_extractor'].extract_features(results["cleaned_text"])
                results["extracted_features"] = extracted_features
                logging.info(f"✓ Extracted {len(extracted_features)} medical features for ML")
            
            results["processing_time"]["feature_extraction"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            logging.error(f"Feature extraction failed: {str(e)}")
            results["errors"].append(f"Feature extraction failed: {str(e)}")
            results["extracted_features"] = {}
        
        # Step 6: NEW - Diagnosis Extraction (for AI/ML prediction)
        try:
            start_time = datetime.now()
            
            if procs['diagnosis_extractor']:
                diagnosis_data = procs['diagnosis_extractor'].extract_diagnoses(results["summary"])
                results["diagnosis_data"] = diagnosis_data
                
                # Prepare for ML prediction
                patient_data_for_ml = procs['diagnosis_extractor'].prepare_for_ml_prediction(
                    diagnosis_data,
                    results.get("extracted_features", {})
                )
                results["patient_data_for_ml"] = patient_data_for_ml
                
                logging.info(f"✓ Extracted diagnoses: {diagnosis_data.get('primary_diagnosis', 'Unknown')}")
                logging.info(f"✓ Prepared patient data for ML with {len(patient_data_for_ml)} features")
            
            results["processing_time"]["diagnosis_extraction"] = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            logging.error(f"Diagnosis extraction failed: {str(e)}")
            results["errors"].append(f"Diagnosis extraction failed: {str(e)}")
            results["diagnosis_data"] = {}
            results["patient_data_for_ml"] = {}
        
        # Calculate total time
        results["processing_time"]["total"] = sum(results["processing_time"].values())
        
        # Step 7: Save Results
        try:
            procs['data'].save_complete_report(
                unique_filename,
                results["transcript"],
                results["summary"],
                results["prescriptions"]
            )
        except Exception as e:
            results["errors"].append(f"Failed to save results: {str(e)}")
        
        # Clean up uploaded file (optional)
        # file_path.unlink()
        
        return {
            "success": True,
            "data": results,
            "message": "Consultation processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})

@router.get('/history')
async def get_consultation_history(
    current_user = Depends(get_current_user) if auth_available else None
):
    """
    Get consultation history for the authenticated user
    """
    try:
        procs = get_processors()
        if not procs or procs['data'] is None:
            raise HTTPException(status_code=503, detail="Service not available")
        
        # Get all summaries
        files = procs['data'].list_files("summaries")
        
        return {
            "success": True,
            "consultations": files,
            "count": len(files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})

@router.get('/report/{filename}')
async def get_consultation_report(
    filename: str,
    current_user = Depends(get_current_user) if auth_available else None
):
    """
    Get a specific consultation report
    """
    try:
        procs = get_processors()
        if not procs or procs['data'] is None:
            raise HTTPException(status_code=503, detail="Service not available")
        
        # Load the report
        report = procs['data'].load_summary(filename)
        
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "success": True,
            "report": report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})

@router.get('/status')
async def get_service_status():
    """
    Check if consultation service is available
    """
    procs = get_processors()
    
    return {
        "available": procs is not None and procs['audio'] is not None,
        "models_loaded": {
            "audio_processor": procs['audio'] is not None if procs else False,
            "summarizer": procs['summarizer'] is not None if procs else False,
            "prescription_extractor": procs['prescription'] is not None if procs else False
        } if procs else {}
    }
