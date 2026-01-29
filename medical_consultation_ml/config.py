"""
Configuration file for Medical Consultation ML Project
Contains all model paths, parameters, and settings
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
AUDIO_DIR = DATA_DIR / "audio_files"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
SUMMARIES_DIR = DATA_DIR / "summaries"
PRESCRIPTIONS_DIR = DATA_DIR / "prescriptions"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, SUMMARIES_DIR, PRESCRIPTIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===== DEVICE DETECTION =====
# Automatically detect if CUDA is available
DEVICE_AVAILABLE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device detected: {DEVICE_AVAILABLE.upper()}")
if DEVICE_AVAILABLE == "cpu":
    print("ℹ️  Running on CPU - Processing will be slower but functional")

# ===== MODEL CONFIGURATIONS =====

# Speech-to-Text Model Configuration
SPEECH_TO_TEXT_CONFIG = {
    "model_name": "base",  # Options: tiny, base, small, medium, large, large-v2, large-v3
    "model_type": "faster-whisper",  # Options: whisper, faster-whisper
    "device": DEVICE_AVAILABLE,  # Automatically detected: cuda or cpu
    "compute_type": "float16" if DEVICE_AVAILABLE == "cuda" else "int8",  # float16 for GPU, int8 for CPU
    "language": "en",  # Options: en, es, fr, etc. or None for auto-detection
    "task": "transcribe",  # Options: transcribe, translate
}

# Text Summarization Model Configuration
SUMMARIZATION_CONFIG = {
    "model_name": "facebook/bart-large-cnn",  # Options: facebook/bart-large-cnn, google/pegasus-xsum, t5-base
    "max_length": 512,
    "min_length": 100,
    "length_penalty": 2.0,
    "num_beams": 4,
    "early_stopping": True,
    "device": DEVICE_AVAILABLE,  # Automatically detected: cuda or cpu
}

# Alternative summarization models (commented out)
# "model_name": "t5-base"  # Good for general summarization
# "model_name": "google/pegasus-xsum"  # Good for extreme summarization
# "model_name": "facebook/bart-large-cnn"  # Good for news-style summaries

# Named Entity Recognition for Medical Terms
NER_CONFIG = {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",  # Medical domain BERT
    "device": DEVICE_AVAILABLE,  # Automatically detected: cuda or cpu
}

# Prescription Extraction Configuration
PRESCRIPTION_CONFIG = {
    "medicine_patterns": [
        r'\b([A-Z][a-z]+(?:ine|ol|pam|xib|cin|ate|ide|one|pril|sartan|ide|mycin|cillin))\b',
        r'\b([A-Z][a-z]+\s+\d+(?:mg|g|ml|mcg))\b',
    ],
    "dosage_patterns": [
        r'\b(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|IU|units?))\b',
    ],
    "frequency_patterns": [
        r'\b(once|twice|thrice|\d+\s*times?)\s+(?:daily|per\s+day|a\s+day)\b',
        r'\b(every\s+\d+\s+hours?)\b',
        r'\b(morning|afternoon|evening|night|bedtime)\b',
    ],
    "duration_patterns": [
        r'\b(for\s+\d+\s+(?:days?|weeks?|months?))\b',
        r'\b(\d+\s+(?:days?|weeks?|months?)\s+course)\b',
    ],
}

# ===== PREPROCESSING CONFIGURATIONS =====

# Audio Preprocessing
AUDIO_PREPROCESSING = {
    "sample_rate": 16000,
    "max_duration_seconds": 3600,  # 1 hour max
    "supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
    "normalize": True,
}

# Text Preprocessing
TEXT_PREPROCESSING = {
    "remove_filler_words": True,
    "filler_words": ["um", "uh", "hmm", "like", "you know", "sort of", "kind of"],
    "lowercase": False,  # Keep original case for medical terms
    "remove_extra_whitespace": True,
    "min_transcript_length": 50,  # Minimum characters
}

# ===== OUTPUT CONFIGURATIONS =====

# Output Format
OUTPUT_CONFIG = {
    "save_transcript": True,
    "save_summary": True,
    "save_prescription": True,
    "format": "json",  # Options: json, csv, both
    "pretty_print": True,
    "include_timestamp": True,
    "include_metadata": True,
}

# ===== STREAMLIT UI CONFIGURATIONS =====

STREAMLIT_CONFIG = {
    "page_title": "Medical Consultation Transcription & Summarization",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 200,  # MB
}

# ===== LOGGING CONFIGURATIONS =====

LOGGING_CONFIG = {
    "level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "log_file": BASE_DIR / "logs" / "app.log",
    "rotation": "10 MB",
    "retention": "30 days",
}

# Create logs directory
(BASE_DIR / "logs").mkdir(exist_ok=True)

# ===== PERFORMANCE CONFIGURATIONS =====

PERFORMANCE_CONFIG = {
    "batch_size": 8,
    "num_workers": 4,
    "use_gpu": True,
    "mixed_precision": True,  # For faster inference
    "cache_models": True,  # Cache loaded models in memory
}

# ===== MEDICAL DOMAIN VOCABULARY =====

MEDICAL_VOCABULARY = {
    "symptoms": [
        "fever", "cough", "headache", "nausea", "vomiting", "diarrhea", 
        "fatigue", "weakness", "dizziness", "pain", "shortness of breath",
        "chest pain", "abdominal pain", "back pain", "joint pain"
    ],
    "diagnoses": [
        "hypertension", "diabetes", "asthma", "pneumonia", "bronchitis",
        "gastritis", "arthritis", "infection", "allergy", "migraine"
    ],
    "procedures": [
        "blood test", "x-ray", "CT scan", "MRI", "ultrasound", 
        "ECG", "EKG", "biopsy", "examination"
    ],
}

# ===== API KEYS (if needed for advanced features) =====
# Load from environment variables for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# ===== ERROR HANDLING =====

ERROR_MESSAGES = {
    "audio_load_failed": "Failed to load audio file. Please check the file format and integrity.",
    "transcription_failed": "Speech-to-text conversion failed. Please try again.",
    "summarization_failed": "Failed to generate summary. The transcript may be too short.",
    "extraction_failed": "Could not extract prescription details from the transcript.",
    "model_load_failed": "Failed to load AI model. Please check your installation.",
    "invalid_file": "Invalid file format. Please upload a valid audio file.",
}

# ===== SUCCESS MESSAGES =====

SUCCESS_MESSAGES = {
    "processing_complete": "✅ Processing completed successfully!",
    "transcript_saved": "📝 Transcript saved successfully.",
    "summary_generated": "📋 Medical summary generated.",
    "prescription_extracted": "💊 Prescription details extracted.",
}

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Data Directory: {DATA_DIR}")
