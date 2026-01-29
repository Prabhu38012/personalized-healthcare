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

# ===== PERFORMANCE MODE =====
# Choose speed vs accuracy tradeoff:
# "FAST" - 10-20x faster, good quality (base + distilbart) - RECOMMENDED for most users
# "BALANCED" - 5x faster, better quality (small + distilbart)
# "QUALITY" - Slower but best results (large-v2 + bart-large) - Use if time is not a concern
PERFORMANCE_MODE = os.getenv("CONSULTATION_MODE", "FAST")  # Change to BALANCED or QUALITY if needed

print(f"⚡ Performance Mode: {PERFORMANCE_MODE}")

# ===== DEVICE DETECTION =====
# Automatically detect if CUDA is available
# Force CPU to avoid out of memory errors with large models
DEVICE_AVAILABLE = "cpu"  # Force CPU for stability with large models
print(f"🔧 Device configured: {DEVICE_AVAILABLE.upper()}")
if DEVICE_AVAILABLE == "cpu":
    print("ℹ️  Using CPU for Medical Consultation (more stable for large models)")

# ===== MODEL CONFIGURATIONS =====

# Model selection based on performance mode
MODEL_CONFIGS = {
    "FAST": {
        "whisper": "base",  # 10-20x faster than large-v2
        "summarizer": "sshleifer/distilbart-cnn-12-6",  # 2x faster than bart-large
        "description": "Fastest processing (~30 sec for 5min audio)"
    },
    "BALANCED": {
        "whisper": "small",  # 5x faster, better accuracy
        "summarizer": "sshleifer/distilbart-cnn-12-6",
        "description": "Good balance (~1-2 min for 5min audio)"
    },
    "QUALITY": {
        "whisper": "large-v2",  # Most accurate but slow
        "summarizer": "facebook/bart-large-cnn",
        "description": "Best quality but slow (~5-10 min for 5min audio)"
    }
}

# Get models for current mode
current_config = MODEL_CONFIGS.get(PERFORMANCE_MODE, MODEL_CONFIGS["FAST"])
print(f"   {current_config['description']}")

# Speech-to-Text Model Configuration
SPEECH_TO_TEXT_CONFIG = {
    "model_name": current_config["whisper"],
    "model_type": "faster-whisper",  # Options: whisper, faster-whisper
    "device": DEVICE_AVAILABLE,  # Automatically detected: cuda or cpu
    "compute_type": "float16" if DEVICE_AVAILABLE == "cuda" else "int8",  # float16 for GPU, int8 for CPU
    "language": "en",  # Options: en, es, fr, etc. or None for auto-detection
    "task": "transcribe",  # Options: transcribe, translate
}

# Text Summarization Model Configuration
SUMMARIZATION_CONFIG = {
    "model_name": current_config["summarizer"],
    "max_length": 512,
    "min_length": 100,
    "length_penalty": 2.0,
    "num_beams": 4,
    "early_stopping": True,
    "device": DEVICE_AVAILABLE,  # Automatically detected: cuda or cpu
}

print(f"   📢 Speech-to-Text: {SPEECH_TO_TEXT_CONFIG['model_name']}")
print(f"   📝 Summarization: {SUMMARIZATION_CONFIG['model_name'].split('/')[-1]}")
print(f"   💡 To change: Edit PERFORMANCE_MODE in config.py (FAST/BALANCED/QUALITY)")
print()

# Alternative models (for reference):
# Speech-to-Text: tiny (fastest) < base < small < medium < large-v2 (slowest/best)
# Summarization: distilbart (fast) < bart-large (slow/best)
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
