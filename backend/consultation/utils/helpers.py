"""
Helper utility functions for the Medical Consultation ML Project
Contains common utilities for file handling, validation, and formatting
"""

import json
import torch
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger


def format_timestamp(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    Format a datetime object as a string
    
    Args:
        dt: Datetime object (default: current time)
        format_str: Format string for output
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)


def validate_audio_file(file_path: Path, supported_formats: List[str]) -> bool:
    """
    Validate if the audio file exists and has a supported format
    
    Args:
        file_path: Path to the audio file
        supported_formats: List of supported file extensions (e.g., ['.wav', '.mp3'])
        
    Returns:
        True if valid, False otherwise
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    if file_path.suffix.lower() not in supported_formats:
        logger.error(f"Unsupported format: {file_path.suffix}. Supported: {supported_formats}")
        return False
    
    return True


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device (CUDA/CPU)
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def clean_text(text: str, remove_filler_words: bool = True, 
               filler_words: Optional[List[str]] = None) -> str:
    """
    Clean and preprocess text
    
    Args:
        text: Input text to clean
        remove_filler_words: Whether to remove common filler words
        filler_words: List of filler words to remove
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove filler words if specified
    if remove_filler_words and filler_words:
        pattern = r'\b(' + '|'.join(map(re.escape, filler_words)) + r')\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()  # Clean up extra spaces again
    
    return text


def save_json(data: Dict[Any, Any], file_path: Path, pretty: bool = True) -> bool:
    """
    Save data to a JSON file
    
    Args:
        data: Dictionary to save
        file_path: Output file path
        pretty: Whether to use pretty printing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        logger.info(f"Saved JSON to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        return False


def load_json(file_path: Path) -> Optional[Dict[Any, Any]]:
    """
    Load data from a JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary if successful, None otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return None


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_patient_info(text: str) -> Dict[str, Optional[str]]:
    """
    Extract basic patient information from transcript
    
    Args:
        text: Transcript text
        
    Returns:
        Dictionary with patient information
    """
    info = {
        "age": None,
        "gender": None,
        "chief_complaint": None,
    }
    
    # Extract age
    age_match = re.search(r'\b(\d{1,3})\s*(?:years?|year-old|yo|y/o)\b', text, re.IGNORECASE)
    if age_match:
        info["age"] = age_match.group(1)
    
    # Extract gender
    gender_match = re.search(r'\b(male|female|man|woman|boy|girl)\b', text, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).lower()
        if gender in ['male', 'man', 'boy']:
            info["gender"] = "Male"
        elif gender in ['female', 'woman', 'girl']:
            info["gender"] = "Female"
    
    # Extract chief complaint (first sentence mentioning symptoms)
    sentences = text.split('.')
    for sentence in sentences[:5]:  # Check first 5 sentences
        if any(word in sentence.lower() for word in ['complain', 'problem', 'issue', 'pain', 'fever']):
            info["chief_complaint"] = sentence.strip()
            break
    
    return info


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    max_length = 200
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        name = name[:max_length - len(ext) - 1]
        filename = f"{name}.{ext}" if ext else name
    
    return filename


def calculate_confidence_score(text: str, min_length: int = 100) -> float:
    """
    Calculate a simple confidence score for the transcription
    Based on text length and quality indicators
    
    Args:
        text: Transcribed text
        min_length: Minimum expected length
        
    Returns:
        Confidence score (0-1)
    """
    if not text:
        return 0.0
    
    score = 0.0
    
    # Length score (up to 0.4)
    length_ratio = min(len(text) / min_length, 1.0)
    score += length_ratio * 0.4
    
    # Word count score (up to 0.3)
    word_count = len(text.split())
    word_score = min(word_count / 50, 1.0)  # 50 words = full score
    score += word_score * 0.3
    
    # Sentence structure score (up to 0.3)
    sentence_count = len([s for s in text.split('.') if s.strip()])
    sentence_score = min(sentence_count / 5, 1.0)  # 5 sentences = full score
    score += sentence_score * 0.3
    
    return round(min(score, 1.0), 2)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test timestamp
    timestamp = format_timestamp()
    print(f"Timestamp: {timestamp}")
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test text cleaning
    sample_text = "Um, the patient, like, has a fever and, you know, cough."
    cleaned = clean_text(sample_text, remove_filler_words=True, 
                        filler_words=["um", "like", "you know"])
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    
    print("\n✅ Utilities test completed!")
