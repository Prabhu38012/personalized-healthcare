"""
Text Processing Module
Handles text cleaning, preprocessing, and preparation for NLP tasks
"""

import re
import string
from typing import List, Dict, Optional
from loguru import logger


class TextProcessor:
    """
    Processes and cleans transcribed text for medical summarization
    """
    
    def __init__(self, config: Dict):
        """
        Initialize TextProcessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.filler_words = config.get("filler_words", [])
        self.should_remove_fillers = config.get("remove_filler_words", True)
        self.min_length = config.get("min_transcript_length", 50)
        
        logger.info("TextProcessor initialized")
    
    def remove_filler_words(self, text: str) -> str:
        """
        Remove common filler words from text
        
        Args:
            text: Input text
            
        Returns:
            Text with filler words removed
        """
        if not self.filler_words:
            return text
        
        # Create pattern for filler words
        pattern = r'\b(' + '|'.join(map(re.escape, self.filler_words)) + r')\b'
        cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Add space after punctuation if missing
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
        
        return text.strip()
    
    def fix_capitalization(self, text: str) -> str:
        """
        Fix capitalization in text (start of sentences)
        
        Args:
            text: Input text
            
        Returns:
            Text with proper capitalization
        """
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Capitalize after sentence-ending punctuation
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text
    
    def remove_repetitions(self, text: str) -> str:
        """
        Remove repeated words or phrases
        
        Args:
            text: Input text
            
        Returns:
            Text with repetitions removed
        """
        # Remove repeated words (e.g., "the the patient")
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        
        return text
    
    def standardize_medical_terms(self, text: str) -> str:
        """
        Standardize common medical abbreviations and terms
        
        Args:
            text: Input text
            
        Returns:
            Text with standardized medical terms
        """
        # Common medical abbreviations
        replacements = {
            r'\btemp\b': 'temperature',
            r'\bbp\b': 'blood pressure',
            r'\bhr\b': 'heart rate',
            r'\brr\b': 'respiratory rate',
            r'\bwt\b': 'weight',
            r'\bht\b': 'height',
            r'\bbmi\b': 'BMI',
            r'\bdx\b': 'diagnosis',
            r'\btx\b': 'treatment',
            r'\brx\b': 'prescription',
            r'\bhx\b': 'history',
            r'\bpt\b': 'patient',
            r'\bdr\b': 'doctor',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def segment_conversation(self, text: str) -> Dict[str, List[str]]:
        """
        Attempt to segment conversation into doctor and patient parts
        
        Args:
            text: Input transcript
            
        Returns:
            Dictionary with 'doctor' and 'patient' segments
        """
        segments = {
            "doctor": [],
            "patient": [],
            "unknown": []
        }
        
        # Split by common indicators
        lines = text.split('.')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for speaker indicators
            if any(word in line.lower() for word in ['doctor:', 'dr:', 'physician:']):
                segments["doctor"].append(line)
            elif any(word in line.lower() for word in ['patient:', 'pt:', 'i have', 'i feel', 'my']):
                segments["patient"].append(line)
            else:
                segments["unknown"].append(line)
        
        return segments
    
    def extract_medical_sections(self, text: str) -> Dict[str, str]:
        """
        Extract different sections from medical conversation
        
        Args:
            text: Input transcript
            
        Returns:
            Dictionary with different medical sections
        """
        sections = {
            "chief_complaint": "",
            "symptoms": "",
            "diagnosis": "",
            "treatment": "",
            "follow_up": ""
        }
        
        # Look for section keywords
        text_lower = text.lower()
        
        # Chief complaint (usually at the beginning)
        complaint_patterns = [
            r'(?:chief complaint|presenting complaint|main problem|complaining of)[:\s]+([^.]+)',
            r'(?:patient (?:complains|reports|states|presents with))[:\s]+([^.]+)'
        ]
        for pattern in complaint_patterns:
            match = re.search(pattern, text_lower)
            if match:
                sections["chief_complaint"] = match.group(1).strip()
                break
        
        # Symptoms
        symptom_keywords = ['symptom', 'pain', 'fever', 'cough', 'headache', 'nausea']
        symptom_sentences = [s for s in text.split('.') 
                           if any(kw in s.lower() for kw in symptom_keywords)]
        sections["symptoms"] = '. '.join(symptom_sentences[:3]) if symptom_sentences else ""
        
        # Diagnosis
        diagnosis_patterns = [
            r'(?:diagnosis|diagnosed with|appears to be|likely)[:\s]+([^.]+)',
            r'(?:suffering from|has)[:\s]+([^.]+)'
        ]
        for pattern in diagnosis_patterns:
            match = re.search(pattern, text_lower)
            if match:
                sections["diagnosis"] = match.group(1).strip()
                break
        
        # Treatment
        treatment_keywords = ['prescribe', 'medication', 'treatment', 'take', 'medicine']
        treatment_sentences = [s for s in text.split('.') 
                             if any(kw in s.lower() for kw in treatment_keywords)]
        sections["treatment"] = '. '.join(treatment_sentences) if treatment_sentences else ""
        
        # Follow-up
        followup_patterns = [
            r'(?:follow.?up|come back|see me again|return)[:\s]+([^.]+)',
            r'(?:in \d+ (?:days?|weeks?|months?))'
        ]
        for pattern in followup_patterns:
            match = re.search(pattern, text_lower)
            if match:
                sections["follow_up"] = match.group(0).strip()
                break
        
        return sections
    
    def clean_transcript(self, text: str) -> str:
        """
        Complete cleaning pipeline for transcript
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned transcript
        """
        if not text or len(text) < self.min_length:
            logger.warning(f"Transcript too short: {len(text)} characters")
            return text
        
        logger.info("Cleaning transcript...")
        
        # Apply cleaning steps
        cleaned = text
        
        # 1. Remove filler words
        if self.should_remove_fillers:
            cleaned = self.remove_filler_words(cleaned)
        
        # 2. Remove repetitions
        cleaned = self.remove_repetitions(cleaned)
        
        # 3. Normalize whitespace
        cleaned = self.normalize_whitespace(cleaned)
        
        # 4. Fix capitalization
        cleaned = self.fix_capitalization(cleaned)
        
        # 5. Standardize medical terms
        cleaned = self.standardize_medical_terms(cleaned)
        
        logger.info(f"✓ Cleaned transcript: {len(text)} -> {len(cleaned)} characters")
        
        return cleaned
    
    def prepare_for_summarization(self, text: str, 
                                  max_length: int = 1024) -> str:
        """
        Prepare text for summarization model
        
        Args:
            text: Cleaned text
            max_length: Maximum length for model input
            
        Returns:
            Prepared text for summarization
        """
        # Clean the text first
        prepared = self.clean_transcript(text)
        
        # Truncate if too long
        if len(prepared) > max_length:
            # Try to truncate at sentence boundary
            sentences = prepared.split('.')
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) < max_length:
                    truncated += sentence + '.'
                else:
                    break
            prepared = truncated.strip()
            logger.warning(f"Text truncated to {len(prepared)} characters for summarization")
        
        return prepared
    
    def validate_transcript(self, text: str) -> Dict[str, any]:
        """
        Validate transcript quality
        
        Args:
            text: Transcript text
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": True,
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len([s for s in text.split('.') if s.strip()]),
            "warnings": []
        }
        
        # Check minimum length
        if len(text) < self.min_length:
            validation["is_valid"] = False
            validation["warnings"].append(f"Transcript too short (min: {self.min_length})")
        
        # Check word count
        if validation["word_count"] < 10:
            validation["is_valid"] = False
            validation["warnings"].append("Too few words in transcript")
        
        # Check if mostly numbers or special characters
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars / len(text) < 0.5:
            validation["warnings"].append("Low alphabetic character ratio")
        
        return validation
    
    def post_process_summary(self, summary: str) -> str:
        """
        Post-process generated summary
        
        Args:
            summary: Raw summary from model
            
        Returns:
            Polished summary
        """
        # Remove redundant phrases
        summary = re.sub(r'(?i)\b(in summary|to summarize|in conclusion)\b[,:]?\s*', '', summary)
        
        # Ensure proper capitalization
        summary = self.fix_capitalization(summary)
        
        # Normalize whitespace
        summary = self.normalize_whitespace(summary)
        
        # Ensure ends with proper punctuation
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        return summary.strip()


# Test function
if __name__ == "__main__":
    print("Testing TextProcessor...")
    
    # Sample configuration
    config = {
        "filler_words": ["um", "uh", "like", "you know"],
        "remove_filler_words": True,
        "min_transcript_length": 50
    }
    
    processor = TextProcessor(config)
    
    # Test text
    sample = "Um, the the patient, like, complains of, you know, fever and cough. temp is 101. bp is normal."
    
    print(f"\nOriginal: {sample}")
    cleaned = processor.clean_transcript(sample)
    print(f"Cleaned: {cleaned}")
    
    # Test validation
    validation = processor.validate_transcript(cleaned)
    print(f"\nValidation: {validation}")
    
    print("\n✅ TextProcessor test completed!")
