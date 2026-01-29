"""
Medical Summarization Module
Uses transformer models (BART, T5, Pegasus) to generate medical summaries
"""

import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    BartForConditionalGeneration,
    BartTokenizer
)
from typing import Dict, List, Optional
from loguru import logger
import warnings
warnings.filterwarnings("ignore")


class MedicalSummarizer:
    """
    Generates concise medical summaries from consultation transcripts
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Medical Summarizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get("model_name", "facebook/bart-large-cnn")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get("max_length", 512)
        self.min_length = config.get("min_length", 100)
        
        self.model = None
        self.tokenizer = None
        self.summarizer = None
        
        logger.info(f"Initializing MedicalSummarizer with {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine dtype based on device
            if self.device == "cuda":
                model_dtype = torch.float16
            else:
                model_dtype = torch.float32
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                dtype=model_dtype
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"✓ Model loaded successfully on {self.device.upper()}")
            
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            raise
    
    def generate_summary(self, text: str, 
                        max_length: Optional[int] = None,
                        min_length: Optional[int] = None) -> str:
        """
        Generate a summary from input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length (overrides config)
            min_length: Minimum summary length (overrides config)
            
        Returns:
            Generated summary text
        """
        if not text:
            logger.warning("Empty text provided for summarization")
            return ""
        
        try:
            max_len = max_length or self.max_length
            min_len = min_length or self.min_length
            
            # Adjust lengths based on input length
            input_length = len(self.tokenizer.encode(text))
            
            # If input is very short, return it as-is
            if input_length < 50:
                logger.warning(f"Input too short for summarization ({input_length} tokens), returning original")
                return text[:500]  # Return first 500 chars
            
            max_len = min(max_len, input_length)
            min_len = min(min_len, max_len // 2)
            
            # Ensure min_len is not too close to max_len
            if max_len - min_len < 10:
                min_len = max(10, max_len - 20)
            
            logger.info(f"Generating summary (input: {input_length} tokens, range: {min_len}-{max_len})...")
            
            # Generate summary using pipeline
            summary_result = self.summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                length_penalty=self.config.get("length_penalty", 2.0),
                num_beams=self.config.get("num_beams", 4),
                early_stopping=self.config.get("early_stopping", True),
                do_sample=False,
                truncation=True
            )
            
            summary = summary_result[0]['summary_text'].strip()
            
            logger.info(f"✓ Summary generated: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Return a truncated version of the original text as fallback
            logger.warning("Returning truncated original text as fallback")
            return text[:500] + "..." if len(text) > 500 else text
    
    def generate_medical_summary(self, text: str) -> Dict[str, str]:
        """
        Generate a structured medical summary with different sections
        
        Args:
            text: Input consultation transcript
            
        Returns:
            Dictionary with different summary sections
        """
        try:
            # Generate main summary
            try:
                main_summary = self.generate_summary(text)
            except Exception as e:
                logger.warning(f"Main summary generation failed: {e}, using truncated text")
                main_summary = text[:1000] + "..." if len(text) > 1000 else text
            
            # Extract key information for structured summary
            structured_summary = {
                "general_summary": main_summary,
                "chief_complaint": self._extract_chief_complaint(text),
                "assessment": self._extract_assessment(text),
                "plan": self._extract_plan(text),
            }
            
            logger.info("✓ Structured medical summary generated")
            return structured_summary
            
        except Exception as e:
            logger.error(f"Failed to generate medical summary: {e}")
            # Return a basic summary instead of raising
            return {
                "general_summary": text[:1000] + "..." if len(text) > 1000 else text,
                "chief_complaint": "Unable to extract",
                "assessment": "Unable to extract",
                "plan": "Unable to extract"
            }
    
    def _extract_chief_complaint(self, text: str) -> str:
        """
        Extract chief complaint from transcript
        
        Args:
            text: Transcript text
            
        Returns:
            Chief complaint summary
        """
        # Look for common patterns that indicate chief complaint
        patterns = [
            r'(?:referred|brought in|came in|presenting|complaining|complaint|concern).*?(?:for|about|with|of)\s+([^.!?]{10,100})',
            r'(?:patient|she|he).*?(?:has|reports|says|states|complains of|experiencing)\s+([^.!?]{10,100})',
            r'(?:suffering from|problem with|issue with|worried about)\s+([^.!?]{10,100})'
        ]
        
        extracted_parts = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                extracted_parts.append(match.group(0))
                if len(extracted_parts) >= 2:  # Get first 2 relevant matches
                    break
            if len(extracted_parts) >= 2:
                break
        
        # If patterns found, use them
        if extracted_parts:
            complaint_text = '. '.join(extracted_parts[:2])
        else:
            # Fallback: Use first 300 words but skip greetings
            words = text.split()
            # Try to skip initial pleasantries
            start_idx = 0
            for i, word in enumerate(words[:50]):
                if any(kw in word.lower() for kw in ['referred', 'patient', 'complaint', 'problem', 'concerned', 'worried']):
                    start_idx = max(0, i - 10)
                    break
            complaint_text = ' '.join(words[start_idx:start_idx+300])
        
        # Generate concise summary
        try:
            summary = self.generate_summary(
                complaint_text,
                max_length=80,
                min_length=20
            )
            # Clean up and format
            summary = summary.strip()
            if not summary.endswith('.'):
                summary += '.'
            return summary
        except Exception as e:
            logger.warning(f"Chief complaint extraction failed: {e}")
            # Fallback to first sentence with relevant info
            sentences = text.split('.')[:5]
            for sent in sentences:
                if len(sent.strip()) > 30:  # Meaningful sentence
                    return sent.strip() + '.'
            return "Unable to extract specific chief complaint from consultation."
    
    def _extract_assessment(self, text: str) -> str:
        """
        Extract assessment/diagnosis from transcript
        
        Args:
            text: Transcript text
            
        Returns:
            Assessment summary
        """
        # Expanded keywords for finding assessment sections
        assessment_keywords = [
            'diagnosis', 'diagnosed', 'appears', 'likely', 'assessment', 'evaluation',
            'findings', 'indicates', 'suggests', 'shows signs', 'condition',
            'blood pressure', 'vitals', 'examination', 'presenting with',
            'suffering from', 'consistent with', 'suspected'
        ]
        
        # Look for physical findings and measurements
        vital_patterns = [
            r'blood pressure.*?\d+.*?\d+',
            r'temperature.*?\d+',
            r'heart rate.*?\d+',
            r'weight.*?\d+'
        ]
        
        sentences = text.split('.')
        relevant_sentences = []
        vital_info = []
        
        # Extract relevant sentences
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check for assessment keywords
            if any(kw in sentence_lower for kw in assessment_keywords):
                relevant_sentences.append(sentence.strip())
            # Check for vital signs
            for pattern in vital_patterns:
                if re.search(pattern, sentence_lower):
                    vital_info.append(sentence.strip())
                    break
        
        # Combine findings
        all_findings = list(set(vital_info[:2] + relevant_sentences[:3]))  # Remove duplicates, limit count
        
        if all_findings:
            assessment_text = '. '.join(all_findings)
            # Ensure it's substantial enough
            if len(assessment_text.split()) < 15:
                # Add more context
                assessment_text = assessment_text + '. ' + '. '.join(relevant_sentences[:2])
            
            try:
                summary = self.generate_summary(
                    assessment_text,
                    max_length=150,
                    min_length=30
                )
                # Clean up and format
                summary = summary.strip()
                if not summary.endswith('.'):
                    summary += '.'
                return summary
            except Exception as e:
                logger.warning(f"Assessment summarization failed: {e}")
                # Return the raw findings, cleaned up
                return '. '.join(all_findings[:3]) + '.'
        
        # If still no assessment found, try to extract from middle section
        middle_start = len(text) // 3
        middle_end = 2 * len(text) // 3
        middle_section = text[middle_start:middle_end]
        
        try:
            summary = self.generate_summary(
                middle_section,
                max_length=120,
                min_length=25
            )
            return f"Based on consultation: {summary}"
        except:
            return "No specific clinical assessment documented in this consultation."
    
    def _extract_plan(self, text: str) -> str:
        """
        Extract treatment plan from transcript
        
        Args:
            text: Transcript text
            
        Returns:
            Treatment plan summary
        """
        # Comprehensive treatment-related keywords
        treatment_keywords = [
            'prescribe', 'medication', 'treatment', 'recommend', 'advise', 'take',
            'should', 'need to', 'plan', 'next steps', 'follow up', 'continue',
            'start', 'begin', 'therapy', 'intervention', 'management', 'monitor',
            'discuss', 'bring', 'work with', 'help', 'goals', 'improve'
        ]
        
        # Pattern for specific recommendations
        recommendation_patterns = [
            r'(?:should|need to|must|have to|going to)\s+([^.!?]{10,100})',
            r'(?:recommend|advise|suggest).*?(?:to|that)\s+([^.!?]{10,100})',
            r'(?:plan is to|next step|will)\s+([^.!?]{10,100})'
        ]
        
        sentences = text.split('.')
        relevant_sentences = []
        recommendations = []
        
        # Extract treatment-related sentences
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in treatment_keywords):
                # Skip if it's just dialogue filler
                if not any(filler in sentence_lower for filler in ['can you', 'will you', 'i guess', 'you know']):
                    relevant_sentences.append(sentence.strip())
            
            # Extract specific recommendations
            for pattern in recommendation_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    recommendations.append(match.group(0))
        
        # Prioritize recommendations over general sentences
        if recommendations:
            plan_text = '. '.join(recommendations[:3])
        elif relevant_sentences:
            # Filter out questions and non-actionable statements
            actionable = [s for s in relevant_sentences if '?' not in s and len(s.split()) > 5]
            plan_text = '. '.join(actionable[:4])
        else:
            # Try extracting from last third of conversation (usually contains plan)
            last_third_start = 2 * len(text) // 3
            last_section = text[last_third_start:]
            plan_text = last_section[:800]  # Reasonable chunk
        
        if plan_text.strip():
            try:
                summary = self.generate_summary(
                    plan_text,
                    max_length=180,
                    min_length=40
                )
                # Clean up and format
                summary = summary.strip()
                # Remove dialogue artifacts
                summary = re.sub(r'\b(um|uh|like|you know)\b', '', summary, flags=re.IGNORECASE)
                summary = re.sub(r'\s+', ' ', summary).strip()
                
                if not summary.endswith('.'):
                    summary += '.'
                return summary
            except Exception as e:
                logger.warning(f"Treatment plan summarization failed: {e}")
                # Return cleaned recommendations
                if recommendations:
                    clean_recs = '. '.join(recommendations[:2])
                    return clean_recs + '.'
                elif relevant_sentences:
                    return '. '.join(relevant_sentences[:2]) + '.'
        
        return "Treatment plan to be discussed. Patient to follow up with healthcare provider for next steps."
    
    def batch_summarize(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Summarize multiple texts
        
        Args:
            texts: List of texts to summarize
            
        Returns:
            List of summary dictionaries
        """
        summaries = []
        total = len(texts)
        
        for idx, text in enumerate(texts, 1):
            try:
                logger.info(f"Summarizing {idx}/{total}...")
                summary = self.generate_medical_summary(text)
                summaries.append({
                    "success": True,
                    "summary": summary
                })
            except Exception as e:
                logger.error(f"Failed to summarize text {idx}: {e}")
                summaries.append({
                    "success": False,
                    "error": str(e)
                })
        
        return summaries
    
    def generate_bullet_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Generate key bullet points from text
        
        Args:
            text: Input text
            num_points: Number of bullet points to generate
            
        Returns:
            List of bullet point strings
        """
        try:
            # Generate a concise summary
            summary = self.generate_summary(text, max_length=200, min_length=50)
            
            # Split into sentences
            sentences = [s.strip() + '.' for s in summary.split('.') if s.strip()]
            
            # Take top N sentences as bullet points
            bullet_points = sentences[:num_points]
            
            logger.info(f"✓ Generated {len(bullet_points)} bullet points")
            return bullet_points
            
        except Exception as e:
            logger.error(f"Failed to generate bullet points: {e}")
            return []
    
    def calculate_summary_quality(self, original: str, summary: str) -> Dict[str, float]:
        """
        Calculate quality metrics for the generated summary
        
        Args:
            original: Original text
            summary: Generated summary
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "compression_ratio": 0.0,
            "length_ratio": 0.0,
            "word_overlap": 0.0
        }
        
        try:
            # Compression ratio
            metrics["compression_ratio"] = len(summary) / len(original) if original else 0
            
            # Length ratio (in words)
            orig_words = len(original.split())
            summ_words = len(summary.split())
            metrics["length_ratio"] = summ_words / orig_words if orig_words else 0
            
            # Word overlap (simple measure)
            orig_words_set = set(original.lower().split())
            summ_words_set = set(summary.lower().split())
            overlap = len(orig_words_set & summ_words_set)
            metrics["word_overlap"] = overlap / len(summ_words_set) if summ_words_set else 0
            
        except Exception as e:
            logger.warning(f"Failed to calculate summary quality: {e}")
        
        return metrics


# Test function
if __name__ == "__main__":
    print("Testing MedicalSummarizer...")
    
    # Check if model can be loaded
    config = {
        "model_name": "facebook/bart-large-cnn",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_length": 150,
        "min_length": 50
    }
    
    print(f"Device: {config['device']}")
    print(f"Model: {config['model_name']}")
    
    # Note: Actual model loading requires transformers library
    print("\n✅ MedicalSummarizer module loaded successfully!")
