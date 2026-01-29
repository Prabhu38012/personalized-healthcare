"""
Medical Summarization Module
Uses transformer models (BART, T5, Pegasus) to generate medical summaries
"""

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
            max_len = min(max_len, input_length)
            min_len = min(min_len, max_len // 2)
            
            logger.info(f"Generating summary (input: {input_length} tokens)...")
            
            # Generate summary using pipeline
            summary_result = self.summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                length_penalty=self.config.get("length_penalty", 2.0),
                num_beams=self.config.get("num_beams", 4),
                early_stopping=self.config.get("early_stopping", True),
                do_sample=False
            )
            
            summary = summary_result[0]['summary_text'].strip()
            
            logger.info(f"✓ Summary generated: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise
    
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
            main_summary = self.generate_summary(text)
            
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
            raise
    
    def _extract_chief_complaint(self, text: str) -> str:
        """
        Extract chief complaint from transcript
        
        Args:
            text: Transcript text
            
        Returns:
            Chief complaint summary
        """
        # Look for the beginning of the conversation (first 200 words)
        words = text.split()[:200]
        complaint_text = ' '.join(words)
        
        # Generate short summary of chief complaint
        try:
            summary = self.generate_summary(
                complaint_text,
                max_length=50,
                min_length=10
            )
            return summary
        except:
            return "Unable to extract chief complaint"
    
    def _extract_assessment(self, text: str) -> str:
        """
        Extract assessment/diagnosis from transcript
        
        Args:
            text: Transcript text
            
        Returns:
            Assessment summary
        """
        # Look for diagnosis-related keywords
        keywords = ['diagnosis', 'diagnosed', 'appears', 'likely', 'assessment']
        sentences = text.split('.')
        
        relevant_sentences = []
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keywords):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            assessment_text = '. '.join(relevant_sentences[:3])
            try:
                return self.generate_summary(
                    assessment_text,
                    max_length=100,
                    min_length=20
                )
            except:
                return '. '.join(relevant_sentences[:2])
        
        return "No specific assessment mentioned"
    
    def _extract_plan(self, text: str) -> str:
        """
        Extract treatment plan from transcript
        
        Args:
            text: Transcript text
            
        Returns:
            Treatment plan summary
        """
        # Look for treatment-related keywords
        keywords = ['prescribe', 'medication', 'treatment', 'recommend', 'advise', 'take']
        sentences = text.split('.')
        
        relevant_sentences = []
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keywords):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            plan_text = '. '.join(relevant_sentences[:3])
            try:
                return self.generate_summary(
                    plan_text,
                    max_length=150,
                    min_length=30
                )
            except:
                return '. '.join(relevant_sentences[:2])
        
        return "No specific treatment plan mentioned"
    
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
