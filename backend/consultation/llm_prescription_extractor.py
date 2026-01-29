"""
LLM-Based Prescription Extractor
Uses GROQ/Gemini to accurately extract prescriptions from consultations
Much more accurate than regex patterns!
"""

import os
import json
from typing import Dict, List, Any
from loguru import logger

# Import LLM analyzer
try:
    from backend.utils.llm_analyzer import llm_analyzer
except ImportError:
    from utils.llm_analyzer import llm_analyzer


class LLMPrescriptionExtractor:
    """Extract prescriptions using LLM (GROQ/Gemini) - much more accurate!"""
    
    def __init__(self):
        """Initialize the LLM prescription extractor"""
        self.llm = llm_analyzer
        logger.info("LLM Prescription Extractor initialized")
    
    def extract_prescriptions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract prescriptions from consultation text using LLM
        
        Args:
            text: Consultation transcript
            
        Returns:
            List of prescription dictionaries
        """
        if not self.llm.is_available():
            logger.warning("LLM not available, falling back to empty list")
            return []
        
        # Check if there are any prescription keywords
        prescription_keywords = ['prescribe', 'prescribing', 'medication', 'take', 'tablet', 'capsule', 'mg', 'milligram']
        if not any(keyword in text.lower() for keyword in prescription_keywords):
            logger.info("No prescription keywords found in text")
            return []
        
        try:
            # Create focused prompt for prescription extraction
            prompt = f"""Extract ONLY actual prescriptions from this medical consultation transcript.

TRANSCRIPT:
{text[:3000]}

Return ONLY valid JSON array. Each prescription must have these exact fields:
{{
  "medicine": "Exact medication name (e.g., Amoxicillin, Lisinopril)",
  "dosage": "Dosage with unit (e.g., 500mg, 10mg) or None",
  "frequency": "How often (e.g., three times daily, once daily, every 6 hours) or None",
  "duration": "How long (e.g., 7 days, 2 weeks) or None",
  "instructions": ["Additional instructions if any"]
}}

CRITICAL RULES:
1. ONLY include actual medications prescribed by the doctor
2. Do NOT include: patient names, places, random words, dates
3. If medicine ends with -cillin, -mycin, -pril, -olol, -statin - it's likely valid
4. If no real prescriptions found, return empty array []
5. Return ONLY the JSON array, no other text
6. Invalid entries: "Anyone", "Appreciate", "Private", "State", "Liverpool"

Example valid output:
[
  {{"medicine": "Amoxicillin", "dosage": "500mg", "frequency": "three times daily", "duration": "7 days", "instructions": ["Take with food", "Complete full course"]}},
  {{"medicine": "Lisinopril", "dosage": "10mg", "frequency": "once daily", "duration": None, "instructions": ["Take in the morning"]}}
]

Extract prescriptions now:"""

            # Call LLM
            if self.llm.provider == "groq":
                response = self.llm.groq_client.chat.completions.create(
                    model=self.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                response_text = response.choices[0].message.content.strip()
            else:
                response = self.llm.model.generate_content(prompt)
                response_text = response.text.strip()
            
            # Clean response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            # Parse JSON
            prescriptions = json.loads(response_text)
            
            # Validate prescriptions
            valid_prescriptions = []
            invalid_words = {'anyone', 'appreciate', 'private', 'state', 'liverpool', 
                           'telephone', 'address', 'hospital', 'patient', 'doctor',
                           'thank', 'welcome', 'take', 'care', 'nurse', 'okay'}
            
            for rx in prescriptions:
                medicine = rx.get('medicine', '').lower()
                # Must have a medicine name
                if not medicine or len(medicine) < 4:
                    continue
                # Filter out obvious non-medicines
                if medicine in invalid_words:
                    continue
                # Must have at least dosage OR frequency to be valid
                if not rx.get('dosage') and not rx.get('frequency'):
                    # Check if medicine name looks valid (has common suffixes)
                    if not any(suffix in medicine for suffix in ['cillin', 'mycin', 'pril', 'olol', 'zole', 'statin', 'formin', 'pam']):
                        continue
                
                valid_prescriptions.append(rx)
            
            logger.info(f"✓ Extracted {len(valid_prescriptions)} valid prescription(s) using LLM")
            return valid_prescriptions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response_text[:200]}")
            return []
        except Exception as e:
            logger.error(f"Error extracting prescriptions with LLM: {e}")
            return []


# Test function
if __name__ == "__main__":
    print("Testing LLM Prescription Extractor...")
    
    # Test with sample consultation
    test_text = """
    Doctor: For the infection, I'm prescribing Amoxicillin 500 milligrams. 
    Take one capsule three times daily for seven days.
    
    For the cough, I'm prescribing Dextromethorphan 30 milligrams. 
    Take one tablet every six hours as needed.
    
    For your elevated blood pressure, I want you to start on Lisinopril 10 milligrams 
    once daily in the morning.
    
    Thank you, doctor. I really appreciate your help.
    """
    
    extractor = LLMPrescriptionExtractor()
    
    if extractor.llm.is_available():
        prescriptions = extractor.extract_prescriptions(test_text)
        
        print(f"\n✓ Extracted {len(prescriptions)} prescriptions:")
        for i, rx in enumerate(prescriptions, 1):
            print(f"\n{i}. {rx['medicine']}")
            print(f"   Dosage: {rx.get('dosage', 'Not specified')}")
            print(f"   Frequency: {rx.get('frequency', 'Not specified')}")
            print(f"   Duration: {rx.get('duration', 'Not specified')}")
    else:
        print("❌ LLM not available - configure GROQ_API_KEY or GEMINI_API_KEY")
