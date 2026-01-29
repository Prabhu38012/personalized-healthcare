"""
Prescription Extraction Module
Extracts medication details from medical consultation transcripts
Extracts: medicine name, dosage, frequency, duration
"""

import re
from typing import Dict, List, Optional, Tuple
from loguru import logger


class PrescriptionExtractor:
    """
    Extracts structured prescription information from medical transcripts
    """
    
    def __init__(self, config: Dict):
        """
        Initialize PrescriptionExtractor with configuration
        
        Args:
            config: Configuration dictionary with extraction patterns
        """
        self.config = config
        self.medicine_patterns = config.get("medicine_patterns", [])
        self.dosage_patterns = config.get("dosage_patterns", [])
        self.frequency_patterns = config.get("frequency_patterns", [])
        self.duration_patterns = config.get("duration_patterns", [])
        
        logger.info("PrescriptionExtractor initialized")
    
    def extract_medicines(self, text: str) -> List[str]:
        """
        Extract medicine names from text
        
        Args:
            text: Input transcript text
            
        Returns:
            List of extracted medicine names
        """
        medicines = []
        
        # Use configured patterns
        for pattern in self.medicine_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medicines.extend(matches)
        
        # Common medicine name patterns
        # Pattern 1: Capitalized words ending with common suffixes
        common_suffixes = r'\b([A-Z][a-z]{3,}(?:cillin|mycin|cycline|oxacin|azole|prazole|dipine|olol|pril|sartan|statin|formin))\b'
        medicines.extend(re.findall(common_suffixes, text))
        
        # Pattern 2: Medicine with dosage (e.g., "Amoxicillin 500mg")
        medicine_with_dose = r'\b([A-Z][a-z]{3,})\s+\d+\s*(?:mg|g|ml|mcg)\b'
        medicines.extend([m[0] for m in re.findall(medicine_with_dose, text)])
        
        # Filter out common false positives (formatting words, names, etc.)
        false_positives = {
            'underline', 'bold', 'italic', 'heading', 'paragraph', 'line', 'stop',
            'next', 'first', 'second', 'bullet', 'comma', 'full', 'new', 'inverted',
            'close', 'open', 'janet', 'james', 'jones', 'smith', 'john', 'medical',
            'report', 'client', 'accident', 'solicitor', 'street', 'drive', 'road',
            'private', 'state', 'liverpool', 'telephone', 'address', 'hospital',
            'reference', 'letter', 'consultant', 'surgeon', 'patient', 'doctor',
            'march', 'april', 'january', 'february', 'house', 'birth', 'thomas',
            'jason', 'spring', 'wiston', 'examination', 'opinion', 'expert', 'grateful'
        }
        
        # Remove duplicates and false positives while preserving order
        unique_medicines = []
        seen = set()
        for med in medicines:
            med_clean = med.strip().title()
            if (med_clean and 
                med_clean.lower() not in seen and 
                med_clean.lower() not in false_positives and
                len(med_clean) >= 4 and  # At least 4 characters
                not med_clean.lower().startswith(('mr', 'mrs', 'ms', 'dr'))):  # Not titles
                seen.add(med_clean.lower())
                unique_medicines.append(med_clean)
        
        logger.info(f"✓ Extracted {len(unique_medicines)} medicine(s)")
        return unique_medicines
    
    def extract_dosages(self, text: str) -> List[str]:
        """
        Extract dosage information from text
        
        Args:
            text: Input transcript text
            
        Returns:
            List of extracted dosages
        """
        dosages = []
        
        # Use configured patterns
        for pattern in self.dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dosages.extend(matches)
        
        # Additional common dosage patterns
        # Pattern: number + unit (e.g., "500mg", "2.5ml")
        dosage_pattern = r'\b(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|µg|IU|units?))\b'
        additional_dosages = re.findall(dosage_pattern, text, re.IGNORECASE)
        dosages.extend(additional_dosages)
        
        # Standardize units
        standardized = []
        for dosage in dosages:
            dosage = dosage.strip()
            # Standardize spacing
            dosage = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', dosage)
            dosage = re.sub(r'\s+', ' ', dosage)
            standardized.append(dosage)
        
        # Remove duplicates
        unique_dosages = list(dict.fromkeys(standardized))
        
        logger.info(f"✓ Extracted {len(unique_dosages)} dosage(s)")
        return unique_dosages
    
    def extract_frequencies(self, text: str) -> List[str]:
        """
        Extract frequency information from text
        
        Args:
            text: Input transcript text
            
        Returns:
            List of extracted frequencies
        """
        frequencies = []
        
        # Use configured patterns
        for pattern in self.frequency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            frequencies.extend(matches)
        
        # Additional frequency patterns
        patterns = [
            r'\b(once|twice|thrice|three times)\s+(?:a\s+)?day\b',
            r'\b(\d+)\s+times?\s+(?:a\s+|per\s+)?day\b',
            r'\b(every\s+\d+\s+hours?)\b',
            r'\b(every\s+(?:morning|evening|night))\b',
            r'\b(before|after)\s+(?:meals?|food|breakfast|lunch|dinner)\b',
            r'\b(with|without)\s+food\b',
            r'\b(at\s+bedtime|before\s+sleep)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            frequencies.extend(matches)
        
        # Remove duplicates
        unique_frequencies = list(dict.fromkeys([f.strip() for f in frequencies if f]))
        
        logger.info(f"✓ Extracted {len(unique_frequencies)} frequency instruction(s)")
        return unique_frequencies
    
    def extract_durations(self, text: str) -> List[str]:
        """
        Extract duration information from text
        
        Args:
            text: Input transcript text
            
        Returns:
            List of extracted durations
        """
        durations = []
        
        # Use configured patterns
        for pattern in self.duration_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            durations.extend(matches)
        
        # Additional duration patterns
        patterns = [
            r'\b(for\s+\d+\s+(?:day|days|week|weeks|month|months))\b',
            r'\b(\d+\s+(?:day|days|week|weeks|month|months)\s+course)\b',
            r'\b(until\s+(?:finished|symptoms\s+improve|better))\b',
            r'\b(continue\s+for\s+\d+\s+\w+)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            durations.extend(matches)
        
        # Remove duplicates
        unique_durations = list(dict.fromkeys([d.strip() for d in durations if d]))
        
        logger.info(f"✓ Extracted {len(unique_durations)} duration instruction(s)")
        return unique_durations
    
    def extract_prescription_context(self, text: str, medicine: str) -> str:
        """
        Extract the context around a specific medicine mention
        
        Args:
            text: Input transcript
            medicine: Medicine name to find context for
            
        Returns:
            Context text around the medicine
        """
        # Find sentences containing the medicine
        sentences = text.split('.')
        context_sentences = []
        
        for i, sentence in enumerate(sentences):
            if medicine.lower() in sentence.lower():
                # Include previous and next sentence for context
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                context_sentences.extend(sentences[start:end])
        
        context = '. '.join(context_sentences).strip()
        return context + '.' if context else ""
    
    def extract_prescription_details(self, text: str, medicine: str) -> Dict[str, any]:
        """
        Extract detailed prescription information for a specific medicine
        
        Args:
            text: Input transcript
            medicine: Medicine name
            
        Returns:
            Dictionary with prescription details
        """
        # Get context around this medicine
        context = self.extract_prescription_context(text, medicine)
        
        if not context:
            context = text  # Use full text if no specific context found
        
        # Extract details from context
        details = {
            "medicine": medicine,
            "dosage": None,
            "frequency": None,
            "duration": None,
            "instructions": []
        }
        
        # Extract dosage (closest to medicine name)
        dosages = self.extract_dosages(context)
        if dosages:
            details["dosage"] = dosages[0]
        
        # Extract frequency
        frequencies = self.extract_frequencies(context)
        if frequencies:
            details["frequency"] = frequencies[0]
        
        # Extract duration
        durations = self.extract_durations(context)
        if durations:
            details["duration"] = durations[0]
        
        # Extract additional instructions (limit size)
        instruction_keywords = ['before', 'after', 'with', 'without', 'take', 'apply', 'use']
        for sentence in context.split('.'):
            if any(kw in sentence.lower() for kw in instruction_keywords):
                instruction = sentence.strip()
                if instruction and 10 < len(instruction) < 200:  # Between 10-200 chars
                    details["instructions"].append(instruction)
        
        # Limit total instructions to prevent data dump
        if len(details["instructions"]) > 3:
            details["instructions"] = details["instructions"][:3]
        
        return details
    
    def extract_all_prescriptions(self, text: str) -> List[Dict[str, any]]:
        """
        Extract all prescription information from transcript
        
        Args:
            text: Input transcript text
            
        Returns:
            List of prescription dictionaries
        """
        try:
            logger.info("Extracting prescriptions from transcript...")
            
            # Extract all medicines first
            medicines = self.extract_medicines(text)
            
            if not medicines:
                logger.warning("No medicines found in transcript")
                return []
            
            # Extract detailed information for each medicine
            prescriptions = []
            for medicine in medicines:
                details = self.extract_prescription_details(text, medicine)
                prescriptions.append(details)
            
            logger.info(f"✓ Extracted {len(prescriptions)} prescription(s)")
            return prescriptions
            
        except Exception as e:
            logger.error(f"Prescription extraction failed: {e}")
            return []
    
    def format_prescription(self, prescription: Dict[str, any]) -> str:
        """
        Format prescription details as human-readable text
        
        Args:
            prescription: Prescription dictionary
            
        Returns:
            Formatted prescription string
        """
        parts = [f"Medicine: {prescription['medicine']}"]
        
        if prescription.get('dosage'):
            parts.append(f"Dosage: {prescription['dosage']}")
        
        if prescription.get('frequency'):
            parts.append(f"Frequency: {prescription['frequency']}")
        
        if prescription.get('duration'):
            parts.append(f"Duration: {prescription['duration']}")
        
        if prescription.get('instructions'):
            parts.append(f"Instructions: {'; '.join(prescription['instructions'][:2])}")
        
        return " | ".join(parts)
    
    def validate_prescription(self, prescription: Dict[str, any]) -> Dict[str, any]:
        """
        Validate extracted prescription information
        
        Args:
            prescription: Prescription dictionary
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": True,
            "completeness": 0,
            "warnings": []
        }
        
        # Check required fields
        required_fields = ['medicine', 'dosage', 'frequency']
        present_fields = sum(1 for field in required_fields if prescription.get(field))
        validation["completeness"] = present_fields / len(required_fields)
        
        if not prescription.get('medicine'):
            validation["is_valid"] = False
            validation["warnings"].append("Missing medicine name")
        
        if not prescription.get('dosage'):
            validation["warnings"].append("Missing dosage information")
        
        if not prescription.get('frequency'):
            validation["warnings"].append("Missing frequency information")
        
        if not prescription.get('duration'):
            validation["warnings"].append("Missing duration information")
        
        return validation


# Test function
if __name__ == "__main__":
    print("Testing PrescriptionExtractor...")
    
    # Sample configuration
    config = {
        "medicine_patterns": [r'\b([A-Z][a-z]+cillin)\b'],
        "dosage_patterns": [r'\b(\d+\s*mg)\b'],
        "frequency_patterns": [r'\b(twice\s+daily)\b'],
        "duration_patterns": [r'\b(for\s+\d+\s+days)\b']
    }
    
    extractor = PrescriptionExtractor(config)
    
    # Test text
    sample = """
    I'm prescribing Amoxicillin 500mg twice daily for 7 days.
    Also take Ibuprofen 400mg three times a day after meals for pain.
    Continue Metformin 850mg once daily in the morning.
    """
    
    print(f"\nSample text: {sample}\n")
    
    # Extract prescriptions
    prescriptions = extractor.extract_all_prescriptions(sample)
    
    print(f"Found {len(prescriptions)} prescription(s):\n")
    for i, rx in enumerate(prescriptions, 1):
        print(f"{i}. {extractor.format_prescription(rx)}")
        validation = extractor.validate_prescription(rx)
        print(f"   Completeness: {validation['completeness']*100:.0f}%")
        if validation['warnings']:
            print(f"   Warnings: {', '.join(validation['warnings'])}")
        print()
    
    print("✅ PrescriptionExtractor test completed!")
