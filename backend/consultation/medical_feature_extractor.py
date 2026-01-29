"""
Medical Feature Extractor
Extracts numerical medical features from consultation transcripts for AI/ML predictions
Converts raw conversation into structured patient data for prediction models
"""

import re
import json
from typing import Dict, Any, List, Tuple
from loguru import logger


class MedicalFeatureExtractor:
    """Extract medical features from consultation text for AI predictions"""
    
    # Patterns for extracting medical values
    PATTERNS = {
        # Blood pressure: "BP 120/80", "blood pressure 130/90", etc.
        'blood_pressure': [
            r'(?:BP|blood pressure)(?:\s+(?:is|of|around|about))?\s+(\d{2,3})[/-](\d{2,3})',
            r'(\d{2,3})[/-](\d{2,3})\s*(?:mmHg|mm\s*Hg)',
        ],
        # Cholesterol: "cholesterol 240", "cholesterol is high at 250"
        'cholesterol': [
            r'(?:cholesterol|total cholesterol)(?:\s+(?:is|of|around|about))?\s+(\d{2,3})',
            r'cholesterol.*?(\d{2,3})\s*(?:mg/dL|mg/dl)',
        ],
        # Blood glucose/Fasting blood sugar: "glucose 120", "blood sugar 150", "fasting BS 110"
        'glucose': [
            r'(?:glucose|blood\s+sugar|fasting\s+(?:blood\s+)?sugar|fasting\s+BS)(?:\s+(?:is|of|around|about))?\s+(\d{2,3})',
            r'(?:glucose|blood\s+sugar).*?(\d{2,3})\s*(?:mg/dL|mg/dl)',
        ],
        # Heart rate: "heart rate 85", "pulse 92", "resting HR 78"
        'heart_rate': [
            r'(?:heart\s+rate|pulse|HR|resting\s+heart\s+rate)(?:\s+(?:is|of|around|about))?\s+(\d{2,3})',
            r'(?:heart\s+rate|pulse).*?(\d{2,3})\s*(?:bpm|beats?\s+per\s+minute)',
        ],
        # Age: "45 years old", "age 50", "38 year old patient"
        'age': [
            r'(?:age|aged|years?\s+old)[\s:]*(\d{1,3})(?:\s+years?)?',
        ],
        # BMI: "BMI 28", "body mass index 32"
        'bmi': [
            r'(?:BMI|body\s+mass\s+index)(?:\s+(?:is|of|around|about))?\s+([\d.]+)',
        ],
        # Weight: "weighs 85kg", "weight 180 lbs", "75 kilograms"
        'weight': [
            r'(?:weight|weighs|weigh)(?:\s+(?:is|of|around|about))?\s+([\d.]+)\s*(?:kg|kilograms?|lbs?|pounds?)',
        ],
        # Height: "height 6 feet", "175 cm", "5'10"
        'height': [
            r'(?:height|tall|cm|inches?)[\s:]*(\d+)[.\d]*\s*(?:cm|centimeters?|feet|ft|\'|"|inches?|in)?',
        ],
    }
    
    # Keywords for boolean conditions
    CONDITIONS = {
        'smoking': {
            'present': [r'\bsmoke|smoking|cigarette|tobacco|nicotine', r'smoke.*(?:day|daily|week)'],
            'absent': [r'(?:don\'?t|don\'?t|no|never|non[- ]?smoker|quit|stopped)\s+(?:smoking|smoke|cigarette)',
                      r'(?:smoke|smoking).*(?:never|don\'?t|no)\s'],
        },
        'family_history': {
            'present': [r'father.*(?:heart|attack|MI|disease|died)',
                       r'mother.*(?:diabetes|heart|attack|disease)',
                       r'family.*history|family.*(?:heart|disease)',
                       r'(?:brother|sister|parent|grandfather|grandmother).*(?:heart|diabetes|disease)'],
        },
        'chest_pain': {
            'present': [r'(?:chest|thoracic)[\s\-]?(?:pain|discomfort|tightness|pressure)',
                       r'(?:pain|discomfort).*chest',
                       r'angina|MI\s|heart\s+attack'],
        },
        'shortness_of_breath': {
            'present': [r'(?:shortness|short).*breath|dyspnea|breathless|cannot\s+breathe',
                       r'(?:breath|breathing|breathing\s+difficulty)',
                       r'suffocation|gasping'],
        },
        'palpitations': {
            'present': [r'palpitations?|heart\s+(?:racing|pounding|fluttering)',
                       r'(?:racing|pounding)\s+heart'],
        },
        'anxiety': {
            'present': [r'anxiety|anxious|nervous|worried|scared|panic'],
        },
        'dizziness': {
            'present': [r'dizzy|dizziness|vertigo|lightheaded|faint|syncope'],
        },
    }
    
    def __init__(self):
        """Initialize the feature extractor"""
        logger.info("Medical Feature Extractor initialized")
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract medical features from consultation text
        
        Args:
            text: Consultation transcript
            
        Returns:
            Dictionary with extracted medical features
        """
        text_lower = text.lower()
        features = {}
        
        # Extract numerical values
        features.update(self._extract_numerical_features(text_lower))
        
        # Extract boolean conditions
        features.update(self._extract_conditions(text_lower))
        
        # Extract symptom severity indicators
        features.update(self._extract_symptom_severity(text_lower))
        
        # Extract temporal information (how long symptoms last)
        features.update(self._extract_temporal_features(text_lower))
        
        logger.info(f"Extracted {len(features)} medical features")
        return features
    
    def _extract_numerical_features(self, text: str) -> Dict[str, Any]:
        """Extract numerical medical values from text"""
        features = {}
        
        for feature_name, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    try:
                        if len(match.groups()) == 2:
                            # For BP: extract systolic and diastolic
                            systolic = int(match.group(1))
                            diastolic = int(match.group(2))
                            # Calculate mean arterial pressure
                            features['blood_pressure'] = systolic
                            features['blood_pressure_systolic'] = systolic
                            features['blood_pressure_diastolic'] = diastolic
                            logger.info(f"Found BP: {systolic}/{diastolic} mmHg")
                        else:
                            # Single value extraction
                            value_str = match.group(1)
                            value = float(value_str) if '.' in value_str else int(value_str)
                            
                            # Validate reasonable ranges
                            if feature_name == 'blood_pressure' and 60 <= value <= 200:
                                features['blood_pressure'] = value
                            elif feature_name == 'cholesterol' and 100 <= value <= 400:
                                features['cholesterol'] = value
                            elif feature_name == 'glucose' and 50 <= value <= 400:
                                features['glucose'] = value
                                features['fasting_bs'] = value  # Alias for model
                            elif feature_name == 'heart_rate' and 30 <= value <= 200:
                                features['heart_rate'] = value
                            elif feature_name == 'age' and 1 <= value <= 120:
                                features['age'] = int(value)
                            elif feature_name == 'bmi' and 10 <= value <= 60:
                                features['bmi'] = value
                            elif feature_name in ['weight', 'height']:
                                features[feature_name] = value
                    except (ValueError, IndexError):
                        continue
                
                # Only use first found value for each feature
                if feature_name in features:
                    break
        
        return features
    
    def _extract_conditions(self, text: str) -> Dict[str, bool]:
        """Extract boolean health conditions from text"""
        conditions = {}
        
        for condition_name, keywords in self.CONDITIONS.items():
            if condition_name == 'family_history':
                # Special handling for family history
                if 'present' in keywords:
                    for pattern in keywords['present']:
                        if re.search(pattern, text):
                            conditions['family_history'] = True
                            logger.info(f"Found condition: {condition_name}")
                            break
                if condition_name not in conditions:
                    conditions['family_history'] = False
            else:
                # Check for presence
                present_found = False
                if 'present' in keywords:
                    for pattern in keywords['present']:
                        if re.search(pattern, text):
                            present_found = True
                            break
                
                # Check for absence
                absent_found = False
                if 'absent' in keywords:
                    for pattern in keywords['absent']:
                        if re.search(pattern, text):
                            absent_found = True
                            break
                
                # Determine final value (presence overrides absence)
                if present_found and not absent_found:
                    conditions[condition_name] = True
                    logger.info(f"Found condition: {condition_name}")
                else:
                    conditions[condition_name] = False
        
        return conditions
    
    def _extract_symptom_severity(self, text: str) -> Dict[str, str]:
        """Extract severity indicators for symptoms"""
        severity = {}
        
        severity_keywords = {
            'severe': [r'\bsevere|very\s+(?:bad|painful|worse)', r'(?:extreme|terrible|awful)'],
            'moderate': [r'\bmoderate|fairly|quite|somewhat'],
            'mild': [r'\bmild|slight|little|small'],
        }
        
        for symptom in ['chest_pain', 'shortness_of_breath', 'anxiety']:
            for level, patterns in severity_keywords.items():
                for pattern in patterns:
                    # Check if severity word appears near symptom mention
                    symptom_patterns = self.CONDITIONS.get(symptom, {}).get('present', [])
                    for symptom_pattern in symptom_patterns:
                        combined = f"(?:{pattern}).*(?:{symptom_pattern})|(?:{symptom_pattern}).*(?:{pattern})"
                        if re.search(combined, text):
                            severity[f"{symptom}_severity"] = level
                            logger.info(f"Found {symptom} severity: {level}")
                            break
        
        return severity
    
    def _extract_temporal_features(self, text: str) -> Dict[str, Any]:
        """Extract temporal information about symptoms"""
        temporal = {}
        
        # Duration patterns: "for 3 weeks", "last few days", "past month"
        duration_patterns = [
            r'(?:for|past|last|over)?\s*(\d+)\s*(?:weeks?|days?|months?|years?)',
            r'(?:few|several|couple)\s+(?:weeks?|days?|months?)',
        ]
        
        for pattern in duration_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                temporal['symptom_duration'] = match.group(0)
                logger.info(f"Found symptom duration: {match.group(0)}")
                break
        
        # Frequency patterns: "every day", "twice a week", "sometimes"
        frequency_patterns = [
            r'(?:every|almost|frequently)\s+(?:day|morning|evening|night)',
            r'(?:daily|nightly|weekly)',
        ]
        
        for pattern in frequency_patterns:
            if re.search(pattern, text):
                temporal['symptom_frequency'] = re.search(pattern, text).group(0)
                logger.info(f"Found symptom frequency: {temporal['symptom_frequency']}")
                break
        
        return temporal
    
    def extract_diagnosis_clues(self, summary_data: Dict[str, str]) -> List[str]:
        """
        Extract possible diagnoses based on extracted features and summary
        
        Args:
            summary_data: Medical summary from LLMSummarizer
            
        Returns:
            List of suspected diagnoses
        """
        diagnoses = []
        text = json.dumps(summary_data).lower()
        
        # Cardiac disease indicators
        cardiac_indicators = [
            'chest',
            'heart attack',
            'cardiac',
            'coronary',
            'angina',
            'palpitations',
            'heart disease',
            'acute coronary',
            'ACS',
            'MI'
        ]
        
        # Hypertension indicators
        hypertension_indicators = [
            'high blood pressure',
            'hypertension',
            'elevated BP',
            'blood pressure management'
        ]
        
        # Diabetes indicators
        diabetes_indicators = [
            'diabetes',
            'blood sugar',
            'glucose',
            'blood glucose',
            'diabetic'
        ]
        
        # Count indicators
        cardiac_count = sum(1 for indicator in cardiac_indicators if indicator in text)
        hypertension_count = sum(1 for indicator in hypertension_indicators if indicator in text)
        diabetes_count = sum(1 for indicator in diabetes_indicators if indicator in text)
        
        # Add likely diagnoses based on indicator count
        if cardiac_count >= 2:
            diagnoses.append('Cardiac Disease/Heart Problem')
        if hypertension_count >= 1:
            diagnoses.append('Hypertension')
        if diabetes_count >= 1:
            diagnoses.append('Diabetes')
        
        logger.info(f"Extracted diagnosis clues: {diagnoses}")
        return diagnoses


# Example usage and testing
if __name__ == "__main__":
    extractor = MedicalFeatureExtractor()
    
    sample_text = """Patient is 45 years old with blood pressure of 150/95 mmHg. 
    Has high cholesterol at 280 mg/dL. Reports chest discomfort for past 3 weeks.
    Smokes 5 cigarettes per day. Father had heart attack at age 55. 
    Patient is overweight with BMI of 29. Gets shortness of breath with exertion."""
    
    features = extractor.extract_features(sample_text)
    print("Extracted features:", json.dumps(features, indent=2))
