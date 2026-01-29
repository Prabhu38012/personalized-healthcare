"""
Diagnosis Extractor
Uses LLM to extract and classify diagnoses from consultation summaries
Bridges between consultation text and AI/ML prediction models
"""

import json
from typing import Dict, List, Any
from loguru import logger

try:
    from backend.utils.llm_analyzer import llm_analyzer
except ImportError:
    from utils.llm_analyzer import llm_analyzer


class DiagnosisExtractor:
    """Extract and map diagnoses to ML model features"""
    
    # Mapping between diagnosis names and model input features
    DIAGNOSIS_TO_FEATURES = {
        'Hypertension': {
            'blood_pressure_flag': 1,
            'condition': 'hypertension',
            'risk_increase': 0.3
        },
        'Hypercholesterolemia': {
            'cholesterol_flag': 1,
            'condition': 'high_cholesterol',
            'risk_increase': 0.25
        },
        'Type 2 Diabetes': {
            'diabetes_flag': 1,
            'condition': 'diabetes',
            'risk_increase': 0.35
        },
        'Coronary Artery Disease': {
            'cad_flag': 1,
            'condition': 'heart_disease',
            'risk_increase': 0.5
        },
        'Heart Failure': {
            'heart_failure_flag': 1,
            'condition': 'heart_disease',
            'risk_increase': 0.6
        },
        'Acute Coronary Syndrome': {
            'acs_flag': 1,
            'condition': 'cardiac_emergency',
            'risk_increase': 0.8
        },
        'Anxiety Disorder': {
            'anxiety_flag': 1,
            'condition': 'mental_health',
            'risk_increase': 0.1
        },
        'Obesity': {
            'obesity_flag': 1,
            'condition': 'weight_management',
            'risk_increase': 0.2
        }
    }
    
    def __init__(self):
        """Initialize diagnosis extractor"""
        self.llm = llm_analyzer
        logger.info("Diagnosis Extractor initialized")
    
    def extract_diagnoses(self, summary_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract diagnoses from medical summary using LLM
        
        Args:
            summary_data: Output from LLMSummarizer with chief_complaint, assessment, etc.
            
        Returns:
            Dictionary with extracted diagnoses and severity
        """
        if not self.llm.is_available():
            logger.warning("LLM not available, using rule-based diagnosis extraction")
            return self._extract_diagnoses_rule_based(summary_data)
        
        try:
            # Build context from summary
            context = f"""
Chief Complaint: {summary_data.get('chief_complaint', '')}
Assessment: {summary_data.get('assessment', '')}
Diagnosis Section: {summary_data.get('diagnosis', '')}
Risk Factors: {summary_data.get('risk_factors', '')}
Symptoms: {summary_data.get('symptoms_detailed', '')}
            """
            
            prompt = f"""Based on the following medical consultation summary, extract the specific diagnoses, 
their confidence levels, and recommended next steps for AI/ML prediction models.

MEDICAL SUMMARY:
{context}

Return ONLY valid JSON with this EXACT structure:
{{
  "primary_diagnosis": "Most likely/confirmed diagnosis",
  "differential_diagnoses": ["diagnosis1", "diagnosis2", "diagnosis3"],
  "diagnosis_confidence": "HIGH/MODERATE/LOW",
  "urgency": "EMERGENT/URGENT/ROUTINE",
  "risk_level": "HIGH/MODERATE/LOW",
  "key_findings": ["finding1", "finding2", "finding3"],
  "requires_immediate_action": true/false,
  "recommended_tests": ["test1", "test2"],
  "model_input_features": {{
    "hypertension": true/false,
    "high_cholesterol": true/false,
    "diabetes": true/false,
    "smoking": true/false,
    "family_history": true/false,
    "obesity": true/false,
    "chest_pain": true/false,
    "shortness_of_breath": true/false,
    "anxiety": true/false
  }}
}}

Be specific about diagnoses - use actual medical condition names.
For cardiac symptoms: consider ACS, CAD, heart failure, hypertensive emergency
For metabolic: hypertension, diabetes, hypercholesterolemia
For anxiety-related chest pain: note psychological component but don't exclude cardiac
Always err on side of caution for cardiac symptoms.

Return ONLY the JSON:"""
            
            # Call LLM
            if self.llm.provider == "groq":
                response = self.llm.groq_client.chat.completions.create(
                    model=self.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Lower temp for consistent diagnosis
                    max_tokens=1200
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
            
            diagnosis_data = json.loads(response_text)
            
            # Validate required fields
            required = ["primary_diagnosis", "differential_diagnoses", "urgency"]
            for field in required:
                if field not in diagnosis_data:
                    logger.warning(f"Missing diagnosis field: {field}")
                    return self._extract_diagnoses_rule_based(summary_data)
            
            logger.info(f"✓ Extracted diagnosis: {diagnosis_data['primary_diagnosis']}")
            return diagnosis_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse diagnosis JSON: {e}")
            return self._extract_diagnoses_rule_based(summary_data)
        except Exception as e:
            logger.error(f"Diagnosis extraction failed: {e}")
            return self._extract_diagnoses_rule_based(summary_data)
    
    def _extract_diagnoses_rule_based(self, summary_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Rule-based diagnosis extraction as fallback
        
        Args:
            summary_data: Medical summary
            
        Returns:
            Extracted diagnoses
        """
        full_text = json.dumps(summary_data).lower()
        
        diagnoses = {
            "primary_diagnosis": "Medical Condition Requiring Evaluation",
            "differential_diagnoses": [],
            "diagnosis_confidence": "MODERATE",
            "urgency": "ROUTINE",
            "risk_level": "MODERATE",
            "key_findings": [],
            "requires_immediate_action": False,
            "recommended_tests": [],
            "model_input_features": {
                "hypertension": False,
                "high_cholesterol": False,
                "diabetes": False,
                "smoking": False,
                "family_history": False,
                "obesity": False,
                "chest_pain": False,
                "shortness_of_breath": False,
                "anxiety": False
            }
        }
        
        # Check for cardiac indicators
        cardiac_keywords = ['chest', 'heart', 'cardiac', 'palpitation', 'angina', 'ACS', 'MI', 'coronary']
        cardiac_count = sum(1 for kw in cardiac_keywords if kw in full_text)
        
        if cardiac_count >= 2:
            diagnoses["primary_diagnosis"] = "Cardiac Concern - Requires Evaluation"
            diagnoses["differential_diagnoses"] = ["Acute Coronary Syndrome", "Coronary Artery Disease", "Anxiety-related chest pain"]
            diagnoses["urgency"] = "URGENT" if 'emergency' in full_text or 'severe' in full_text else "URGENT"
            diagnoses["risk_level"] = "HIGH"
            diagnoses["model_input_features"]["chest_pain"] = True
            diagnoses["requires_immediate_action"] = True
            diagnoses["recommended_tests"] = ["ECG", "Troponin", "Echocardiogram", "Stress Test"]
        
        # Check for hypertension
        if any(kw in full_text for kw in ['blood pressure', 'hypertension', 'bp', 'elevated']):
            if "Hypertension" not in diagnoses["differential_diagnoses"]:
                diagnoses["differential_diagnoses"].insert(0, "Hypertension")
            diagnoses["model_input_features"]["hypertension"] = True
        
        # Check for diabetes/glucose issues
        if any(kw in full_text for kw in ['diabetes', 'glucose', 'blood sugar', 'fasting bs']):
            if "Type 2 Diabetes" not in diagnoses["differential_diagnoses"]:
                diagnoses["differential_diagnoses"].append("Type 2 Diabetes")
            diagnoses["model_input_features"]["diabetes"] = True
        
        # Check for cholesterol
        if any(kw in full_text for kw in ['cholesterol', 'hypercholesterol']):
            if "Hypercholesterolemia" not in diagnoses["differential_diagnoses"]:
                diagnoses["differential_diagnoses"].append("Hypercholesterolemia")
            diagnoses["model_input_features"]["high_cholesterol"] = True
        
        # Check for smoking
        if any(kw in full_text for kw in ['smoke', 'smoking', 'cigarette', 'tobacco']):
            diagnoses["model_input_features"]["smoking"] = True
        
        # Check for family history
        if any(kw in full_text for kw in ['father', 'mother', 'family history', 'parent']):
            diagnoses["model_input_features"]["family_history"] = True
        
        # Check for shortness of breath
        if any(kw in full_text for kw in ['shortness of breath', 'dyspnea', 'breathless', 'suffocation']):
            diagnoses["model_input_features"]["shortness_of_breath"] = True
        
        # Check for anxiety
        if any(kw in full_text for kw in ['anxiety', 'anxious', 'worried', 'scared', 'panic']):
            diagnoses["model_input_features"]["anxiety"] = True
            if "Anxiety Disorder" not in diagnoses["differential_diagnoses"]:
                diagnoses["differential_diagnoses"].append("Anxiety Disorder")
        
        return diagnoses
    
    def prepare_for_ml_prediction(self, diagnosis_data: Dict[str, Any], 
                                  extracted_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for ML model prediction
        
        Args:
            diagnosis_data: Output from extract_diagnoses()
            extracted_features: Output from MedicalFeatureExtractor
            
        Returns:
            Patient data ready for ML model prediction
        """
        patient_data = extracted_features.copy()
        
        # Add model input features from diagnosis
        if "model_input_features" in diagnosis_data:
            patient_data.update(diagnosis_data["model_input_features"])
        
        # Add derived features
        patient_data["has_cardiac_risk_factors"] = (
            patient_data.get("hypertension", False) or
            patient_data.get("high_cholesterol", False) or
            patient_data.get("smoking", False) or
            patient_data.get("diabetes", False)
        )
        
        patient_data["cardiac_symptom_present"] = (
            patient_data.get("chest_pain", False) or
            patient_data.get("shortness_of_breath", False) or
            patient_data.get("palpitations", False)
        )
        
        patient_data["urgency"] = diagnosis_data.get("urgency", "ROUTINE")
        patient_data["risk_level"] = diagnosis_data.get("risk_level", "MODERATE")
        patient_data["primary_diagnosis"] = diagnosis_data.get("primary_diagnosis", "")
        
        # Set defaults for missing ML features
        model_features = [
            'age', 'blood_pressure', 'cholesterol', 'glucose', 'heart_rate', 'bmi',
            'smoking', 'hypertension', 'diabetes', 'high_cholesterol', 'family_history'
        ]
        
        for feature in model_features:
            if feature not in patient_data:
                # Set reasonable defaults based on feature type
                if feature in ['age']:
                    patient_data[feature] = 50
                elif feature in ['blood_pressure', 'resting_bp']:
                    patient_data[feature] = 120
                elif feature in ['cholesterol']:
                    patient_data[feature] = 200
                elif feature in ['glucose', 'fasting_bs']:
                    patient_data[feature] = 100
                elif feature in ['heart_rate']:
                    patient_data[feature] = 70
                elif feature in ['bmi']:
                    patient_data[feature] = 25
                elif isinstance(patient_data.get(feature), bool):
                    patient_data[feature] = False
        
        logger.info(f"Prepared patient data with {len(patient_data)} features for ML prediction")
        return patient_data
