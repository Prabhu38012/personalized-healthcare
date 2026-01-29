"""
LLM-Based Medical Summarizer
Uses GROQ/Gemini to accurately extract structured medical information
"""

import os
import json
from typing import Dict, Any
from loguru import logger

# Import LLM analyzer
try:
    from backend.utils.llm_analyzer import llm_analyzer
except ImportError:
    from utils.llm_analyzer import llm_analyzer


class LLMSummarizer:
    """Extract structured medical information using LLM"""
    
    def __init__(self):
        """Initialize the LLM summarizer"""
        self.llm = llm_analyzer
        logger.info("LLM Medical Summarizer initialized")
    
    def generate_medical_summary(self, text: str) -> Dict[str, str]:
        """
        Generate structured medical summary from consultation transcript
        
        Args:
            text: Full consultation transcript
            
        Returns:
            Dictionary with chief_complaint, assessment, and plan
        """
        if not self.llm.is_available():
            logger.warning("LLM not available, using fallback extraction")
            return self._fallback_extraction(text)
        
        try:
            # Create prompt for structured extraction with focus on diagnosis
            prompt = f"""Analyze this medical consultation transcript and extract comprehensive medical information including diagnosis details and patient risk factors.

TRANSCRIPT:
{text[:4000]}

Extract and return ONLY valid JSON with these EXACT fields:
{{
  "chief_complaint": "Why did the patient come? Main symptoms and duration. What triggered the visit? (1-2 sentences)",
  "assessment": "Detailed clinical findings, diagnosis(es), vital signs, examination findings. Identify ALL mentioned health conditions. (3-4 sentences)",
  "diagnosis": "Specific diagnosis or differential diagnoses mentioned. If no clear diagnosis, list suspected conditions based on symptoms. (1-2 sentences)",
  "risk_factors": "Patient's risk factors: age, smoking, family history, medical history, lifestyle factors mentioned (1-2 sentences)",
  "symptoms_detailed": "List of ALL symptoms mentioned: chest pain, shortness of breath, anxiety, headaches, etc. with their characteristics",
  "plan": "Treatment plan, medications (dosage & frequency), lifestyle advice, follow-up instructions (2-3 sentences)"
}}

CRITICAL RULES:
1. DIAGNOSIS IS MOST IMPORTANT - Extract the actual suspected or confirmed diseases/conditions
2. List ALL symptoms explicitly mentioned, not just main ones
3. Include family history (e.g., "Father had heart attack", "Mother has diabetes")
4. Include vital measurements if mentioned (BP, heart rate, temperature, weight, BMI)
5. Extract smoking status, exercise level, medication adherence
6. For cardiac symptoms: chest pain, palpitations, shortness of breath, syncope, anxiety
7. For hypertension: BP readings, duration of condition, medication status
8. For anxiety/stress symptoms: identify and note them
9. Use past tense for patient statements ("Patient reported...", "Patient experienced...")
10. Use present tense for diagnoses ("Patient has...", "Findings indicate...")
11. Be COMPLETE - capture all medical information mentioned, not just main points
12. Remove filler words and conversational markers

Example output (Cardiac Case):
{{
  "chief_complaint": "Patient presented with chest discomfort over past 3-4 weeks, intermittently. Associated with anxiety and difficulty breathing at night.",
  "assessment": "Patient has 6-year history of hypertension on irregular medications. Recent symptoms suggest cardiac concerns given family history. Examination findings include anxiety manifestations, chest tenderness on palpation, and reported suffocation episodes at night. Risk stratification indicates moderate-to-high cardiac risk.",
  "diagnosis": "Likely acute coronary syndrome with hypertensive urgency versus anxiety-induced chest pain. Requires urgent cardiac workup including ECG and troponin levels.",
  "risk_factors": "Age factor relevant, 6-year hypertension history, high cholesterol, smoking 5-6 cigarettes daily, sedentary lifestyle, strong family history of MI in father (age 55), maternal diabetes. Irregular medication adherence.",
  "symptoms_detailed": "Chest discomfort (3-4 weeks, intermittent, worse at rest/after stress), shortness of breath (especially at night), anxiety, fear of serious illness, headaches, blurred vision, palpitations on anxiety episodes, sudden sleep interruptions with suffocation feeling, pain on exertion and during stress.",
  "plan": "Immediate: Refer to cardiology for ECG, troponin, echocardiogram, stress test. Optimize antihypertensive regimen - consider ACE inhibitor + beta blocker + statin. Stress management and psychiatric evaluation for anxiety. Smoking cessation program. Lifestyle modification: DASH diet, aerobic exercise. Follow-up with cardiology within 48 hours."
}}

Return ONLY the JSON object, no other text:"""

            # Call LLM
            if self.llm.provider == "groq":
                response = self.llm.groq_client.chat.completions.create(
                    model=self.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1500
                )
                response_text = response.choices[0].message.content.strip()
            else:
                response = self.llm.model.generate_content(prompt)
                response_text = response.text.strip()
            
            # Clean response - remove markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            # Parse JSON
            try:
                summary_data = json.loads(response_text)
                
                # Validate required fields - be flexible with new fields
                required_fields = ["chief_complaint", "assessment", "plan"]
                for field in required_fields:
                    if field not in summary_data:
                        logger.warning(f"Missing field: {field}, using fallback")
                        return self._fallback_extraction(text)
                
                # Add missing optional fields with defaults if not present
                if "diagnosis" not in summary_data:
                    summary_data["diagnosis"] = summary_data.get("assessment", "")
                if "risk_factors" not in summary_data:
                    summary_data["risk_factors"] = ""
                if "symptoms_detailed" not in summary_data:
                    summary_data["symptoms_detailed"] = ""
                
                # Add general summary (combination of all parts)
                summary_data["general_summary"] = (
                    f"{summary_data['chief_complaint']} "
                    f"{summary_data['assessment']} "
                    f"{summary_data['plan']}"
                )
                
                logger.info("✓ LLM medical summary generated successfully")
                return summary_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.debug(f"Raw response: {response_text[:500]}")
                return self._fallback_extraction(text)
                
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Dict[str, str]:
        """
        Simple fallback extraction if LLM is unavailable
        
        Args:
            text: Consultation transcript
            
        Returns:
            Basic structured summary
        """
        import re
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Try to find chief complaint (first 1/3 of conversation)
        chief_start = 0
        chief_end = len(sentences) // 3
        chief_sentences = []
        
        for sent in sentences[chief_start:chief_end]:
            if any(kw in sent.lower() for kw in ['cough', 'pain', 'fever', 'problem', 'issue', 'weeks', 'days']):
                chief_sentences.append(sent)
                if len(chief_sentences) >= 2:
                    break
        
        chief_complaint = '. '.join(chief_sentences[:2]) + '.' if chief_sentences else "Patient presented for medical consultation."
        
        # Try to find assessment (middle 1/3 of conversation)
        assess_start = len(sentences) // 3
        assess_end = 2 * len(sentences) // 3
        assess_sentences = []
        
        for sent in sentences[assess_start:assess_end]:
            if any(kw in sent.lower() for kw in ['diagnosis', 'infection', 'pressure', 'temperature', 'appears', 'examination']):
                assess_sentences.append(sent)
                if len(assess_sentences) >= 2:
                    break
        
        assessment = '. '.join(assess_sentences[:2]) + '.' if assess_sentences else "Clinical assessment documented."
        
        # Try to find plan (last 1/3 of conversation)
        plan_start = 2 * len(sentences) // 3
        plan_sentences = []
        
        for sent in sentences[plan_start:]:
            if any(kw in sent.lower() for kw in ['prescri', 'take', 'medication', 'follow', 'recommend']):
                plan_sentences.append(sent)
                if len(plan_sentences) >= 2:
                    break
        
        plan = '. '.join(plan_sentences[:2]) + '.' if plan_sentences else "Treatment plan discussed with patient."
        
        return {
            "general_summary": f"{chief_complaint} {assessment} {plan}",
            "chief_complaint": chief_complaint,
            "assessment": assessment,
            "plan": plan
        }
