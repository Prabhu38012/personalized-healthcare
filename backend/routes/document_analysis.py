"""
Document Analysis API Routes
Handles medical report and prescription document uploads and AI-powered analysis
"""

import os
import logging
import uuid
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import io

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import sqlalchemy as sa
from sqlalchemy.orm import Session

# Import database and authentication
try:
    from backend.auth.database_store import get_db
    from backend.auth.routes import get_current_user
    from backend.utils.llm_analyzer import llm_analyzer
    from backend.routes.ai_decision_support import generate_lifestyle_diet_plan
except ImportError:
    from auth.database_store import get_db
    from auth.routes import get_current_user
    from utils.llm_analyzer import llm_analyzer
    try:
        from routes.ai_decision_support import generate_lifestyle_diet_plan
    except ImportError:
        # Fallback if module structure is different
        generate_lifestyle_diet_plan = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router (no prefix here, will be added in app.py)
router = APIRouter(tags=["document-analysis"])

# Pydantic models
class DocumentAnalysisResponse(BaseModel):
    """Document analysis response model"""
    id: str
    user_id: str
    document_type: str
    patient_name: Optional[str]
    filename: str
    extracted_text: str
    analysis: Dict[str, Any]
    lifestyle_diet_plan: Optional[Dict[str, Any]] = None
    created_at: datetime

class DocumentListResponse(BaseModel):
    """Document list response model"""
    id: str
    user_id: str
    document_type: str
    patient_name: Optional[str]
    filename: str
    created_at: datetime
    summary: Optional[str]

# Database table creation
def create_document_analysis_table(db: Session):
    """Create document analysis table if it doesn't exist"""
    try:
        result = db.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='document_analyses'"))
        if not result.fetchone():
            db.execute(sa.text("""
                CREATE TABLE document_analyses (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    patient_name TEXT,
                    filename TEXT NOT NULL,
                    file_size INTEGER,
                    extracted_text TEXT,
                    analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """))
            db.commit()
            logger.info("Document analyses table created successfully")
    except Exception as e:
        logger.error(f"Error creating document analyses table: {e}")
        db.rollback()

# Document processing utilities
async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded file (PDF, image, or text)"""
    try:
        content = await file.read()
        filename_lower = file.filename.lower()
        
        # Handle PDF files
        if filename_lower.endswith('.pdf'):
            try:
                import PyPDF2
                pdf_file = io.BytesIO(content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # Validate extracted text
                text = text.strip()
                
                # Check if we got actual text or just PDF structure
                if not text or len(text) < 50:
                    raise ValueError("No readable text extracted from PDF")
                
                # Check for PDF structure markers (indicates failed extraction)
                if any(marker in text[:200] for marker in ['obj', 'endobj', '/Type', '/Page', '/Parent']):
                    logger.warning("PDF extraction returned structure instead of text")
                    raise ValueError("PDF text extraction failed - got structure instead of content")
                
                logger.info(f"Successfully extracted {len(text)} characters from PDF")
                return text
                
            except ImportError:
                logger.error("PyPDF2 not installed")
                raise HTTPException(
                    status_code=400,
                    detail="PyPDF2 required for PDF processing. Install with: pip install PyPDF2"
                )
            except Exception as e:
                logger.error(f"PDF extraction failed: {str(e)}")
                # Try alternative PDF extraction method
                try:
                    import pdfplumber
                    pdf_file = io.BytesIO(content)
                    text = ""
                    with pdfplumber.open(pdf_file) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    if text.strip() and len(text.strip()) >= 50:
                        logger.info(f"Successfully extracted {len(text)} chars with pdfplumber")
                        return text.strip()
                except ImportError:
                    logger.warning("pdfplumber not available as fallback")
                except Exception as plumber_error:
                    logger.error(f"pdfplumber also failed: {plumber_error}")
                
                # Final fallback: inform user
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to extract text from PDF: {str(e)}. The PDF might be scanned/image-based. Try using OCR or convert to text first."
                )
        
        # Handle image files (OCR)
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            try:
                from PIL import Image
                import pytesseract
                import os
                
                # Configure Tesseract path for Windows
                if os.name == 'nt':  # Windows
                    tesseract_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
                    ]
                    
                    for path in tesseract_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            logger.info(f"Using Tesseract OCR from: {path}")
                            break
                
                image = Image.open(io.BytesIO(content))
                text = pytesseract.image_to_string(image)
                return text.strip()
            except ImportError:
                logger.warning("PIL or pytesseract not installed, cannot process images")
                raise HTTPException(
                    status_code=400,
                    detail="Image processing not available. Please install Pillow and pytesseract."
                )
            except pytesseract.TesseractNotFoundError:
                raise HTTPException(
                    status_code=400,
                    detail="tesseract is not installed or it's not in your PATH. See README file for more information."
                )
        
        # Handle text files
        elif filename_lower.endswith(('.txt', '.doc', '.docx')):
            if filename_lower.endswith('.docx'):
                try:
                    import docx
                    doc_file = io.BytesIO(content)
                    doc = docx.Document(doc_file)
                    text = "\n".join([para.text for para in doc.paragraphs])
                    return text.strip()
                except ImportError:
                    logger.warning("python-docx not installed")
                    return content.decode('utf-8', errors='ignore')
            else:
                return content.decode('utf-8', errors='ignore')
        
        else:
            # Try to decode as text
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from document: {str(e)}")

def extract_patient_data_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Extract patient health data from document analysis to generate lifestyle plan"""
    patient_data = {}
    
    # Extract metrics if available
    metrics = analysis.get('metrics', {})
    
    # Try to extract common health values
    for key, value in metrics.items():
        key_lower = key.lower()
        
        # Extract cholesterol
        if 'cholesterol' in key_lower or 'ldl' in key_lower:
            try:
                num = float(''.join(c for c in str(value).split('(')[0] if c.isdigit() or c == '.'))
                if 'ldl' in key_lower:
                    patient_data['cholesterol'] = num
                elif 'total' in key_lower and 'cholesterol' not in patient_data:
                    patient_data['cholesterol'] = num
            except:
                pass
        
        # Extract blood pressure
        if 'blood pressure' in key_lower or 'bp' in key_lower or 'systolic' in key_lower:
            try:
                num = float(''.join(c for c in str(value).split('/')[0] if c.isdigit() or c == '.'))
                patient_data['blood_pressure'] = num
            except:
                pass
        
        # Extract glucose
        if 'glucose' in key_lower or 'blood sugar' in key_lower or 'hba1c' in key_lower:
            try:
                num = float(''.join(c for c in str(value).split('(')[0] if c.isdigit() or c == '.'))
                patient_data['glucose'] = num
            except:
                pass
        
        # Extract BMI
        if 'bmi' in key_lower:
            try:
                num = float(''.join(c for c in str(value).split('(')[0] if c.isdigit() or c == '.'))
                patient_data['bmi'] = num
            except:
                pass
    
    # Default values if not found
    if 'age' not in patient_data:
        patient_data['age'] = 50  # Default
    if 'bmi' not in patient_data:
        patient_data['bmi'] = 25  # Default normal BMI
    if 'blood_pressure' not in patient_data:
        patient_data['blood_pressure'] = 120  # Default
    if 'cholesterol' not in patient_data:
        patient_data['cholesterol'] = 200  # Default
    if 'glucose' not in patient_data:
        patient_data['glucose'] = 100  # Default
    
    patient_data['smoking'] = False  # Default
    patient_data['exercise'] = 0  # Default
    
    return patient_data

def calculate_risk_from_analysis(analysis: Dict[str, Any]) -> float:
    """Calculate risk score from document analysis"""
    # Get diagnosis severity
    diagnosis = analysis.get('diagnosis', {})
    severity = diagnosis.get('severity', 'mild').lower()
    
    # Base risk on severity
    if severity == 'critical':
        base_risk = 0.85
    elif severity == 'severe':
        base_risk = 0.70
    elif severity == 'moderate':
        base_risk = 0.50
    else:  # mild or unknown
        base_risk = 0.30
    
    # Adjust based on concerns
    concerns = analysis.get('concerns', [])
    concern_adjustment = min(len(concerns) * 0.05, 0.20)
    
    # Adjust based on disease indicators
    disease_indicators = analysis.get('disease_indicators', [])
    indicator_adjustment = min(len(disease_indicators) * 0.03, 0.15)
    
    final_risk = min(base_risk + concern_adjustment + indicator_adjustment, 0.95)
    
    return final_risk

async def analyze_medical_document(
    extracted_text: str,
    document_type: str,
    patient_name: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze medical document using LLM"""
    
    # Validate extracted text before processing
    if not extracted_text or len(extracted_text.strip()) < 20:
        logger.error("Extracted text too short or empty")
        return {
            "status": "error",
            "summary": "Document text extraction failed",
            "key_findings": ["Unable to extract readable text from document"],
            "concerns": ["Document may be corrupted, password-protected, or image-based"],
            "recommendations": [
                "Verify the document is not corrupted",
                "If it's a scanned document, use OCR-enabled PDF",
                "Try converting to text format first"
            ]
        }
    
    # Check for PDF structure markers (failed extraction)
    if any(marker in extracted_text[:500] for marker in ['endobj', '/Type /Page', '/Parent', 'stream\n']):
        logger.error("Extracted text contains PDF structure - extraction failed")
        return {
            "status": "error",
            "summary": "PDF text extraction failed - received structure instead of content",
            "key_findings": ["Document appears to be scanned or image-based PDF"],
            "concerns": ["Text extraction returned PDF structure codes instead of readable text"],
            "recommendations": [
                "Use an OCR tool to extract text from scanned PDFs",
                "Try converting the PDF to text format using Adobe Acrobat or similar",
                "If possible, obtain a text-searchable version of the document"
            ]
        }
    
    # Check text quality
    letter_count = sum(1 for c in extracted_text[:1000] if c.isalpha())
    if letter_count < 100:  # Less than 100 letters in first 1000 chars
        logger.warning("Extracted text has very few letters - may be corrupted")
        return {
            "status": "error",
            "summary": "Extracted text appears corrupted or unreadable",
            "key_findings": ["Text quality check failed - insufficient readable content"],
            "concerns": ["Document may be in unsupported format or corrupted"],
            "recommendations": [
                "Check if document is in a supported format (PDF, DOCX, TXT, images)",
                "Ensure document is not password-protected or encrypted",
                "Try re-saving the document in a standard format"
            ]
        }
    
    if not llm_analyzer.is_available():
        logger.warning("LLM analyzer not available, returning basic analysis")
        return {
            "status": "processed",
            "message": "Document processed but AI analysis unavailable (GEMINI_API_KEY not configured)",
            "extracted_text_length": len(extracted_text),
            "recommendations": ["Configure GEMINI_API_KEY for AI-powered analysis"]
        }
    
    try:
        # Create appropriate prompt based on document type
        if document_type == "medical_report":
            prompt = f"""You are an expert physician reviewing a patient's medical report. Provide a comprehensive clinical analysis with disease identification.

Patient: {patient_name or "Not specified"}

Medical Report:
{extracted_text[:4000]}

Provide a professional medical analysis in valid JSON format with these EXACT keys (no markdown code blocks, pure JSON only):
{{
  "summary": "Write a detailed 4-5 sentence executive summary as a physician would write for a medical record. Include: primary diagnosis/impression, critical lab values or findings, overall health status assessment, and immediate clinical significance. Be specific with numbers and medical terminology.",
  "diagnosis": {{
    "primary_condition": "Main disease or condition identified (e.g., 'Type 2 Diabetes Mellitus', 'Hyperlipidemia', 'Hypertension')",
    "icd10_code": "ICD-10 code if applicable (e.g., 'E11.9 for Type 2 Diabetes')",
    "severity": "mild|moderate|severe|critical",
    "confidence": "high|medium|low - based on available test results",
    "additional_conditions": ["List any secondary conditions or comorbidities identified"]
  }},
  "clinical_interpretation": "2-3 sentences explaining what the findings mean for the patient's health, the disease progression stage, and prognosis",
  "key_findings": [
    "Finding 1: Include specific test name, actual value, unit, normal range, and clinical interpretation (e.g., 'LDL Cholesterol: 160 mg/dL (Optimal: <100 mg/dL) - Elevated, indicates increased cardiovascular risk')",
    "Finding 2: Same format with specific clinical context and how it relates to the diagnosis",
    "Finding 3: Same format with pathological significance if applicable",
    "Finding 4: Additional critical findings supporting the diagnosis",
    "Finding 5: Include all abnormal or noteworthy results"
  ],
  "metrics": {{
    "Complete Test Name": "Value with Unit (Normal Range: X-Y) - Clinical Status",
    "Example - Total Cholesterol": "240 mg/dL (Desirable: <200 mg/dL) - Borderline High"
  }},
  "disease_indicators": [
    "Specific biomarker or symptom that indicates the diagnosed condition",
    "Pattern of results supporting the diagnosis",
    "Any pathognomonic (definitive) findings"
  ],
  "concerns": [
    "Specific health risk 1 with severity level and clinical implications for the diagnosed condition",
    "Specific health risk 2 with timeframe for intervention if applicable",
    "Complications that may arise from the identified disease",
    "Any contraindications or warnings based on findings"
  ],
  "recommendations": [
    "Immediate action 1: Specific medication/intervention with dosage if mentioned (e.g., 'Initiate statin therapy - discuss with cardiologist')",
    "Disease management: Specific treatments for the identified condition",
    "Lifestyle modification: Specific dietary changes with targets (e.g., 'Reduce saturated fat to <7% of total calories')",
    "Follow-up testing: Specific tests with recommended timeframe (e.g., 'Repeat lipid panel in 3 months')",
    "Specialist referral: Which specialist and reason (e.g., 'Refer to endocrinologist for diabetes management')",
    "Preventive measures: Evidence-based recommendations to prevent complications"
  ]
}}

CRITICAL REQUIREMENTS:
1. MUST identify and name any diseases or medical conditions present
2. Extract ALL numeric values with units and reference ranges
3. Classify every finding as Normal, Borderline, Elevated, Critical, etc.
4. Use proper medical terminology and ICD-10 codes for diagnoses
5. Prioritize abnormal findings first and link them to potential diseases
6. Include temporal context (acute vs chronic findings)
7. Return ONLY valid JSON - no markdown formatting, no code blocks, no extra text
8. If a field has no data, use empty array [] or empty string "", never omit the key
9. Make clinical connections between test results and disease diagnosis"""

        elif document_type == "prescription":
            prompt = f"""You are an expert clinical pharmacist analyzing a prescription. Extract medications EXACTLY as they appear in the prescription.

Patient: {patient_name or "Not specified"}

Prescription Text:
{extracted_text[:3000]}

Provide analysis in JSON format:
{{
  "medications": [
    {{
      "name": "Brand name + Generic name (if both shown)",
      "dosage": "Exact dosage per unit",
      "frequency": "Exact frequency from prescription",
      "indication": "Primary medical use"
    }}
  ],
  "interactions": ["Warning if applicable"],
  "safety_info": ["Important information"],
  "dosages": ["Detailed instructions"],
  "recommendations": ["Patient guidance"]
}}

CRITICAL RULES FOR INDIAN PRESCRIPTIONS:

1. **MEDICATION EXTRACTION - Use the FIRST/MAIN name visible in each row**:
   - Example: "Levoflox 500 Tablet (Levofloxacin(500mg))" → name: "Levoflox 500"
   - Example: "Montek LC Tablet (Levocetirizine(5mg) + Montelukast(10mg))" → name: "Montek LC"
   - Example: "Ascoril D Plus Syrup" → name: "Ascoril D Plus Syrup"
   - DO NOT split combination drugs into separate entries
   - Use brand name + strength if visible (e.g., "Rantac 300")

2. **DOSAGE EXTRACTION**:
   - Look for the strength in parentheses: "(500mg)" → dosage: "500mg"
   - For combination drugs, include all: "(5mg + 10mg)" → dosage: "5mg + 10mg"
   - For syrups with volume: "10 ml" → dosage: "10ml"
   - If dosage is per tablet/unit, mention it: "1 tablet (500mg)" → dosage: "500mg per tablet"

3. **FREQUENCY INTERPRETATION** (Indian format "Daily: X-Y-Z"):
   - "Daily: 1-0-1" means Morning-Afternoon-Evening → frequency: "Twice daily (morning and evening)"
   - "Daily: 0-0-1" means only evening → frequency: "Once daily (evening)"
   - "Daily: 1-0-0" means only morning → frequency: "Once daily (morning)"
   - "Daily: 1-1-1" means all three times → frequency: "Three times daily (after each meal)"
   - "Daily: 1-1-0" means morning and afternoon → frequency: "Twice daily (morning and afternoon)"
   - If syrup mentions "10 ml" → add "10ml per dose"

4. **MEAL TIMING**:
   - "After Meal" → include in frequency: "after meals"
   - "Before Meal" → include in frequency: "before meals"
   - Example: "Daily: 1-0-1, After Meal" → "Twice daily after meals (morning and evening)"

5. **INDICATION/PURPOSE - MANDATORY FOR ALL MEDICATIONS**:
   - Look for generic name(s) in parentheses: "Tab. CEFMOOV-O (Cefpodoxime(200mg))" → identify drug class
   - NEVER leave indication as "Not specified" or empty
   - Use generic drug name to determine purpose:
     * Cefpodoxime/Cefixime → "Treats bacterial infections (cephalosporin antibiotic)"
     * Levofloxacin → "Treats bacterial infections (fluoroquinolone antibiotic)"
     * Levocetirizine → "Treats allergic reactions and symptoms"
     * Montelukast → "Treats asthma and allergic rhinitis"
     * Paracetamol → "Relieves pain and reduces fever"
     * Esomeprazole/Omeprazole → "Reduces stomach acid, treats GERD and ulcers"
     * Domperidone → "Treats nausea and improves digestion"
     * Sucralfate → "Protects stomach lining, treats ulcers"
     * Ranitidine/Pantoprazole → "Reduces stomach acid"
     * Dextromethorphan → "Suppresses cough"
     * Phenylephrine → "Relieves nasal congestion"
     * Chlorpheniramine → "Treats allergic symptoms"
     * Methylprednisolone/Prednisolone → "Reduces inflammation and swelling"
     * Azithromycin/Amoxicillin → "Treats bacterial infections"
   - If generic name not visible, infer from brand name pattern:
     * Names with "CEF-" prefix → Antibiotic for bacterial infections
     * Names with "-D" suffix → Often indicates combination with Domperidone (anti-nausea)
     * Names with "ESO-" prefix → Proton pump inhibitor for stomach acid
     * Syrups with "SUCRAL" → Stomach protectant for ulcers
   - Always provide a meaningful indication, never leave blank

6. **COMBINATION DRUGS** - Keep as ONE medication:
   - Montek LC contains Levocetirizine + Montelukast → ONE entry "Montek LC"
   - Ascoril D Plus contains multiple ingredients → ONE entry "Ascoril D Plus Syrup"
   - Include all active ingredients in indication

7. **DURATION**:
   - If "5 Days" or "7 Days" mentioned, include in recommendations
   - Example: "Take for 5 days as prescribed"

EXAMPLES OF CORRECT EXTRACTION:

Input: "Levoflox 500 Tablet (Levofloxacin(500mg)) Daily: 1-0-1 5 Days: After Meal"
Output:
{{
  "name": "Levoflox 500",
  "dosage": "500mg per tablet",
  "frequency": "Twice daily after meals (morning and evening)",
  "indication": "Treats bacterial infections (fluoroquinolone antibiotic)"
}}

Input: "Montek LC Tablet (Levocetirizine(5mg) + Montelukast(10mg)) Daily: 0-0-1 After Meal"
Output:
{{
  "name": "Montek LC",
  "dosage": "5mg + 10mg per tablet",
  "frequency": "Once daily in the evening after meal",
  "indication": "Treats allergic reactions, asthma, and allergic rhinitis"
}}

Input: "Ascoril D Plus Syrup (Phenylephrine(5mg) + Chlorpheniramine Maleate(2mg)) Daily: 1-1-1 10 ml"
Output:
{{
  "name": "Ascoril D Plus Syrup",
  "dosage": "10ml per dose",
  "frequency": "Three times daily after meals",
  "indication": "Relieves cough, cold, and nasal congestion"
}}

Input: "Tab. CEFMOOV-O (Cefpodoxime(200mg)) Daily: 1-0-1 After Meal"
Output:
{{
  "name": "CEFMOOV-O",
  "dosage": "200mg per tablet",
  "frequency": "Twice daily after meals (morning and evening)",
  "indication": "Treats bacterial infections (cephalosporin antibiotic)"
}}

Input: "Cap. ESOCALL-D 40 (Esomeprazole(40mg) + Domperidone(30mg)) Daily: 1-0-1 Before Meal"
Output:
{{
  "name": "ESOCALL-D 40",
  "dosage": "40mg + 30mg per capsule",
  "frequency": "Twice daily before meals (morning and evening)",
  "indication": "Reduces stomach acid and treats GERD, improves digestion"
}}

Input: "Syr. SUCRAL-O 10 (Sucralfate + Oxetacaine) Daily: 1-1-1"
Output:
{{
  "name": "SUCRAL-O Syrup",
  "dosage": "10ml per dose",
  "frequency": "Three times daily",
  "indication": "Protects stomach lining and relieves ulcer pain"
}}

Input: "Tab. FIMOLYBD-M (Trypsin + Bromelain + Diclofenac) Daily: 1-0-1 Before Meal"
Output:
{{
  "name": "FIMOLYBD-M",
  "dosage": "1 tablet",
  "frequency": "Twice daily before meals (morning and evening)",
  "indication": "Reduces inflammation, swelling, and pain (enzyme therapy)"
}}

CRITICAL: Every medication MUST have a valid indication. Use generic name or drug class patterns to determine purpose. NEVER output "Not specified" for indication.

Return ONLY valid JSON - no markdown, no code blocks."""

        else:
            prompt = f"""Analyze the following medical document and provide insights.

Document Type: {document_type}
Patient Name: {patient_name or "Not specified"}

Content:
{extracted_text}

Provide a structured analysis with key points, findings, and recommendations."""

        # Call LLM for analysis - support both GROQ and Gemini
        if llm_analyzer.provider == "groq":
            # Use GROQ API
            response = llm_analyzer.groq_client.chat.completions.create(
                model=llm_analyzer.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content.strip()
        else:
            # Use Gemini API
            response = llm_analyzer.model.generate_content(prompt)
            response_text = response.text.strip()
        
        # Clean the response text - handle encoding issues
        import re
        
        # Log original response for debugging
        logger.info(f"Raw LLM response length: {len(response_text)}")
        
        # Check if response is corrupted (all special characters, no letters)
        letter_count = sum(1 for c in response_text if c.isalpha())
        special_count = sum(1 for c in response_text if c in '@#$%^&*~`+=')
        
        # If response is mostly garbage, try to regenerate with simpler prompt
        if len(response_text) > 100 and (letter_count < len(response_text) * 0.1 or special_count > len(response_text) * 0.3):
            logger.warning(f"Detected corrupted LLM response ({letter_count} letters, {special_count} special chars in {len(response_text)} chars)")
            logger.warning(f"Corrupted preview: {response_text[:100]}")
            
            # Try a simpler, more direct prompt
            simple_prompt = f"""Extract key information from this medical report and format as JSON.

Text: {extracted_text[:2000]}

Return only valid JSON with keys: summary, key_findings, concerns, recommendations"""
            
            try:
                if llm_analyzer.provider == "groq":
                    response = llm_analyzer.groq_client.chat.completions.create(
                        model=llm_analyzer.model,
                        messages=[{"role": "user", "content": simple_prompt}],
                        temperature=0.1,
                        max_tokens=1500
                    )
                    response_text = response.choices[0].message.content.strip()
                else:
                    response = llm_analyzer.model.generate_content(simple_prompt)
                    response_text = response.text.strip()
                    
                logger.info(f"Retry response length: {len(response_text)}")
            except Exception as retry_error:
                logger.error(f"Retry also failed: {retry_error}")
        
        # Fix common encoding corruption patterns
        # Remove @QC= P pattern and similar corruptions
        response_text = re.sub(r'@[A-Z]+[=\s]+[A-Z](?:@[A-Z]+[=\s]+[A-Z])*', '', response_text)
        
        # Remove any remaining problematic patterns
        response_text = re.sub(r'[@#$%^&*]{3,}', '', response_text)  # Multiple special chars (3+ in a row)
        
        # If still mostly garbage after cleaning, return structured fallback
        cleaned_letter_count = sum(1 for c in response_text if c.isalpha())
        if len(response_text) > 50 and cleaned_letter_count < 20:
            logger.error(f"Response still corrupted after cleaning, using fallback analysis")
            raise ValueError("LLM response corrupted, using fallback")
        
        # Try to extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        # Final check: ensure response has actual content
        if not response_text or len(response_text.strip()) < 10:
            logger.error("Response too short or empty after cleaning")
            raise ValueError("Empty response after cleaning")
        
        logger.info(f"Cleaned response preview: {response_text[:200]}")
        
        try:
            analysis = json.loads(response_text)
            logger.info("Successfully parsed JSON response from LLM")
            
            # Deep clean function to remove any remaining corruption
            def clean_text(text):
                if isinstance(text, str):
                    # Remove encoding corruption patterns
                    text = re.sub(r'@[A-Z]+[=\s]+[A-Z](?:@[A-Z]+[=\s]+[A-Z])*', '', text)
                    text = re.sub(r'[@#$%^&*]{2,}', '', text)
                    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)  # ASCII printable only
                    # Clean up extra whitespace
                    text = ' '.join(text.split())
                    return text.strip()
                return text
            
            # Recursively clean all string values in the analysis
            def deep_clean(obj):
                if isinstance(obj, str):
                    return clean_text(obj)
                elif isinstance(obj, list):
                    return [deep_clean(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: deep_clean(value) for key, value in obj.items()}
                return obj
            
            analysis = deep_clean(analysis)
                            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}, response: {response_text[:200]}")
            # If not JSON, create structured analysis from extracted text
            logger.info("Creating fallback structured analysis from extracted text")
            
            # Extract key information from the text
            sentences = extracted_text.split('.')[:10]  # First 10 sentences
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            analysis = {
                "summary": f"Medical report analyzed for {patient_name or 'patient'}. " + 
                          (meaningful_sentences[0] + '.' if meaningful_sentences else "Document processed successfully."),
                "key_findings": meaningful_sentences[:5] if meaningful_sentences else ["Document text extracted - manual review recommended"],
                "concerns": ["AI analysis encountered formatting issues - manual review recommended", 
                            "Please verify all findings with healthcare provider"],
                "recommendations": [
                    "Consult with healthcare provider for interpretation",
                    "Review extracted text manually for accuracy",
                    "Consider re-uploading document if analysis seems incomplete"
                ],
                "status": "processed_with_warnings",
                "note": "Automatic analysis had issues, fallback extraction used"
            }
        except ValueError as e:
            # This catches our "corrupted response" error
            logger.error(f"Fallback analysis triggered: {str(e)}")
            
            # Create basic analysis from text extraction
            sentences = extracted_text.split('.')[:8]
            meaningful = [s.strip() + '.' for s in sentences if len(s.strip()) > 15]
            
            analysis = {
                "summary": f"Medical document processed for {patient_name or 'patient'}. AI analysis unavailable, text extracted successfully.",
                "key_findings": meaningful[:4] if meaningful else ["Text extraction completed"],
                "concerns": ["AI analysis service temporarily unavailable - manual review required"],
                "recommendations": [
                    "Review the extracted text below for medical information",
                    "Consult healthcare provider for proper interpretation",
                    "Consider manual analysis of the document"
                ],
                "status": "text_extracted_only",
                "note": "AI analysis failed, showing extracted text only"
            }
        
        # Ensure required fields exist for medical reports
        if document_type == "medical_report":
            if "summary" not in analysis:
                analysis["summary"] = "Analysis completed"
            if "key_findings" not in analysis:
                analysis["key_findings"] = ["Document processed successfully"]
            if "metrics" not in analysis:
                analysis["metrics"] = {}
            if "concerns" not in analysis:
                analysis["concerns"] = []
            if "recommendations" not in analysis:
                analysis["recommendations"] = ["Consult healthcare provider for interpretation"]
        
        # Ensure required fields for prescriptions
        elif document_type == "prescription":
            if "medications" not in analysis:
                analysis["medications"] = ["Medication information extracted"]
            if "interactions" not in analysis:
                analysis["interactions"] = []
            if "safety_info" not in analysis:
                analysis["safety_info"] = ["Please verify all medications with pharmacist"]
            if "recommendations" not in analysis:
                analysis["recommendations"] = ["Follow prescription instructions carefully"]
            
            # Enhance medications with indication information from database
            try:    
                from backend.utils.medication_database import enhance_medication_list
                analysis["medications"] = enhance_medication_list(analysis["medications"])
                logger.info(f"Enhanced {len(analysis['medications'])} medications with indication info")
            except Exception as e:
                logger.warning(f"Could not enhance medications with database: {e}")
        
        analysis["status"] = "completed"
        analysis["analyzed_at"] = datetime.utcnow().isoformat()
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing document with LLM: {str(e)}")
        # Return structured error with guidance
        return {
            "status": "error",
            "summary": "Analysis encountered an error",
            "key_findings": ["Document text was extracted successfully"],
            "concerns": [f"AI analysis failed: {str(e)}"],
            "recommendations": [
                "Review the extracted text manually",
                "Ensure API credentials are configured correctly",
                "Contact support if issue persists"
            ],
            "message": f"Analysis failed: {str(e)}",
            "extracted_text_length": len(extracted_text)
        }

# API Routes

@router.post("/upload/medical-report", response_model=DocumentAnalysisResponse)
async def upload_medical_report(
    file: UploadFile = File(...),
    patient_name: Optional[str] = Form(None),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and analyze a medical report document"""
    try:
        # Ensure table exists
        create_document_analysis_table(db)
        
        # Validate file type
        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.txt', '.doc', '.docx']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Extract text from document
        logger.info(f"Extracting text from {file.filename}")
        extracted_text = await extract_text_from_file(file)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Could not extract meaningful text from document. Please ensure the document is readable."
            )
        
        # Analyze document with LLM
        logger.info(f"Analyzing medical report for {patient_name or 'unknown patient'}")
        analysis = await analyze_medical_document(extracted_text, "medical_report", patient_name)
        
        # Generate lifestyle and diet plan based on analysis
        lifestyle_diet_plan = None
        if generate_lifestyle_diet_plan and analysis.get('status') == 'completed':
            try:
                # Extract patient data from analysis
                patient_data = extract_patient_data_from_analysis(analysis)
                
                # Calculate risk score
                risk_score = calculate_risk_from_analysis(analysis)
                
                # Generate personalized lifestyle plan
                lifestyle_diet_plan_obj = generate_lifestyle_diet_plan(patient_data, risk_score)
                lifestyle_diet_plan = lifestyle_diet_plan_obj.dict() if hasattr(lifestyle_diet_plan_obj, 'dict') else lifestyle_diet_plan_obj
                
                logger.info(f"Generated lifestyle plan for risk level: {risk_score:.2f}")
            except Exception as e:
                logger.warning(f"Failed to generate lifestyle plan: {e}")
        
        # Save to database
        analysis_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        db.execute(sa.text("""
            INSERT INTO document_analyses (
                id, user_id, document_type, patient_name, filename,
                file_size, extracted_text, analysis, created_at
            ) VALUES (
                :id, :user_id, :document_type, :patient_name, :filename,
                :file_size, :extracted_text, :analysis, :created_at
            )
        """), {
            "id": analysis_id,
            "user_id": current_user.id,
            "document_type": "medical_report",
            "patient_name": patient_name,
            "filename": file.filename,
            "file_size": len(extracted_text),
            "extracted_text": extracted_text[:10000],  # Store first 10k chars
            "analysis": json.dumps(analysis),
            "created_at": now
        })
        db.commit()
        
        logger.info(f"Medical report analysis saved with ID: {analysis_id}")
        
        return DocumentAnalysisResponse(
            id=analysis_id,
            user_id=current_user.id,
            document_type="medical_report",
            patient_name=patient_name,
            filename=file.filename,
            extracted_text=extracted_text,
            analysis=analysis,
            lifestyle_diet_plan=lifestyle_diet_plan,
            created_at=now
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading medical report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process medical report: {str(e)}")

@router.post("/upload/prescription", response_model=DocumentAnalysisResponse)
async def upload_prescription(
    file: UploadFile = File(...),
    patient_name: Optional[str] = Form(None),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and analyze a prescription document"""
    try:
        # Ensure table exists
        create_document_analysis_table(db)
        
        # Validate file type
        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.txt', '.doc', '.docx']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Extract text from document
        logger.info(f"Extracting text from {file.filename}")
        extracted_text = await extract_text_from_file(file)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Could not extract meaningful text from document. Please ensure the document is readable."
            )
        
        # Analyze document with LLM
        logger.info(f"Analyzing prescription for {patient_name or 'unknown patient'}")
        analysis = await analyze_medical_document(extracted_text, "prescription", patient_name)
        
        # Save to database
        analysis_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        db.execute(sa.text("""
            INSERT INTO document_analyses (
                id, user_id, document_type, patient_name, filename,
                file_size, extracted_text, analysis, created_at
            ) VALUES (
                :id, :user_id, :document_type, :patient_name, :filename,
                :file_size, :extracted_text, :analysis, :created_at
            )
        """), {
            "id": analysis_id,
            "user_id": current_user.id,
            "document_type": "prescription",
            "patient_name": patient_name,
            "filename": file.filename,
            "file_size": len(extracted_text),
            "extracted_text": extracted_text[:10000],  # Store first 10k chars
            "analysis": json.dumps(analysis),
            "created_at": now
        })
        db.commit()
        
        logger.info(f"Prescription analysis saved with ID: {analysis_id}")
        
        return DocumentAnalysisResponse(
            id=analysis_id,
            user_id=current_user.id,
            document_type="prescription",
            patient_name=patient_name,
            filename=file.filename,
            extracted_text=extracted_text,
            analysis=analysis,
            lifestyle_diet_plan=None,  # Not applicable for prescriptions
            created_at=now
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading prescription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process prescription: {str(e)}")

@router.get("/analysis/{analysis_id}", response_model=DocumentAnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document analysis by ID"""
    try:
        result = db.execute(sa.text("""
            SELECT * FROM document_analyses 
            WHERE id = :id AND user_id = :user_id
        """), {"id": analysis_id, "user_id": current_user.id}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        analysis = json.loads(result[7]) if result[7] else {}
        
        # Generate lifestyle plan if it's a medical report
        lifestyle_diet_plan = None
        if result[2] == "medical_report" and generate_lifestyle_diet_plan and analysis.get('status') == 'completed':
            try:
                patient_data = extract_patient_data_from_analysis(analysis)
                risk_score = calculate_risk_from_analysis(analysis)
                lifestyle_diet_plan_obj = generate_lifestyle_diet_plan(patient_data, risk_score)
                lifestyle_diet_plan = lifestyle_diet_plan_obj.dict() if hasattr(lifestyle_diet_plan_obj, 'dict') else lifestyle_diet_plan_obj
            except Exception as e:
                logger.warning(f"Failed to generate lifestyle plan: {e}")
        
        return DocumentAnalysisResponse(
            id=result[0],
            user_id=result[1],
            document_type=result[2],
            patient_name=result[3],
            filename=result[4],
            extracted_text=result[6],
            analysis=analysis,
            lifestyle_diet_plan=lifestyle_diet_plan,
            created_at=result[8]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@router.get("/list", response_model=List[DocumentListResponse])
async def list_analyses(
    patient_name: Optional[str] = None,
    document_type: Optional[str] = None,
    limit: int = 50,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List document analyses for current user"""
    try:
        query = "SELECT * FROM document_analyses WHERE user_id = :user_id"
        params = {"user_id": current_user.id}
        
        if patient_name:
            query += " AND patient_name = :patient_name"
            params["patient_name"] = patient_name
        
        if document_type:
            query += " AND document_type = :document_type"
            params["document_type"] = document_type
        
        query += " ORDER BY created_at DESC LIMIT :limit"
        params["limit"] = limit
        
        results = db.execute(sa.text(query), params).fetchall()
        
        analyses = []
        for row in results:
            # Extract summary from analysis
            analysis_data = json.loads(row[7]) if row[7] else {}
            summary = (
                analysis_data.get("summary") or 
                analysis_data.get("message") or 
                "Analysis completed"
            )
            
            analyses.append(DocumentListResponse(
                id=row[0],
                user_id=row[1],
                document_type=row[2],
                patient_name=row[3],
                filename=row[4],
                created_at=row[8],
                summary=summary[:200]  # Truncate summary
            ))
        
        return analyses
        
    except Exception as e:
        logger.error(f"Error listing analyses: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list analyses")

@router.delete("/analysis/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete document analysis"""
    try:
        result = db.execute(sa.text("""
            DELETE FROM document_analyses 
            WHERE id = :id AND user_id = :user_id
        """), {"id": analysis_id, "user_id": current_user.id})
        db.commit()
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {"message": "Analysis deleted successfully", "id": analysis_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete analysis")

# Health check endpoint
@router.get("/health")
async def document_health_check():
    """Check if document analysis service is available"""
    return {
        "status": "healthy",
        "llm_available": llm_analyzer.is_available(),
        "supported_formats": [".pdf", ".png", ".jpg", ".jpeg", ".txt", ".doc", ".docx"],
        "timestamp": datetime.utcnow().isoformat()
    }
