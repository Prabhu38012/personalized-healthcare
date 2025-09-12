import os
import logging
from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator, model_validator
import joblib
import pandas as pd
import numpy as np

# Import authentication dependencies with absolute imports first
try:
    # Try absolute import first (more reliable)
    from backend.auth.routes import get_current_user, get_doctor_or_admin
    auth_available = True
except ImportError:
    try:
        # Fallback to relative import
        from auth.routes import get_current_user, get_doctor_or_admin
        auth_available = True
    except ImportError:
        # Fallback for when auth is not available
        auth_available = False
        async def get_current_user() -> Any:
            return {"id": "demo_user", "email": "demo@demo.com", "role": "patient"}
        async def get_doctor_or_admin() -> Any:
            return {"id": "demo_admin", "email": "admin@demo.com", "role": "admin"}

# Initialize logger first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EHRDataProcessor import with proper fallback handling
def get_ehr_processor():
    """Get EHRDataProcessor with fallback"""
    try:
        # Try relative import first (when running from backend directory)
        from utils.ehr_processor import EHRDataProcessor
        return EHRDataProcessor
    except ImportError:
        try:
            # Fallback to absolute import (when running from project root)
            from backend.utils.ehr_processor import EHRDataProcessor
            return EHRDataProcessor
        except ImportError:
            # Final fallback - create a dummy processor
            logger.warning("EHRDataProcessor not found, creating dummy processor")
            class DummyEHRDataProcessor:
                def process_single_record(self, fhir_bundle):
                    return pd.DataFrame()
            return DummyEHRDataProcessor

router = APIRouter()

# Global variables for model and preprocessor
model_data = None

# Try multiple possible model paths to handle different execution contexts
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "ehr_risk_model.pkl"),  # From backend/routes/
    os.path.join(os.getcwd(), "models", "ehr_risk_model.pkl"),  # From project root
    r"d:\personalized-healthcare\models\ehr_risk_model.pkl",  # Absolute path
    "models/ehr_risk_model.pkl",  # Relative from current directory
]

# Find the actual model path
MODEL_PATH = None
for path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break
        
if MODEL_PATH is None:
    MODEL_PATH = POSSIBLE_MODEL_PATHS[0]  # Use first path as fallback
    logger.warning(f"Model file not found in any expected location. Will use fallback: {MODEL_PATH}")

@router.get("/reload-model")
async def reload_model(current_user = Depends(get_doctor_or_admin) if auth_available else None):
    """Force reload the model (requires doctor or admin access if auth is enabled)"""
    global model_data
    model_data = None
    try:
        load_model()
        return {"message": "Model reloaded successfully", "status": "success"}
    except Exception as e:
        return {"message": f"Failed to reload model: {str(e)}", "status": "error"}

class FHIRPatientData(BaseModel):
    fhir_bundle: Dict

class SimplePatientData(BaseModel):
    """Patient data model optimized for EHR dataset features
    
    Based on your EHR dataset's top predictive features:
    - birth_date (22.3%) - Most important feature
    - SystolicBloodPressure (16.9%) - Second most important  
    - DiastolicBloodPressure (8.5%) - Fourth most important
    - BodyWeight (7.0%) - Fifth most important
    - BodyHeight (5.7%) - Sixth most important
    """
    # TOP PRIORITY - Most predictive EHR features (with backward compatibility)
    age: int = Field(..., ge=1, le=120, description="Patient age in years (used to calculate birth_date - 22.3% importance)")
    systolic_bp: Optional[int] = Field(None, ge=80, le=250, description="Systolic blood pressure in mmHg (16.9% importance)")
    weight: float = Field(..., ge=30, le=300, description="Body weight in kg (7.0% importance)")
    height: float = Field(..., ge=100, le=250, description="Body height in cm (5.7% importance)")
    sex: str = Field(..., description="Patient sex (M/F)")
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v not in ['M', 'F']:
            raise ValueError('Sex must be either M or F')
        return v
    
    # SECONDARY PRIORITY - Important clinical markers
    diastolic_bp: Optional[int] = Field(None, ge=40, le=150, description="Diastolic blood pressure mmHg (8.5% importance) - calculated if not provided")
    bmi: Optional[float] = Field(None, ge=10, le=50, description="BMI - calculated if not provided (5.0% importance)")
    total_cholesterol: Optional[int] = Field(None, ge=100, le=600, description="Total cholesterol mg/dL (2.8% importance)")
    ldl_cholesterol: Optional[int] = Field(None, ge=50, le=300, description="LDL cholesterol mg/dL (2.9% importance)")
    hdl_cholesterol: Optional[int] = Field(None, ge=20, le=100, description="HDL cholesterol mg/dL")
    triglycerides: Optional[int] = Field(None, ge=50, le=500, description="Triglycerides mg/dL (3.1% importance)")
    
    # MEDICAL CONDITIONS - High impact on risk
    diabetes: Optional[int] = Field(0, ge=0, le=1, description="Diabetes status (1=diabetic, 0=non-diabetic)")
    smoking: Optional[int] = Field(0, ge=0, le=1, description="Smoking status (1=smoker, 0=non-smoker)")
    
    # OPTIONAL - Additional clinical data
    hba1c: Optional[float] = Field(None, ge=4.0, le=15.0, description="HbA1c percentage (diabetes marker)")
    family_history: Optional[int] = Field(0, ge=0, le=1, description="Family history of heart disease (1=yes, 0=no)")
    fasting_blood_sugar: Optional[int] = Field(0, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    
    # BACKWARD COMPATIBILITY - Legacy fields (automatically map to new fields)
    resting_bp: Optional[int] = Field(None, description="Legacy: Use systolic_bp instead - will auto-map")
    cholesterol: Optional[int] = Field(None, description="Legacy: Use total_cholesterol instead - will auto-map")
    max_heart_rate: Optional[int] = Field(None, ge=60, le=220, description="Maximum heart rate (low importance for EHR model)")
    chest_pain_type: Optional[str] = Field(None, description="Type of chest pain (not in EHR top features)")
    resting_ecg: Optional[str] = Field(None, description="Resting ECG (not in EHR top features)")
    exercise_angina: Optional[int] = Field(None, description="Exercise angina (not in EHR top features)")
    oldpeak: Optional[float] = Field(None, description="ST depression (not in EHR top features)")
    slope: Optional[str] = Field(None, description="ST slope (not in EHR top features)")
    ca: Optional[int] = Field(None, description="Fluoroscopy vessels (not in EHR top features)")
    thal: Optional[str] = Field(None, description="Thalassemia (not in EHR top features)")
    
    @model_validator(mode='after')
    def validate_legacy_fields(self):
        """Auto-map legacy fields to new fields for backward compatibility"""
        # Map resting_bp to systolic_bp if systolic_bp is not provided
        if self.systolic_bp is None and self.resting_bp is not None:
            self.systolic_bp = self.resting_bp
            
        # Map cholesterol to total_cholesterol if total_cholesterol is not provided
        if self.total_cholesterol is None and self.cholesterol is not None:
            self.total_cholesterol = self.cholesterol
            
        # Ensure we have minimum required values
        if self.systolic_bp is None:
            raise ValueError("Either systolic_bp or resting_bp must be provided")
        if self.total_cholesterol is None:
            raise ValueError("Either total_cholesterol or cholesterol must be provided")
            
        return self

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: str
    recommendations: List[str]
    risk_factors: List[str]
    confidence: float

def load_model():
    """Load the trained model and preprocessor"""
    global model_data
    
    if model_data is None:
        try:
            if MODEL_PATH and os.path.exists(MODEL_PATH):
                model_data = joblib.load(MODEL_PATH)
                logger.info("EHR model loaded successfully")
                
                # Safe model type logging with error handling
                try:
                    if isinstance(model_data, dict) and 'model' in model_data and model_data['model'] is not None:
                        logger.info(f"Model type: {type(model_data['model']).__name__}")
                    else:
                        logger.warning("Model data structure is unexpected or missing 'model' key")
                        logger.info(f"Available keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dictionary'}")
                except Exception as e:
                    logger.warning(f"Could not determine model type: {e}")
            else:
                logger.warning(f"Model file not found at {MODEL_PATH}")
                # Create a simple fallback model
                from sklearn.ensemble import RandomForestClassifier
                import numpy as np
                
                # Create a dummy model with the expected features
                # Use a more dynamic approach to avoid same results
                X_dummy = np.random.rand(100, 20)  # 20 features
                y_dummy = np.random.randint(0, 2, 100)
                fallback_model = RandomForestClassifier(n_estimators=10, random_state=None)  # Remove fixed seed
                fallback_model.fit(X_dummy, y_dummy)
                
                model_data = {
                    'model': fallback_model,
                    'model_type': 'FallbackRandomForest'
                }
                logger.warning("Using fallback model - predictions may not be accurate. Please train a proper model using train_config.py")
                    
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Create emergency fallback
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            
            X_dummy = np.random.rand(50, 20)
            y_dummy = np.random.randint(0, 2, 50)
            emergency_model = RandomForestClassifier(n_estimators=5, random_state=42)
            emergency_model.fit(X_dummy, y_dummy)
            
            model_data = {
                'model': emergency_model,
                'model_type': 'EmergencyFallback'
            }
            logger.info("Using emergency fallback model")
    
    return model_data


def calculate_simple_risk_score(patient_data: dict) -> float:
    """Calculate a simple risk score based on patient data (0-1 scale) for fallback prediction
    
    Prioritizes EHR dataset's top predictive features:
    - age/birth_date (22.3%), systolic_bp (16.9%), weight (7.0%), height (5.7%)
    """
    score = 0.0
    
    # Age factor (0-0.3) - Most important feature in EHR dataset
    age = patient_data.get('age', 40)
    if age > 70:
        score += 0.3
    elif age > 60:
        score += 0.25
    elif age > 50:
        score += 0.15
    elif age > 40:
        score += 0.1
    
    # Systolic blood pressure factor (0-0.25) - Second most important in EHR
    systolic_bp = patient_data.get('systolic_bp', patient_data.get('resting_bp', 120))
    if systolic_bp >= 160:
        score += 0.25
    elif systolic_bp >= 140:
        score += 0.2
    elif systolic_bp >= 130:
        score += 0.1
    
    # Total cholesterol factor (0-0.2) - Important EHR feature
    total_chol = patient_data.get('total_cholesterol', patient_data.get('cholesterol', 200))
    if total_chol >= 280:
        score += 0.2
    elif total_chol >= 240:
        score += 0.15
    elif total_chol >= 200:
        score += 0.05
    
    # BMI factor (0-0.15) - Calculated from weight/height (important EHR features)
    if 'bmi' in patient_data and patient_data['bmi']:
        bmi = patient_data['bmi']
    else:
        weight = patient_data.get('weight', 70)
        height_m = patient_data.get('height', 170) / 100
        bmi = weight / (height_m ** 2)
    
    if bmi >= 35:
        score += 0.15
    elif bmi >= 30:
        score += 0.1
    elif bmi >= 25:
        score += 0.05
    
    # Smoking factor (0-0.1)
    smoking = patient_data.get('smoking', 0)
    if smoking == 1:  # Smoker
        score += 0.1
    
    # Diabetes factor (0-0.1)
    diabetes = patient_data.get('diabetes', 0)
    if diabetes == 1:  # Diabetes
        score += 0.1
    
    # Family history (0-0.05)
    if patient_data.get('family_history', 0) == 1:
        score += 0.05
    
    # Diastolic BP factor (0-0.08) - Fourth most important EHR feature
    diastolic_bp = patient_data.get('diastolic_bp')
    if not diastolic_bp:
        # Calculate from systolic if not provided
        diastolic_bp = max(60, systolic_bp - 40)
    
    if diastolic_bp >= 100:
        score += 0.08
    elif diastolic_bp >= 90:
        score += 0.05
    
    # Legacy clinical assessment fields (lower weight for EHR model)
    chest_pain = patient_data.get('chest_pain_type')
    if chest_pain == 'typical':
        score += 0.03
    elif chest_pain == 'atypical':
        score += 0.02
    
    # Exercise angina (0-0.02)
    if patient_data.get('exercise_angina', 0) == 1:
        score += 0.02
    
    # Ensure score is between 0 and 1
    return min(max(score, 0.05), 0.95)  # Keep between 5% and 95%



def generate_recommendations(patient_data: dict, risk_prob: float) -> List[str]:
    """Generate personalized health recommendations based on patient data"""
    recommendations = []
    
    # Age-based recommendations
    age = patient_data.get('age', 0)
    if age > 65:
        recommendations.append("Annual comprehensive geriatric assessment recommended")
    elif age > 50:
        recommendations.append("Regular health screenings recommended based on age")

    # Blood pressure management (using new field names)
    systolic_bp = patient_data.get('systolic_bp', patient_data.get('resting_bp', 0))
    if systolic_bp >= 140:
        recommendations.append("Monitor blood pressure regularly and consult with your healthcare provider")
    elif systolic_bp >= 130:
        recommendations.append("Lifestyle modifications recommended for blood pressure management")
    
    # Cholesterol management (using new field names)
    total_cholesterol = patient_data.get('total_cholesterol', patient_data.get('cholesterol', 0))
    if total_cholesterol >= 240:
        recommendations.append("Discuss cholesterol management strategies with your doctor")
    elif total_cholesterol >= 200:
        recommendations.append("Consider cholesterol screening and dietary modifications")
    
    # BMI recommendations (calculate if needed)
    if 'bmi' in patient_data and patient_data['bmi']:
        bmi = patient_data['bmi']
    else:
        weight = patient_data.get('weight', 70)
        height_m = patient_data.get('height', 170) / 100
        bmi = weight / (height_m ** 2)
    
    if bmi >= 30:
        recommendations.append("Weight management program recommended for obesity")
    elif bmi >= 25:
        recommendations.append("Healthy weight maintenance recommended")
    
    # Risk-based recommendations
    if risk_prob > 0.7:
        recommendations.extend([
            "Immediate consultation with a healthcare provider recommended",
            "Consider comprehensive cardiovascular assessment",
            "Close monitoring of vital signs recommended"
        ])
    elif risk_prob > 0.4:
        recommendations.extend([
            "Schedule a follow-up with your healthcare provider",
            "Focus on modifiable risk factors",
            "Regular health monitoring recommended"
        ])
    else:
        recommendations.extend([
            "Maintain regular health check-ups",
            "Continue healthy lifestyle habits",
            "Preventive health measures recommended"
        ])
    
    # Ensure we have at least some recommendations
    if not recommendations:
        recommendations = [
            "Maintain regular health check-ups",
            "Follow a balanced diet and exercise regularly",
            "Monitor your health metrics regularly"
        ]
    
    return recommendations

def identify_ehr_risk_factors(patient_data: dict) -> List[str]:
    """Identify current risk factors based on EHR data"""
    risk_factors = []
    
    # Age risk factors
    if 'birth_date' in patient_data:
        try:
            birth_date = pd.to_datetime(patient_data['birth_date'])
            age = (pd.to_datetime('today') - birth_date).days / 365.25
            if age > 65:
                risk_factors.append(f"Advanced age ({int(age)} years)")
            elif age > 50:
                risk_factors.append(f"Middle age ({int(age)} years)")
        except:
            pass

    # Blood pressure risk factors
    systolic = patient_data.get('SystolicBloodPressure')
    diastolic = patient_data.get('DiastolicBloodPressure')
    
    if systolic and diastolic:
        if systolic >= 140 or diastolic >= 90:
            risk_factors.append(f"Hypertension ({systolic}/{diastolic} mmHg)")
        elif systolic >= 130 or diastolic >= 80:
            risk_factors.append(f"Elevated blood pressure ({systolic}/{diastolic} mmHg)")
    
    # Cholesterol risk factors
    cholesterol = patient_data.get('TotalCholesterol')
    if cholesterol and cholesterol >= 240:
        risk_factors.append(f"High cholesterol ({cholesterol} mg/dL)")

    if 'conditions' in patient_data:
        risk_factors.extend(patient_data['conditions'].split(' '))
    
    return risk_factors

def map_simple_to_ehr_features(patient_dict: dict) -> dict:
    """Map simple patient data to EHR feature format based on your actual dataset
    
    Uses the top predictive features from your EHR dataset:
    - birth_date (22.3%), SystolicBloodPressure (16.9%), patient_id (12.7%),
    - DiastolicBloodPressure (8.5%), BodyWeight (7.0%)
    """
    from datetime import datetime
    import numpy as np
    
    age = patient_dict.get('age', 30)
    
    # Create deterministic seed for consistent synthetic data
    patient_str = f"{age}_{patient_dict.get('sex', 'M')}_{patient_dict.get('systolic_bp', patient_dict.get('resting_bp', 120))}_{patient_dict.get('total_cholesterol', patient_dict.get('cholesterol', 200))}"
    np.random.seed(abs(hash(patient_str)) % 1000)
    
    ehr_data = {}
    
    # TOP PREDICTIVE FEATURES from your EHR dataset
    # 1. birth_date (22.3% importance) - Most important feature!
    birth_year = datetime.now().year - age
    ehr_data['birth_date'] = f"{birth_year}-01-01"
    
    # 2. SystolicBloodPressure (16.9% importance) - Second most important!
    ehr_data['SystolicBloodPressure'] = patient_dict.get('systolic_bp', patient_dict.get('resting_bp', 120))
    
    # 3. patient_id (12.7% importance) - Third most important!
    ehr_data['patient_id'] = str(abs(hash(patient_str)) % 100000)
    
    # 4. DiastolicBloodPressure (8.5% importance)
    if 'diastolic_bp' in patient_dict and patient_dict['diastolic_bp']:
        ehr_data['DiastolicBloodPressure'] = patient_dict['diastolic_bp']
    else:
        systolic = patient_dict.get('systolic_bp', patient_dict.get('resting_bp', 120))
        # Realistic diastolic calculation with variation based on risk
        base_diastolic = systolic - 40
        if patient_dict.get('diabetes', 0) or patient_dict.get('smoking', 0):
            base_diastolic += 5  # Higher diastolic for risk factors
        ehr_data['DiastolicBloodPressure'] = max(60, base_diastolic)
    
    # 5. BodyWeight (7.0% importance)
    ehr_data['BodyWeight'] = patient_dict.get('weight', 70)
    
    # Other important physical measurements
    ehr_data['BodyHeight'] = patient_dict.get('height', 170)
    
    # Calculate BMI if not provided
    if 'bmi' in patient_dict and patient_dict['bmi']:
        ehr_data['BodyMassIndex'] = patient_dict['bmi']
    else:
        weight = patient_dict.get('weight', 70)
        height_m = patient_dict.get('height', 170) / 100
        ehr_data['BodyMassIndex'] = weight / (height_m ** 2)
    
    # Cholesterol profile - important for cardiac risk
    ehr_data['TotalCholesterol'] = patient_dict.get('total_cholesterol', patient_dict.get('cholesterol', 200))
    ehr_data['LowDensityLipoproteinCholesterol'] = patient_dict.get('ldl_cholesterol', 100)
    ehr_data['HighDensityLipoproteinCholesterol'] = patient_dict.get('hdl_cholesterol', 50)
    ehr_data['Triglycerides'] = patient_dict.get('triglycerides', 150)
    
    # Diabetes marker
    hba1c_base = 5.5
    if patient_dict.get('hba1c'):
        hba1c_base = patient_dict['hba1c']
    elif patient_dict.get('diabetes', 0) == 1:
        hba1c_base = 7.0 + np.random.uniform(0, 1.5)  # Diabetic range
    ehr_data['HemoglobinA1cHemoglobintotalinBlood'] = hba1c_base
    
    # Gender mapping
    ehr_data['gender'] = patient_dict.get('sex', 'M')
    
    # Conditions - very important for risk assessment
    conditions = []
    if patient_dict.get('diabetes', 0) == 1:
        conditions.append('diabetes')
    if patient_dict.get('smoking', 0) == 1:
        conditions.append('smoking')
    systolic = patient_dict.get('systolic_bp', patient_dict.get('resting_bp', 120))
    if systolic >= 140:
        conditions.append('hypertension')
    total_chol = patient_dict.get('total_cholesterol', patient_dict.get('cholesterol', 200))
    if total_chol >= 240:
        conditions.append('dyslipidemia')
    bmi = ehr_data['BodyMassIndex']
    if bmi >= 30:
        conditions.append('obesity')
    
    ehr_data['conditions'] = ' '.join(conditions) if conditions else 'none'
    
    # Lab values with risk-adjusted ranges
    diabetes_multiplier = 1.3 if patient_dict.get('diabetes', 0) else 1.0
    
    # Core lab values
    ehr_data['Glucose'] = (85 * diabetes_multiplier) + np.random.uniform(-15, 25)
    ehr_data['Creatinine'] = 0.9 + (0.2 if age > 65 else 0) + np.random.uniform(-0.1, 0.2)
    ehr_data['Calcium'] = 9.5 + np.random.uniform(-0.3, 0.3)
    ehr_data['Sodium'] = 140 + np.random.uniform(-3, 3)
    ehr_data['Potassium'] = 4.0 + np.random.uniform(-0.3, 0.3)
    ehr_data['Chloride'] = 100 + np.random.uniform(-3, 3)
    ehr_data['CarbonDioxide'] = 24 + np.random.uniform(-2, 2)
    
    # Kidney function
    age_factor = max(0.7, 1.2 - (age / 100))  # Decline with age
    ehr_data['EstimatedGlomerularFiltrationRate'] = 90 * age_factor + np.random.uniform(-10, 10)
    
    # Lung function
    smoking_penalty = 0.1 if patient_dict.get('smoking', 0) else 0
    ehr_data['FEV1FVC'] = (0.8 - smoking_penalty) + np.random.uniform(-0.05, 0.05)
    
    # Allergy markers (typically low unless allergic)
    allergy_features = [
        'AmericanhousedustmiteIgEAbinSerum', 'CatdanderIgEAbinSerum',
        'CladosporiumherbarumIgEAbinSerum', 'CodfishIgEAbinSerum',
        'CommonRagweedIgEAbinSerum', 'CowmilkIgEAbinSerum', 'EggwhiteIgEAbinSerum'
    ]
    
    for feature in allergy_features:
        ehr_data[feature] = np.random.exponential(2)  # Exponential distribution for allergy markers
    
    # Social/demographic factors
    ehr_data['AbuseStatusOMAHA'] = np.random.uniform(0, 10)
    ehr_data['AreyoucoveredbyhealthinsuranceorsomeotherkindofhealthcareplanPhenX'] = np.random.uniform(80, 100)
    
    # Add the exact missing features that the model was trained on
    missing_trained_features = [
        'MicroalbuminCreatineRatio', 'Oraltemperature', 'PeanutIgEAbinSerum',
        'PolypsizegreatestdimensionbyCAPcancerprotocols', 'Sexualorientation',
        'ShrimpIgEAbinSerum', 'SoybeanIgEAbinSerum', 'TotalscoreMMSE',
        'WalnutIgEAbinSerum', 'WheatIgEAbinSerum', 'WhiteoakIgEAbinSerum'
    ]
    
    for feature in missing_trained_features:
        if 'IgE' in feature:
            ehr_data[feature] = np.random.exponential(2)  # Allergy markers
        elif feature == 'MicroalbuminCreatineRatio':
            ehr_data[feature] = np.random.uniform(0, 30)  # Normal kidney function
        elif feature == 'Oraltemperature':
            ehr_data[feature] = np.random.uniform(97, 99)  # Normal temperature
        elif feature == 'PolypsizegreatestdimensionbyCAPcancerprotocols':
            ehr_data[feature] = 0  # No polyps for most people
        elif feature == 'Sexualorientation':
            ehr_data[feature] = np.random.choice([0, 1, 2])  # Various orientations
        elif feature == 'TotalscoreMMSE':
            ehr_data[feature] = np.random.uniform(24, 30)  # Normal cognitive function
        else:
            ehr_data[feature] = np.random.uniform(0, 5)  # Generic default
    
    return ehr_data
    
@router.post("/predict-simple", response_model=PredictionResponse)
async def predict_health_risk_simple(patient: SimplePatientData, current_user = Depends(get_current_user) if auth_available else None):
    """Predict health risk for a patient using simple data format (requires authentication if enabled)"""
    try:
        logger.info("=== STARTING PREDICTION ===")
        # Load model
        model_data = load_model()
        model = model_data['model']
        
        # Log which model is being used for debugging
        model_type = model_data.get('model_type', type(model).__name__)
        logger.info(f"Using model type: {model_type}")
        logger.info(f"Model data keys: {list(model_data.keys())}")
        
        # Convert simple data to model format
        patient_dict = patient.dict()
        
        # Check if we have the EHR model with preprocessor
        if 'preprocessor' in model_data and 'feature_columns' in model_data:
            # Map simple patient data to EHR format
            ehr_data = map_simple_to_ehr_features(patient_dict)
            logger.info(f"Mapped to EHR features: {list(ehr_data.keys())[:10]}...")
            
            # Use preprocessor to transform data
            try:
                preprocessor = model_data['preprocessor']
                expected_features = model_data['feature_columns']
                
                # Create DataFrame with EHR data
                df = pd.DataFrame([ehr_data])
                logger.info(f"Generated features: {len(df.columns)}")
                
                # Ensure we have exactly the features the model expects
                aligned_df = pd.DataFrame()
                for feature in expected_features:
                    if feature in df.columns:
                        aligned_df[feature] = df[feature]
                    else:
                        # Set default values for missing features
                        import numpy as np
                        if 'IgE' in feature:
                            aligned_df[feature] = [np.random.exponential(2)]
                        elif feature in ['MicroalbuminCreatineRatio']:
                            aligned_df[feature] = [np.random.uniform(0, 30)]
                        elif feature in ['Oraltemperature']:
                            aligned_df[feature] = [np.random.uniform(97, 99)]
                        elif feature in ['TotalscoreMMSE']:
                            aligned_df[feature] = [np.random.uniform(24, 30)]
                        else:
                            aligned_df[feature] = [0]
                
                logger.info(f"Aligned features: {len(aligned_df.columns)}")
                
                # Apply the same preprocessing steps as during training
                df_processed = preprocessor.clean_data(aligned_df.copy())
                df_encoded = preprocessor.encode_categorical_features(df_processed, fit=False)
                df_scaled = preprocessor.scale_features(df_encoded, fit=False)
                
                # Final check - ensure exact feature match
                if len(df_scaled.columns) != len(expected_features):
                    logger.warning(f"Feature count mismatch: {len(df_scaled.columns)} vs {len(expected_features)}")
                    # Force exact alignment
                    final_df = pd.DataFrame()
                    for feature in expected_features:
                        if feature in df_scaled.columns:
                            final_df[feature] = df_scaled[feature]
                        else:
                            final_df[feature] = [0]
                    df_scaled = final_df
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    risk_probability = float(model.predict_proba(df_scaled.values)[0][1])
                    logger.info(f"EHR model prediction: {risk_probability:.3f}")
                else:
                    risk_probability = float(model.predict(df_scaled.values)[0])
                    
            except Exception as transform_error:
                logger.error(f"EHR transformation error: {str(transform_error)}")
                # Fall back to simple calculation
                risk_probability = calculate_simple_risk_score(patient_dict)
                logger.info(f"Using fallback calculation: {risk_probability:.3f}")
                
        elif 'label_encoders' in model_data and 'feature_columns' in model_data:
            # Use the label encoder format
            df = pd.DataFrame([patient_dict])
            
            # Apply label encoding
            for feature, encoder in model_data['label_encoders'].items():
                if feature in df.columns:
                    try:
                        df[feature] = encoder.transform(df[feature].astype(str))
                    except ValueError:
                        # Handle unknown categories
                        df[feature] = 0
            
            # Ensure all features are present
            for col in model_data['feature_columns']:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training
            df = df[model_data['feature_columns']]
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                risk_probability = float(model.predict_proba(df)[0][1])
            else:
                risk_probability = float(model.predict(df)[0])
                
        elif model_data.get('model_type') in ['FallbackRandomForest', 'EmergencyFallback']:
            # Use a simple rule-based prediction for fallback models to provide dynamic results
            risk_probability = calculate_simple_risk_score(patient_dict)
            logger.info(f"Using simple risk calculation: {risk_probability:.3f}")
        else:
            # Fallback to old method for older models
            categorical_mappings = {
                'sex': {'M': 1, 'F': 0},
                'chest_pain_type': {'typical': 0, 'atypical': 1, 'non_anginal': 2, 'asymptomatic': 3},
                'resting_ecg': {'normal': 0, 'abnormal': 1, 'hypertrophy': 2},
                'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
                'thal': {'normal': 0, 'fixed': 1, 'reversible': 2}
            }
            
            # Convert categorical features
            processed_data = patient_dict.copy()
            for feature, mapping in categorical_mappings.items():
                if feature in processed_data:
                    processed_data[feature] = mapping.get(processed_data[feature], 0)
            
            # Create feature array in the correct order
            feature_order = [
                'age', 'sex', 'height', 'weight', 'bmi', 'resting_bp', 'max_heart_rate',
                'cholesterol', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
                'fasting_blood_sugar', 'hba1c', 'chest_pain_type', 'resting_ecg',
                'exercise_angina', 'oldpeak', 'slope', 'ca', 'thal'
            ]
            
            feature_vector = []
            for feature in feature_order:
                value = processed_data.get(feature, 0)
                feature_vector.append(float(value))
            
            # Make prediction
            import numpy as np
            feature_array = np.array([feature_vector])
            
            if hasattr(model, 'predict_proba'):
                risk_probability = float(model.predict_proba(feature_array)[0][1])
            else:
                risk_probability = float(model.predict(feature_array)[0])
        
        # Calculate confidence
        confidence = abs(risk_probability - 0.5) * 2
        
        # Determine risk level
        if risk_probability > 0.7:
            risk_level = "High"
        elif risk_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate recommendations and identify risk factors
        recommendations = generate_recommendations(patient_dict, risk_probability)
        risk_factors = identify_simple_risk_factors(patient_dict)
        
        return PredictionResponse(
            risk_probability=float(risk_probability),
            risk_level=risk_level,
            recommendations=recommendations,
            risk_factors=risk_factors,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Simple prediction error: {str(e)}")
        # Return a default response if prediction fails
        return PredictionResponse(
            risk_probability=0.3,
            risk_level="Low",
            recommendations=[
                "Unable to process prediction at this time",
                "Please consult with your healthcare provider",
                "Regular health check-ups recommended"
            ],
            risk_factors=["Assessment temporarily unavailable"],
            confidence=0.5
        )
def identify_simple_risk_factors(patient_data: dict) -> List[str]:
    """Identify current risk factors based on simple patient data"""
    risk_factors = []
    
    # Age risk factors
    age = patient_data.get('age', 0)
    if age > 65:
        risk_factors.append(f"Advanced age ({age} years)")
    elif age > 50:
        risk_factors.append(f"Middle age ({age} years)")
    
    # Blood pressure risk factors
    systolic_bp = patient_data.get('systolic_bp', patient_data.get('resting_bp', 0))
    if systolic_bp and systolic_bp >= 140:
        risk_factors.append(f"Hypertension ({systolic_bp} mmHg)")
    elif systolic_bp and systolic_bp >= 130:
        risk_factors.append(f"Elevated blood pressure ({systolic_bp} mmHg)")
    
    # Cholesterol risk factors
    cholesterol = patient_data.get('total_cholesterol', patient_data.get('cholesterol', 0))
    if cholesterol and cholesterol >= 240:
        risk_factors.append(f"High total cholesterol ({cholesterol} mg/dL)")
    elif cholesterol and cholesterol >= 200:
        risk_factors.append(f"Borderline high cholesterol ({cholesterol} mg/dL)")
    
    ldl = patient_data.get('ldl_cholesterol', 0)
    if ldl >= 160:
        risk_factors.append(f"Very high LDL cholesterol ({ldl} mg/dL)")
    elif ldl >= 130:
        risk_factors.append(f"High LDL cholesterol ({ldl} mg/dL)")
    
    hdl = patient_data.get('hdl_cholesterol', 0)
    if hdl < 40:
        risk_factors.append(f"Low HDL cholesterol ({hdl} mg/dL)")
    
    triglycerides = patient_data.get('triglycerides', 0)
    if triglycerides >= 200:
        risk_factors.append(f"High triglycerides ({triglycerides} mg/dL)")
    
    # BMI risk factors
    bmi = patient_data.get('bmi', 0)
    if bmi >= 30:
        risk_factors.append(f"Obesity (BMI: {bmi:.1f})")
    elif bmi >= 25:
        risk_factors.append(f"Overweight (BMI: {bmi:.1f})")
    
    # Blood sugar risk factors
    fasting_glucose = patient_data.get('fasting_blood_sugar', 0)
    if isinstance(fasting_glucose, int) and fasting_glucose > 0:
        risk_factors.append("Elevated fasting blood sugar")
    
    hba1c = patient_data.get('hba1c', 0)
    if hba1c >= 6.5:
        risk_factors.append(f"Diabetes (HbA1c: {hba1c}%)")
    elif hba1c >= 5.7:
        risk_factors.append(f"Prediabetes (HbA1c: {hba1c}%)")
    
    # Clinical assessment risk factors
    if patient_data.get('chest_pain_type') == 'typical':
        risk_factors.append("Typical chest pain")
    
    if patient_data.get('exercise_angina', 0) == 1:
        risk_factors.append("Exercise-induced angina")
    
    oldpeak = patient_data.get('oldpeak', 0)
    if oldpeak > 2.0:
        risk_factors.append(f"Significant ST depression ({oldpeak})")
    
    ca = patient_data.get('ca', 0)
    if ca > 0:
        risk_factors.append(f"Coronary artery disease ({ca} vessels affected)")
    
    if patient_data.get('thal') == 'reversible':
        risk_factors.append("Reversible defect in stress test")
    
    # Lifestyle risk factors (if available)
    smoking = patient_data.get('smoking', 0)
    if smoking == 2:
        risk_factors.append("Current smoker")
    elif smoking == 1:
        risk_factors.append("Former smoker")
    
    diabetes = patient_data.get('diabetes', 0)
    if diabetes == 1:
        risk_factors.append("Diabetes mellitus")
    elif diabetes == 2:
        risk_factors.append("Prediabetes")
    
    if patient_data.get('family_history', 0) == 1:
        risk_factors.append("Family history of heart disease")
    
    return risk_factors

@router.post("/predict", response_model=PredictionResponse)
async def predict_health_risk(patient: FHIRPatientData, current_user = Depends(get_current_user) if auth_available else None):
    """Predict health risk for a patient using EHR data (requires authentication if enabled)"""
    try:
        # Load model
        model_data = load_model()
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        feature_columns = preprocessor.feature_columns
        
        # Process the raw FHIR bundle
        ehr_processor_class = get_ehr_processor()
        ehr_processor = ehr_processor_class()
        patient_df = ehr_processor.process_single_record(patient.fhir_bundle)

        if patient_df.empty:
            raise HTTPException(status_code=400, detail="Could not process the provided FHIR data.")

        # Align columns with the training data
        aligned_df = pd.DataFrame(columns=feature_columns)
        for col in feature_columns:
            if col in patient_df.columns:
                aligned_df[col] = patient_df[col]
            else:
                aligned_df[col] = 0 # or np.nan

        # Prepare features for the model
        processed_data, _ = preprocessor.prepare_features(aligned_df, fit=False)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            risk_probability = model.predict_proba(processed_data)[0][1]
        else:
            risk_probability = model.predict(processed_data)[0]
        
        # Calculate confidence (based on probability distance from 0.5)
        confidence = abs(risk_probability - 0.5) * 2
        
        # Determine risk level
        if risk_probability > 0.7:
            risk_level = "High"
        elif risk_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate recommendations and identify risk factors
        patient_dict = aligned_df.to_dict(orient='records')[0]
        recommendations = generate_recommendations(patient_dict, risk_probability)
        risk_factors = identify_ehr_risk_factors(patient_dict)
        
        return PredictionResponse(
            risk_probability=float(risk_probability),
            risk_level=risk_level,
            recommendations=recommendations,
            risk_factors=risk_factors,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded EHR model"""
    try:
        model_data = load_model()
        
        # Get model type safely
        model_type = model_data.get('model_type')
        if not model_type:
            model_type = type(model_data['model']).__name__
        
        # Get feature information safely
        feature_columns = []
        feature_count = 0
        
        if 'feature_columns' in model_data:
            feature_columns = model_data['feature_columns']
            feature_count = len(feature_columns)
        elif 'preprocessor' in model_data and hasattr(model_data['preprocessor'], 'feature_columns'):
            feature_columns = model_data['preprocessor'].feature_columns
            feature_count = len(feature_columns)
        elif hasattr(model_data['model'], 'feature_names_in_'):
            feature_columns = list(model_data['model'].feature_names_in_)
            feature_count = len(feature_columns)
        else:
            # Fallback for models without feature info
            feature_columns = ["Feature information not available"]
            feature_count = 0
        
        # Get additional model information
        model_info = {
            "model_type": model_type,
            "feature_count": feature_count,
            "features": feature_columns[:10] if len(feature_columns) > 10 else feature_columns,  # Limit to first 10 features
            "total_features": feature_count,
            "status": "loaded",
            "dataset": "EHR Data from Kaggle",
            "model_keys": list(model_data.keys())
        }
        
        # Add performance metrics if available
        if 'metrics' in model_data:
            model_info['performance'] = model_data['metrics']
        
        # Add training info if available
        if 'training_info' in model_data:
            model_info['training_info'] = model_data['training_info']
            
        return model_info
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        # Return a safe fallback response instead of raising error
        return {
            "model_type": "Unknown",
            "feature_count": 0,
            "features": ["Model information temporarily unavailable"],
            "total_features": 0,
            "status": "error",
            "dataset": "EHR Data from Kaggle",
            "error": str(e)
        }

@router.get("/health")
async def health_check():
    """Health check endpoint for the prediction API"""
    try:
        # Try to load model to verify it's working
        model_data = load_model()
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": model_data.get('model_type', type(model_data['model']).__name__),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@router.post("/batch-predict")
async def batch_predict_health_risk(patients: List[FHIRPatientData], current_user = Depends(get_doctor_or_admin) if auth_available else None):
    """Predict health risk for multiple patients using EHR data (requires doctor or admin access if auth enabled)"""
    try:
        predictions = []
        
        for patient in patients:
            # Convert single patient data to the expected format
            patient_dict = patient.fhir_bundle if hasattr(patient, 'fhir_bundle') else patient.dict()
            
            # Use the existing prediction logic
            prediction_response = await predict_health_risk(FHIRPatientData(fhir_bundle=patient_dict))
            predictions.append(prediction_response)
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")