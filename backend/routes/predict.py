from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import joblib
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Global variables for model and preprocessor
model_data = None
MODEL_PATH = "models/risk_model.pkl"

class PatientData(BaseModel):
    age: int
    sex: str
    chest_pain_type: str
    resting_bp: int
    cholesterol: int
    fasting_blood_sugar: int
    resting_ecg: str
    max_heart_rate: int
    exercise_angina: int
    oldpeak: float
    slope: str
    ca: int
    thal: str
    bmi: Optional[float] = 25.0
    smoking: Optional[int] = 0
    diabetes: Optional[int] = 0
    family_history: Optional[int] = 0

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: str
    recommendations: List[str]
    risk_factors: List[str]

def load_model():
    """Load the trained model and preprocessor"""
    global model_data
    
    if model_data is None:
        try:
            if os.path.exists(MODEL_PATH):
                model_data = joblib.load(MODEL_PATH)
                logger.info("Model loaded successfully")
            else:
                logger.warning("Model file not found. Training new model...")
                # Import and train model if not exists
                from models.train_model import main as train_model
                model, preprocessor = train_model()
                model_data = {
                    'model': model,
                    'preprocessor': preprocessor,
                    'feature_columns': preprocessor.feature_columns
                }
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    
    return model_data

def generate_recommendations(patient_data: dict, risk_prob: float) -> List[str]:
    """Generate personalized health recommendations"""
    recommendations = []
    
    # Age-based recommendations
    if patient_data['age'] > 50:
        recommendations.append("Regular cardiac screening recommended due to age")
    
    # Blood pressure recommendations
    if patient_data['resting_bp'] > 140:
        recommendations.append("Monitor blood pressure regularly and consider lifestyle modifications")
    
    # Cholesterol recommendations
    if patient_data['cholesterol'] > 240:
        recommendations.append("Adopt a heart-healthy diet low in saturated fats")
    
    # BMI recommendations
    if patient_data.get('bmi', 25) > 30:
        recommendations.append("Weight management through diet and exercise")
    
    # Smoking recommendations
    if patient_data.get('smoking', 0) == 1:
        recommendations.append("Smoking cessation programs strongly recommended")
    
    # Exercise recommendations
    if patient_data['max_heart_rate'] < 100:
        recommendations.append("Gradual increase in physical activity under medical supervision")
    
    # General recommendations based on risk level
    if risk_prob > 0.7:
        recommendations.extend([
            "Immediate consultation with cardiologist recommended",
            "Consider stress testing and advanced cardiac imaging"
        ])
    elif risk_prob > 0.4:
        recommendations.extend([
            "Regular follow-up with primary care physician",
            "Annual cardiac risk assessment"
        ])
    else:
        recommendations.extend([
            "Maintain current healthy lifestyle",
            "Annual routine health checkup"
        ])
    
    return recommendations

def identify_risk_factors(patient_data: dict) -> List[str]:
    """Identify current risk factors"""
    risk_factors = []
    
    if patient_data['age'] > 50:
        risk_factors.append("Advanced age")
    
    if patient_data['resting_bp'] > 140:
        risk_factors.append("High blood pressure")
    
    if patient_data['cholesterol'] > 240:
        risk_factors.append("High cholesterol")
    
    if patient_data.get('smoking', 0) == 1:
        risk_factors.append("Smoking")
    
    if patient_data.get('diabetes', 0) == 1:
        risk_factors.append("Diabetes")
    
    if patient_data.get('family_history', 0) == 1:
        risk_factors.append("Family history of heart disease")
    
    if patient_data.get('bmi', 25) > 30:
        risk_factors.append("Obesity")
    
    if patient_data['exercise_angina'] == 1:
        risk_factors.append("Exercise-induced angina")
    
    return risk_factors

@router.post("/predict", response_model=PredictionResponse)
async def predict_health_risk(patient: PatientData):
    """Predict health risk for a patient"""
    try:
        # Load model
        model_data = load_model()
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        
        # Convert patient data to dictionary
        patient_dict = patient.dict()
        
        # Preprocess data
        processed_data = preprocessor.transform_single_record(patient_dict)
        
        # Make prediction
        risk_probability = model.predict_proba(processed_data)[0][1]
        
        # Determine risk level
        if risk_probability > 0.7:
            risk_level = "High"
        elif risk_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate recommendations and identify risk factors
        recommendations = generate_recommendations(patient_dict, risk_probability)
        risk_factors = identify_risk_factors(patient_dict)
        
        return PredictionResponse(
            risk_probability=float(risk_probability),
            risk_level=risk_level,
            recommendations=recommendations,
            risk_factors=risk_factors
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        model_data = load_model()
        
        return {
            "model_type": type(model_data['model']).__name__,
            "feature_count": len(model_data['feature_columns']),
            "features": model_data['feature_columns']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/batch-predict")
async def batch_predict(patients: List[PatientData]):
    """Predict health risk for multiple patients"""
    try:
        results = []
        
        for patient in patients:
            prediction = await predict_health_risk(patient)
            results.append({
                "patient_id": len(results) + 1,
                "prediction": prediction
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
