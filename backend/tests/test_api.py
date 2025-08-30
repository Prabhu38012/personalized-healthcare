import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Healthcare Recommendation API is running"}

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    """Test prediction endpoint with valid data"""
    patient_data = {
        "age": 45,
        "sex": "M",
        "chest_pain_type": "typical",
        "resting_bp": 130,
        "cholesterol": 200,
        "fasting_blood_sugar": 0,
        "resting_ecg": "normal",
        "max_heart_rate": 150,
        "exercise_angina": 0,
        "oldpeak": 1.0,
        "slope": "upsloping",
        "ca": 0,
        "thal": "normal",
        "bmi": 25.0,
        "smoking": 0,
        "diabetes": 0,
        "family_history": 0
    }
    
    response = client.post("/api/predict", json=patient_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "risk_probability" in result
    assert "risk_level" in result
    assert "recommendations" in result
    assert "risk_factors" in result
    assert 0 <= result["risk_probability"] <= 1

def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    
    result = response.json()
    assert "model_type" in result
    assert "feature_count" in result
    assert "features" in result

def test_invalid_patient_data():
    """Test prediction with invalid data"""
    invalid_data = {
        "age": "invalid",  # Should be int
        "sex": "M"
    }
    
    response = client.post("/api/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_batch_predict():
    """Test batch prediction endpoint"""
    patients_data = [
        {
            "age": 45,
            "sex": "M",
            "chest_pain_type": "typical",
            "resting_bp": 130,
            "cholesterol": 200,
            "fasting_blood_sugar": 0,
            "resting_ecg": "normal",
            "max_heart_rate": 150,
            "exercise_angina": 0,
            "oldpeak": 1.0,
            "slope": "upsloping",
            "ca": 0,
            "thal": "normal"
        },
        {
            "age": 60,
            "sex": "F",
            "chest_pain_type": "atypical",
            "resting_bp": 160,
            "cholesterol": 280,
            "fasting_blood_sugar": 1,
            "resting_ecg": "abnormal",
            "max_heart_rate": 120,
            "exercise_angina": 1,
            "oldpeak": 2.5,
            "slope": "flat",
            "ca": 2,
            "thal": "reversible"
        }
    ]
    
    response = client.post("/api/batch-predict", json=patients_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 2
