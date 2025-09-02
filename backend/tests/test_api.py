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
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    """Test prediction endpoint with valid data"""
    patient_data = {
        "fhir_bundle": {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient1",
                        "gender": "male",
                        "birthDate": "1978-01-01"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"display": "Systolic Blood Pressure"}]},
                        "valueQuantity": {"value": 130}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"display": "Total Cholesterol"}]},
                        "valueQuantity": {"value": 200}
                    }
                }
            ]
        }
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
            "fhir_bundle": {
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "patient1",
                            "gender": "male",
                            "birthDate": "1978-01-01"
                        }
                    },
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "code": {"coding": [{"display": "Systolic Blood Pressure"}]},
                            "valueQuantity": {"value": 130}
                        }
                    }
                ]
            }
        },
        {
            "fhir_bundle": {
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "patient2",
                            "gender": "female",
                            "birthDate": "1963-01-01"
                        }
                    },
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "code": {"coding": [{"display": "Systolic Blood Pressure"}]},
                            "valueQuantity": {"value": 160}
                        }
                    }
                ]
            }
        }
    ]
    
    response = client.post("/api/batch-predict", json=patients_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 2
