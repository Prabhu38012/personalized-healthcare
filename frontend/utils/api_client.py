"""
API utilities for frontend communication with backend services
"""
import requests
import streamlit as st
import time
from .caching import cached_health_check, cached_model_info


class HealthcareAPI:
    """Healthcare API client for making requests to the backend"""
    
    def __init__(self, base_url="http://localhost:8000/api"):
        self.base_url = base_url
        self.auth_headers = {}
    
    def set_auth_headers(self, headers):
        """Set authentication headers"""
        self.auth_headers = headers or {}
    
    def make_prediction(self, patient_data, force_llm=False):
        """Make API call to get prediction with only expected features
        
        Args:
            patient_data (dict): Dictionary containing patient data
            force_llm (bool): Whether to force LLM analysis even if it's disabled by default
            
        Returns:
            dict: Prediction results including risk assessment and recommendations
        """
        # Define the expected features based on the backend's PatientData model
        expected_features = [
            # Basic Information
            'age', 'sex', 'height', 'weight', 'bmi',
            # Vital Signs
            'resting_bp', 'cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
            'triglycerides', 'fasting_blood_sugar', 'hba1c',
            # Medical History
            'chest_pain_type', 'resting_ecg', 'max_heart_rate', 'exercise_angina',
            'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Filter the patient data to only include expected features
        filtered_data = {k: v for k, v in patient_data.items() if k in expected_features}
        
        # Add force_llm flag to the request data
        filtered_data['force_llm'] = force_llm
        
        # Convert numeric fields to appropriate types
        numeric_fields = ['age', 'resting_bp', 'cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
                         'triglycerides', 'fasting_blood_sugar', 'max_heart_rate', 'ca']
        
        for field in numeric_fields:
            if field in filtered_data:
                try:
                    filtered_data[field] = int(float(filtered_data[field]))
                except (ValueError, TypeError):
                    pass
        
        # Convert float fields
        float_fields = ['bmi', 'hba1c', 'oldpeak']
        for field in float_fields:
            if field in filtered_data:
                try:
                    filtered_data[field] = float(filtered_data[field])
                except (ValueError, TypeError):
                    pass
        
        try:
            headers = {**self.auth_headers, "Content-Type": "application/json"}
            try:
                response = requests.post(
                    f"{self.base_url}/predict/simple",
                    json=filtered_data,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error occurred: {http_err} - {response.text if 'response' in locals() else 'No response'}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the prediction service. Please ensure the backend server is running.\nError: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    def check_backend_health(self):
        """Check if backend is healthy - using global cached function"""
        return cached_health_check(self.base_url)
    
    def get_model_info(self):
        """Get model information from backend - using global cached function"""
        return cached_model_info(self.base_url)
    
    def reload_model(self):
        """Force reload the model on backend"""
        try:
            response = requests.get(f"{self.base_url}/reload-model", timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None