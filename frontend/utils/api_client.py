"""
API utilities for frontend communication with backend services
"""
import requests
import streamlit as st
import time
from .caching import cached_health_check, cached_model_info


class APIClient:
    """General API client for backend communication"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.auth_headers = {}
    
    def set_auth_headers(self, headers):
        """Set authentication headers"""
        self.auth_headers = headers or {}


class HealthcareAPI:
    """Healthcare API client for making requests to the backend"""
    
    def __init__(self, base_url="http://localhost:8000/api"):
        self.base_url = base_url
        self.auth_headers = {}
    
    def set_auth_headers(self, headers):
        """Set authentication headers"""
        self.auth_headers = headers or {}
    
    def make_lab_prediction(self, patient_data, force_llm=False):
        """Make API call to get lab-enhanced AI prediction
        
        Args:
            patient_data (dict): Dictionary containing patient data with lab values
            force_llm (bool): Whether to force LLM analysis even if it's disabled by default
            
        Returns:
            dict: Lab-enhanced prediction results including risk assessment and recommendations from AI
        """
        # Include all patient data for AI analysis - it can handle comprehensive lab data
        filtered_data = dict(patient_data)  # Create a copy
        
        # Add force_llm flag to the request data
        filtered_data['force_llm'] = force_llm
        
        # Convert numeric fields to appropriate types for better AI analysis
        numeric_fields = ['age', 'resting_bp', 'systolic_bp', 'diastolic_bp', 'cholesterol', 
                         'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
                         'triglycerides', 'fasting_blood_sugar', 'max_heart_rate', 'ca',
                         'hemoglobin', 'total_leukocyte_count', 'red_blood_cell_count',
                         'hematocrit', 'platelet_count']
        
        for field in numeric_fields:
            if field in filtered_data and filtered_data[field] is not None:
                try:
                    filtered_data[field] = int(float(filtered_data[field]))
                except (ValueError, TypeError):
                    pass
        
        # Convert float fields including lab parameters
        float_fields = ['bmi', 'hba1c', 'oldpeak', 'weight', 'height',
                       'mean_corpuscular_volume', 'mean_corpuscular_hb', 'mean_corpuscular_hb_conc',
                       'red_cell_distribution_width', 'mean_platelet_volume', 'platelet_distribution_width',
                       'erythrocyte_sedimentation_rate', 'neutrophils_percent', 'lymphocytes_percent',
                       'monocytes_percent', 'eosinophils_percent', 'basophils_percent',
                       'absolute_neutrophil_count', 'absolute_lymphocyte_count', 'absolute_monocyte_count',
                       'absolute_eosinophil_count', 'absolute_basophil_count']
        for field in float_fields:
            if field in filtered_data and filtered_data[field] is not None:
                try:
                    filtered_data[field] = float(filtered_data[field])
                except (ValueError, TypeError):
                    pass
        
        # Map legacy field names to new ones for better compatibility
        if 'resting_bp' in filtered_data and 'systolic_bp' not in filtered_data:
            filtered_data['systolic_bp'] = filtered_data['resting_bp']
        if 'cholesterol' in filtered_data and 'total_cholesterol' not in filtered_data:
            filtered_data['total_cholesterol'] = filtered_data['cholesterol']
        
        try:
            headers = {**self.auth_headers, "Content-Type": "application/json"}
            
            # Use traditional model only for fast, consistent results
            filtered_data['force_llm'] = False
            
            with st.spinner("🔬 Analyzing lab data (traditional model)..."):
                response = requests.post(
                    f"{self.base_url}/predict/lab",
                    json=filtered_data,
                    headers=headers,
                    timeout=30  # Short timeout for traditional analysis
                )
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                
                result = response.json()
                
                # Add flags to indicate this is traditional lab prediction
                if result:
                    result['powered_by_ai'] = False
                    result['lab_enhanced'] = True
                    result['analysis_type'] = 'Traditional Lab Analysis'
                    
                return result
                
        except requests.exceptions.HTTPError as http_err:
            error_msg = f"HTTP error occurred: {http_err}"
            if 'response' in locals() and response.text:
                try:
                    error_detail = response.json().get('detail', response.text)
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text}"
            st.error(error_msg)
            return None
        except requests.exceptions.Timeout as e:
            st.error("⏱️ Analysis timed out. Please try again.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the AI lab prediction service. Please ensure the backend server is running.\nError: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error making AI lab prediction: {str(e)}")
            return None

    def make_prediction(self, patient_data, force_llm=False):
        """Make API call to get AI-powered prediction
        
        Args:
            patient_data (dict): Dictionary containing patient data
            force_llm (bool): Whether to force LLM analysis even if it's disabled by default
            
        Returns:
            dict: Prediction results including risk assessment and recommendations from AI
        """
        # Include all patient data for AI analysis - it can handle comprehensive data
        filtered_data = dict(patient_data)  # Create a copy
        
        # Add force_llm flag to the request data
        filtered_data['force_llm'] = force_llm
        
        # Convert numeric fields to appropriate types for better AI analysis
        numeric_fields = ['age', 'resting_bp', 'systolic_bp', 'diastolic_bp', 'cholesterol', 
                         'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
                         'triglycerides', 'fasting_blood_sugar', 'max_heart_rate', 'ca']
        
        for field in numeric_fields:
            if field in filtered_data and filtered_data[field] is not None:
                try:
                    filtered_data[field] = int(float(filtered_data[field]))
                except (ValueError, TypeError):
                    pass
        
        # Convert float fields
        float_fields = ['bmi', 'hba1c', 'oldpeak', 'weight', 'height']
        for field in float_fields:
            if field in filtered_data and filtered_data[field] is not None:
                try:
                    filtered_data[field] = float(filtered_data[field])
                except (ValueError, TypeError):
                    pass
        
        # Map legacy field names to new ones for better compatibility
        if 'resting_bp' in filtered_data and 'systolic_bp' not in filtered_data:
            filtered_data['systolic_bp'] = filtered_data['resting_bp']
        if 'cholesterol' in filtered_data and 'total_cholesterol' not in filtered_data:
            filtered_data['total_cholesterol'] = filtered_data['cholesterol']
        
        try:
            headers = {**self.auth_headers, "Content-Type": "application/json"}
            try:
                # Show user that we're using AI
                with st.spinner("🤖 Analyzing with AI..."):
                    response = requests.post(
                        f"{self.base_url}/predict/simple",
                        json=filtered_data,
                        headers=headers,
                        timeout=45  # Increased timeout for AI API calls
                    )
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                
                result = response.json()
                
                # Add a flag to indicate this is an AI-powered prediction
                if result:
                    result['powered_by_ai'] = True
                    
                return result
                
            except requests.exceptions.HTTPError as http_err:
                error_msg = f"HTTP error occurred: {http_err}"
                if 'response' in locals() and response.text:
                    try:
                        error_detail = response.json().get('detail', response.text)
                        error_msg += f" - {error_detail}"
                    except:
                        error_msg += f" - {response.text}"
            st.error(error_msg)
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the AI prediction service. Please ensure the backend server is running.\nError: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error making AI prediction: {str(e)}")
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
    
    def upload_medical_report(self, file_data, patient_name):
        """Upload and analyze medical report"""
        try:
            headers = {**self.auth_headers}
            files = {"file": file_data}
            data = {"patient_name": patient_name}
            
            with st.spinner("📄 Uploading and analyzing medical report..."):
                response = requests.post(
                    f"{self.base_url}/medical-report/upload",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=120  # Longer timeout for file processing
                )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as http_err:
            error_msg = f"HTTP error occurred: {http_err}"
            if 'response' in locals() and response.text:
                try:
                    error_detail = response.json().get('detail', response.text)
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text}"
            st.error(error_msg)
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to medical report service. Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error uploading medical report: {str(e)}")
            return None
    
    def get_analysis(self, analysis_id):
        """Get medical report analysis by ID"""
        try:
            headers = {**self.auth_headers}
            response = requests.get(
                f"{self.base_url}/medical-report/analysis/{analysis_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error retrieving analysis: {str(e)}")
            return None
    
    def list_analyses(self, patient_name=None, limit=50):
        """List medical report analyses"""
        try:
            headers = {**self.auth_headers}
            params = {"limit": limit}
            if patient_name:
                params["patient_name"] = patient_name
                
            response = requests.get(
                f"{self.base_url}/medical-report/list",
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error listing analyses: {str(e)}")
            return None
    
    def download_report(self, analysis_id):
        """Download PDF report"""
        try:
            headers = {**self.auth_headers}
            response = requests.get(
                f"{self.base_url}/medical-report/download/{analysis_id}",
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Error downloading report: {str(e)}")
            return None
    
    def delete_analysis(self, analysis_id):
        """Delete medical report analysis"""
        try:
            headers = {**self.auth_headers}
            response = requests.delete(
                f"{self.base_url}/medical-report/analysis/{analysis_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error deleting analysis: {str(e)}")
            return None