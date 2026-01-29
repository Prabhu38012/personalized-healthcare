"""
Input validation utilities for healthcare data
"""
from typing import Dict, Any, List, Optional, Tuple
import re
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class HealthMetricsValidator:
    """Validate health metrics and patient data"""
    
    # Valid ranges for health metrics
    VALID_RANGES = {
        'age': (0, 120),
        'blood_pressure': (60, 250),
        'systolic_bp': (70, 250),
        'diastolic_bp': (40, 150),
        'heart_rate': (30, 220),
        'cholesterol': (100, 600),
        'bmi': (10, 80),
        'glucose': (40, 600),
        'hba1c': (4.0, 15.0),
        'weight': (20, 300),  # kg
        'height': (50, 250),  # cm
        'temperature': (35.0, 42.0),  # celsius
        'oxygen_saturation': (70, 100),
        'creatinine': (0.1, 15.0),
        'gfr': (1, 150),
        'ejection_fraction': (10, 80),
        'exercise': (0, 1000),  # minutes per week
        'sleep_hours': (0, 24),
        'stress_level': (0, 10),
        'waist_circumference': (40, 200),  # cm
    }
    
    @staticmethod
    def validate_numeric_field(field_name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a numeric health field
        Returns: (is_valid, error_message)
        """
        if value is None:
            return True, None  # Optional fields
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return False, f"{field_name} must be a number"
        
        if field_name in HealthMetricsValidator.VALID_RANGES:
            min_val, max_val = HealthMetricsValidator.VALID_RANGES[field_name]
            if not (min_val <= numeric_value <= max_val):
                return False, f"{field_name} must be between {min_val} and {max_val}"
        
        return True, None
    
    @staticmethod
    def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate complete patient data dictionary
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate numeric fields
        for field, value in data.items():
            if field in HealthMetricsValidator.VALID_RANGES:
                is_valid, error_msg = HealthMetricsValidator.validate_numeric_field(field, value)
                if not is_valid:
                    errors.append(error_msg)
        
        # Cross-field validation
        if 'systolic_bp' in data and 'diastolic_bp' in data:
            try:
                if float(data['systolic_bp']) <= float(data['diastolic_bp']):
                    errors.append("Systolic blood pressure must be greater than diastolic")
            except (ValueError, TypeError):
                pass
        
        # BMI calculation validation
        if 'height' in data and 'weight' in data:
            try:
                height_m = float(data['height']) / 100
                calculated_bmi = float(data['weight']) / (height_m ** 2)
                if 'bmi' in data and abs(float(data['bmi']) - calculated_bmi) > 2:
                    logger.warning(f"BMI mismatch: provided {data['bmi']}, calculated {calculated_bmi:.1f}")
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_patient_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize and normalize patient data
        """
        sanitized = {}
        
        for key, value in data.items():
            # Skip None values
            if value is None:
                continue
            
            # Convert numeric strings to numbers
            if key in HealthMetricsValidator.VALID_RANGES:
                try:
                    sanitized[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {key}={value} to float")
                    continue
            
            # Convert boolean strings
            elif isinstance(value, str) and value.lower() in ['true', 'false', 'yes', 'no']:
                sanitized[key] = value.lower() in ['true', 'yes']
            
            # Pass through other values
            else:
                sanitized[key] = value
        
        return sanitized


class TextValidator:
    """Validate text inputs for security"""
    
    # Potentially dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script',  # XSS
        r'javascript:',  # XSS
        r'onerror=',  # XSS
        r'onclick=',  # XSS
        r'DROP TABLE',  # SQL injection
        r'DELETE FROM',  # SQL injection
        r'INSERT INTO',  # SQL injection
        r'UPDATE.*SET',  # SQL injection
        r'../../',  # Path traversal
        r'\.\.\\',  # Path traversal
    ]
    
    @staticmethod
    def is_safe_text(text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is safe from common attacks
        Returns: (is_safe, reason)
        """
        if not isinstance(text, str):
            return False, "Input must be text"
        
        # Check length
        if len(text) > 10000:
            return False, "Text too long (max 10000 characters)"
        
        # Check for dangerous patterns
        text_upper = text.upper()
        for pattern in TextValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return False, f"Potentially dangerous input detected"
        
        return True, None
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename for safe storage
        """
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """Validate email format"""
    if not isinstance(email, str):
        return False, "Email must be a string"
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    if len(email) > 254:
        return False, "Email too long"
    
    return True, None


def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password strength
    Returns: (is_strong, list_of_issues)
    """
    issues = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters")
    
    if not re.search(r'[A-Z]', password):
        issues.append("Password must contain an uppercase letter")
    
    if not re.search(r'[a-z]', password):
        issues.append("Password must contain a lowercase letter")
    
    if not re.search(r'\d', password):
        issues.append("Password must contain a number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Password must contain a special character")
    
    return len(issues) == 0, issues
