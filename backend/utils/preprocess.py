import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        
    def clean_data(self, df):
        """Clean and prepare healthcare data"""
        logger.info("Starting data cleaning process")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)
            
        # Fill categorical missing values with mode
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
            
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical variables"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['target', 'diagnosis']:  # Skip target variables
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(df[col].astype(str))
                        known_values = set(self.label_encoders[col].classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            # Map unknown values to a default category
                            df[col] = df[col].astype(str).replace(list(unknown_values), 'Unknown')
                        
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if fit:
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        else:
            df[numeric_columns] = self.scaler.transform(df[numeric_columns])
            
        return df
    
    def prepare_features(self, df, target_column=None, fit=True):
        """Complete preprocessing pipeline"""
        logger.info("Starting feature preparation")
        
        # Clean data
        df = self.clean_data(df.copy())
        
        # Separate features and target
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df.copy()
            y = None
            
        # Encode categorical features
        X = self.encode_categorical_features(X, fit=fit)
        
        # Scale features
        X = self.scale_features(X, fit=fit)
        
        if fit:
            self.feature_columns = X.columns.tolist()
            
        logger.info(f"Feature preparation complete. Features: {len(X.columns)}")
        
        return X, y
    
    def transform_single_record(self, record_dict):
        """Transform a single patient record for prediction"""
        df = pd.DataFrame([record_dict])
        X, _ = self.prepare_features(df, fit=False)
        
        # Ensure all expected features are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0  # Default value for missing features
            X = X[self.feature_columns]  # Reorder columns
            
        return X.iloc[0].values.reshape(1, -1)

def create_synthetic_patient_data(n_samples=1000):
    """Create synthetic patient data for demonstration"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'chest_pain_type': np.random.choice(['typical', 'atypical', 'non_anginal', 'asymptomatic'], n_samples),
        'resting_bp': np.random.randint(90, 200, n_samples),
        'cholesterol': np.random.randint(120, 400, n_samples),
        'fasting_blood_sugar': np.random.choice([0, 1], n_samples),
        'resting_ecg': np.random.choice(['normal', 'abnormal', 'hypertrophy'], n_samples),
        'max_heart_rate': np.random.randint(60, 220, n_samples),
        'exercise_angina': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.choice(['upsloping', 'flat', 'downsloping'], n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.choice(['normal', 'fixed', 'reversible'], n_samples),
        'bmi': np.random.uniform(18.5, 40, n_samples),
        'smoking': np.random.choice([0, 1], n_samples),
        'diabetes': np.random.choice([0, 1], n_samples),
        'family_history': np.random.choice([0, 1], n_samples)
    }
    
    # Create target variable (heart disease risk)
    risk_factors = (
        (data['age'] > 50).astype(int) +
        (data['resting_bp'] > 140).astype(int) +
        (data['cholesterol'] > 240).astype(int) +
        data['smoking'] +
        data['diabetes'] +
        data['family_history'] +
        (data['bmi'] > 30).astype(int)
    )
    
    # Add some randomness
    data['target'] = (risk_factors >= 3).astype(int)
    noise = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['target'] = np.logical_xor(data['target'], noise).astype(int)
    
    return pd.DataFrame(data)
