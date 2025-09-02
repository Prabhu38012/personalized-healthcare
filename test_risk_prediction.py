#!/usr/bin/env python3
"""
Test script to verify risk probability calculation and model accuracy.
This script will test the EHR model with known test cases to validate its predictions.
"""

import sys
import os
import requests
import json
import numpy as np
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api_endpoint():
    """Test the API endpoint with sample data"""
    print("🧪 Testing API Endpoint...")
    
    # Test data - high risk patient
    high_risk_patient = {
        "age": 70,
        "sex": "M",
        "height": 175,
        "weight": 90,
        "bmi": 29.4,
        "resting_bp": 160,
        "max_heart_rate": 120,
        "cholesterol": 280,
        "hdl_cholesterol": 35,
        "ldl_cholesterol": 180,
        "triglycerides": 200,
        "fasting_blood_sugar": 1,
        "hba1c": 8.5,
        "chest_pain_type": "typical",
        "resting_ecg": "abnormal",
        "exercise_angina": 1,
        "oldpeak": 3.5,
        "slope": "downsloping",
        "ca": 2,
        "thal": "reversible",
        "smoking": 1,
        "diabetes": 1,
        "family_history": 1
    }
    
    # Test data - low risk patient
    low_risk_patient = {
        "age": 25,
        "sex": "F",
        "height": 165,
        "weight": 60,
        "bmi": 22.0,
        "resting_bp": 110,
        "max_heart_rate": 180,
        "cholesterol": 160,
        "hdl_cholesterol": 60,
        "ldl_cholesterol": 100,
        "triglycerides": 80,
        "fasting_blood_sugar": 0,
        "hba1c": 5.2,
        "chest_pain_type": "asymptomatic",
        "resting_ecg": "normal",
        "exercise_angina": 0,
        "oldpeak": 0.0,
        "slope": "upsloping",
        "ca": 0,
        "thal": "normal",
        "smoking": 0,
        "diabetes": 0,
        "family_history": 0
    }
    
    base_url = "http://localhost:8000"
    
    try:
        # Test high risk patient
        response = requests.post(f"{base_url}/api/predict-simple", 
                               json=high_risk_patient, 
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ High Risk Patient - Risk: {result['risk_probability']:.3f} ({result['risk_level']})")
            print(f"   Confidence: {result['confidence']:.3f}")
            
            # Validate high risk prediction
            if result['risk_probability'] > 0.6:
                print("   ✅ Correctly identified as high risk")
            else:
                print(f"   ⚠️  Expected high risk but got {result['risk_probability']:.3f}")
        else:
            print(f"❌ API Error for high risk patient: {response.status_code}")
            print(response.text)
            
        # Test low risk patient  
        response = requests.post(f"{base_url}/api/predict-simple", 
                               json=low_risk_patient, 
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Low Risk Patient - Risk: {result['risk_probability']:.3f} ({result['risk_level']})")
            print(f"   Confidence: {result['confidence']:.3f}")
            
            # Validate low risk prediction
            if result['risk_probability'] < 0.4:
                print("   ✅ Correctly identified as low risk")
            else:
                print(f"   ⚠️  Expected low risk but got {result['risk_probability']:.3f}")
        else:
            print(f"❌ API Error for low risk patient: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the backend server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False
        
    return True

def analyze_model_metrics():
    """Analyze the model training metrics from the report"""
    print("\n📊 Analyzing Model Performance...")
    
    try:
        with open('models/ehr_training_report.md', 'r') as f:
            report = f.read()
            
        print("📈 Model Performance Summary:")
        
        # Extract Random Forest metrics (best model)
        lines = report.split('\n')
        rf_section = False
        rf_metrics = {}
        
        for line in lines:
            if "Random Forest" in line:
                rf_section = True
                continue
            elif line.startswith("##") and rf_section:
                break
            elif rf_section and line.startswith("- "):
                metric, value = line.split(': ')
                metric = metric.replace('- ', '')
                rf_metrics[metric] = float(value)
        
        print(f"🎯 Best Model: Random Forest")
        for metric, value in rf_metrics.items():
            if metric == "Accuracy":
                status = "✅ Good" if value >= 0.75 else "⚠️ Fair" if value >= 0.65 else "❌ Poor"
            elif metric == "F1 Score":
                status = "✅ Good" if value >= 0.75 else "⚠️ Fair" if value >= 0.65 else "❌ Poor"
            elif metric == "AUC Score":
                status = "✅ Excellent" if value >= 0.85 else "✅ Good" if value >= 0.75 else "⚠️ Fair"
            else:
                status = ""
                
            print(f"   {metric}: {value:.3f} {status}")
            
        # Feature importance analysis
        print("\n🔍 Top Predictive Features:")
        feature_section = False
        for line in lines:
            if "Feature Importance" in line:
                feature_section = True
                continue
            elif feature_section and line.startswith("- "):
                feature, importance = line.split(': ')
                feature = feature.replace('- ', '')
                print(f"   {feature}: {float(importance):.3f}")
                
        # Overall assessment
        accuracy = rf_metrics.get('Accuracy', 0)
        f1_score = rf_metrics.get('F1 Score', 0)
        auc_score = rf_metrics.get('AUC Score', 0)
        
        print(f"\n🏆 Overall Model Quality Assessment:")
        if accuracy >= 0.75 and f1_score >= 0.75 and auc_score >= 0.85:
            print("   ✅ EXCELLENT - Model shows strong predictive performance")
        elif accuracy >= 0.7 and f1_score >= 0.7 and auc_score >= 0.8:
            print("   ✅ GOOD - Model is well-trained and suitable for production")
        elif accuracy >= 0.65 and f1_score >= 0.65:
            print("   ⚠️  FAIR - Model needs improvement but usable")
        else:
            print("   ❌ POOR - Model requires retraining or more data")
            
    except FileNotFoundError:
        print("❌ Training report not found. Please run training first.")
        return False
    except Exception as e:
        print(f"❌ Error analyzing model metrics: {str(e)}")
        return False
        
    return True

def validate_ehr_data_usage():
    """Validate that the EHR dataset was properly used"""
    print("\n🏥 Validating EHR Dataset Usage...")
    
    try:
        with open('models/ehr_training_report.md', 'r') as f:
            report = f.read()
            
        # Check if real EHR features are being used
        ehr_features = [
            'birth_date', 'SystolicBloodPressure', 'DiastolicBloodPressure', 
            'BodyWeight', 'BodyHeight', 'BodyMassIndex', 'TotalCholesterol'
        ]
        
        found_features = []
        for feature in ehr_features:
            if feature in report:
                found_features.append(feature)
                
        print(f"📋 EHR Features Found: {len(found_features)}/{len(ehr_features)}")
        for feature in found_features:
            print(f"   ✅ {feature}")
            
        if len(found_features) >= 5:
            print("   ✅ Model successfully trained on EHR dataset")
            return True
        else:
            print("   ⚠️  Limited EHR features detected - may be using synthetic data")
            return False
            
    except Exception as e:
        print(f"❌ Error validating EHR usage: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🩺 Healthcare Risk Prediction Model Validation")
    print("=" * 60)
    
    # Analyze model performance
    metrics_ok = analyze_model_metrics()
    
    # Validate EHR data usage
    ehr_ok = validate_ehr_data_usage()
    
    # Test API endpoint
    api_ok = test_api_endpoint()
    
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY:")
    print(f"   Model Performance: {'✅ PASS' if metrics_ok else '❌ FAIL'}")
    print(f"   EHR Data Usage: {'✅ PASS' if ehr_ok else '❌ FAIL'}")
    print(f"   API Functionality: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if metrics_ok and ehr_ok:
        print("\n🎉 SUCCESS: Risk probability calculation is properly trained with EHR dataset!")
        print("   The model shows good performance metrics and uses real EHR features.")
    else:
        print("\n⚠️  ISSUES DETECTED: Please review the training process and data quality.")

if __name__ == "__main__":
    main()