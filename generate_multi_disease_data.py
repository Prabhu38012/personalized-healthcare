"""
Generate comprehensive multi-disease healthcare dataset
Covers: Heart Disease, Diabetes, Hypertension, Obesity, Kidney Disease, Liver Disease
"""
import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

def generate_multi_disease_data(n_samples=10000):
    """
    Generate realistic multi-disease healthcare dataset
    Predicts multiple conditions simultaneously
    """
    
    print(f"🔄 Generating {n_samples} patient records with multiple disease risks...")
    
    # Demographics
    ages = np.random.randint(18, 90, n_samples)
    gender = np.random.choice([0, 1], n_samples, p=[0.52, 0.48])  # 0=Female, 1=Male
    
    # Physiological measurements (age-correlated)
    base_bp = 110 + (ages - 40) * 0.6 + np.random.normal(0, 15, n_samples)
    blood_pressure = np.clip(base_bp, 90, 220).astype(int)
    
    base_chol = 170 + (ages - 40) * 1.0 + np.random.normal(0, 35, n_samples)
    cholesterol = np.clip(base_chol, 120, 400).astype(int)
    
    # BMI with obesity trends
    base_bmi = 22 + (ages - 30) * 0.12 + np.random.normal(0, 5, n_samples)
    bmi = np.clip(base_bmi, 15, 50).round(1)
    
    # Glucose (critical for diabetes)
    base_glucose = 85 + (ages - 40) * 0.4 + (bmi - 25) * 1.5 + np.random.normal(0, 20, n_samples)
    glucose = np.clip(base_glucose, 60, 300).astype(int)
    
    # HbA1c (diabetes marker)
    base_hba1c = 5.0 + (glucose - 90) * 0.02 + np.random.normal(0, 0.5, n_samples)
    hba1c = np.clip(base_hba1c, 4.0, 14.0).round(1)
    
    # Kidney function markers
    base_creatinine = 0.8 + (ages - 40) * 0.005 + np.random.normal(0, 0.2, n_samples)
    creatinine = np.clip(base_creatinine, 0.5, 5.0).round(2)
    
    base_gfr = 120 - (ages - 30) * 0.8 + np.random.normal(0, 10, n_samples)
    gfr = np.clip(base_gfr, 15, 140).astype(int)
    
    # Liver function
    base_alt = 20 + (bmi - 25) * 2 + np.random.normal(0, 10, n_samples)
    alt = np.clip(base_alt, 7, 200).astype(int)
    
    base_ast = 22 + (bmi - 25) * 1.5 + np.random.normal(0, 8, n_samples)
    ast = np.clip(base_ast, 10, 180).astype(int)
    
    # Heart-specific
    base_hr = 70 + (ages - 50) * 0.2 + np.random.normal(0, 12, n_samples)
    heart_rate = np.clip(base_hr, 50, 130).astype(int)
    
    base_ejection = 65 - (ages - 50) * 0.15 + np.random.normal(0, 8, n_samples)
    ejection_fraction = np.clip(base_ejection, 20, 80).astype(int)
    
    # Lifestyle factors
    smoking = np.random.choice([0, 1], n_samples, p=[0.72, 0.28])
    alcohol = np.random.choice([0, 1, 2, 3], n_samples, p=[0.25, 0.45, 0.20, 0.10])
    
    base_exercise = 200 - (ages - 30) * 2 - (bmi - 25) * 5 + np.random.normal(0, 60, n_samples)
    exercise = np.clip(base_exercise, 0, 500).astype(int)
    
    # Medical history
    family_heart = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])
    family_diabetes = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    
    # Symptoms and vitals
    stress_level = np.random.randint(1, 11, n_samples)
    sleep_hours = np.random.normal(7, 1.8, n_samples).clip(3, 12).round(1)
    
    base_waist = 70 + (bmi - 20) * 3 + gender * 8 + np.random.normal(0, 8, n_samples)
    waist_circumference = np.clip(base_waist, 55, 150).astype(int)
    
    # Medications
    on_bp_meds = ((blood_pressure > 140) | ((ages > 50) & (blood_pressure > 130))).astype(int)
    on_diabetes_meds = ((glucose > 126) | (hba1c > 6.5)).astype(int)
    on_statin = ((cholesterol > 240) | ((ages > 50) & (cholesterol > 200))).astype(int)
    
    # ============================================
    # DISEASE RISK CALCULATIONS
    # ============================================
    
    # 1. HEART DISEASE RISK
    heart_risk = (
        (ages - 40) * 0.012 +
        (blood_pressure - 120) * 0.0045 +
        (cholesterol - 200) * 0.003 +
        (bmi - 25) * 0.025 +
        smoking * 0.28 +
        family_heart * 0.22 +
        (180 - exercise) * 0.0012 +
        stress_level * 0.018 +
        (ejection_fraction < 50).astype(int) * 0.35 +
        gender * 0.08 +  # Males higher risk
        np.random.normal(0, 0.12, n_samples)
    )
    heart_disease = (heart_risk > np.percentile(heart_risk, 65)).astype(int)
    
    # 2. DIABETES RISK
    diabetes_risk = (
        (glucose - 90) * 0.008 +
        (hba1c - 5.0) * 0.15 +
        (bmi - 25) * 0.035 +
        (ages - 40) * 0.008 +
        family_diabetes * 0.25 +
        (waist_circumference - 80) * 0.005 +
        (exercise < 150).astype(int) * 0.15 +
        np.random.normal(0, 0.15, n_samples)
    )
    diabetes = (diabetes_risk > np.percentile(diabetes_risk, 75)).astype(int)
    
    # 3. HYPERTENSION
    hypertension = (blood_pressure >= 140).astype(int)
    
    # 4. OBESITY
    obesity = (bmi >= 30).astype(int)
    
    # 5. KIDNEY DISEASE RISK
    kidney_risk = (
        (ages - 40) * 0.015 +
        (creatinine - 1.0) * 0.4 +
        (120 - gfr) * 0.006 +
        diabetes * 0.3 +
        hypertension * 0.25 +
        (blood_pressure - 120) * 0.003 +
        np.random.normal(0, 0.12, n_samples)
    )
    kidney_disease = (kidney_risk > np.percentile(kidney_risk, 85)).astype(int)
    
    # 6. LIVER DISEASE RISK
    liver_risk = (
        (alt - 30) * 0.008 +
        (ast - 30) * 0.007 +
        (bmi - 25) * 0.03 +
        alcohol * 0.12 +
        (ages - 40) * 0.005 +
        np.random.normal(0, 0.15, n_samples)
    )
    liver_disease = (liver_risk > np.percentile(liver_risk, 88)).astype(int)
    
    # 7. METABOLIC SYNDROME (combination)
    metabolic_syndrome = (
        (bmi >= 30).astype(int) +
        (blood_pressure >= 130).astype(int) +
        (glucose >= 100).astype(int) +
        (cholesterol >= 200).astype(int) +
        (waist_circumference >= 88).astype(int)  # Female threshold
    )
    metabolic_syndrome = (metabolic_syndrome >= 3).astype(int)
    
    # 8. OVERALL HIGH RISK (any serious condition)
    high_risk = (
        heart_disease | 
        diabetes | 
        kidney_disease | 
        (metabolic_syndrome & (ages > 50))
    ).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        # Demographics
        'age': ages,
        'gender': gender,
        
        # Vital Signs
        'blood_pressure': blood_pressure,
        'heart_rate': heart_rate,
        'bmi': bmi,
        'waist_circumference': waist_circumference,
        
        # Lab Tests - General
        'cholesterol': cholesterol,
        'glucose': glucose,
        'hba1c': hba1c,
        
        # Lab Tests - Kidney
        'creatinine': creatinine,
        'gfr': gfr,
        
        # Lab Tests - Liver
        'alt': alt,
        'ast': ast,
        
        # Heart-specific
        'ejection_fraction': ejection_fraction,
        
        # Lifestyle
        'smoking': smoking,
        'alcohol': alcohol,
        'exercise': exercise,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        
        # Medical History
        'family_heart_disease': family_heart,
        'family_diabetes': family_diabetes,
        
        # Medications
        'on_bp_meds': on_bp_meds,
        'on_diabetes_meds': on_diabetes_meds,
        'on_statin': on_statin,
        
        # TARGET VARIABLES (Multiple Diseases)
        'heart_disease': heart_disease,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'obesity': obesity,
        'kidney_disease': kidney_disease,
        'liver_disease': liver_disease,
        'metabolic_syndrome': metabolic_syndrome,
        'high_risk': high_risk  # Overall risk flag
    })
    
    return data


def print_dataset_statistics(data):
    """Print comprehensive statistics"""
    print("\n" + "=" * 70)
    print("📊 Multi-Disease Dataset Statistics")
    print("=" * 70)
    
    print(f"\n📋 Total Samples: {len(data)}")
    print(f"👥 Age Range: {data['age'].min()}-{data['age'].max()} (mean: {data['age'].mean():.1f})")
    print(f"⚖️  BMI Range: {data['bmi'].min():.1f}-{data['bmi'].max():.1f} (mean: {data['bmi'].mean():.1f})")
    
    print("\n🎯 Disease Prevalence:")
    diseases = ['heart_disease', 'diabetes', 'hypertension', 'obesity', 
                'kidney_disease', 'liver_disease', 'metabolic_syndrome', 'high_risk']
    
    for disease in diseases:
        count = data[disease].sum()
        pct = count / len(data) * 100
        print(f"   {disease:20}: {count:5} ({pct:5.1f}%)")
    
    print("\n📈 Key Metrics:")
    print(f"   Mean Blood Pressure: {data['blood_pressure'].mean():.1f} mmHg")
    print(f"   Mean Cholesterol: {data['cholesterol'].mean():.1f} mg/dL")
    print(f"   Mean Glucose: {data['glucose'].mean():.1f} mg/dL")
    print(f"   Mean HbA1c: {data['hba1c'].mean():.1f}%")
    print(f"   Mean GFR: {data['gfr'].mean():.1f} mL/min")
    
    print(f"\n🚬 Lifestyle Factors:")
    print(f"   Smokers: {data['smoking'].sum()} ({data['smoking'].mean()*100:.1f}%)")
    print(f"   Mean Exercise: {data['exercise'].mean():.0f} min/week")
    print(f"   Mean Sleep: {data['sleep_hours'].mean():.1f} hours")


if __name__ == "__main__":
    print("=" * 70)
    print("🏥 Multi-Disease Healthcare Dataset Generator")
    print("=" * 70)
    
    # Generate large dataset
    print("\n📊 Generating comprehensive multi-disease dataset...")
    data = generate_multi_disease_data(n_samples=10000)
    
    # Print statistics
    print_dataset_statistics(data)
    
    # Save dataset
    import os
    os.makedirs('data', exist_ok=True)
    
    data.to_csv('data/multi_disease_training.csv', index=False)
    print(f"\n💾 Saved: data/multi_disease_training.csv ({len(data)} samples)")
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42,
        stratify=data['high_risk']
    )
    
    train_data.to_csv('data/multi_disease_train.csv', index=False)
    test_data.to_csv('data/multi_disease_test.csv', index=False)
    
    print(f"💾 Saved: data/multi_disease_train.csv ({len(train_data)} samples)")
    print(f"💾 Saved: data/multi_disease_test.csv ({len(test_data)} samples)")
    
    print("\n" + "=" * 70)
    print("✅ Multi-Disease Dataset Ready!")
    print("=" * 70)
    print("\nThis dataset can predict:")
    print("  ✅ Heart Disease")
    print("  ✅ Diabetes") 
    print("  ✅ Hypertension")
    print("  ✅ Obesity")
    print("  ✅ Kidney Disease")
    print("  ✅ Liver Disease")
    print("  ✅ Metabolic Syndrome")
    print("  ✅ Overall Health Risk")
    
    print("\n🚀 Next: Train multi-disease models")
    print("   Run: python train_multi_disease_models.py")
