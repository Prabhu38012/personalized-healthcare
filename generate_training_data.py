"""
Generate comprehensive training dataset for AI/ML models
Creates realistic healthcare data for training decision support models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_healthcare_data(n_samples=5000):
    """Generate synthetic but realistic healthcare data"""
    
    print(f"🔄 Generating {n_samples} patient records...")
    
    # Demographics
    ages = np.random.randint(18, 90, n_samples)
    
    # Physiological measurements (correlated with age)
    base_bp = 110 + (ages - 40) * 0.5 + np.random.normal(0, 15, n_samples)
    blood_pressure = np.clip(base_bp, 90, 200).astype(int)
    
    base_chol = 180 + (ages - 40) * 0.8 + np.random.normal(0, 30, n_samples)
    cholesterol = np.clip(base_chol, 120, 350).astype(int)
    
    base_bmi = 22 + (ages - 30) * 0.08 + np.random.normal(0, 4, n_samples)
    bmi = np.clip(base_bmi, 16, 45).round(1)
    
    base_glucose = 85 + (ages - 40) * 0.3 + np.random.normal(0, 15, n_samples)
    glucose = np.clip(base_glucose, 60, 200).astype(int)
    
    # Lifestyle factors
    smoking = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    
    # Younger people tend to exercise more
    base_exercise = 180 - (ages - 30) * 1.5 + np.random.normal(0, 60, n_samples)
    exercise = np.clip(base_exercise, 0, 400).astype(int)
    
    # Medical history (correlated with age)
    diabetes_prob = 0.05 + (ages - 40) * 0.005
    diabetes_prob = np.clip(diabetes_prob, 0, 0.4)
    diabetes = np.random.binomial(1, diabetes_prob)
    
    family_history = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    
    # Additional health metrics
    heart_rate = np.random.randint(55, 105, n_samples)
    
    stress_level = np.random.randint(1, 11, n_samples)
    
    sleep_hours = np.random.normal(7, 1.5, n_samples)
    sleep_hours = np.clip(sleep_hours, 3, 11).round(1)
    
    alcohol = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    # Calculate sophisticated risk score
    risk_components = (
        (ages - 50) * 0.015 +                          # Age factor
        (blood_pressure - 120) * 0.004 +               # BP factor
        (cholesterol - 200) * 0.003 +                  # Cholesterol factor
        (bmi - 25) * 0.025 +                          # BMI factor
        (glucose - 100) * 0.008 +                     # Glucose factor
        smoking * 0.25 +                              # Smoking impact
        diabetes * 0.3 +                              # Diabetes impact
        family_history * 0.2 +                        # Family history
        (180 - exercise) * 0.001 +                    # Exercise (inverse)
        (heart_rate - 70) * 0.003 +                   # Heart rate
        stress_level * 0.02 +                         # Stress
        (7 - sleep_hours) * 0.03 +                    # Sleep (inverse)
        alcohol * 0.05                                # Alcohol
    )
    
    # Add non-linear interactions
    interaction_effects = (
        smoking * diabetes * 0.15 +                    # Smoking + diabetes
        (bmi > 30).astype(int) * (blood_pressure > 140).astype(int) * 0.2 +  # Obesity + hypertension
        (ages > 60).astype(int) * family_history * 0.15  # Age + family history
    )
    
    total_risk = risk_components + interaction_effects + np.random.normal(0, 0.15, n_samples)
    
    # Create binary target with some probability transition zone
    risk_threshold = np.percentile(total_risk, 70)  # Top 30% are high risk
    risk_prob = 1 / (1 + np.exp(-(total_risk - risk_threshold) * 5))  # Sigmoid
    target = np.random.binomial(1, risk_prob)
    
    # Medication (people with high risk are more likely on medication)
    medication_prob = 0.1 + target * 0.4
    on_medication = np.random.binomial(1, medication_prob)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': ages,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'bmi': bmi,
        'glucose': glucose,
        'smoking': smoking,
        'exercise': exercise,
        'diabetes': diabetes,
        'family_history': family_history,
        'heart_rate': heart_rate,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'alcohol': alcohol,
        'on_medication': on_medication,
        'high_risk': target  # Target variable
    })
    
    return data


def add_data_quality_issues(data, noise_ratio=0.02):
    """Add realistic data quality issues"""
    n_samples = len(data)
    n_noise = int(n_samples * noise_ratio)
    
    # Add some missing values
    for col in ['exercise', 'sleep_hours', 'stress_level']:
        missing_idx = np.random.choice(n_samples, size=n_noise // 3, replace=False)
        data.loc[missing_idx, col] = np.nan
    
    return data


def generate_temporal_data(base_data, n_patients=500):
    """Generate time-series data for pattern recognition"""
    
    print(f"🔄 Generating temporal data for {n_patients} patients...")
    
    temporal_records = []
    patient_ids = range(1, n_patients + 1)
    
    for patient_id in patient_ids:
        # Select base patient data
        base_patient = base_data.sample(1).iloc[0]
        
        # Generate 6 months of weekly measurements
        num_weeks = 24
        
        for week in range(num_weeks):
            record = base_patient.copy()
            
            # Add temporal variation
            if record['high_risk'] == 1:
                # High-risk patients show worsening trends
                trend_factor = week * 0.02
                record['blood_pressure'] += np.random.randint(-3, 8) + trend_factor * 10
                record['cholesterol'] += np.random.randint(-5, 10) + trend_factor * 5
                record['bmi'] += np.random.uniform(-0.1, 0.3)
            else:
                # Low-risk patients are more stable
                record['blood_pressure'] += np.random.randint(-5, 5)
                record['cholesterol'] += np.random.randint(-10, 10)
                record['bmi'] += np.random.uniform(-0.2, 0.2)
            
            record['glucose'] += np.random.randint(-10, 10)
            record['heart_rate'] += np.random.randint(-5, 5)
            record['exercise'] += np.random.randint(-20, 20)
            
            # Clip to realistic ranges
            record['blood_pressure'] = np.clip(record['blood_pressure'], 90, 200)
            record['cholesterol'] = np.clip(record['cholesterol'], 120, 350)
            record['bmi'] = np.clip(record['bmi'], 16, 45)
            record['glucose'] = np.clip(record['glucose'], 60, 200)
            record['heart_rate'] = np.clip(record['heart_rate'], 55, 120)
            record['exercise'] = np.clip(record['exercise'], 0, 400)
            
            # Add metadata
            timestamp = datetime.now() - timedelta(weeks=(num_weeks - week))
            
            temporal_records.append({
                'patient_id': patient_id,
                'week': week,
                'timestamp': timestamp.strftime('%Y-%m-%d'),
                **record.to_dict()
            })
    
    return pd.DataFrame(temporal_records)


if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Healthcare ML Training Data Generator")
    print("=" * 60)
    
    # Generate main training dataset
    print("\n📊 Step 1: Generating main training dataset...")
    train_data = generate_healthcare_data(n_samples=5000)
    
    print("\n📊 Step 2: Adding realistic data quality variations...")
    train_data = add_data_quality_issues(train_data, noise_ratio=0.02)
    
    # Save main dataset
    train_data.to_csv('data/ml_training_data.csv', index=False)
    print(f"✅ Saved main training data: data/ml_training_data.csv ({len(train_data)} samples)")
    
    # Generate temporal dataset for pattern recognition
    print("\n📊 Step 3: Generating temporal dataset...")
    temporal_data = generate_temporal_data(train_data, n_patients=500)
    temporal_data.to_csv('data/temporal_training_data.csv', index=False)
    print(f"✅ Saved temporal data: data/temporal_training_data.csv ({len(temporal_data)} records)")
    
    # Generate test dataset (separate from training)
    print("\n📊 Step 4: Generating test dataset...")
    test_data = generate_healthcare_data(n_samples=1000)
    test_data.to_csv('data/ml_test_data.csv', index=False)
    print(f"✅ Saved test data: data/ml_test_data.csv ({len(test_data)} samples)")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("📈 Dataset Statistics")
    print("=" * 60)
    print(f"\n🎯 Target Distribution (Training):")
    print(train_data['high_risk'].value_counts())
    print(f"\n📊 Risk Rate: {train_data['high_risk'].mean():.1%}")
    
    print(f"\n📋 Feature Summary:")
    print(train_data.describe())
    
    print("\n✅ Data generation complete! Ready for model training.")
    print("\nNext step: Run 'python train_ai_models.py' to train ML models")
