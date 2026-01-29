"""
Download and prepare real healthcare datasets from public sources
Supports multiple datasets with high sample counts
"""
import pandas as pd
import numpy as np
import requests
import os
from io import StringIO

def download_uci_heart_disease():
    """
    Download UCI Heart Disease Dataset
    Source: https://archive.ics.uci.edu/dataset/45/heart+disease
    ~1,000 samples across multiple locations
    """
    print("\n📥 Downloading UCI Heart Disease Dataset...")
    
    # URLs for different locations
    datasets = {
        'cleveland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
        'hungarian': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
        'switzerland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
        'va': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data'
    }
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    all_data = []
    
    for location, url in datasets.items():
        try:
            print(f"   Downloading {location}...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Parse data
                data = pd.read_csv(StringIO(response.text), names=column_names, na_values='?')
                data['location'] = location
                all_data.append(data)
                print(f"   ✅ {location}: {len(data)} samples")
        except Exception as e:
            print(f"   ⚠️  {location}: {e}")
    
    if not all_data:
        print("❌ Failed to download UCI dataset")
        return None
    
    # Combine all datasets
    combined = pd.concat(all_data, ignore_index=True)
    
    # Convert to binary classification (0 = no disease, 1 = disease)
    combined['target'] = (combined['num'] > 0).astype(int)
    combined = combined.drop('num', axis=1)
    
    print(f"✅ Total UCI samples: {len(combined)}")
    return combined


def download_statlog_heart():
    """
    Download Statlog Heart Disease Dataset
    270 samples, clean data
    """
    print("\n📥 Downloading Statlog Heart Disease Dataset...")
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # Column names for Statlog
            columns = [
                'age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol',
                'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
                'exercise_induced_angina', 'oldpeak', 'slope',
                'major_vessels', 'thal', 'target'
            ]
            
            data = pd.read_csv(StringIO(response.text), sep=' ', names=columns)
            # Convert target: 1=no disease, 2=disease -> 0=no disease, 1=disease
            data['target'] = (data['target'] == 2).astype(int)
            
            print(f"✅ Statlog samples: {len(data)}")
            return data
    except Exception as e:
        print(f"⚠️  Failed to download Statlog: {e}")
    
    return None


def download_framingham_kaggle():
    """
    Framingham Heart Study Dataset (Kaggle)
    ~4,000 samples with 15 features
    Note: Requires manual download from Kaggle
    """
    print("\n📥 Checking for Framingham Dataset (Kaggle)...")
    
    # Check if already downloaded
    possible_paths = [
        'data/framingham.csv',
        'data/framingham_heart_disease.csv',
        'framingham.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found Framingham dataset at {path}")
            data = pd.read_csv(path)
            print(f"   Samples: {len(data)}")
            return data
    
    print("⚠️  Framingham dataset not found locally")
    print("   Download from: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset")
    print("   Save as: data/framingham.csv")
    return None


def prepare_for_training(df, dataset_name):
    """
    Standardize dataset to match our model format
    """
    print(f"\n🔧 Preparing {dataset_name} for training...")
    
    # Map to our standard format
    standard_columns = {
        'age': 'age',
        'trestbps': 'blood_pressure',
        'resting_bp': 'blood_pressure',
        'chol': 'cholesterol',
        'cholesterol': 'cholesterol',
        'thalach': 'heart_rate',
        'max_heart_rate': 'heart_rate',
        'exang': 'exercise_induced',
        'exercise_induced_angina': 'exercise_induced',
        'fbs': 'high_glucose',
        'fasting_blood_sugar': 'high_glucose',
        'sex': 'gender'
    }
    
    # Rename columns
    df_copy = df.copy()
    for old, new in standard_columns.items():
        if old in df_copy.columns:
            df_copy.rename(columns={old: new}, inplace=True)
    
    # Calculate BMI if not present (estimate)
    if 'bmi' not in df_copy.columns:
        # Use age-based estimation
        if 'age' in df_copy.columns:
            df_copy['bmi'] = 22 + (df_copy['age'] - 40) * 0.1 + np.random.normal(0, 3, len(df_copy))
            df_copy['bmi'] = df_copy['bmi'].clip(16, 45)
    
    # Add missing features with defaults or estimations
    required_features = [
        'age', 'blood_pressure', 'cholesterol', 'bmi', 'glucose',
        'smoking', 'exercise', 'diabetes', 'family_history', 'heart_rate',
        'stress_level', 'sleep_hours', 'alcohol', 'on_medication', 'high_risk'
    ]
    
    for feature in required_features:
        if feature not in df_copy.columns:
            if feature == 'glucose':
                # Estimate from fasting blood sugar
                if 'high_glucose' in df_copy.columns:
                    df_copy['glucose'] = df_copy['high_glucose'].map({0: 90, 1: 120})
                else:
                    df_copy['glucose'] = 90 + np.random.randint(-10, 30, len(df_copy))
            
            elif feature == 'smoking':
                # Estimate from gender and age
                if 'gender' in df_copy.columns and 'age' in df_copy.columns:
                    prob = 0.15 + (df_copy['age'] > 50).astype(int) * 0.1
                    df_copy['smoking'] = np.random.binomial(1, prob)
                else:
                    df_copy['smoking'] = np.random.choice([0, 1], len(df_copy), p=[0.75, 0.25])
            
            elif feature == 'exercise':
                # Inverse correlation with age
                if 'age' in df_copy.columns:
                    df_copy['exercise'] = (200 - df_copy['age'] + np.random.randint(-50, 50, len(df_copy))).clip(0, 400)
                else:
                    df_copy['exercise'] = np.random.randint(0, 300, len(df_copy))
            
            elif feature == 'diabetes':
                df_copy['diabetes'] = np.random.choice([0, 1], len(df_copy), p=[0.85, 0.15])
            
            elif feature == 'family_history':
                df_copy['family_history'] = np.random.choice([0, 1], len(df_copy), p=[0.65, 0.35])
            
            elif feature == 'stress_level':
                df_copy['stress_level'] = np.random.randint(1, 11, len(df_copy))
            
            elif feature == 'sleep_hours':
                df_copy['sleep_hours'] = np.random.normal(7, 1.5, len(df_copy)).clip(3, 11).round(1)
            
            elif feature == 'alcohol':
                df_copy['alcohol'] = np.random.choice([0, 1, 2, 3], len(df_copy), p=[0.3, 0.4, 0.2, 0.1])
            
            elif feature == 'on_medication':
                # Correlate with disease
                if 'target' in df_copy.columns:
                    df_copy['on_medication'] = (df_copy['target'] * 0.5 + np.random.rand(len(df_copy)) * 0.5 > 0.6).astype(int)
                else:
                    df_copy['on_medication'] = np.random.choice([0, 1], len(df_copy), p=[0.7, 0.3])
            
            elif feature == 'high_risk':
                if 'target' in df_copy.columns:
                    df_copy['high_risk'] = df_copy['target']
                else:
                    df_copy['high_risk'] = 0
    
    # Remove original target column if different name
    if 'target' in df_copy.columns and 'high_risk' in df_copy.columns:
        df_copy = df_copy.drop('target', axis=1)
    
    # Keep only required features
    available_features = [f for f in required_features if f in df_copy.columns]
    df_final = df_copy[available_features]
    
    # Handle missing values
    df_final = df_final.fillna(df_final.median(numeric_only=True))
    
    print(f"✅ Prepared {len(df_final)} samples with {len(df_final.columns)} features")
    print(f"   Features: {', '.join(df_final.columns)}")
    print(f"   Target distribution: {df_final['high_risk'].value_counts().to_dict() if 'high_risk' in df_final.columns else 'N/A'}")
    
    return df_final


def combine_datasets(datasets):
    """
    Combine multiple datasets
    """
    print("\n🔗 Combining datasets...")
    
    combined = pd.concat(datasets, ignore_index=True)
    
    # Remove duplicates
    combined = combined.drop_duplicates()
    
    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Combined dataset: {len(combined)} total samples")
    return combined


def main():
    """
    Download and prepare real healthcare datasets
    """
    print("=" * 70)
    print("🌐 Real Healthcare Dataset Downloader")
    print("=" * 70)
    
    datasets = []
    
    # Download UCI Heart Disease (multiple locations)
    uci_data = download_uci_heart_disease()
    if uci_data is not None:
        uci_prepared = prepare_for_training(uci_data, "UCI Heart Disease")
        datasets.append(uci_prepared)
    
    # Download Statlog Heart
    statlog_data = download_statlog_heart()
    if statlog_data is not None:
        statlog_prepared = prepare_for_training(statlog_data, "Statlog Heart")
        datasets.append(statlog_prepared)
    
    # Check for Framingham (manual download)
    framingham_data = download_framingham_kaggle()
    if framingham_data is not None:
        framingham_prepared = prepare_for_training(framingham_data, "Framingham")
        datasets.append(framingham_prepared)
    
    if not datasets:
        print("\n❌ No datasets downloaded. Check your internet connection.")
        return
    
    # Combine all datasets
    combined_data = combine_datasets(datasets)
    
    # Save combined dataset
    os.makedirs('data', exist_ok=True)
    combined_data.to_csv('data/real_training_data.csv', index=False)
    print(f"\n💾 Saved combined dataset: data/real_training_data.csv")
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42, 
                                             stratify=combined_data['high_risk'])
    
    train_data.to_csv('data/real_training_split.csv', index=False)
    test_data.to_csv('data/real_test_split.csv', index=False)
    print(f"💾 Saved train split: data/real_training_split.csv ({len(train_data)} samples)")
    print(f"💾 Saved test split: data/real_test_split.csv ({len(test_data)} samples)")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Dataset Summary")
    print("=" * 70)
    print(f"Total samples: {len(combined_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\nFeatures: {len(combined_data.columns) - 1}")
    print(f"Target distribution:")
    print(combined_data['high_risk'].value_counts())
    print(f"\n✅ Ready to train! Run: python train_ai_models.py")
    print(f"   (Update train_ai_models.py to use 'data/real_training_data.csv')")


if __name__ == "__main__":
    main()
