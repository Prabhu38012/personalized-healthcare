"""
Train multi-disease prediction models
Predicts: Heart Disease, Diabetes, Hypertension, Kidney Disease, Liver Disease, etc.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class MultiDiseaseTrainer:
    def __init__(self, data_path='data/multi_disease_training.csv'):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        
        # Disease targets
        self.disease_targets = [
            'heart_disease', 'diabetes', 'hypertension', 'obesity',
            'kidney_disease', 'liver_disease', 'metabolic_syndrome', 'high_risk'
        ]
        
        # Input features
        self.feature_cols = [
            'age', 'gender', 'blood_pressure', 'heart_rate', 'bmi',
            'waist_circumference', 'cholesterol', 'glucose', 'hba1c',
            'creatinine', 'gfr', 'alt', 'ast', 'ejection_fraction',
            'smoking', 'alcohol', 'exercise', 'stress_level', 'sleep_hours',
            'family_heart_disease', 'family_diabetes',
            'on_bp_meds', 'on_diabetes_meds', 'on_statin'
        ]
    
    def load_data(self):
        """Load multi-disease dataset"""
        print(f"\n📂 Loading data from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            print(f"❌ File not found. Run: python generate_multi_disease_data.py")
            return None, None
        
        data = pd.read_csv(self.data_path)
        
        X = data[self.feature_cols]
        y = data[self.disease_targets]
        
        print(f"✅ Loaded {len(data)} samples")
        print(f"📊 Features: {len(self.feature_cols)}")
        print(f"🎯 Target Diseases: {len(self.disease_targets)}")
        print(f"\nDisease Prevalence:")
        for disease in self.disease_targets:
            pct = y[disease].mean() * 100
            print(f"   {disease:20}: {pct:5.1f}%")
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train multi-output models for all diseases"""
        
        print("\n" + "=" * 70)
        print("🤖 Training Multi-Disease Prediction Models")
        print("=" * 70)
        
        # 1. Random Forest Multi-Output
        print("\n🌲 Training Random Forest (Multi-Disease)...")
        rf = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        print("✅ Random Forest trained")
        
        # 2. Gradient Boosting Multi-Output
        print("\n🚀 Training Gradient Boosting (Multi-Disease)...")
        gb = MultiOutputClassifier(
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        )
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        print("✅ Gradient Boosting trained")
        
        # 3. Neural Network
        print("\n🧠 Training Neural Network (Multi-Disease)...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        nn = MultiOutputClassifier(
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                verbose=False
            )
        )
        nn.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn
        print("✅ Neural Network trained")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on all diseases"""
        
        print("\n" + "=" * 70)
        print("📊 Model Evaluation - Multi-Disease Performance")
        print("=" * 70)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'=' * 70}")
            print(f"📈 {model_name.upper()}")
            print('=' * 70)
            
            # Prepare test data
            if model_name == 'neural_network':
                X_test_eval = self.scaler.transform(X_test)
            else:
                X_test_eval = X_test
            
            # Predict all diseases
            y_pred = model.predict(X_test_eval)
            
            # Evaluate each disease
            disease_results = {}
            for i, disease in enumerate(self.disease_targets):
                acc = accuracy_score(y_test[disease], y_pred[:, i])
                
                # Handle cases where there might be only one class
                try:
                    prec = precision_score(y_test[disease], y_pred[:, i], zero_division=0)
                    rec = recall_score(y_test[disease], y_pred[:, i], zero_division=0)
                    f1 = f1_score(y_test[disease], y_pred[:, i], zero_division=0)
                except:
                    prec = rec = f1 = 0.0
                
                disease_results[disease] = {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1_score': f1
                }
                
                print(f"\n{disease}:")
                print(f"   Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
                print(f"   Precision: {prec:.4f} ({prec*100:.1f}%)")
                print(f"   Recall:    {rec:.4f} ({rec*100:.1f}%)")
                print(f"   F1-Score:  {f1:.4f} ({f1*100:.1f}%)")
            
            # Calculate average performance
            avg_acc = np.mean([r['accuracy'] for r in disease_results.values()])
            avg_f1 = np.mean([r['f1_score'] for r in disease_results.values()])
            
            print(f"\n📊 Average Performance:")
            print(f"   Accuracy: {avg_acc:.4f} ({avg_acc*100:.1f}%)")
            print(f"   F1-Score: {avg_f1:.4f} ({avg_f1*100:.1f}%)")
            
            results[model_name] = {
                'diseases': disease_results,
                'average_accuracy': avg_acc,
                'average_f1': avg_f1
            }
        
        return results
    
    def save_models(self, output_dir='models'):
        """Save all models"""
        print(f"\n💾 Saving models to {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filename = os.path.join(output_dir, f'multi_disease_{name}_model.pkl')
            joblib.dump(model, filename)
            print(f"✅ Saved {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(output_dir, 'multi_disease_scaler.pkl'))
        print(f"✅ Saved {output_dir}/multi_disease_scaler.pkl")
        
        # Save feature names and targets
        joblib.dump(self.feature_cols, os.path.join(output_dir, 'multi_disease_features.pkl'))
        joblib.dump(self.disease_targets, os.path.join(output_dir, 'multi_disease_targets.pkl'))
        print(f"✅ Saved feature and target lists")
        
        # Save configuration
        config = {
            'features': self.feature_cols,
            'targets': self.disease_targets,
            'model_type': 'multi_output',
            'diseases': self.disease_targets
        }
        joblib.dump(config, os.path.join(output_dir, 'multi_disease_config.pkl'))
        print(f"✅ Saved {output_dir}/multi_disease_config.pkl")
    
    def generate_report(self, results, output_file='models/multi_disease_report.md'):
        """Generate training report"""
        print(f"\n📝 Generating report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("# Multi-Disease Prediction Model - Training Report\n\n")
            f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset**: {self.data_path}\n\n")
            
            f.write("## Diseases Covered\n\n")
            for disease in self.disease_targets:
                f.write(f"- {disease.replace('_', ' ').title()}\n")
            
            f.write("\n## Model Performance\n\n")
            
            # Overall comparison
            f.write("### Average Performance Across All Diseases\n\n")
            f.write("| Model | Avg Accuracy | Avg F1-Score |\n")
            f.write("|-------|--------------|-------------|\n")
            for model_name, result in results.items():
                f.write(f"| {model_name} | {result['average_accuracy']:.4f} | {result['average_f1']:.4f} |\n")
            
            # Per-disease performance
            for model_name, result in results.items():
                f.write(f"\n### {model_name.upper()} - Disease-Specific Performance\n\n")
                f.write("| Disease | Accuracy | Precision | Recall | F1-Score |\n")
                f.write("|---------|----------|-----------|--------|----------|\n")
                
                for disease, metrics in result['diseases'].items():
                    f.write(f"| {disease} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | ")
                    f.write(f"{metrics['recall']:.4f} | {metrics['f1_score']:.4f} |\n")
            
            f.write("\n## Usage\n\n")
            f.write("Models can predict multiple diseases simultaneously:\n\n")
            f.write("```python\n")
            f.write("import joblib\n")
            f.write("import pandas as pd\n\n")
            f.write("# Load model\n")
            f.write("model = joblib.load('models/multi_disease_random_forest_model.pkl')\n\n")
            f.write("# Predict\n")
            f.write("predictions = model.predict(patient_data)\n")
            f.write("# Returns: [heart_disease, diabetes, hypertension, obesity, ...]\n")
            f.write("```\n")
        
        print(f"✅ Report saved: {output_file}")


def main():
    print("=" * 70)
    print("🏥 Multi-Disease Prediction Model Training")
    print("=" * 70)
    
    # Initialize trainer
    trainer = MultiDiseaseTrainer()
    
    # Load data
    X, y = trainer.load_data()
    if X is None:
        return
    
    # Split data
    print("\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['high_risk']
    )
    print(f"✅ Training: {len(X_train)} samples")
    print(f"✅ Testing: {len(X_test)} samples")
    
    # Train models
    trainer.train_models(X_train, y_train)
    
    # Evaluate
    results = trainer.evaluate_models(X_test, y_test)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Final Model Comparison")
    print("=" * 70)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"   Average Accuracy: {result['average_accuracy']:.4f} ({result['average_accuracy']*100:.1f}%)")
        print(f"   Average F1-Score: {result['average_f1']:.4f} ({result['average_f1']*100:.1f}%)")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['average_f1'])
    print(f"\n🏆 Best Model: {best_model[0]} (F1: {best_model[1]['average_f1']:.4f})")
    
    # Save models
    trainer.save_models()
    
    # Generate report
    trainer.generate_report(results)
    
    print("\n" + "=" * 70)
    print("✅ Multi-Disease Model Training Complete!")
    print("=" * 70)
    print("\n🎉 Models can now predict 8 different health conditions!")
    print("📂 Models saved in 'models/' directory")
    print("📊 Report: models/multi_disease_report.md")


if __name__ == "__main__":
    main()
