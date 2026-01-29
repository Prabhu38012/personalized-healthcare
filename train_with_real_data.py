"""
Train ML models with real healthcare data
Supports large datasets and advanced configurations
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class RealDataModelTrainer:
    def __init__(self, data_path='data/real_training_data.csv'):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self):
        """Load real healthcare data"""
        print(f"\n📂 Loading data from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            print(f"❌ File not found: {self.data_path}")
            print("   Run: python download_real_datasets.py")
            return None, None
        
        data = pd.read_csv(self.data_path)
        
        # Handle missing values
        print("🔧 Handling missing values...")
        data = data.fillna(data.median(numeric_only=True))
        
        # Separate features and target
        target_col = 'high_risk' if 'high_risk' in data.columns else 'target'
        self.feature_names = [col for col in data.columns if col not in [target_col, 'target', 'high_risk']]
        
        X = data[self.feature_names]
        y = data[target_col]
        
        print(f"✅ Loaded {len(data)} samples")
        print(f"📊 Features ({len(self.feature_names)}): {', '.join(self.feature_names[:10])}{'...' if len(self.feature_names) > 10 else ''}")
        print(f"🎯 Target distribution:\n{y.value_counts()}")
        print(f"   Disease rate: {y.mean():.1%}")
        
        return X, y
    
    def train_advanced_models(self, X_train, y_train):
        """Train advanced ML models with tuning"""
        
        print("\n" + "=" * 70)
        print("🤖 Training Advanced ML Models")
        print("=" * 70)
        
        # 1. Random Forest with tuning
        print("\n🌲 Training Random Forest (tuned)...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        print("✅ Random Forest trained")
        
        # 2. Gradient Boosting with tuning
        print("\n🚀 Training Gradient Boosting (tuned)...")
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.85,
            random_state=42
        )
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        print("✅ Gradient Boosting trained")
        
        # 3. Neural Network
        print("\n🧠 Training Neural Network (deep)...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        nn = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=False
        )
        nn.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn
        print("✅ Neural Network trained")
        
        # 4. Voting Classifier (Ensemble)
        print("\n🎭 Creating Voting Ensemble...")
        voting = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb)
            ],
            voting='soft',
            weights=[0.45, 0.55]
        )
        voting.fit(X_train, y_train)
        self.models['voting_ensemble'] = voting
        print("✅ Voting Ensemble created")
    
    def evaluate_model(self, model, X_test, y_test, model_name, scale=False):
        """Comprehensive model evaluation"""
        print(f"\n{'=' * 70}")
        print(f"📊 Evaluating: {model_name}")
        print('=' * 70)
        
        # Prepare test data
        X_test_eval = self.scaler.transform(X_test) if scale else X_test
        
        # Predictions
        y_pred = model.predict(X_test_eval)
        y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"\n📈 Performance Metrics:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name:15}: {value:.4f} ({value*100:.2f}%)")
        
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
        
        print(f"\n🎯 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"   True Negatives:  {cm[0, 0]}")
        print(f"   False Positives: {cm[0, 1]}")
        print(f"   False Negatives: {cm[1, 0]}")
        print(f"   True Positives:  {cm[1, 1]}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 10 Most Important Features:")
            for idx, row in importances.head(10).iterrows():
                print(f"   {row['feature']:20}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")
        
        return metrics
    
    def save_models(self, output_dir='models'):
        """Save all trained models"""
        print(f"\n💾 Saving models to {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filename = os.path.join(output_dir, f'real_ai_{name}_model.pkl')
            joblib.dump(model, filename)
            print(f"✅ Saved {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(output_dir, 'real_ai_scaler.pkl'))
        print(f"✅ Saved {output_dir}/real_ai_scaler.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, os.path.join(output_dir, 'real_ai_feature_names.pkl'))
        print(f"✅ Saved {output_dir}/real_ai_feature_names.pkl")
        
        # Save ensemble config
        ensemble_config = {
            'models': ['random_forest', 'gradient_boosting', 'neural_network', 'voting_ensemble'],
            'weights': {
                'random_forest': 0.30,
                'gradient_boosting': 0.35,
                'neural_network': 0.20,
                'voting_ensemble': 0.15
            },
            'feature_names': self.feature_names,
            'data_source': 'real_datasets',
            'training_samples': len(self.feature_names)
        }
        joblib.dump(ensemble_config, os.path.join(output_dir, 'real_ai_ensemble_config.pkl'))
        print(f"✅ Saved {output_dir}/real_ai_ensemble_config.pkl")
    
    def generate_report(self, all_metrics, output_file='models/real_ai_training_report.md'):
        """Generate comprehensive training report"""
        print(f"\n📝 Generating report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("# Real Healthcare Data - ML Model Training Report\n\n")
            f.write(f"**Training Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Data Source**: Real public healthcare datasets\n")
            f.write(f"**Dataset**: {self.data_path}\n\n")
            
            f.write("## Model Performance Comparison\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 Score | AUC |\n")
            f.write("|-------|----------|-----------|--------|----------|---------|\n")
            
            for model_name, metrics in all_metrics.items():
                f.write(f"| {model_name:20} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | ")
                f.write(f"{metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} |\n")
            
            best_model = max(all_metrics.items(), key=lambda x: x[1]['roc_auc'])
            f.write(f"\n**Best Model**: {best_model[0]} (AUC: {best_model[1]['roc_auc']:.4f})\n\n")
            
            f.write("## Features\n\n")
            f.write(f"Total features: {len(self.feature_names)}\n\n")
            f.write("```\n")
            f.write(", ".join(self.feature_names))
            f.write("\n```\n\n")
            
            f.write("## Usage\n\n")
            f.write("Models trained on real healthcare data:\n")
            f.write("- `real_ai_random_forest_model.pkl`\n")
            f.write("- `real_ai_gradient_boosting_model.pkl`\n")
            f.write("- `real_ai_neural_network_model.pkl`\n")
            f.write("- `real_ai_voting_ensemble_model.pkl`\n")
            f.write("- `real_ai_scaler.pkl`\n")
            f.write("- `real_ai_feature_names.pkl`\n")
            f.write("- `real_ai_ensemble_config.pkl`\n\n")
            
            f.write("## Integration\n\n")
            f.write("To use these models in the backend, update `backend/routes/ai_decision_support.py`:\n\n")
            f.write("```python\n")
            f.write("# Change model file paths to use real data models\n")
            f.write("rf_model = joblib.load('models/real_ai_random_forest_model.pkl')\n")
            f.write("gb_model = joblib.load('models/real_ai_gradient_boosting_model.pkl')\n")
            f.write("nn_model = joblib.load('models/real_ai_neural_network_model.pkl')\n")
            f.write("```\n")
        
        print(f"✅ Report saved: {output_file}")


def main():
    print("=" * 70)
    print("🌐 Training ML Models on REAL Healthcare Data")
    print("=" * 70)
    
    # Initialize trainer
    trainer = RealDataModelTrainer('data/real_training_data.csv')
    
    # Load data
    X, y = trainer.load_data()
    if X is None:
        return
    
    # Split data
    print("\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training: {len(X_train)} samples")
    print(f"✅ Testing: {len(X_test)} samples")
    
    # Train models
    trainer.train_advanced_models(X_train, y_train)
    
    # Evaluate all models
    all_metrics = {}
    
    all_metrics['Random Forest'] = trainer.evaluate_model(
        trainer.models['random_forest'], X_test, y_test, 'Random Forest', scale=False
    )
    
    all_metrics['Gradient Boosting'] = trainer.evaluate_model(
        trainer.models['gradient_boosting'], X_test, y_test, 'Gradient Boosting', scale=False
    )
    
    all_metrics['Neural Network'] = trainer.evaluate_model(
        trainer.models['neural_network'], X_test, y_test, 'Neural Network', scale=True
    )
    
    all_metrics['Voting Ensemble'] = trainer.evaluate_model(
        trainer.models['voting_ensemble'], X_test, y_test, 'Voting Ensemble', scale=False
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Final Model Comparison")
    print("=" * 70)
    comparison_df = pd.DataFrame(all_metrics).T
    print(comparison_df)
    
    best_model = comparison_df['roc_auc'].idxmax()
    print(f"\n🏆 Best Model: {best_model} (AUC: {comparison_df.loc[best_model, 'roc_auc']:.4f})")
    
    # Save models
    trainer.save_models()
    
    # Generate report
    trainer.generate_report(all_metrics)
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print("\n🎉 Models trained on REAL healthcare data!")
    print(f"📂 Models saved in 'models/' directory")
    print(f"📊 Report: models/real_ai_training_report.md")
    print("\n🚀 Next: Update backend to use these models (see report)")


if __name__ == "__main__":
    main()
