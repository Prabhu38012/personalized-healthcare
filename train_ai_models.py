"""
Train actual ML models for AI Decision Support
Trains Random Forest, Gradient Boosting, and Neural Network models
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

class AIModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, filepath='data/ml_training_data.csv'):
        """Load and prepare training data"""
        print(f"📂 Loading data from {filepath}...")
        data = pd.read_csv(filepath)
        
        # Handle missing values
        data = data.fillna(data.median(numeric_only=True))
        
        # Separate features and target
        self.feature_names = [col for col in data.columns if col != 'high_risk']
        X = data[self.feature_names]
        y = data['high_risk']
        
        print(f"✅ Loaded {len(data)} samples with {len(self.feature_names)} features")
        print(f"📊 Features: {', '.join(self.feature_names)}")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_random_forest(self, X_train, y_train, tune=True):
        """Train Random Forest classifier"""
        print("\n🌲 Training Random Forest Classifier...")
        
        if tune:
            print("🔍 Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train, tune=True):
        """Train Gradient Boosting classifier"""
        print("\n🚀 Training Gradient Boosting Classifier...")
        
        if tune:
            print("🔍 Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                gb, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
        else:
            model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.9,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        return model
    
    def train_neural_network(self, X_train, y_train):
        """Train Neural Network classifier"""
        print("\n🧠 Training Neural Network Classifier...")
        
        # Scale features for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        )
        
        model.fit(X_train_scaled, y_train)
        
        self.models['neural_network'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name, scale=False):
        """Evaluate model performance"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Prepare test data
        if scale:
            X_test_eval = self.scaler.transform(X_test)
        else:
            X_test_eval = X_test
        
        # Predictions
        y_pred = model.predict(X_test_eval)
        y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print results
        print(f"✅ {model_name} Results:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f} ({value*100:.2f}%)")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n🎯 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 10 Important Features:")
            print(importances.head(10))
        
        return metrics
    
    def save_models(self):
        """Save all trained models"""
        print("\n💾 Saving models...")
        
        # Save individual models
        for name, model in self.models.items():
            filename = f'models/ai_{name}_model.pkl'
            joblib.dump(model, filename)
            print(f"✅ Saved {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, 'models/ai_scaler.pkl')
        print("✅ Saved models/ai_scaler.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, 'models/ai_feature_names.pkl')
        print("✅ Saved models/ai_feature_names.pkl")
    
    def create_ensemble_predictor(self):
        """Create ensemble prediction function"""
        print("\n🎭 Creating ensemble predictor...")
        
        ensemble_config = {
            'models': list(self.models.keys()),
            'weights': {
                'random_forest': 0.35,
                'gradient_boosting': 0.40,
                'neural_network': 0.25
            },
            'feature_names': self.feature_names
        }
        
        joblib.dump(ensemble_config, 'models/ai_ensemble_config.pkl')
        print("✅ Saved ensemble configuration")


def main():
    print("=" * 70)
    print("🤖 AI Decision Support - Machine Learning Model Training")
    print("=" * 70)
    
    # Initialize trainer
    trainer = AIModelTrainer()
    
    # Load data
    X, y = trainer.load_data('data/ml_training_data.csv')
    
    # Split data
    print("\n📊 Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training set: {len(X_train)} samples")
    print(f"✅ Test set: {len(X_test)} samples")
    
    # Train models
    print("\n" + "=" * 70)
    print("🎯 Training Models")
    print("=" * 70)
    
    all_metrics = {}
    
    # Random Forest
    rf_model = trainer.train_random_forest(X_train, y_train, tune=False)
    all_metrics['Random Forest'] = trainer.evaluate_model(
        rf_model, X_test, y_test, 'Random Forest', scale=False
    )
    
    # Gradient Boosting
    gb_model = trainer.train_gradient_boosting(X_train, y_train, tune=False)
    all_metrics['Gradient Boosting'] = trainer.evaluate_model(
        gb_model, X_test, y_test, 'Gradient Boosting', scale=False
    )
    
    # Neural Network
    nn_model = trainer.train_neural_network(X_train, y_train)
    all_metrics['Neural Network'] = trainer.evaluate_model(
        nn_model, X_test, y_test, 'Neural Network', scale=True
    )
    
    # Compare models
    print("\n" + "=" * 70)
    print("📊 Model Comparison")
    print("=" * 70)
    
    comparison_df = pd.DataFrame(all_metrics).T
    print(comparison_df)
    
    # Find best model
    best_model_name = comparison_df['roc_auc'].idxmax()
    print(f"\n🏆 Best Model: {best_model_name} (AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f})")
    
    # Save models
    trainer.save_models()
    trainer.create_ensemble_predictor()
    
    # Generate report
    print("\n📝 Generating training report...")
    report_path = 'models/ai_training_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# AI Decision Support - Model Training Report\n\n")
        f.write(f"**Training Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Dataset Size**: {len(X)} samples\n")
        f.write(f"**Features**: {len(trainer.feature_names)}\n")
        f.write(f"**Train/Test Split**: {len(X_train)}/{len(X_test)}\n\n")
        
        f.write("## Model Performance\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 Score | AUC |\n")
        f.write("|-------|----------|-----------|--------|----------|---------|\n")
        
        for model_name, metrics in all_metrics.items():
            f.write(f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} |\n")
        
        f.write(f"\n**Best Model**: {best_model_name}\n\n")
        
        f.write("## Feature Names\n\n")
        f.write("```\n")
        f.write(", ".join(trainer.feature_names))
        f.write("\n```\n\n")
        
        f.write("## Ensemble Configuration\n\n")
        f.write("- Random Forest: 35%\n")
        f.write("- Gradient Boosting: 40%\n")
        f.write("- Neural Network: 25%\n\n")
        
        f.write("## Usage\n\n")
        f.write("Models are saved in `models/` directory:\n")
        f.write("- `ai_random_forest_model.pkl`\n")
        f.write("- `ai_gradient_boosting_model.pkl`\n")
        f.write("- `ai_neural_network_model.pkl`\n")
        f.write("- `ai_scaler.pkl` (for neural network)\n")
        f.write("- `ai_feature_names.pkl`\n")
        f.write("- `ai_ensemble_config.pkl`\n")
    
    print(f"✅ Saved training report: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print("\n🎉 All models trained and saved successfully!")
    print("📂 Models saved in 'models/' directory")
    print("📊 Training report: models/ai_training_report.md")
    print("\n🚀 Next step: Models are ready to be integrated into AI Decision Support")


if __name__ == "__main__":
    main()
