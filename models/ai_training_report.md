# AI Decision Support - Model Training Report

**Training Date**: 2026-01-25 09:48:11

**Dataset Size**: 5000 samples
**Features**: 14
**Train/Test Split**: 4000/1000

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.8700 | 0.8105 | 0.7524 | 0.7804 | 0.9407 |
| Gradient Boosting | 0.8720 | 0.7993 | 0.7785 | 0.7888 | 0.9358 |
| Neural Network | 0.8730 | 0.8082 | 0.7687 | 0.7880 | 0.9437 |

**Best Model**: Neural Network

## Feature Names

```
age, blood_pressure, cholesterol, bmi, glucose, smoking, exercise, diabetes, family_history, heart_rate, stress_level, sleep_hours, alcohol, on_medication
```

## Ensemble Configuration

- Random Forest: 35%
- Gradient Boosting: 40%
- Neural Network: 25%

## Usage

Models are saved in `models/` directory:
- `ai_random_forest_model.pkl`
- `ai_gradient_boosting_model.pkl`
- `ai_neural_network_model.pkl`
- `ai_scaler.pkl` (for neural network)
- `ai_feature_names.pkl`
- `ai_ensemble_config.pkl`
