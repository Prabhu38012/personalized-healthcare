# Real Healthcare Data - ML Model Training Report

**Training Date**: 2026-01-25 10:06:27

**Data Source**: Real public healthcare datasets
**Dataset**: data/real_training_data.csv

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest        | 0.9202 | 0.9652 | 0.8810 | 0.9212 | 0.9621 |
| Gradient Boosting    | 0.9034 | 0.9328 | 0.8810 | 0.9061 | 0.9582 |
| Neural Network       | 0.8950 | 0.9469 | 0.8492 | 0.8954 | 0.9537 |
| Voting Ensemble      | 0.9034 | 0.9328 | 0.8810 | 0.9061 | 0.9627 |

**Best Model**: Voting Ensemble (AUC: 0.9627)

## Features

Total features: 14

```
age, blood_pressure, cholesterol, bmi, glucose, smoking, exercise, diabetes, family_history, heart_rate, stress_level, sleep_hours, alcohol, on_medication
```

## Usage

Models trained on real healthcare data:
- `real_ai_random_forest_model.pkl`
- `real_ai_gradient_boosting_model.pkl`
- `real_ai_neural_network_model.pkl`
- `real_ai_voting_ensemble_model.pkl`
- `real_ai_scaler.pkl`
- `real_ai_feature_names.pkl`
- `real_ai_ensemble_config.pkl`

## Integration

To use these models in the backend, update `backend/routes/ai_decision_support.py`:

```python
# Change model file paths to use real data models
rf_model = joblib.load('models/real_ai_random_forest_model.pkl')
gb_model = joblib.load('models/real_ai_gradient_boosting_model.pkl')
nn_model = joblib.load('models/real_ai_neural_network_model.pkl')
```
