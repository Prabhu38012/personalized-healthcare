# Multi-Disease Prediction Model - Training Report

**Date**: 2026-01-25 11:47:43

**Dataset**: data/multi_disease_training.csv

## Diseases Covered

- Heart Disease
- Diabetes
- Hypertension
- Obesity
- Kidney Disease
- Liver Disease
- Metabolic Syndrome
- High Risk

## Model Performance

### Average Performance Across All Diseases

| Model | Avg Accuracy | Avg F1-Score |
|-------|--------------|-------------|
| random_forest | 0.9576 | 0.8985 |
| gradient_boosting | 0.9619 | 0.9061 |
| neural_network | 0.9426 | 0.8634 |

### RANDOM_FOREST - Disease-Specific Performance

| Disease | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| heart_disease | 0.9135 | 0.8858 | 0.8670 | 0.8763 |
| diabetes | 0.9355 | 0.8985 | 0.8399 | 0.8682 |
| hypertension | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| obesity | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kidney_disease | 0.9505 | 0.8723 | 0.7961 | 0.8325 |
| liver_disease | 0.9365 | 0.7402 | 0.6711 | 0.7040 |
| metabolic_syndrome | 0.9945 | 1.0000 | 0.9804 | 0.9901 |
| high_risk | 0.9305 | 0.9254 | 0.9090 | 0.9171 |

### GRADIENT_BOOSTING - Disease-Specific Performance

| Disease | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| heart_disease | 0.9310 | 0.9178 | 0.8840 | 0.9006 |
| diabetes | 0.9415 | 0.8961 | 0.8696 | 0.8826 |
| hypertension | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| obesity | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kidney_disease | 0.9510 | 0.8576 | 0.8188 | 0.8377 |
| liver_disease | 0.9310 | 0.6867 | 0.7111 | 0.6987 |
| metabolic_syndrome | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| high_risk | 0.9410 | 0.9428 | 0.9161 | 0.9293 |

### NEURAL_NETWORK - Disease-Specific Performance

| Disease | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| heart_disease | 0.9210 | 0.8950 | 0.8798 | 0.8873 |
| diabetes | 0.9300 | 0.8560 | 0.8696 | 0.8627 |
| hypertension | 0.9850 | 0.9634 | 0.9293 | 0.9460 |
| obesity | 0.9775 | 0.9449 | 0.9321 | 0.9384 |
| kidney_disease | 0.9355 | 0.7961 | 0.7832 | 0.7896 |
| liver_disease | 0.9235 | 0.6452 | 0.7111 | 0.6765 |
| metabolic_syndrome | 0.9370 | 0.8820 | 0.8946 | 0.8883 |
| high_risk | 0.9315 | 0.9307 | 0.9054 | 0.9179 |

## Usage

Models can predict multiple diseases simultaneously:

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/multi_disease_random_forest_model.pkl')

# Predict
predictions = model.predict(patient_data)
# Returns: [heart_disease, diabetes, hypertension, obesity, ...]
```
