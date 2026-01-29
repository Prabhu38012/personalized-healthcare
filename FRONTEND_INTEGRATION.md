# Frontend Integration Guide

## Displaying Enhanced Diagnosis Results

Your frontend should now display the enriched data from the consultation endpoint.

## Updated API Response Structure

```typescript
interface ConsultationResponse {
  success: boolean;
  data: {
    // Existing fields
    transcript: string;
    summary: {
      chief_complaint: string;
      assessment: string;
      plan: string;
      // NEW FIELDS
      diagnosis: string;
      risk_factors: string;
      symptoms_detailed: string;
    };
    prescriptions: Prescription[];
    
    // NEW SECTIONS
    extracted_features: {
      blood_pressure?: number;
      blood_pressure_systolic?: number;
      blood_pressure_diastolic?: number;
      cholesterol?: number;
      glucose?: number;
      heart_rate?: number;
      age?: number;
      bmi?: number;
      smoking: boolean;
      family_history: boolean;
      chest_pain: boolean;
      shortness_of_breath: boolean;
      anxiety: boolean;
      [key: string]: any;
    };
    
    diagnosis_data: {
      primary_diagnosis: string;
      differential_diagnoses: string[];
      urgency: 'EMERGENT' | 'URGENT' | 'ROUTINE';
      risk_level: 'HIGH' | 'MODERATE' | 'LOW';
      diagnosis_confidence: 'HIGH' | 'MODERATE' | 'LOW';
      requires_immediate_action: boolean;
      recommended_tests: string[];
      model_input_features: {
        hypertension: boolean;
        high_cholesterol: boolean;
        diabetes: boolean;
        smoking: boolean;
        family_history: boolean;
        chest_pain: boolean;
        shortness_of_breath: boolean;
        anxiety: boolean;
        [key: string]: boolean;
      };
    };
    
    patient_data_for_ml: {
      [key: string]: any;  // All features ready for ML prediction
    };
  };
}
```

## UI/UX Recommendations

### 1. Urgency Alert Banner

Show at top of results page if urgent:

```jsx
{data.diagnosis_data.requires_immediate_action && (
  <Alert severity="error" icon={<WarningIcon />}>
    <AlertTitle>⚠️ URGENT - Immediate Action Required</AlertTitle>
    <strong>Urgency: {data.diagnosis_data.urgency}</strong>
    <br />
    <strong>Risk Level: {data.diagnosis_data.risk_level}</strong>
    <br />
    This case requires immediate clinical attention and specialist referral.
  </Alert>
)}
```

### 2. Diagnosis Section (NEW)

```jsx
<Card>
  <CardHeader title="🎯 Diagnosis" />
  <CardContent>
    <Typography variant="h6" color="primary">
      Primary Diagnosis
    </Typography>
    <Typography variant="body1" gutterBottom>
      {data.summary.diagnosis || data.diagnosis_data.primary_diagnosis}
    </Typography>
    
    {data.diagnosis_data.differential_diagnoses?.length > 0 && (
      <>
        <Typography variant="h6" color="textSecondary" style={{marginTop: 16}}>
          Differential Diagnoses
        </Typography>
        <List>
          {data.diagnosis_data.differential_diagnoses.map((dx, i) => (
            <ListItem key={i}>
              <ListItemIcon><CheckCircle /></ListItemIcon>
              <ListItemText primary={dx} />
            </ListItem>
          ))}
        </List>
      </>
    )}
    
    <Divider style={{margin: '16px 0'}} />
    
    <Grid container spacing={2}>
      <Grid item xs={4}>
        <Chip 
          label={`Urgency: ${data.diagnosis_data.urgency}`}
          color={data.diagnosis_data.urgency === 'URGENT' ? 'error' : 'default'}
        />
      </Grid>
      <Grid item xs={4}>
        <Chip 
          label={`Risk: ${data.diagnosis_data.risk_level}`}
          color={data.diagnosis_data.risk_level === 'HIGH' ? 'error' : 
                 data.diagnosis_data.risk_level === 'MODERATE' ? 'warning' : 'success'}
        />
      </Grid>
      <Grid item xs={4}>
        <Chip 
          label={`Confidence: ${data.diagnosis_data.diagnosis_confidence}`}
          color="primary"
        />
      </Grid>
    </Grid>
  </CardContent>
</Card>
```

### 3. Risk Factors Section (NEW)

```jsx
<Card>
  <CardHeader title="⚠️ Risk Factors" />
  <CardContent>
    <Typography variant="body1" paragraph>
      {data.summary.risk_factors}
    </Typography>
    
    <Typography variant="h6" gutterBottom style={{marginTop: 16}}>
      Identified Conditions
    </Typography>
    <Grid container spacing={1}>
      {Object.entries(data.diagnosis_data.model_input_features).map(([key, value]) => (
        value && (
          <Grid item key={key}>
            <Chip 
              label={key.replace(/_/g, ' ').toUpperCase()}
              color="warning"
              icon={<Warning />}
            />
          </Grid>
        )
      ))}
    </Grid>
  </CardContent>
</Card>
```

### 4. Extracted Features Panel (NEW)

```jsx
<Card>
  <CardHeader title="📊 Medical Features Extracted" />
  <CardContent>
    <Grid container spacing={2}>
      {data.extracted_features.blood_pressure && (
        <Grid item xs={6} md={3}>
          <Paper elevation={2} style={{padding: 16, textAlign: 'center'}}>
            <Typography variant="caption" color="textSecondary">
              Blood Pressure
            </Typography>
            <Typography variant="h6">
              {data.extracted_features.blood_pressure_systolic || data.extracted_features.blood_pressure}
              /{data.extracted_features.blood_pressure_diastolic || '?'} mmHg
            </Typography>
          </Paper>
        </Grid>
      )}
      
      {data.extracted_features.cholesterol && (
        <Grid item xs={6} md={3}>
          <Paper elevation={2} style={{padding: 16, textAlign: 'center'}}>
            <Typography variant="caption" color="textSecondary">
              Cholesterol
            </Typography>
            <Typography variant="h6">
              {data.extracted_features.cholesterol} mg/dL
            </Typography>
          </Paper>
        </Grid>
      )}
      
      {data.extracted_features.heart_rate && (
        <Grid item xs={6} md={3}>
          <Paper elevation={2} style={{padding: 16, textAlign: 'center'}}>
            <Typography variant="caption" color="textSecondary">
              Heart Rate
            </Typography>
            <Typography variant="h6">
              {data.extracted_features.heart_rate} bpm
            </Typography>
          </Paper>
        </Grid>
      )}
      
      {data.extracted_features.glucose && (
        <Grid item xs={6} md={3}>
          <Paper elevation={2} style={{padding: 16, textAlign: 'center'}}>
            <Typography variant="caption" color="textSecondary">
              Blood Glucose
            </Typography>
            <Typography variant="h6">
              {data.extracted_features.glucose} mg/dL
            </Typography>
          </Paper>
        </Grid>
      )}
    </Grid>
    
    <Divider style={{margin: '16px 0'}} />
    
    <Typography variant="h6" gutterBottom>
      Symptoms Present
    </Typography>
    <Grid container spacing={1}>
      {data.extracted_features.chest_pain && (
        <Grid item>
          <Chip label="Chest Pain" color="error" icon={<Favorite />} />
        </Grid>
      )}
      {data.extracted_features.shortness_of_breath && (
        <Grid item>
          <Chip label="Shortness of Breath" color="error" icon={<Air />} />
        </Grid>
      )}
      {data.extracted_features.anxiety && (
        <Grid item>
          <Chip label="Anxiety" color="warning" icon={<Psychology />} />
        </Grid>
      )}
      {data.extracted_features.smoking && (
        <Grid item>
          <Chip label="Smoking" color="warning" icon={<SmokingRooms />} />
        </Grid>
      )}
      {data.extracted_features.family_history && (
        <Grid item>
          <Chip label="Family History" color="info" icon={<FamilyRestroom />} />
        </Grid>
      )}
    </Grid>
  </CardContent>
</Card>
```

### 5. Recommended Tests Section (NEW)

```jsx
<Card>
  <CardHeader title="🧪 Recommended Tests" />
  <CardContent>
    {data.diagnosis_data.recommended_tests?.length > 0 ? (
      <List>
        {data.diagnosis_data.recommended_tests.map((test, i) => (
          <ListItem key={i}>
            <ListItemIcon>
              <Science color={data.diagnosis_data.requires_immediate_action ? "error" : "primary"} />
            </ListItemIcon>
            <ListItemText 
              primary={test}
              secondary={data.diagnosis_data.requires_immediate_action ? "Urgent - Order immediately" : "Recommended"}
            />
            <ListItemSecondaryAction>
              <Button 
                variant="outlined" 
                size="small"
                color={data.diagnosis_data.requires_immediate_action ? "error" : "primary"}
              >
                Order Test
              </Button>
            </ListItemSecondaryAction>
          </ListItem>
        ))}
      </List>
    ) : (
      <Typography color="textSecondary">
        No specific tests recommended at this time.
      </Typography>
    )}
  </CardContent>
</Card>
```

### 6. AI Prediction Button (Integration)

Add button to send data to AI prediction:

```jsx
<Button
  variant="contained"
  color="primary"
  startIcon={<Psychology />}
  onClick={() => handleAIPrediction(data.patient_data_for_ml)}
  disabled={!data.patient_data_for_ml || Object.keys(data.patient_data_for_ml).length === 0}
>
  Get AI Risk Assessment
</Button>

// Handler
const handleAIPrediction = async (patientData) => {
  try {
    const response = await fetch('/ai-decision/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        patient_data: patientData,
        include_explanation: true,
        include_lifestyle_plan: true
      })
    });
    const prediction = await response.json();
    
    // Display prediction results
    setAIPrediction(prediction);
    setShowPredictionDialog(true);
  } catch (error) {
    console.error('AI Prediction failed:', error);
    showErrorNotification('Failed to get AI prediction');
  }
};
```

### 7. Complete Page Layout

```jsx
function ConsultationResultsPage({ data }) {
  return (
    <Container maxWidth="lg">
      {/* Urgency Alert */}
      {data.diagnosis_data.requires_immediate_action && <UrgencyAlert />}
      
      <Grid container spacing={3}>
        {/* Transcript Section */}
        <Grid item xs={12}>
          <TranscriptCard transcript={data.transcript} />
        </Grid>
        
        {/* Diagnosis - PROMINENT */}
        <Grid item xs={12}>
          <DiagnosisCard 
            summary={data.summary}
            diagnosis={data.diagnosis_data}
          />
        </Grid>
        
        {/* Risk Factors */}
        <Grid item xs={12} md={6}>
          <RiskFactorsCard 
            riskFactors={data.summary.risk_factors}
            modelFeatures={data.diagnosis_data.model_input_features}
          />
        </Grid>
        
        {/* Extracted Features */}
        <Grid item xs={12} md={6}>
          <ExtractedFeaturesCard features={data.extracted_features} />
        </Grid>
        
        {/* Recommended Tests */}
        <Grid item xs={12} md={6}>
          <RecommendedTestsCard 
            tests={data.diagnosis_data.recommended_tests}
            urgent={data.diagnosis_data.requires_immediate_action}
          />
        </Grid>
        
        {/* Treatment Plan */}
        <Grid item xs={12} md={6}>
          <TreatmentPlanCard plan={data.summary.plan} />
        </Grid>
        
        {/* Prescriptions */}
        {data.prescriptions?.length > 0 && (
          <Grid item xs={12}>
            <PrescriptionsCard prescriptions={data.prescriptions} />
          </Grid>
        )}
        
        {/* AI Prediction Button */}
        <Grid item xs={12}>
          <Paper style={{padding: 24, textAlign: 'center'}}>
            <Typography variant="h6" gutterBottom>
              Get Advanced AI Risk Assessment
            </Typography>
            <Button
              variant="contained"
              color="primary"
              size="large"
              startIcon={<Psychology />}
              onClick={() => handleAIPrediction(data.patient_data_for_ml)}
            >
              Calculate Risk Score with AI/ML Models
            </Button>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}
```

## Color Coding Recommendations

### Urgency
- `EMERGENT`: Red (#d32f2f)
- `URGENT`: Orange (#f57c00)
- `ROUTINE`: Blue (#1976d2)

### Risk Level
- `HIGH`: Red (#d32f2f)
- `MODERATE`: Orange (#f57c00)
- `LOW`: Green (#388e3c)

### Confidence
- `HIGH`: Green (#388e3c)
- `MODERATE`: Orange (#f57c00)
- `LOW`: Grey (#757575)

## Example Complete Component

```jsx
import React, { useState } from 'react';
import {
  Container, Grid, Card, CardHeader, CardContent,
  Typography, Chip, Button, Alert, AlertTitle,
  List, ListItem, ListItemIcon, ListItemText,
  Paper, Divider
} from '@mui/material';
import {
  Warning, CheckCircle, Science, Psychology,
  Favorite, Air, SmokingRooms, FamilyRestroom
} from '@mui/icons-material';

export function EnhancedConsultationResults({ consultationData }) {
  const { data } = consultationData;
  const [aiPrediction, setAIPrediction] = useState(null);

  return (
    <Container maxWidth="lg" style={{paddingTop: 24}}>
      {/* Urgency Banner */}
      {data.diagnosis_data.requires_immediate_action && (
        <Alert severity="error" style={{marginBottom: 24}}>
          <AlertTitle>⚠️ URGENT - Immediate Action Required</AlertTitle>
          <strong>Urgency: {data.diagnosis_data.urgency}</strong> | 
          <strong> Risk Level: {data.diagnosis_data.risk_level}</strong>
          <br />
          This case requires immediate clinical attention.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Diagnosis Card */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardHeader 
              title="🎯 Diagnosis & Assessment"
              style={{backgroundColor: '#1976d2', color: 'white'}}
            />
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Primary Diagnosis
              </Typography>
              <Typography variant="body1" paragraph>
                {data.summary.diagnosis || data.diagnosis_data.primary_diagnosis}
              </Typography>

              {data.diagnosis_data.differential_diagnoses?.length > 0 && (
                <>
                  <Typography variant="h6" color="textSecondary" gutterBottom>
                    Differential Diagnoses
                  </Typography>
                  <List dense>
                    {data.diagnosis_data.differential_diagnoses.map((dx, i) => (
                      <ListItem key={i}>
                        <ListItemIcon><CheckCircle /></ListItemIcon>
                        <ListItemText primary={dx} />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}

              <Divider style={{margin: '16px 0'}} />

              <Grid container spacing={2}>
                <Grid item>
                  <Chip 
                    label={`Urgency: ${data.diagnosis_data.urgency}`}
                    color={data.diagnosis_data.urgency === 'URGENT' ? 'error' : 'default'}
                  />
                </Grid>
                <Grid item>
                  <Chip 
                    label={`Risk: ${data.diagnosis_data.risk_level}`}
                    color={data.diagnosis_data.risk_level === 'HIGH' ? 'error' : 'warning'}
                  />
                </Grid>
                <Grid item>
                  <Chip 
                    label={`Confidence: ${data.diagnosis_data.diagnosis_confidence}`}
                    color="primary"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Add other sections here... */}
      </Grid>
    </Container>
  );
}
```

## Notes

1. **Backward Compatible**: All new fields are optional, old code still works
2. **Progressive Enhancement**: Show new sections only if data exists
3. **Responsive**: Use Material-UI Grid for mobile-friendly layout
4. **Accessibility**: Use semantic HTML and ARIA labels
5. **Error Handling**: Check for field existence before displaying

## Testing

Test with the heart problem case:
1. Upload the audio file
2. Verify all new sections display
3. Check urgency banner appears
4. Confirm AI prediction button works
5. Validate data flows to AI endpoint

---

**Need help with implementation?** Check the API response in browser DevTools Network tab to see the actual structure.
