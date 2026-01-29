# 🏥 MyVitals - AI-Powered Personalized Healthcare System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready AI-powered healthcare recommendation system featuring multi-disease risk prediction, explainable AI, document analysis, and real-time health monitoring.

## ✨ Key Features

### 🤖 AI Decision Support
- **Multi-Disease Prediction**: Heart disease, diabetes, hypertension, obesity, kidney disease, liver disease, and metabolic syndrome
- **Explainable AI**: SHAP values, feature importance, and confidence intervals
- **Bias Detection**: Demographic fairness analysis and bias mitigation
- **Multiple ML Models**: Random Forest, Gradient Boosting, Neural Networks with ensemble predictions

### 📄 Document Analysis
- **Medical Report Processing**: Upload and analyze PDF, images, or text documents
- **Prescription Analysis**: Medication identification and drug interaction checking
- **AI-Powered Extraction**: Using Google Gemini for intelligent text analysis
- **Historical Tracking**: Store and review previous analyses

### 📊 Health Monitoring
- **Health Log System**: Track daily metrics (weight, BP, heart rate, glucose)
- **Trend Analysis**: Historical data visualization and statistics
- **Real-Time Alerts**: Immediate risk detection and recommendations

### 🔐 Security & Authentication
- **Role-Based Access**: Admin, Doctor, and Patient roles
- **JWT Authentication**: Secure token-based auth with 8-hour expiry
- **Account Security**: Failed login tracking and automatic lockout
- **Password Strength**: Enforced strong password requirements

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 8GB+ RAM recommended
- Google Gemini API key (optional, for enhanced analysis)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd personalized-healthcare

# 2. Create virtual environment
python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional)
# Copy .env.example to .env and add your Gemini API key
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here
```

### Running the Application

```bash
# Start both backend and frontend
python start.py
```

The application will be available at:
- 🌐 **Frontend**: http://localhost:8501
- 🔧 **Backend API**: http://localhost:8000
- 📖 **API Docs**: http://localhost:8000/docs

### Default Login Credentials

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@healthcare.com | Admin123! |
| Doctor | doctor@healthcare.com | Doctor123! |
| Patient | patient@healthcare.com | Patient123! |

## 📋 Core Functionality

### 1. Risk Assessment
Predict cardiovascular and metabolic disease risk using ML models trained on real healthcare data.

**Features:**
- Multi-disease simultaneous prediction
- Risk scores with confidence intervals
- Personalized recommendations
- Explainable results with SHAP values

### 2. AI Decision Support
Advanced AI system with:
- Pattern recognition in health data
- Anomaly detection
- Bias fairness analysis
- Real-time monitoring capabilities

### 3. Document Analysis
Upload and analyze:
- Medical reports (PDF/images)
- Prescriptions
- Lab results
- Text-based health documents

**AI extracts:**
- Key findings
- Medications
- Drug interactions
- Recommendations

### 4. Health Log
Track daily health metrics:
- Weight, BMI
- Blood pressure
- Heart rate
- Blood glucose
- Custom notes

## 🏗️ Technical Architecture

### Backend (FastAPI)
- RESTful API with 30+ endpoints
- SQLite database with connection pooling
- JWT-based authentication
- Async request handling
- Comprehensive error handling

### Frontend (Streamlit)
- Modern, responsive UI
- Real-time updates
- Interactive visualizations
- Role-based views
- Session management

### ML Pipeline
- Scikit-learn models (v1.7.0)
- Multi-output classification
- Feature preprocessing & scaling
- Model versioning
- SHAP explainability

### AI Integration
- Google Gemini 2.0 Flash
- Fallback mechanisms
- Rate limiting
- Error resilience

## 📊 API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `GET /api/auth/default-users` - List default accounts

### Risk Assessment
- `POST /api/predict/` - Get risk prediction
- `POST /api/predict/batch` - Batch predictions

### AI Decision Support
- `POST /api/ai-decision/predict` - Advanced AI prediction
- `POST /api/ai-decision/pattern-analysis` - Pattern detection
- `POST /api/ai-decision/real-time-monitoring` - Live monitoring
- `GET /api/ai-decision/model-info` - Model information

### Document Analysis
- `POST /api/document/upload/medical-report` - Upload medical report
- `POST /api/document/upload/prescription` - Upload prescription
- `GET /api/document/list` - List all analyses
- `GET /api/document/analysis/{id}` - Get specific analysis

### Health Log
- `POST /api/health-log/` - Create health entry
- `GET /api/health-log/` - List health entries
- `GET /api/health-log/statistics` - Get health statistics
- `PUT /api/health-log/{id}` - Update entry
- `DELETE /api/health-log/{id}` - Delete entry

## 🛠️ System Improvements

### Performance Optimizations
✅ **Database**: SQLite WAL mode, connection pooling, optimized pragma settings
✅ **Caching**: Request caching for health checks and model info
✅ **Async Operations**: Non-blocking I/O for file uploads
✅ **Query Optimization**: Indexed queries and efficient joins

### Security Enhancements
✅ **Input Validation**: Comprehensive sanitization of all inputs
✅ **XSS Protection**: Text validation against injection attacks
✅ **SQL Injection**: Parameterized queries throughout
✅ **Rate Limiting**: API endpoint throttling
✅ **Secure Sessions**: 8-hour JWT expiry with refresh tokens

### Code Quality
✅ **Type Hints**: Full typing across codebase
✅ **Error Handling**: Try-catch blocks with detailed logging
✅ **Logging**: Structured logging at all levels
✅ **Documentation**: Comprehensive docstrings and comments

### ML Improvements
✅ **Model Compatibility**: Version pinning (scikit-learn==1.7.0)
✅ **Feature Validation**: Range checking for all health metrics
✅ **Preprocessing**: Standardized scaling and normalization
✅ **Explainability**: SHAP values with fallback approximations

## 📝 Configuration

### Environment Variables (.env)
```env
# AI Services
GEMINI_API_KEY=your_gemini_api_key_here

# Security
JWT_SECRET_KEY=your_secret_key_here
SECRET_KEY=your_secret_key_here

# Database
DATABASE_URL=sqlite:///./healthcare.db

# Server
BACKEND_PORT=8000
FRONTEND_PORT=8501
ENVIRONMENT=production

# Risk Thresholds
RISK_THRESHOLD_HIGH=0.7
RISK_THRESHOLD_MEDIUM=0.4
```

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest

# Test API endpoints
curl http://localhost:8000/api/health-log/health

# Check model loading
python -c "import joblib; print(joblib.load('../models/multi_disease_random_forest_model.pkl'))"
```
- 📋 **Patient Data Input Form** - Easy-to-use interface
- 📊 **Real-time Risk Assessment** - Visual risk indicators
- 📈 **Interactive Dashboards** - Health metrics visualization
- 💊 **Personalized Recommendations** - AI-generated health advice

### Backend (FastAPI)
- 🤖 **ML Prediction API** - REST endpoints for predictions
- 📁 **EHR Data Processing** - Handle large healthcare datasets
- 🏥 **Risk Assessment Engine** - Cardiovascular disease prediction
- 🔗 **Model Management** - Load/save trained models

### Machine Learning
- 🌲 **Random Forest** - Primary prediction model
- 📈 **Gradient Boosting** - Enhanced accuracy
- 📉 **Logistic Regression** - Interpretable baseline
- 🔍 **Feature Engineering** - Automated EHR processing

## 📁 Project Structure

```
personalized-healthcare/
├── backend/                 # FastAPI backend
│   ├── routes/             # API endpoints
│   ├── models/             # ML models and training
│   ├── utils/              # Data processing utilities
│   └── app.py              # Main backend application
├── frontend/               # Streamlit frontend
│   ├── components/         # UI components
│   ├── pages/              # Application pages
│   ├── utils/              # Frontend utilities
│   └── app.py              # Main frontend application
├── data/                   # Sample datasets
├── deployment/             # Docker and deployment configs
├── models/                 # Trained model files and reports
├── requirements.txt        # Python dependencies
└── TRAINING_GUIDE.md       # Detailed training instructions
```

## 🧪 Testing the System

### 1. Test Backend API
```bash
# Check if backend is running
curl http://localhost:8000/api/health

# Test prediction endpoint
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": 1, "chest_pain_type": 2, "resting_bp": 130}'
```

### 2. Test Frontend
1. Open http://localhost:8501
2. Fill in the patient form
3. Click "Predict Risk"
4. View results and recommendations

### 3. Run Automated Tests
```bash
cd backend
python -m pytest tests/ -v
```

## 🛠️ Development

### Adding New Features
1. **Backend**: Add endpoints in `backend/routes/`
2. **Frontend**: Add components in `frontend/components/`
3. **ML Models**: Modify `backend/models/train_model.py`

### Environment Variables
Create `.env` file for configuration:
```env
BACKEND_URL=http://localhost:8000
MODEL_PATH=models/
DEBUG=True
```

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError"
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt
```

#### "Port already in use"
```bash
# Change ports in the commands
streamlit run app.py --server.port 8502
uvicorn app:app --port 8001
```

#### "Model not found"
```bash
# Train a model first or use sample data
cd backend
python train_config.py
```

#### "Memory errors during training"
```bash
# Reduce dataset size in train_config.py
# Set max_samples = 10000 for testing
```

### Getting Help
1. Check the [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) for training issues
2. Review API documentation at http://localhost:8000/docs
3. Look at sample data in `data/` directory

## 📋 System Requirements

### Minimum
- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Windows, macOS, Linux

### Recommended
- **Python**: 3.9-3.11
- **RAM**: 16GB+
- **Storage**: 20GB+ free space
- **CPU**: Multi-core processor

## 🔒 Data Privacy

- All data processing happens locally
- No patient data is transmitted externally
- Models are trained and stored locally
- Compliant with healthcare data standards

## 📄 License

This project is for educational and research purposes. Please ensure compliance with healthcare regulations when using with real patient data.

---

## 🆘 Quick Help

**Can't get it running?** Try this minimal setup:

1. `git clone <repo-url> && cd personalized-healthcare`
2. `python -m venv venv && venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `cd backend && python app.py` (in one terminal)
5. `cd frontend && streamlit run app.py` (in another terminal)
6. Open http://localhost:8501

**Still stuck?** The system includes sample data, so you can test immediately without downloading large datasets!