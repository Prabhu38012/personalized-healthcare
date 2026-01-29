import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import prediction routes
predict_router = None
predict_available = False

try:
    from routes.predict import router as predict_router
    predict_available = True
    print("✓ Prediction routes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import prediction routes: {e}")
    predict_router = None
    predict_available = False
    print("⚠️  Prediction module not available")

# Import health log routes
health_log_router = None
health_log_available = False

try:
    from routes.health_log import router as health_log_router
    health_log_available = True
    print("✓ Health log routes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import health log routes: {e}")
    health_log_router = None
    health_log_available = False
    print("⚠️  Health log module not available")

# Import document analysis routes
document_router = None
document_available = False

try:
    from routes.document_analysis import router as document_router
    document_available = True
    print("✓ Document analysis routes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import document analysis routes: {e}")
    document_router = None
    document_available = False
    print("⚠️  Document analysis module not available")

# Import consultation routes
consultation_router = None
consultation_available = False

try:
    from routes.consultation import router as consultation_router
    consultation_available = True
    print("✓ Medical consultation routes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import consultation routes: {e}")
    consultation_router = None
    consultation_available = False
    print("⚠️  Medical consultation module not available")

# Import authentication routes with absolute imports first
auth_router = None
auth_available = False

try:
    # Try relative import first when running from backend directory
    from auth.database_store import get_db
    from auth.routes import router as auth_router
    auth_available = True
    print("✓ Authentication routes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import auth routes: {e}")
    auth_router = None
    auth_available = False
    print("⚠️  Authentication module not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personalized Healthcare Recommendation API",
    description="AI-powered healthcare recommendation system",
    version="1.0.0"
)

# CORS middleware for frontend integration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501,http://localhost:8503,http://localhost:8000,http://localhost:8002").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Load from environment variables
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
if predict_available and predict_router:
    app.include_router(predict_router, prefix="/api", tags=["predictions"])
    print("✓ Prediction routes registered at /api")
else:
    print("⚠️  Running without prediction functionality")

if health_log_available and health_log_router:
    app.include_router(health_log_router, prefix="/api/health-log", tags=["health-log"])
    print("✓ Health log routes registered at /api/health-log")
    print("Available health log endpoints:")
    print("  - POST /api/health-log/")
    print("  - GET /api/health-log/")
    print("  - PUT /api/health-log/{entry_id}")
    print("  - DELETE /api/health-log/{entry_id}")
    print("  - GET /api/health-log/statistics")
    print("  - GET /api/health-log/health")
else:
    print("⚠️  Running without health log functionality")

if auth_available and auth_router:
    app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
    print("✓ Authentication routes registered at /api/auth")
    print("Available auth endpoints:")
    print("  - POST /api/auth/login")
    print("  - GET /api/auth/default-users")
    print("  - GET /api/auth/health")
    print("  - GET /api/auth/test-connection")
else:
    print("⚠️  Running in demo mode without authentication")
    print("  - Authentication endpoints will return 404")

if consultation_available and consultation_router:
    app.include_router(consultation_router, prefix="/api/consultation", tags=["consultation"])
    print("✓ Medical consultation routes registered at /api/consultation")
    print("Available consultation endpoints:")
    print("  - POST /api/consultation/process")
    print("  - GET /api/consultation/history")
    print("  - GET /api/consultation/report/<filename>")
    print("  - GET /api/consultation/status")
else:
    print("⚠️  Running without medical consultation functionality")

if document_available and document_router:
    app.include_router(document_router, prefix="/api/document", tags=["document-analysis"])
    print("✓ Document analysis routes registered at /api/document")
    print("Available document endpoints:")
    print("  - POST /api/document/upload/medical-report")
    print("  - POST /api/document/upload/prescription")
    print("  - GET /api/document/analysis/{analysis_id}")
    print("  - GET /api/document/list")
    print("  - DELETE /api/document/analysis/{analysis_id}")
    print("  - GET /api/document/health")
else:
    print("⚠️  Running without document analysis functionality")

# Import AI decision support routes
ai_decision_router = None
ai_decision_available = False

try:
    from routes.ai_decision_support import router as ai_decision_router
    ai_decision_available = True
    print("✓ AI Decision Support routes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import AI decision support routes: {e}")
    ai_decision_router = None
    ai_decision_available = False
    print("⚠️  AI Decision Support module not available")

if ai_decision_available and ai_decision_router:
    app.include_router(ai_decision_router, prefix="/api", tags=["ai-decision-support"])
    print("✓ AI Decision Support routes registered at /api/ai-decision")
    print("Available AI endpoints:")
    print("  - POST /api/ai-decision/predict")
    print("  - POST /api/ai-decision/pattern-analysis")
    print("  - POST /api/ai-decision/real-time-monitoring")
    print("  - GET /api/ai-decision/health")
    print("  - GET /api/ai-decision/model-info")
else:
    print("⚠️  Running without AI Decision Support functionality")

@app.get("/")
async def root():
    return {"message": "Healthcare Recommendation API is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Backend service is running"}

@app.get("/api/version")
async def version_info():
    """Get API version information"""
    try:
        # Import here to avoid circular imports
        try:
            from routes.predict import load_model
        except ImportError:
            from backend.routes.predict import load_model
        
        model_data = load_model()
        return {
            "version": "1.0.0",
            "model_status": "loaded" if model_data else "not_loaded",
            "features": model_data['feature_columns'] if model_data else []
        }
    except:
        return {
            "version": "1.0.0",
            "model_status": "not_loaded",
            "features": []
        }

@app.get("/api/test")
async def test_endpoint():
    return {"message": "This is a test endpoint"}

@app.get("/api/model-info")
async def get_model_info():
    """Get comprehensive model information"""
    info = {
        "version": "1.0.0",
        "status": "operational",
        "models": {}
    }
    
    # Get AI Decision Support model info
    if ai_decision_available:
        try:
            from routes.ai_decision_support import MODEL_SOURCE, disease_targets, feature_names
            info["models"]["ai_decision"] = {
                "source": MODEL_SOURCE,
                "available": True,
                "diseases": disease_targets if 'disease_targets' in dir() else ["high_risk"],
                "features_count": len(feature_names) if 'feature_names' in dir() else 0
            }
        except:
            info["models"]["ai_decision"] = {"available": False}
    
    # Get prediction model info
    if predict_available:
        try:
            from routes.predict import load_model
            model_data = load_model()
            if model_data:
                info["models"]["prediction"] = {
                    "available": True,
                    "features": model_data.get('feature_columns', []),
                    "type": model_data.get('model_type', 'unknown')
                }
            else:
                info["models"]["prediction"] = {"available": False}
        except:
            info["models"]["prediction"] = {"available": False}
    
    return info

if __name__ == "__main__":
    # Increase timeout for large audio file processing with AI models
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        timeout_keep_alive=900  # 15 minutes for large files
    )
