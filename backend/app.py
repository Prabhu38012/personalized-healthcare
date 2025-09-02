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

# Import prediction routes
try:
    # Try absolute import first (more reliable)
    from backend.routes.predict import router as predict_router
    print("✓ Prediction routes imported successfully (absolute)")
except ImportError:
    try:
        # Fallback to relative import
        from routes.predict import router as predict_router
        print("✓ Prediction routes imported successfully (relative)")
    except ImportError as e:
        print(f"✗ Failed to import prediction routes: {e}")
        raise

# Import authentication routes with absolute imports first
auth_router = None
auth_available = False

try:
    # Try absolute import first (more reliable)
    from backend.auth.routes import router as auth_router
    auth_available = True
    print("✓ Authentication routes imported successfully (absolute)")
except ImportError as e:
    print(f"Failed absolute auth import: {e}")
    try:
        # Fallback to relative import
        from auth.routes import router as auth_router
        auth_available = True
        print("✓ Authentication routes imported successfully (relative)")
    except ImportError as e:
        print(f"Failed relative auth import: {e}")
        auth_router = None
        auth_available = False
        print("⚠️  Authentication module not available - running without auth")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personalized Healthcare Recommendation API",
    description="AI-powered healthcare recommendation system",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(predict_router, prefix="/api", tags=["predictions"])
print("✓ Prediction routes registered at /api")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)