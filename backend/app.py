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

try:
    # Try relative import first (when running from backend directory)
    from routes.predict import router as predict_router
except ImportError:
    # Fallback to absolute import (when running from project root)
    from backend.routes.predict import router as predict_router

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

# Include prediction routes
app.include_router(predict_router, prefix="/api", tags=["predictions"])

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