from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import joblib
import pandas as pd
import numpy as np
from routes.predict import router as predict_router

app = FastAPI(
    title="Personalized Healthcare Recommendation API",
    description="AI-powered healthcare recommendation system",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction routes
app.include_router(predict_router, prefix="/api", tags=["predictions"])

@app.get("/")
async def root():
    return {"message": "Healthcare Recommendation API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "healthcare-api"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
