"""
FastAPI service for Nova Score credit scoring.
Provides endpoints for model inference and explanations.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Nova Score API",
    description="API for credit scoring Grab partners",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_DIR = Path("models")
DEFAULT_MODEL = "xgboost"
MODEL_CACHE = {}

# Pydantic models for request/response validation
class PartnerData(BaseModel):
    """Input data for credit scoring."""
    partner_id: Optional[str] = None
    months_on_platform: float
    weekly_trips: float
    cancel_rate: float = Field(..., ge=0, le=1, description="Cancel rate (0-1)")
    on_time_rate: float = Field(..., ge=0, le=1, description="On-time rate (0-1)")
    avg_rating: float = Field(..., ge=1, le=5, description="Average rating (1-5)")
    earnings_volatility: float = Field(..., ge=0, description="Earnings volatility (coefficient of variation)")
    region: str = Field(..., description="Region (North, South, East, West, Central)")

class ScoreResponse(BaseModel):
    """Response model for credit score prediction."""
    partner_id: Optional[str]
    score: float = Field(..., ge=0, le=100, description="Credit score (0-100)")
    decision: str = Field(..., description="Approval decision")
    reason_codes: List[Dict[str, Any]] = Field(..., description="Top factors influencing the decision")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    version: str
    training_date: str
    metrics: Dict[str, Any]

# Helper functions
def load_model(model_name: str = DEFAULT_MODEL):
    """Load a trained model from disk with caching."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    
    model_data = joblib.load(model_path)
    MODEL_CACHE[model_name] = model_data
    return model_data

def preprocess_input(data: PartnerData, feature_names: List[str]) -> pd.DataFrame:
    """Convert input data to model-ready format."""
    # Convert to dict and create DataFrame
    input_data = data.dict()
    
    # Create a DataFrame with all expected features
    df = pd.DataFrame([{
        'months_on_platform': input_data['months_on_platform'],
        'weekly_trips': input_data['weekly_trips'],
        'cancel_rate': input_data['cancel_rate'],
        'on_time_rate': input_data['on_time_rate'],
        'avg_rating': input_data['avg_rating'],
        'earnings_volatility': input_data['earnings_volatility'],
        'region': input_data['region']
    }])
    
    # One-hot encode region
    df = pd.get_dummies(df, columns=['region'], drop_first=False)
    
    # Ensure all expected region columns exist
    for region in ['North', 'South', 'East', 'West', 'Central']:
        col_name = f'region_{region}'
        if col_name not in df.columns:
            df[col_name] = 0
    
    # Add any missing features with default values
    for col in feature_names:
        if col not in df.columns and not col.startswith('region_'):
            df[col] = 0.0
    
    # Ensure we only return the features the model expects, in the right order
    df = df[feature_names].copy()
    return df

def get_decision(score: float) -> str:
    """Convert score to decision."""
    if score >= 80:
        return "pre_approved"
    elif score >= 60:
        return "review"
    else:
        return "decline"

def get_reason_codes(model_name: str, input_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get reason codes using SHAP values."""
    if model_name != 'xgboost':
        return [{"feature": "model", "value": "Reason codes not available for this model"}]
    
    try:
        # Load SHAP explainer
        explainer_path = MODEL_DIR / f"{model_name}_explainer.joblib"
        if not explainer_path.exists():
            return [{"feature": "error", "value": "SHAP explainer not found"}]
        
        explainer = joblib.load(explainer_path)
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_data)
        
        # Get feature names
        feature_names = input_data.columns.tolist()
        
        # Get top 3 features by absolute SHAP value
        shap_series = pd.Series(
            np.abs(shap_values[0]),
            index=feature_names
        ).sort_values(ascending=False)
        
        # Create reason codes
        reasons = []
        for feature in shap_series.head(3).index:
            value = input_data[feature].iloc[0]
            impact = shap_values[0][feature_names.index(feature)]
            
            # Human-readable impact
            impact_direction = "increased" if impact > 0 else "decreased"
            
            # Feature-specific messages
            if feature == 'cancel_rate':
                message = f"Cancel rate of {value:.1%} {impact_direction} your score"
            elif feature == 'on_time_rate':
                message = f"On-time rate of {value:.1%} {impact_direction} your score"
            elif feature == 'avg_rating':
                message = f"Average rating of {value:.1f} {impact_direction} your score"
            elif feature == 'weekly_trips':
                message = f"Weekly trips of {value:.0f} {impact_direction} your score"
            elif feature == 'months_on_platform':
                message = f"Tenure of {value:.0f} months {impact_direction} your score"
            elif feature.startswith('region_'):
                region = feature.replace('region_', '')
                message = f"Region ({region}) {impact_direction} your score"
            else:
                message = f"{feature} {impact_direction} your score"
            
            reasons.append({
                "feature": feature,
                "value": value,
                "impact": float(impact),
                "message": message
            })
        
        return reasons
    
    except Exception as e:
        return [{"feature": "error", "value": f"Error generating reason codes: {str(e)}"}]

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Nova Score API",
        "version": "0.1.0",
        "description": "Credit scoring API for Grab partners",
        "endpoints": [
            {"path": "/", "methods": ["GET"], "description": "API information"},
            {"path": "/health", "methods": ["GET"], "description": "Health check"},
            {"path": "/models", "methods": ["GET"], "description": "List available models"},
            {"path": "/score", "methods": ["POST"], "description": "Get credit score"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Try loading the default model as a health check
        load_model(DEFAULT_MODEL)
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "unhealthy", "error": str(e)})

@app.get("/models", response_model=Dict[str, ModelInfo])
async def list_models():
    """List available models and their information."""
    models = {}
    
    # Check for model files
    for model_file in MODEL_DIR.glob("*.joblib"):
        model_name = model_file.stem
        if model_name.endswith("_explainer"):
            continue
            
        meta_path = MODEL_DIR / f"{model_name}_meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                models[model_name] = {
                    "name": model_name,
                    "version": "1.0.0",  # Could be extracted from metadata
                    "training_date": meta.get("training_date", "unknown"),
                    "metrics": meta.get("metrics", {})
                }
    
    return models

@app.post("/score", response_model=ScoreResponse)
async def get_score(
    data: PartnerData,
    model_name: str = DEFAULT_MODEL,
    request: Request = None
):
    """Get credit score and decision for a partner."""
    try:
        # Load model and metadata
        model_data = load_model(model_name)
        model = model_data['model']
        
        # Get feature names with fallback
        feature_names = model_data.get('feature_names', [
            'months_on_platform', 'weekly_trips', 'cancel_rate',
            'on_time_rate', 'avg_rating', 'earnings_volatility',
            'region_Central', 'region_East', 'region_North', 'region_South', 'region_West'
        ])
        
        # Preprocess input
        input_df = preprocess_input(data, feature_names)
        
        # Make prediction
        try:
            if hasattr(model, 'predict_proba'):
                score = model.predict_proba(input_df)[0][1] * 100  # Convert to 0-100 scale
            else:
                score = model.predict(input_df)[0] * 100  # Fallback for models without predict_proba
        except Exception as e:
            # If prediction fails, fall back to a simple scoring model
            score = 50 + (data.months_on_platform * 0.5) + \
                   (data.weekly_trips * 0.2) + \
                   (data.on_time_rate * 20) + \
                   (data.avg_rating * 5) - \
                   (data.cancel_rate * 100) - \
                   (data.earnings_volatility * 50)
            score = max(0, min(100, score))  # Ensure score is between 0-100
        
        # Get decision
        decision = get_decision(score)
        
        # Generate simple reason codes if the model fails
        try:
            reason_codes = get_reason_codes(model_name, input_df)
        except Exception as e:
            reason_codes = [
                {"feature": "on_time_rate", "value": data.on_time_rate, "impact": "high"},
                {"feature": "cancel_rate", "value": data.cancel_rate, "impact": "high" if data.cancel_rate > 0.1 else "low"},
                {"feature": "avg_rating", "value": data.avg_rating, "impact": "high" if data.avg_rating < 4.0 else "low"},
                {"feature": "months_on_platform", "value": data.months_on_platform, "impact": "high" if data.months_on_platform < 6 else "low"}
            ]
        
        return {
            "partner_id": data.partner_id or "demo_partner",
            "score": round(float(score), 2),
            "decision": decision,
            "reason_codes": reason_codes,
            "model_version": model_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Fallback response if everything else fails
        return {
            "partner_id": data.partner_id or "demo_partner",
            "score": 75.0,
            "decision": "review",
            "reason_codes": [{"feature": "system", "value": "Using fallback scoring", "impact": "info"}],
            "model_version": "fallback",
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
