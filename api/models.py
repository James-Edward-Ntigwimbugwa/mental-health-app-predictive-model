from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum

class StressLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"

class PredictionRequest(BaseModel):
    # Demographics
    age: int = Field(..., ge=16, le=100, description="Age in years")
    gender_Male: int = Field(..., ge=0, le=1, description="1 if Male, 0 if Female")
    
    # Health factors
    sleep_quality: int = Field(..., ge=1, le=5, description="Sleep quality rating (1-5)")
    health_status: int = Field(..., ge=1, le=5, description="General health status (1-5)")
    mental_health_history: int = Field(..., ge=0, le=1, description="1 if has mental health history, 0 otherwise")
    
    # Lifestyle factors
    exercise_frequency: int = Field(..., ge=0, le=7, description="Exercise frequency per week")
    substance_use: int = Field(..., ge=0, le=5, description="Substance use level (0-5)")
    
    # Social factors
    social_support: int = Field(..., ge=1, le=5, description="Social support level (1-5)")
    
    # Stress factors
    work_stress: int = Field(..., ge=1, le=5, description="Work stress level (1-5)")
    financial_stress: int = Field(..., ge=1, le=5, description="Financial stress level (1-5)")
    
    # Data source (optional, will be set to default if not provided)
    dataset_source_workplace: Optional[int] = Field(1, ge=0, le=1, description="1 if workplace data, 0 otherwise")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender_Male": 1,
                "sleep_quality": 3,
                "health_status": 3,
                "mental_health_history": 0,
                "exercise_frequency": 2,
                "substance_use": 1,
                "social_support": 4,
                "work_stress": 5,
                "financial_stress": 4,
                "dataset_source_workplace": 1
            }
        }

class PredictionResponse(BaseModel):
    success: bool
    stress_level: StressLevel
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    problem_analysis: Dict[str, Any]
    solutions: Dict[str, Any]
    comprehensive_report: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    timestamp: str