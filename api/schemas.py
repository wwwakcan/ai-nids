from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class FlowRequest(BaseModel):
    src_ip:   str = Field(..., example="192.168.1.100")
    dst_ip:   str = Field(..., example="10.0.0.1")
    protocol: str = Field("tcp", example="tcp")
    src_bytes: Optional[float] = 0.0
    dst_bytes: Optional[float] = 0.0
    duration:  Optional[float] = 0.0
    features: List[float] = Field(..., min_length=10,
        description="Pre-scaled feature vector (output of MinMaxScaler)")

    class Config:
        json_schema_extra = {
            "example": {
                "src_ip":   "192.168.1.100",
                "dst_ip":   "10.0.0.1",
                "protocol": "tcp",
                "src_bytes": 1024,
                "dst_bytes": 512,
                "duration":  0.5,
                "features": [0.1, 0.9, 0.3, 0.2, 0.7, 0.1, 0.0, 0.0,
                             0.0, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11, 0.09,
                             0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }
        }


class PredictionResponse(BaseModel):
    label:      str   = Field(..., example="DoS")
    severity:   str   = Field(..., example="HIGH")
    ae_score:   float = Field(..., example=0.087)
    confidence: float = Field(..., example=0.94)
    latency_ms: float = Field(..., example=12.3)
    timestamp:  str   = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class HealthResponse(BaseModel):
    status:              str  = Field(..., example="ok")
    models_loaded:       bool = Field(..., example=True)
    autoencoder_loaded:  bool = Field(..., example=True)
    total_predictions:   int  = Field(..., example=42000)
