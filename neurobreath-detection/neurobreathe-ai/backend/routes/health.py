"""
NeuroBreathe AI - Health & System Status Endpoints
"""

import sys
from fastapi import APIRouter
from backend.models_loader import models
from backend.config import PARKINSON_DATA_DIR, RESPIRATORY_AUDIO_DIR

router = APIRouter(prefix="/api", tags=["System & Diagnostics"])

@router.get("/health")
async def health_check():
    """Returns application health and model readiness statuses."""
    status = models.get_status()
    all_ready = status["parkinson"]["ready"] and status["respiratory"]["ready"]

    audio_samples_count = 0
    if RESPIRATORY_AUDIO_DIR.exists():
        audio_samples_count = len(list(RESPIRATORY_AUDIO_DIR.glob("*.wav")))

    return {
        "status": "healthy" if all_ready else "degraded",
        "system": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "models": status,
        "datasets": {
            "parkinson_data_present": (PARKINSON_DATA_DIR / "parkinsons.data").exists(),
            "respiratory_samples_count": audio_samples_count
        }
    }
