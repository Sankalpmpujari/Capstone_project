"""
NeuroBreathe AI - Configuration Module
Resolves paths for ML models, clinical datasets, audio storage, and server settings.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

def resolve_dir(*subpaths):
    candidates = [
        BASE_DIR / "neurobreathe-ai" / "ml_training" / Path(*subpaths),
        BASE_DIR / "ml_training" / Path(*subpaths),
        BASE_DIR / Path(*subpaths)
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

PARKINSON_MODEL_DIR = resolve_dir("models", "parkinson")
RESPIRATORY_MODEL_DIR = resolve_dir("models", "respiratory")
RESPIRATORY_AUDIO_DIR = resolve_dir("data", "respiratory", "audio_and_txt_files")
RESPIRATORY_DATA_DIR = resolve_dir("data", "respiratory")
PARKINSON_DATA_DIR = resolve_dir("data", "parkinson")

FRONTEND_DIR = BASE_DIR / "frontend"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_SAMPLE_RATE = 4000
MAX_UPLOAD_SIZE_MB = 25

HOST = "127.0.0.1"
PORT = 8000
