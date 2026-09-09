"""
NeuroBreathe AI - Model Loader Singleton
Loads and manages pre-trained ML models:
1. Parkinson's Disease (XGBoost + Scaler + Selector)
2. Respiratory Sound Classifier (LightGBM)
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from backend.config import PARKINSON_MODEL_DIR, RESPIRATORY_MODEL_DIR

logger = logging.getLogger("neurobreathe.models")

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return

        logger.info("Initializing NeuroBreathe ML Models...")
        self.load_parkinson_model()
        self.load_respiratory_model()
        self.initialized = True

    def load_parkinson_model(self):
        try:
            self.parkinson_scaler = joblib.load(PARKINSON_MODEL_DIR / "scaler.pkl")
            self.parkinson_selector = joblib.load(PARKINSON_MODEL_DIR / "selector.pkl")
            self.parkinson_model = joblib.load(PARKINSON_MODEL_DIR / "xgb_model.pkl")
            self.parkinson_features = joblib.load(PARKINSON_MODEL_DIR / "feature_order.pkl")
            self.parkinson_ready = True
            logger.info(f"Parkinson XGBoost model loaded successfully with {len(self.parkinson_features)} features.")
        except Exception as e:
            logger.error(f"Failed to load Parkinson model: {e}")
            self.parkinson_scaler = None
            self.parkinson_selector = None
            self.parkinson_model = None
            self.parkinson_features = []
            self.parkinson_ready = False

    def load_respiratory_model(self):
        try:
            self.respiratory_model = joblib.load(RESPIRATORY_MODEL_DIR / "lgbm_model.pkl")
            self.respiratory_features = joblib.load(RESPIRATORY_MODEL_DIR / "feature_order.pkl")
            label_map_path = RESPIRATORY_MODEL_DIR / "label_map.pkl"
            self.respiratory_label_map = joblib.load(label_map_path) if label_map_path.exists() else None
            self.respiratory_classes = list(getattr(self.respiratory_model, "classes_", ["COPD", "Not_COPD"]))
            self.respiratory_ready = True
            logger.info(f"Respiratory LightGBM model loaded successfully with {len(self.respiratory_features)} features.")
        except Exception as e:
            logger.error(f"Failed to load Respiratory model: {e}")
            self.respiratory_model = None
            self.respiratory_features = []
            self.respiratory_classes = ["COPD", "Not_COPD"]
            self.respiratory_ready = False

    def get_status(self):
        return {
            "parkinson": {
                "ready": getattr(self, "parkinson_ready", False),
                "model_type": "XGBoost Classifier",
                "feature_count": len(getattr(self, "parkinson_features", [])),
                "path": str(PARKINSON_MODEL_DIR)
            },
            "respiratory": {
                "ready": getattr(self, "respiratory_ready", False),
                "model_type": "LightGBM Classifier",
                "feature_count": len(getattr(self, "respiratory_features", [])),
                "classes": getattr(self, "respiratory_classes", []),
                "path": str(RESPIRATORY_MODEL_DIR)
            }
        }

# Global singleton
models = ModelManager()
