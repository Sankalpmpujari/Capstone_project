"""
NeuroBreathe AI - FastAPI Application Server
Multi-modal Neurological & Respiratory Acoustic Diagnostic System
Integrates Parkinson's Disease (XGBoost) and Respiratory Sound (LightGBM) ML models.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import FRONTEND_DIR, HOST, PORT
from backend.models_loader import models
from backend.routes.health import router as health_router
from backend.routes.parkinson import router as parkinson_router
from backend.routes.respiratory import router as respiratory_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("neurobreathe.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models into memory
    logger.info("Initializing ML models for NeuroBreathe AI...")
    models.initialize()
    yield
    # Shutdown
    logger.info("NeuroBreathe AI server shutting down.")

app = FastAPI(
    title="NeuroBreathe AI",
    description="Dual-Modal Machine Learning Platform for Parkinson's Voice Biomarker Screening and Respiratory Acoustic Analysis.",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize models immediately on import as well
models.initialize()

# CORS middleware for local development & cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health_router)
app.include_router(parkinson_router)
app.include_router(respiratory_router)

# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"message": "Frontend index.html not found. API is active."})

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        # Allow API routes to be handled naturally
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            return JSONResponse({"error": "Not Found"}, status_code=404)
        
        file_candidate = FRONTEND_DIR / full_path
        if file_candidate.is_file():
            return FileResponse(file_candidate)
            
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"error": "Resource not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
