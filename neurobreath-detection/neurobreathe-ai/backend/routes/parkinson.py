"""
NeuroBreathe AI - Parkinson API Endpoints
Endpoints for single-patient voice analysis, batch CSV screening,
clinical presets, and biomarker metadata.
"""

from typing import Dict, Any, List, Optional
from io import StringIO
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, create_model

import os
import uuid
from pathlib import Path
from backend.config import TEMP_UPLOAD_DIR
from backend.services.parkinson_service import (
    predict_single,
    predict_batch_dataframe,
    analyze_phonation_audio,
    FEATURE_METADATA,
    CLINICAL_PRESETS
)
from backend.services.spiral_service import analyze_spiral_trajectory

router = APIRouter(prefix="/api/parkinson", tags=["Parkinson's Disease"])

class ParkinsonPredictRequest(BaseModel):
    features: Dict[str, float]

class SpiralPoint(BaseModel):
    x: float
    y: float
    t: float

class SpiralRequest(BaseModel):
    points: List[SpiralPoint]

@router.get("/features-info")
async def get_features_info():
    """Returns clinical reference standards and descriptions for all 22 biomarkers."""
    return {
        "count": len(FEATURE_METADATA),
        "features": FEATURE_METADATA
    }

@router.get("/presets")
async def get_presets():
    """Returns pre-configured clinical cases (Healthy controls and Parkinson cases)."""
    return {
        "presets": CLINICAL_PRESETS
    }

@router.post("/predict")
async def predict_parkinson(request: ParkinsonPredictRequest):
    """
    Analyzes 22 voice acoustic biomarkers and returns diagnosis,
    confidence score, risk level, and biomarker deviation metrics.
    """
    try:
        result = predict_single(request.features)
        return {"status": "success", "data": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict-batch")
async def predict_parkinson_batch(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV file containing patient recordings,
    performs batch prediction, and returns summary metrics and individual patient scores.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .csv file.")

    try:
        content = await file.read()
        csv_text = content.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(csv_text))

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

        result = predict_batch_dataframe(df)
        return {
            "status": "success",
            "filename": file.filename,
            "data": result
        }
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process batch CSV: {str(e)}")

@router.post("/predict-audio")
async def predict_parkinson_audio(file: UploadFile = File(...)):
    """
    Accepts recorded sustained phonation audio (/a/ vowel),
    evaluates Maximum Phonation Time (MPT), extracts 22 voice biomarkers,
    and runs XGBoost classification.
    """
    allowed_extensions = [".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a"]
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if file_ext not in allowed_extensions:
        file_ext = ".wav"

    temp_filename = f"parkinson_{uuid.uuid4().hex}{file_ext}"
    temp_path = TEMP_UPLOAD_DIR / temp_filename

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Recorded phonation audio is empty.")

        with open(temp_path, "wb") as f:
            f.write(content)

        result = analyze_phonation_audio(str(temp_path), filename_hint=file.filename or temp_filename)
        return {
            "status": "success",
            "filename": file.filename or "phonation_recording.wav",
            "file_size_bytes": len(content),
            "data": result
        }
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phonation acoustic analysis error: {str(e)}")
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/test-spiral")
async def test_spiral_kinematics(request: SpiralRequest):
    """
    Analyzes digital Archimedes spiral drawing trajectory,
    extracts kinematic jerk, radial velocity decay (micrographia),
    and 4-7 Hz kinematic tremor peak power.
    """
    try:
        points_dicts = [{"x": p.x, "y": p.y, "t": p.t} for p in request.points]
        result = analyze_spiral_trajectory(points_dicts)
        return {"status": "success", "data": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spiral kinematic analysis error: {str(e)}")

class ParkinsonCompositeRequest(BaseModel):
    vocal: Optional[Dict[str, Any]] = None
    spiral: Optional[Dict[str, Any]] = None

@router.post("/composite-predict")
async def predict_parkinson_composite(req: ParkinsonCompositeRequest):
    """
    Combines voice acoustic phonation (MPT & 22 MDVP biomarkers) with digital Archimedes
    spiral motor kinematics into a unified multi-test Parkinson assessment, plain-language
    explanation, and post-prediction care tips.
    """
    vocal = req.vocal
    spiral = req.spiral

    signals = 0
    total_tests = 0
    findings = []

    # 1. Evaluate Vocal Test
    if vocal:
        total_tests += 1
        is_pd_voice = (vocal.get("status") == 1 or vocal.get("diagnosis") == "Parkinson's Disease" or vocal.get("risk_color") == "red")
        mpt_val = None
        if "mpt" in vocal and isinstance(vocal["mpt"], dict):
            mpt_val = vocal["mpt"].get("duration_seconds")
        
        if is_pd_voice:
            signals += 1
            findings.append("Vocal phonation exhibited elevated acoustic jitter/shimmer perturbations characteristic of parkinsonian vocal dysphonia.")
        else:
            findings.append("Vocal phonation demonstrated steady harmonic acoustic stability and normal frequency control.")

        if mpt_val is not None:
            if mpt_val < 10.0:
                findings.append(f"Maximum Phonation Time (MPT) was shortened ({mpt_val}s, normal is >15s), reflecting reduced breath support.")
            else:
                findings.append(f"Maximum Phonation Time was robust ({mpt_val}s), indicating healthy respiratory and laryngeal endurance.")

    # 2. Evaluate Spiral Kinematics
    if spiral:
        total_tests += 1
        spiral_risk = spiral.get("risk_color", "emerald")
        if spiral_risk in ("red", "amber"):
            if spiral_risk == "red":
                signals += 1
            findings.append(f"Archimedes spiral drawing showed kinematic jerkiness and subtle 4-7 Hz action tremor ({spiral.get('risk_level', 'Elevated Concern')}).")
        else:
            findings.append("Archimedes spiral drawing showed smooth, steady motor kinematics without significant tremor or micrographia decay.")

    # Composite Verdict
    if total_tests == 0:
        overall_status = "Awaiting Parkinson Tests"
        overall_badge = "No Tests Completed"
        risk_color = "slate"
        plain_summary = "Please complete at least one test (Vocal Phonation or Archimedes Spiral) to generate a unified assessment."
    elif signals >= 2 or (total_tests == 1 and signals == 1):
        overall_status = "Elevated Risk — Parkinsonian Vocal & Motor Biomarkers Detected"
        overall_badge = "Elevated Likelihood / Specialist Review"
        risk_color = "red"
        plain_summary = (
            "Your results show acoustic vocal frequency variations and/or fine-motor drawing tremors commonly associated "
            "with early-stage neurological changes. A consultation with a neurologist or movement disorder specialist is recommended."
        )
    elif signals == 1 and total_tests > 1:
        overall_status = "Borderline / Mild Motor or Vocal Tremor"
        overall_badge = "Borderline / Clinical Follow-up"
        risk_color = "amber"
        plain_summary = (
            "One test showed subtle signs of tremor or vocal instability, while your other test was within normal ranges. "
            "Mild fatigue, stress, or caffeine can sometimes contribute. Periodic re-testing and clinical monitoring are suggested."
        )
    else:
        overall_status = "Normal & Healthy Vocal and Motor Coordination"
        overall_badge = "Normal / Healthy Control"
        risk_color = "emerald"
        plain_summary = (
            "All completed tests show strong voice stability, healthy phonation endurance, and smooth, steady motor control "
            "without signs of parkinsonian dysphonia or action tremor."
        )

    # Actionable Care Tips
    care_tips = {
        "voice_exercises": [
            {
                "title": "Daily Sustained 'Ahhh' Loudness Drills",
                "how_to": "Take a deep breath and vocalize a loud, clear 'aaah' at comfortable pitch, holding it as steady as you can for 10-15 seconds. Repeat 5 times.",
                "benefit": "Maintains vocal cord closure strength, prevents voice fading, and boosts lung-to-larynx coordination."
            },
            {
                "title": "Pitch Glides & Expressive Reading",
                "how_to": "Glide your voice smoothly from your lowest comfortable note to your highest note like a siren. Read a newspaper paragraph out loud using exaggerated expression.",
                "benefit": "Counters the monotone vocal pitch and reduced facial expression (hypomimia) common in motor conditions."
            }
        ],
        "motor_exercises": [
            {
                "title": "Finger-Tapping & Hand Opening Sequences",
                "how_to": "Open and close your hands as wide and fast as possible for 10 repetitions. Then tap your thumb to each fingertip one by one, forward and backward.",
                "benefit": "Stimulates fine motor dexterity in the fingers and combats finger stiffness (bradykinesia)."
            },
            {
                "title": "Big & Bold Handwriting Practice",
                "how_to": "Spend 5 minutes each morning writing sentences with deliberately large, exaggerated cursive letters across wide-ruled paper.",
                "benefit": "Helps re-calibrate brain motor feedback to prevent gradual handwriting shrinkage (micrographia)."
            }
        ],
        "posture_and_balance": [
            "Practice standing tall with shoulders relaxed and eyes focused forward at eye level while walking.",
            "Take deliberate heel-to-toe strides with normal reciprocal arm swing to prevent foot dragging.",
            "Ensure home pathways are well-lit and free of loose throw rugs, cords, or clutter."
        ],
        "when_to_see_neurologist": [
            "A resting tremor in a hand, finger, or chin that occurs when your muscles are relaxed.",
            "Noticeable muscle stiffness or heaviness when swinging arms or turning around in bed.",
            "Slowness in initiating movements such as standing up from a chair or buttoning a shirt.",
            "Noticeable softening of your speaking voice that others frequently comment on."
        ]
    }

    response = {
        "status": "success",
        "overall_status": overall_status,
        "overall_badge": overall_badge,
        "risk_badge": overall_badge,
        "risk_color": risk_color,
        "composite_risk_level": overall_status,
        "signals": signals,
        "total_tests": total_tests,
        "plain_summary": plain_summary,
        "plain_language_summary": plain_summary,
        "findings": findings,
        "individual_findings": findings,
        "care_tips": care_tips
    }
    response["data"] = {**response}
    return response


