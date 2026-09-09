"""
NeuroBreathe AI - Respiratory Sound API Endpoints
Endpoints for lung sound audio classification (file upload / microphone),
clinical preset sound libraries, and streaming audio playback.
"""

import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from backend.config import TEMP_UPLOAD_DIR
from backend.services.respiratory_service import (
    analyze_audio_file,
    get_preset_samples,
    get_preset_audio_path
)
from backend.services.spirometry_service import (
    analyze_forced_spirometry,
    list_spirometry_presets,
    get_spirometry_preset
)
from backend.services.rppg_service import analyze_rppg_timeseries

router = APIRouter(prefix="/api/respiratory", tags=["Respiratory Acoustic AI"])

class RppgSample(BaseModel):
    t: float
    val: float

class RppgRequest(BaseModel):
    samples: List[RppgSample]

@router.get("/samples")
async def list_clinical_samples():
    """Returns list of curated clinical lung sound recordings available on the server."""
    samples = get_preset_samples()
    return {
        "count": len(samples),
        "samples": samples
    }

@router.get("/sample-audio/{sample_id}")
async def stream_sample_audio(sample_id: str):
    """Streams a clinical lung sound recording for in-browser playback."""
    audio_path = get_preset_audio_path(sample_id)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Clinical audio sample not found.")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=audio_path.name
    )

@router.post("/predict-sample/{sample_id}")
async def predict_preset_sample(sample_id: str):
    """Executes feature extraction and LightGBM classification on an authentic clinical sample."""
    audio_path = get_preset_audio_path(sample_id)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Clinical audio sample not found.")

    try:
        result = analyze_audio_file(str(audio_path), filename_hint=audio_path.name, is_live_mic=False)
        return {
            "status": "success",
            "sample_id": sample_id,
            "filename": audio_path.name,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

@router.post("/predict-audio")
async def predict_audio_file(
    file: UploadFile = File(...),
    is_live_mic: Optional[bool] = Form(None)
):
    """
    Accepts an uploaded lung sound recording (from file or browser microphone),
    extracts 60 acoustic features, calculates respiratory rate & breathing cycle,
    and returns COPD classification results.
    """
    allowed_extensions = [".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a"]
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".wav"

    if file_ext not in allowed_extensions:
        # Default to .wav for raw audio streams
        file_ext = ".wav"

    temp_filename = f"upload_{uuid.uuid4().hex}{file_ext}"
    temp_path = TEMP_UPLOAD_DIR / temp_filename

    try:
        # Save file to temp directory
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        with open(temp_path, "wb") as f:
            f.write(content)

        # Detect if live microphone recording
        is_live = bool(
            is_live_mic if is_live_mic is not None else
            ("recorded" in (file.filename or "").lower() or "mic" in (file.filename or "").lower())
        )

        # Analyze
        result = analyze_audio_file(
            str(temp_path),
            filename_hint=file.filename or temp_filename,
            is_live_mic=is_live
        )

        return {
            "status": "success",
            "filename": file.filename or "microphone_recording.wav",
            "file_size_bytes": len(content),
            "data": result
        }
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis error: {str(e)}")
    finally:
        # Clean up temporary upload
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/test-spirometry")
async def test_virtual_spirometry(file: UploadFile = File(...)):
    """
    Analyzes forced exhalation audio ('candle blow'), models aeroacoustic flow,
    calculates FEV1, FVC, FEV1/FVC ratio, PEF, and generates the Flow-Volume Loop curve.
    """
    allowed_extensions = [".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a"]
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if file_ext not in allowed_extensions:
        file_ext = ".wav"

    temp_filename = f"spiro_{uuid.uuid4().hex}{file_ext}"
    temp_path = TEMP_UPLOAD_DIR / temp_filename

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Exhalation audio file is empty.")

        with open(temp_path, "wb") as f:
            f.write(content)

        result = analyze_forced_spirometry(str(temp_path), filename_hint=file.filename or temp_filename)
        return {
            "status": "success",
            "filename": file.filename or "forced_exhalation.wav",
            "file_size_bytes": len(content),
            "data": result
        }
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spirometry analysis error: {str(e)}")
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/test-rppg")
async def test_cardiopulmonary_rppg(request: RppgRequest):
    """
    Analyzes optical capillary green-channel intensity timeseries,
    extracts heart rate, respiration rate, and Respiratory Sinus Arrhythmia (RSA) E:I ratio.
    """
    try:
        samples_dicts = [{"t": s.t, "val": s.val} for s in request.samples]
        result = analyze_rppg_timeseries(samples_dicts)
        return {"status": "success", "data": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rPPG cardiopulmonary analysis error: {str(e)}")

@router.get("/spirometry-presets")
async def get_spirometry_preset_list():
    """Returns available standardized clinical exhalation presets."""
    return {"presets": list_spirometry_presets()}

@router.post("/spirometry-preset/{preset_id}")
async def load_spirometry_preset_data(preset_id: str):
    """Loads standardized clinical spirometry curve and evaluation by preset ID."""
    preset = get_spirometry_preset(preset_id)
    return {"status": "success", "preset_id": preset_id, "data": preset}

class RespiratoryCompositeRequest(BaseModel):
    auscultation: Optional[Dict[str, Any]] = None
    spirometry: Optional[Dict[str, Any]] = None
    vitals: Optional[Dict[str, Any]] = None

@router.post("/composite-predict")
async def predict_respiratory_composite(req: RespiratoryCompositeRequest):
    """
    Combines lung sound auscultation, virtual spirometry, and optical vitals into
    a unified composite respiratory health diagnosis, plain-language explanation,
    and post-prediction care tips.
    """
    ausc = req.auscultation
    spiro = req.spirometry
    vitals = req.vitals

    # Determine risk signals
    copd_signals = 0
    total_signals = 0
    findings = []

    # 1. Auscultation
    if ausc:
        total_signals += 1
        is_copd_ausc = (ausc.get("prediction") == "COPD" or ausc.get("risk_color") == "red")
        if is_copd_ausc:
            copd_signals += 1
            findings.append("Stethoscope lung sounds showed acoustic characteristics of obstructive airflow / wheezing.")
        else:
            findings.append("Stethoscope breath sounds showed clear vesicular airflow without obstruction.")

    # 2. Spirometry
    if spiro:
        total_signals += 1
        ratio = spiro.get("fev1_fvc_percent", 80.0)
        is_obstructed = (ratio < 70.0 or spiro.get("risk_color") == "red")
        if is_obstructed:
            copd_signals += 1
            findings.append(f"Virtual spirometry demonstrated airflow limitation (FEV1/FVC ratio: {ratio}% vs 70% threshold).")
        else:
            findings.append(f"Virtual spirometry demonstrated healthy dynamic airway emptying (FEV1/FVC ratio: {ratio}%).")

    # 3. Vitals
    if vitals:
        rr = vitals.get("respiratory_rate_bpm") or vitals.get("rr_bpm")
        if rr and (rr > 22 or rr < 10):
            findings.append(f"Resting breathing rate was outside normal reference range ({rr} breaths/min).")

    # Composite Verdict
    if total_signals == 0:
        overall_status = "Awaiting Respiratory Tests"
        overall_badge = "No Tests Completed"
        risk_color = "slate"
        plain_summary = "Please complete at least one respiratory test (Lung Sound, Spirometry, or Vitals) to generate an assessment."
    elif copd_signals >= 2 or (total_signals == 1 and copd_signals == 1):
        overall_status = "Elevated Risk — Obstructive Airway Pattern Detected"
        overall_badge = "High Suspicion of COPD / Obstruction"
        risk_color = "red"
        plain_summary = (
            "Your tests indicate that your breathing airways are narrowed, causing air to leave your lungs more slowly "
            "than normal. This pattern is consistent with Chronic Obstructive Pulmonary Disease (COPD) or significant airway limitation."
        )
    elif copd_signals == 1 and total_signals > 1:
        overall_status = "Borderline / Mild Airway Limitation"
        overall_badge = "Moderate Concern — Monitor Airway"
        risk_color = "amber"
        plain_summary = (
            "Your tests show mixed or mild indications of airway resistance. While one test showed restricted airflow, "
            "others were closer to normal. Early monitoring and medical consultation are advised."
        )
    else:
        overall_status = "Normal & Healthy Respiratory Mechanics"
        overall_badge = "Normal Healthy Lungs"
        risk_color = "emerald"
        plain_summary = (
            "All completed tests demonstrate clear lung breath sounds, healthy airway emptying speed, "
            "and normal breathing volume without signs of chronic airway blockage."
        )

    # Actionable Care Tips
    care_tips = {
        "breathing_exercises": [
            {
                "title": "Pursed-Lip Breathing Technique",
                "how_to": "Inhale slowly through your nose for 2 counts. Pucker your lips like blowing out birthday candles, then exhale gently for 4 counts.",
                "benefit": "Helps keep airways open longer during exhalation, releasing trapped stale air and calming breathlessness."
            },
            {
                "title": "Diaphragmatic (Belly) Breathing",
                "how_to": "Sit back comfortably with one hand on your belly. Breathe in slowly through your nose, feeling your belly push outward. Exhale gently while belly relaxes.",
                "benefit": "Strengthens the primary breathing muscle (diaphragm) and reduces neck/chest muscle strain."
            }
        ],
        "daily_habits": [
            "Stay well-hydrated by sipping water throughout the day to keep mucus thin and easy to clear.",
            "Avoid exposure to secondhand smoke, wood smoke, industrial fumes, and strong aerosol sprays.",
            "Engage in 20-30 minutes of gentle daily walking to maintain cardiovascular and respiratory endurance.",
            "Use a HEPA air purifier in the bedroom if you are sensitive to dust, pollen, or pet dander."
        ],
        "red_flag_warnings": [
            "Sudden severe shortness of breath that does not improve after resting.",
            "Persistent blueness (cyanosis) around the lips, tongue, or fingernails.",
            "Chest pain, heaviness, or coughing up blood.",
            "High fever accompanied by thick yellow/green phlegm and rapid pulse."
        ],
        "doctor_discussion_points": [
            "Share your recorded FEV1/FVC ratio and lung sound classification from this report.",
            "Ask if a clinical pulmonary function test (PFT) or chest radiograph is indicated.",
            "Review whether an inhaled bronchodilator or preventive maintenance inhaler is appropriate."
        ]
    }

    response = {
        "status": "success",
        "overall_status": overall_status,
        "overall_badge": overall_badge,
        "risk_badge": overall_badge,
        "risk_color": risk_color,
        "composite_risk_level": overall_status,
        "copd_signals": copd_signals,
        "total_signals": total_signals,
        "plain_summary": plain_summary,
        "plain_language_summary": plain_summary,
        "findings": findings,
        "individual_findings": findings,
        "care_tips": care_tips
    }
    response["data"] = {**response}
    return response


