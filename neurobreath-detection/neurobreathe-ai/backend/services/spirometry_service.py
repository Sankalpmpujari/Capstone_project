"""
NeuroBreathe AI - Virtual Dynamic Acoustic Spirometry Service
Processes forced expiratory exhalation audio ("candle blow" / maximal expiratory blast),
models aeroacoustic volumetric flow rate Q(t), integrates cumulative expired volume V(t),
computes FEV1, FVC, FEV1/FVC clinical ratio, Peak Expiratory Flow (PEF),
and extracts the Flow-Volume Loop curve with airway concavity analysis.
"""

from typing import Dict, Any, List
import numpy as np
import librosa
from scipy import signal
from backend.services.respiratory_service import decode_any_audio

from typing import Dict, Any, List
import numpy as np
import librosa
from scipy import signal
from backend.services.respiratory_service import decode_any_audio

# Curated Clinical Spirometry Presets for 1-Click Verification
SPIROMETRY_PRESETS = {
    "normal": {
        "id": "normal",
        "title": "Healthy Airway Exhalation",
        "subtitle": "Normal Linear Emptying (Healthy Control)",
        "fev1_liters": 3.42,
        "fvc_liters": 4.15,
        "fev1_fvc_ratio": 0.824,
        "fev1_fvc_percent": 82.4,
        "pef_lps": 8.9,
        "fev1_pct_predicted": 97.7,
        "obstruction_grade": "Normal Spirometry (No Obstruction)",
        "risk_level": "Normal / Low Risk",
        "risk_badge": "Healthy Airway Mechanics",
        "risk_color": "emerald",
        "concavity_depth": 0.08,
        "duration_seconds": 3.8,
        "interpretation": "FEV1/FVC ratio is 82.4% (well above clinical 70% threshold). Airway emptying is rapid and unobstructed with healthy elastic recoil.",
        "simple_status": "Lungs Are Emptying Normally & Clearly",
        "what_this_means": "Your airways are open, flexible, and free of narrowing. You are able to blow out over 80% of your total lung volume in the very first second.",
        "action_advice": "Maintain active cardiovascular exercise, keep hydrated, and avoid air pollutants or tobacco smoke to preserve peak lung capacity.",
        "flow_volume_curve": [
            {"x": 0.0, "y": 0.0}, {"x": 0.25, "y": 8.9}, {"x": 0.65, "y": 8.1},
            {"x": 1.15, "y": 6.9}, {"x": 1.70, "y": 5.6}, {"x": 2.30, "y": 4.3},
            {"x": 2.90, "y": 3.0}, {"x": 3.50, "y": 1.6}, {"x": 4.15, "y": 0.0}
        ],
        "normal_reference_curve": [
            {"x": 0.0, "y": 8.9}, {"x": 1.0, "y": 6.7}, {"x": 2.0, "y": 4.6},
            {"x": 3.0, "y": 2.4}, {"x": 4.15, "y": 0.0}
        ]
    },
    "borderline": {
        "id": "borderline",
        "title": "Mild Peripheral Resistance",
        "subtitle": "Borderline Concavity / Early Small-Airway Irritation",
        "fev1_liters": 2.85,
        "fvc_liters": 4.02,
        "fev1_fvc_ratio": 0.709,
        "fev1_fvc_percent": 70.9,
        "pef_lps": 7.1,
        "fev1_pct_predicted": 81.4,
        "obstruction_grade": "Borderline / Mild Small-Airway Concavity",
        "risk_level": "Borderline Obstruction",
        "risk_badge": "Borderline Airflow / Monitor",
        "risk_color": "amber",
        "concavity_depth": 0.22,
        "duration_seconds": 4.1,
        "interpretation": "FEV1/FVC ratio is 70.9% (close to clinical threshold). Mild expiratory flow concavity indicates slight resistance in peripheral bronchial branches.",
        "simple_status": "Slight Airflow Resistance Detected",
        "what_this_means": "Your lungs empty adequately, but the smaller branches of your breathing tubes show slight resistance or narrowing near the end of exhalation.",
        "action_advice": "Practice daily diaphragmatic breathing, monitor for exercise-induced breathlessness or seasonal allergies, and consider discussing with your doctor if coughing occurs.",
        "flow_volume_curve": [
            {"x": 0.0, "y": 0.0}, {"x": 0.28, "y": 7.1}, {"x": 0.65, "y": 6.0},
            {"x": 1.20, "y": 4.8}, {"x": 1.80, "y": 3.4}, {"x": 2.50, "y": 2.1},
            {"x": 3.20, "y": 1.1}, {"x": 4.02, "y": 0.0}
        ],
        "normal_reference_curve": [
            {"x": 0.0, "y": 7.1}, {"x": 1.0, "y": 5.3}, {"x": 2.0, "y": 3.5},
            {"x": 3.0, "y": 1.8}, {"x": 4.02, "y": 0.0}
        ]
    },
    "copd": {
        "id": "copd",
        "title": "Obstructive Defect (COPD Pattern)",
        "subtitle": "Scooped Flow Concavity (GOLD Stage 2)",
        "fev1_liters": 1.82,
        "fvc_liters": 3.45,
        "fev1_fvc_ratio": 0.528,
        "fev1_fvc_percent": 52.8,
        "pef_lps": 4.9,
        "fev1_pct_predicted": 52.0,
        "obstruction_grade": "GOLD Stage 2 (Moderate Airflow Limitation)",
        "risk_level": "Moderate COPD Obstruction",
        "risk_badge": "Obstructive Pattern / Consult Doctor",
        "risk_color": "red",
        "concavity_depth": 0.44,
        "duration_seconds": 4.6,
        "interpretation": "FEV1/FVC ratio is 52.8% (well below the 70% threshold). The flow-volume curve exhibits a classic 'scooped-out' obstructive pattern from dynamic small-airway collapse.",
        "simple_status": "Airway Narrowing / Obstruction Detected",
        "what_this_means": "Your breathing tubes narrow during exhalation, trapping air in your lungs and slowing down how quickly you can blow it out (only 53% in the first second).",
        "action_advice": "Share this report with a doctor or pulmonologist for formal spirometry evaluation. Use pursed-lip breathing to relieve shortness of breath and avoid respiratory irritants.",
        "flow_volume_curve": [
            {"x": 0.0, "y": 0.0}, {"x": 0.22, "y": 4.9}, {"x": 0.55, "y": 3.4},
            {"x": 1.00, "y": 2.1}, {"x": 1.50, "y": 1.3}, {"x": 2.10, "y": 0.8},
            {"x": 2.75, "y": 0.45}, {"x": 3.45, "y": 0.0}
        ],
        "normal_reference_curve": [
            {"x": 0.0, "y": 7.5}, {"x": 1.0, "y": 5.3}, {"x": 2.0, "y": 3.1},
            {"x": 3.0, "y": 1.0}, {"x": 3.45, "y": 0.0}
        ]
    }
}

def get_spirometry_preset(preset_id: str) -> Dict[str, Any]:
    """Returns precomputed clinical spirometry case by ID."""
    preset = SPIROMETRY_PRESETS.get(preset_id.lower())
    if not preset:
        # Fallback to normal
        preset = SPIROMETRY_PRESETS["normal"]
    return preset

def list_spirometry_presets() -> List[Dict[str, Any]]:
    """Returns list of available clinical spirometry preset cases."""
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "subtitle": p["subtitle"],
            "risk_color": p["risk_color"],
            "obstruction_grade": p["obstruction_grade"]
        }
        for p in SPIROMETRY_PRESETS.values()
    ]

def analyze_forced_spirometry(file_path: str, filename_hint: str = "") -> Dict[str, Any]:
    """
    Analyzes audio of forced expiration into microphone (candle blow maneuver).
    Extracts acoustic FEV1, FVC, FEV1/FVC, PEF, Flow-Volume loop data, and plain-language guidance.
    """
    sr = 8000
    y = decode_any_audio(file_path, target_sr=sr)
    duration_sec = float(len(y) / sr)

    if duration_sec < 0.7:
        raise ValueError("Exhalation recording too brief (minimum 1.0–1.5 seconds required). Please take a deep breath and blow steadily into the microphone.")

    # 1. High-pass filter above 90 Hz to remove DC offset, handling pops, and low-frequency rumble
    b, a = signal.butter(4, 90.0 / (sr / 2.0), btype='highpass')
    y_filt = signal.filtfilt(b, a, y)

    # 2. Extract Acoustic Sound Pressure Power Envelope (Aeroacoustic flow Q(t) ~ sqrt(P(t)))
    frame_len = int(sr * 0.04)  # 40 ms frame
    hop_len = int(sr * 0.01)    # 10 ms hop -> 100 frames/sec
    rms = librosa.feature.rms(y=y_filt, frame_length=frame_len, hop_length=hop_len)[0]
    t_frames = np.arange(len(rms)) * (hop_len / sr)

    # Smooth the envelope
    smooth_rms = signal.savgol_filter(rms, window_length=min(15, len(rms) if len(rms) % 2 != 0 else len(rms) - 1), polyorder=2)
    smooth_rms = np.clip(smooth_rms, 0.0, None)

    # 3. Detect Blast Onset with Adaptive Peak Detection for Low-Sensitivity Microphones
    peak_rms = float(np.max(smooth_rms))
    if peak_rms < 0.0008:
        raise ValueError(
            "Breath sound was too quiet to measure accurately. "
            "Please hold your microphone about 10–15 cm (4–6 inches) away and blow out firmly like blowing out a birthday candle."
        )

    # Auto-normalize envelope if microphone gain is low
    if peak_rms < 0.02:
        norm_factor = 0.05 / (peak_rms + 1e-9)
        smooth_rms = smooth_rms * norm_factor
        peak_rms = float(np.max(smooth_rms))

    onset_thresh = max(0.0010, peak_rms * 0.05)
    onset_idx_candidates = np.where(smooth_rms >= onset_thresh)[0]
    if len(onset_idx_candidates) == 0:
        raise ValueError("Could not detect the start of exhalation. Please blow directly toward the microphone.")
    onset_idx = int(onset_idx_candidates[0])

    # End of forced exhalation: when flow drops and stays below 5% of peak or recording ends
    subsequent = smooth_rms[onset_idx:]
    subsequent_t = t_frames[onset_idx:] - t_frames[onset_idx]

    # 4. Physical Volumetric Flow Rate Modeling
    # Based on Larson et al. (SpiroSmart): Volumetric flow Q(t) ~ sqrt(P_acoustic(t))
    raw_flow = np.sqrt(subsequent + 1e-12)

    # Calibrate scale: average adult forced exhalation has Peak Expiratory Flow around 7.0 - 9.5 L/sec
    # and FVC around 3.5 - 4.8 Liters. We normalize raw_flow to physical clinical units (L/s)
    peak_raw = np.max(raw_flow)
    pef_calibrated = float(np.clip((peak_raw / 0.4) * 8.0, 3.5, 12.0))
    flow_scale = pef_calibrated / (peak_raw + 1e-9)
    flow_lps = raw_flow * flow_scale  # Liters per second

    # Integrate Flow to get Cumulative Volume (Liters)
    dt = hop_len / sr
    cumulative_volume = np.cumsum(flow_lps) * dt

    # Total Forced Vital Capacity (FVC)
    fvc_unscaled = float(cumulative_volume[-1])
    target_fvc = float(np.clip(fvc_unscaled, 2.0, 5.5))
    volume_scale = target_fvc / (fvc_unscaled + 1e-9)
    cumulative_volume = cumulative_volume * volume_scale
    flow_lps = flow_lps * volume_scale
    fvc = float(cumulative_volume[-1])

    # 5. Calculate FEV1 (Volume expelled in the first 1.0 second from onset)
    one_sec_idx = np.searchsorted(subsequent_t, 1.0)
    if one_sec_idx < len(cumulative_volume):
        fev1 = float(cumulative_volume[one_sec_idx])
    else:
        fev1 = float(fvc * 0.85)

    fev1 = float(np.clip(fev1, 0.8, fvc * 0.98))

    # 6. Clinical FEV1 / FVC Ratio
    fev1_fvc_ratio = float(fev1 / (fvc + 1e-9))

    # 7. Flow-Volume Curve Concavity (The "Obstructive Scoop")
    idx_50 = np.searchsorted(cumulative_volume, 0.50 * fvc)
    idx_75 = np.searchsorted(cumulative_volume, 0.75 * fvc)
    idx_peak = int(np.argmax(flow_lps))

    if idx_75 > idx_50 and idx_50 > idx_peak:
        peak_v = cumulative_volume[idx_peak]
        peak_f = flow_lps[idx_peak]
        v_mid = cumulative_volume[idx_50]
        f_mid_actual = flow_lps[idx_50]
        f_mid_linear = peak_f * (1.0 - (v_mid - peak_v) / (fvc - peak_v + 1e-9))
        concavity_depth = float((f_mid_linear - f_mid_actual) / (peak_f + 1e-9))
    else:
        concavity_depth = 0.1

    is_scooped = (concavity_depth > 0.28 or fev1_fvc_ratio < 0.70)

    # 8. Clinical GOLD Obstruction Staging & Plain-Language Explanations
    predicted_fev1 = 3.5  # Standard reference adult norm
    fev1_pct_predicted = float((fev1 / predicted_fev1) * 100.0)

    if fev1_fvc_ratio >= 0.75 and not is_scooped:
        obstruction_grade = "Normal Spirometry (No Obstruction)"
        risk_level = "Normal / Low Risk"
        risk_badge = "Healthy Airway Mechanics"
        risk_color = "emerald"
        simple_status = "Lungs Emptying Normally & Clearly"
        what_this_means = "Your airways show healthy, elastic breathing mechanics. You are clearing air quickly without narrowing."
        action_advice = "No airway obstruction detected. Continue regular aerobic activity and avoid secondhand smoke."
        interpretation = (
            f"Preserved dynamic airway patency with FEV1/FVC ratio of {round(fev1_fvc_ratio * 100, 1)}% "
            f"(clinical threshold >= 70.0%). Flow-Volume curve shows linear expiratory emptying without premature small-airway collapse."
        )
    elif fev1_fvc_ratio >= 0.70:
        obstruction_grade = "Borderline / Mild Small-Airway Concavity"
        risk_level = "Borderline Obstruction"
        risk_badge = "Borderline / Monitor Airway Flow"
        risk_color = "amber"
        simple_status = "Slight Airflow Resistance"
        what_this_means = "Your breathing tubes empty adequately, but smaller branches show slight resistance near the end of your breath."
        action_advice = "Monitor for shortness of breath or persistent cough during exercise. Practice diaphragmatic breathing."
        interpretation = (
            f"FEV1/FVC ratio is {round(fev1_fvc_ratio * 100, 1)}% with mild expiratory flow concavity. "
            "Suggestive of early peripheral small-airway resistance or mild bronchial hyper-reactivity."
        )
    elif fev1_pct_predicted >= 80.0:
        obstruction_grade = "GOLD Stage 1 (Mild Airflow Limitation)"
        risk_level = "Mild COPD / Obstruction"
        risk_badge = "GOLD 1 / Mild Airway Obstruction"
        risk_color = "amber"
        simple_status = "Mild Airway Narrowing Detected"
        what_this_means = "Air is taking slightly longer than normal to leave your lungs, indicating mild airway restriction."
        action_advice = "Consult with a healthcare provider for formal spirometry testing. Learn pursed-lip breathing exercises."
        interpretation = (
            f"FEV1/FVC ratio of {round(fev1_fvc_ratio * 100, 1)}% confirms airflow limitation (<70.0%). "
            f"Preserved FEV1 ({round(fev1_pct_predicted, 1)}% predicted) indicates early-stage obstruction."
        )
    elif fev1_pct_predicted >= 50.0:
        obstruction_grade = "GOLD Stage 2 (Moderate Airflow Limitation)"
        risk_level = "Moderate COPD Obstruction"
        risk_badge = "GOLD 2 / Moderate Airway Obstruction"
        risk_color = "red"
        simple_status = "Moderate Airway Obstruction (Consistent with COPD)"
        what_this_means = "Airway narrowing is significantly slowing your exhale. Air may be getting partially trapped during fast exhalation."
        action_advice = "Schedule a medical consultation with a pulmonologist or physician. Take care to avoid smoke and cold air triggers."
        interpretation = (
            f"Obstructive ventilatory defect with FEV1/FVC ratio of {round(fev1_fvc_ratio * 100, 1)}% and FEV1 at "
            f"{round(fev1_pct_predicted, 1)}% of predicted. Significant scooped-out flow concavity present."
        )
    else:
        obstruction_grade = "GOLD Stage 3-4 (Severe Airflow Limitation)"
        risk_level = "Severe Obstruction / High Risk"
        risk_badge = "GOLD 3-4 / Severe Airway Obstruction"
        risk_color = "red"
        simple_status = "Severe Airflow Obstruction"
        what_this_means = "Airflow is substantially limited. Breathing requires high effort to clear air from the lungs."
        action_advice = "Prompt clinical medical evaluation is strongly recommended. Review prescribed inhalers with your doctor."
        interpretation = (
            f"Severe airflow limitation with FEV1/FVC ratio {round(fev1_fvc_ratio * 100, 1)}% and depressed FEV1 "
            f"({round(fev1_pct_predicted, 1)}% predicted). Marked dynamic expiratory collapse."
        )

    # 9. Downsample Flow-Volume curve coordinates for Chart.js rendering (50 points)
    step = max(1, len(cumulative_volume) // 50)
    flow_volume_points = [
        {"x": round(float(cumulative_volume[i]), 2), "y": round(float(flow_lps[i]), 2)}
        for i in range(0, len(cumulative_volume), step)
    ]

    # Normal reference curve for comparison
    ref_norm_v = np.linspace(0.0, fvc, 30)
    ref_norm_flow = [round(float(pef_calibrated * (1.0 - v / fvc)), 2) for v in ref_norm_v]
    normal_reference_curve = [
        {"x": round(float(ref_norm_v[i]), 2), "y": ref_norm_flow[i]}
        for i in range(len(ref_norm_v))
    ]

    return {
        "fev1_liters": round(fev1, 2),
        "fvc_liters": round(fvc, 2),
        "fev1_fvc_ratio": round(fev1_fvc_ratio, 3),
        "fev1_fvc_percent": round(fev1_fvc_ratio * 100, 1),
        "pef_lps": round(pef_calibrated, 1),
        "fev1_pct_predicted": round(fev1_pct_predicted, 1),
        "obstruction_grade": obstruction_grade,
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "risk_color": risk_color,
        "simple_status": simple_status,
        "what_this_means": what_this_means,
        "action_advice": action_advice,
        "concavity_depth": round(concavity_depth, 2),
        "interpretation": interpretation,
        "flow_volume_curve": flow_volume_points,
        "normal_reference_curve": normal_reference_curve,
        "duration_seconds": round(duration_sec, 2)
    }

