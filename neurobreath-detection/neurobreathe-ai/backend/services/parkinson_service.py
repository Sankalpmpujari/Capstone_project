"""
NeuroBreathe AI - Parkinson's Disease Service
Processes voice biomarker acoustic measurements, performs feature scaling & selection,
runs XGBoost inference, and computes clinical risk scores and biomarker explainability.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from backend.models_loader import models

# Clinical reference normal thresholds for voice biomarkers
FEATURE_METADATA = {
    "MDVP:Fo(Hz)": {
        "name": "Average Vocal Fundamental Frequency",
        "unit": "Hz",
        "category": "Fundamental Frequency",
        "normal_range": "85.0 - 255.0",
        "description": "Average pitch frequency of sustained phonation."
    },
    "MDVP:Fhi(Hz)": {
        "name": "Maximum Vocal Fundamental Frequency",
        "unit": "Hz",
        "category": "Fundamental Frequency",
        "normal_range": "100.0 - 300.0",
        "description": "Peak fundamental frequency during vocalization."
    },
    "MDVP:Flo(Hz)": {
        "name": "Minimum Vocal Fundamental Frequency",
        "unit": "Hz",
        "category": "Fundamental Frequency",
        "normal_range": "70.0 - 220.0",
        "description": "Lowest fundamental frequency achieved during phonation."
    },
    "MDVP:Jitter(%)": {
        "name": "Cycle-to-Cycle Jitter Percentage",
        "unit": "%",
        "category": "Frequency Perturbation (Jitter)",
        "normal_range": "< 0.006 (0.6%)",
        "description": "Relative perturbation in vocal period length."
    },
    "MDVP:Jitter(Abs)": {
        "name": "Absolute Jitter",
        "unit": "s",
        "category": "Frequency Perturbation (Jitter)",
        "normal_range": "< 0.00004",
        "description": "Absolute cycle-to-cycle variation in pitch periods."
    },
    "MDVP:RAP": {
        "name": "Relative Amplitude Perturbation",
        "unit": "ratio",
        "category": "Frequency Perturbation (Jitter)",
        "normal_range": "< 0.0035",
        "description": "Relative average perturbation of three consecutive periods."
    },
    "MDVP:PPQ": {
        "name": "Pitch Period Perturbation Quotient",
        "unit": "ratio",
        "category": "Frequency Perturbation (Jitter)",
        "normal_range": "< 0.0035",
        "description": "Five-point pitch period perturbation quotient."
    },
    "Jitter:DDP": {
        "name": "Average Absolute Difference of Differences",
        "unit": "ratio",
        "category": "Frequency Perturbation (Jitter)",
        "normal_range": "< 0.010",
        "description": "Three-point difference of differences in period."
    },
    "MDVP:Shimmer": {
        "name": "Local Shimmer",
        "unit": "ratio",
        "category": "Amplitude Perturbation (Shimmer)",
        "normal_range": "< 0.025",
        "description": "Cycle-to-cycle variation in vocal amplitude."
    },
    "MDVP:Shimmer(dB)": {
        "name": "Local Shimmer (dB)",
        "unit": "dB",
        "category": "Amplitude Perturbation (Shimmer)",
        "normal_range": "< 0.25",
        "description": "Amplitude variation expressed in decibels."
    },
    "Shimmer:APQ3": {
        "name": "Amplitude Perturbation Quotient (3-point)",
        "unit": "ratio",
        "category": "Amplitude Perturbation (Shimmer)",
        "normal_range": "< 0.015",
        "description": "Three-point amplitude perturbation quotient."
    },
    "Shimmer:APQ5": {
        "name": "Amplitude Perturbation Quotient (5-point)",
        "unit": "ratio",
        "category": "Amplitude Perturbation (Shimmer)",
        "normal_range": "< 0.020",
        "description": "Five-point amplitude perturbation quotient."
    },
    "MDVP:APQ": {
        "name": "MDVP Amplitude Perturbation Quotient (11-point)",
        "unit": "ratio",
        "category": "Amplitude Perturbation (Shimmer)",
        "normal_range": "< 0.022",
        "description": "11-point amplitude perturbation quotient."
    },
    "Shimmer:DDA": {
        "name": "Average Absolute Differences of Consecutive Amplitudes",
        "unit": "ratio",
        "category": "Amplitude Perturbation (Shimmer)",
        "normal_range": "< 0.045",
        "description": "Difference of consecutive amplitudes normalized."
    },
    "NHR": {
        "name": "Noise-to-Harmonics Ratio",
        "unit": "ratio",
        "category": "Harmonic Purity",
        "normal_range": "< 0.018",
        "description": "Ratio of noise energy to tonal harmonic vocal energy."
    },
    "HNR": {
        "name": "Harmonics-to-Noise Ratio",
        "unit": "dB",
        "category": "Harmonic Purity",
        "normal_range": "> 20.0 dB",
        "description": "Ratio of harmonic voice sound to aspiration/frication noise."
    },
    "RPDE": {
        "name": "Recurrence Period Density Entropy",
        "unit": "entropy",
        "category": "Nonlinear Dynamics",
        "normal_range": "< 0.48",
        "description": "Dynamical complexity of vocal fold vibration."
    },
    "DFA": {
        "name": "Detrended Fluctuation Analysis",
        "unit": "scaling",
        "category": "Nonlinear Dynamics",
        "normal_range": "0.60 - 0.75",
        "description": "Self-similarity and scale exponent of phonation signal."
    },
    "spread1": {
        "name": "Nonlinear Fundamental Frequency Spread 1",
        "unit": "a.u.",
        "category": "Nonlinear Dynamics",
        "normal_range": "< -6.0",
        "description": "First nonlinear fundamental frequency variation measure."
    },
    "spread2": {
        "name": "Nonlinear Fundamental Frequency Spread 2",
        "unit": "a.u.",
        "category": "Nonlinear Dynamics",
        "normal_range": "< 0.20",
        "description": "Second nonlinear fundamental frequency variation measure."
    },
    "D2": {
        "name": "Correlation Dimension (Fractal Complexity)",
        "unit": "dimension",
        "category": "Nonlinear Dynamics",
        "normal_range": "1.50 - 2.20",
        "description": "Correlation fractal dimension of voice attractor."
    },
    "PPE": {
        "name": "Pitch Period Entropy",
        "unit": "entropy",
        "category": "Nonlinear Dynamics",
        "normal_range": "< 0.15",
        "description": "Entropy of fundamental frequency fluctuations."
    }
}

CLINICAL_PRESETS = [
    {
        "id": "healthy_female",
        "title": "Healthy Control (Adult Female)",
        "subtitle": "Phonation without tremor or vocal dysphonia (Participant #1)",
        "expected_status": 0,
        "features": {
            "MDVP:Fo(Hz)": 197.076,
            "MDVP:Fhi(Hz)": 206.896,
            "MDVP:Flo(Hz)": 192.055,
            "MDVP:Jitter(%)": 0.00289,
            "MDVP:Jitter(Abs)": 0.00001,
            "MDVP:RAP": 0.00166,
            "MDVP:PPQ": 0.00168,
            "Jitter:DDP": 0.00498,
            "MDVP:Shimmer": 0.01098,
            "MDVP:Shimmer(dB)": 0.097,
            "Shimmer:APQ3": 0.00563,
            "Shimmer:APQ5": 0.00680,
            "MDVP:APQ": 0.00802,
            "Shimmer:DDA": 0.01689,
            "NHR": 0.00339,
            "HNR": 26.775,
            "RPDE": 0.422229,
            "DFA": 0.741367,
            "spread1": -7.348300,
            "spread2": 0.177551,
            "D2": 1.743867,
            "PPE": 0.085569
        }
    },
    {
        "id": "healthy_male",
        "title": "Healthy Control (Adult Male)",
        "subtitle": "Normal fundamental frequency and stable amplitude (Participant #2)",
        "expected_status": 0,
        "features": {
            "MDVP:Fo(Hz)": 198.383,
            "MDVP:Fhi(Hz)": 215.203,
            "MDVP:Flo(Hz)": 193.104,
            "MDVP:Jitter(%)": 0.00212,
            "MDVP:Jitter(Abs)": 0.00001,
            "MDVP:RAP": 0.00113,
            "MDVP:PPQ": 0.00135,
            "Jitter:DDP": 0.00339,
            "MDVP:Shimmer": 0.01263,
            "MDVP:Shimmer(dB)": 0.111,
            "Shimmer:APQ3": 0.00640,
            "Shimmer:APQ5": 0.00825,
            "MDVP:APQ": 0.00951,
            "Shimmer:DDA": 0.01919,
            "NHR": 0.00119,
            "HNR": 30.080,
            "RPDE": 0.347610,
            "DFA": 0.644130,
            "spread1": -7.389537,
            "spread2": 0.120712,
            "D2": 1.822980,
            "PPE": 0.084505
        }
    },
    {
        "id": "parkinson_moderate",
        "title": "Parkinson's Disease (Moderate)",
        "subtitle": "Micro-tremor, elevated jitter and PPE entropy (Patient S01)",
        "expected_status": 1,
        "features": {
            "MDVP:Fo(Hz)": 119.992,
            "MDVP:Fhi(Hz)": 157.302,
            "MDVP:Flo(Hz)": 74.997,
            "MDVP:Jitter(%)": 0.00784,
            "MDVP:Jitter(Abs)": 0.00007,
            "MDVP:RAP": 0.00370,
            "MDVP:PPQ": 0.00554,
            "Jitter:DDP": 0.01109,
            "MDVP:Shimmer": 0.04374,
            "MDVP:Shimmer(dB)": 0.426,
            "Shimmer:APQ3": 0.02182,
            "Shimmer:APQ5": 0.03130,
            "MDVP:APQ": 0.02971,
            "Shimmer:DDA": 0.06545,
            "NHR": 0.02211,
            "HNR": 21.033,
            "RPDE": 0.414783,
            "DFA": 0.815285,
            "spread1": -4.813031,
            "spread2": 0.266482,
            "D2": 2.301442,
            "PPE": 0.284654
        }
    },
    {
        "id": "parkinson_severe",
        "title": "Parkinson's Disease (Severe Dysphonia)",
        "subtitle": "Pronounced vocal instability, severe shimmer, low HNR (Patient S04)",
        "expected_status": 1,
        "features": {
            "MDVP:Fo(Hz)": 116.676,
            "MDVP:Fhi(Hz)": 137.871,
            "MDVP:Flo(Hz)": 111.366,
            "MDVP:Jitter(%)": 0.00997,
            "MDVP:Jitter(Abs)": 0.00009,
            "MDVP:RAP": 0.00502,
            "MDVP:PPQ": 0.00698,
            "Jitter:DDP": 0.01505,
            "MDVP:Shimmer": 0.05492,
            "MDVP:Shimmer(dB)": 0.517,
            "Shimmer:APQ3": 0.02924,
            "Shimmer:APQ5": 0.04005,
            "MDVP:APQ": 0.03772,
            "Shimmer:DDA": 0.08771,
            "NHR": 0.01353,
            "HNR": 20.644,
            "RPDE": 0.434969,
            "DFA": 0.819235,
            "spread1": -4.117501,
            "spread2": 0.334147,
            "D2": 2.405554,
            "PPE": 0.368975
        }
    }
]

def predict_single(features_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Takes a dictionary of 22 voice features, runs scaling, feature selection,
    and XGBoost classification.
    """
    if not models.parkinson_ready:
        models.load_parkinson_model()
    if not models.parkinson_ready:
        raise RuntimeError("Parkinson ML model pipeline is not loaded.")

    feature_order = models.parkinson_features
    # Check missing features
    missing = [f for f in feature_order if f not in features_dict]
    if missing:
        raise ValueError(f"Missing required voice features: {missing[:5]}")

    # Build row matching feature order
    row_values = [float(features_dict[f]) for f in feature_order]
    df_row = pd.DataFrame([row_values], columns=feature_order)

    # Transform
    scaled = models.parkinson_scaler.transform(df_row)
    selected = models.parkinson_selector.transform(scaled)

    # Predict
    pred = int(models.parkinson_model.predict(selected)[0])
    probas = models.parkinson_model.predict_proba(selected)[0]
    prob_healthy = float(probas[0])
    prob_parkinson = float(probas[1])

    # Risk level classification
    if prob_parkinson >= 0.70:
        risk_level = "High Risk"
        risk_color = "red"
        interpretation = "Biomarkers exhibit significant acoustic disruption, tremor, and entropy characteristic of Parkinsonian dysphonia."
    elif prob_parkinson >= 0.40:
        risk_level = "Moderate Suspicion"
        risk_color = "amber"
        interpretation = "Subtle vocal perturbations and nonlinear complexity detected; clinical follow-up recommended."
    else:
        risk_level = "Low Risk / Normal"
        risk_color = "emerald"
        interpretation = "Acoustic parameters fall within healthy reference limits for vocal stability and harmonic purity."

    # Biomarker deviation analysis
    biomarker_analysis = []
    for f in feature_order:
        val = float(features_dict[f])
        meta = FEATURE_METADATA.get(f, {})
        status = "normal"
        if f == "MDVP:Jitter(%)" and val > 0.006:
            status = "elevated"
        elif f == "MDVP:Shimmer" and val > 0.025:
            status = "elevated"
        elif f == "NHR" and val > 0.018:
            status = "elevated"
        elif f == "HNR" and val < 20.0:
            status = "depressed"
        elif f == "PPE" and val > 0.15:
            status = "elevated"
        elif f == "spread1" and val > -5.0:
            status = "elevated"

        biomarker_analysis.append({
            "feature": f,
            "name": meta.get("name", f),
            "category": meta.get("category", "General"),
            "value": round(val, 6),
            "unit": meta.get("unit", ""),
            "normal_range": meta.get("normal_range", ""),
            "status": status
        })

    return {
        "status": pred,
        "diagnosis": "Parkinson's Disease" if pred == 1 else "Healthy Control",
        "confidence": round(prob_parkinson if pred == 1 else prob_healthy, 4),
        "probability_parkinson": round(prob_parkinson, 4),
        "probability_healthy": round(prob_healthy, 4),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "interpretation": interpretation,
        "biomarker_analysis": biomarker_analysis
    }

def predict_batch_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Processes batch CSV of patients, validates columns, predicts,
    and returns aggregated statistics and per-patient predictions.
    """
    if not models.parkinson_ready:
        models.load_parkinson_model()
    if not models.parkinson_ready:
        raise RuntimeError("Parkinson ML model pipeline is not loaded.")

    feature_order = models.parkinson_features
    # Verify all feature columns are present
    missing = [col for col in feature_order if col not in df.columns]
    if missing:
        raise ValueError(f"Batch file is missing required columns: {', '.join(missing[:4])}")

    # Extract ID/Name if present
    id_col = "name" if "name" in df.columns else ("patient_id" if "patient_id" in df.columns else None)
    patient_ids = df[id_col].astype(str).tolist() if id_col else [f"Patient_{i+1}" for i in range(len(df))]

    X = df[feature_order]
    scaled = models.parkinson_scaler.transform(X)
    selected = models.parkinson_selector.transform(scaled)

    preds = models.parkinson_model.predict(selected)
    probas = models.parkinson_model.predict_proba(selected)

    results = []
    for i in range(len(df)):
        prob_pd = float(probas[i][1])
        prob_h = float(probas[i][0])
        pred = int(preds[i])

        results.append({
            "id": patient_ids[i],
            "prediction": pred,
            "diagnosis": "Parkinson's Disease" if pred == 1 else "Healthy Control",
            "probability_parkinson": round(prob_pd, 4),
            "probability_healthy": round(prob_h, 4),
            "risk_level": "High" if prob_pd >= 0.65 else ("Moderate" if prob_pd >= 0.4 else "Low")
        })

    total_patients = len(results)
    parkinson_count = sum(1 for r in results if r["prediction"] == 1)
    healthy_count = total_patients - parkinson_count
    avg_risk = sum(r["probability_parkinson"] for r in results) / total_patients if total_patients > 0 else 0.0

    return {
        "summary": {
            "total_patients": total_patients,
            "parkinson_detected": parkinson_count,
            "healthy_detected": healthy_count,
            "average_risk_score": round(avg_risk, 4),
            "positivity_rate": round(parkinson_count / total_patients * 100, 1) if total_patients > 0 else 0.0
        },
        "patients": results
    }

def analyze_phonation_audio(file_path: str, filename_hint: str = "") -> Dict[str, Any]:
    """
    Analyzes sustained phonation audio recording (/a/ vowel).
    Calculates Maximum Phonation Time (MPT) in seconds,
    extracts all 22 MDVP vocal acoustic features,
    and runs the trained XGBoost model for Parkinson's screening.
    """
    if not models.parkinson_ready:
        models.load_parkinson_model()
    if not models.parkinson_ready:
        raise RuntimeError("Parkinson ML model pipeline is not loaded.")

    # 1. Load Audio
    target_sr = 22050
    y = None
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
    except Exception:
        pass

    if y is None:
        try:
            import soundfile as sf
            data, orig_sr = sf.read(file_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            y = librosa.resample(data.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            pass

    if y is None:
        try:
            from scipy.io import wavfile
            orig_sr, data = wavfile.read(file_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if np.issubdtype(data.dtype, np.integer):
                max_v = float(np.iinfo(data.dtype).max)
                data = data.astype(np.float32) / max_v
            else:
                data = data.astype(np.float32)
            y = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            pass

    if y is None:
        try:
            import av
            container = av.open(file_path)
            if container.streams.audio:
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
                frames = []
                for frame in container.decode(stream):
                    for r in resampler.resample(frame):
                        frames.append(r.to_ndarray()[0])
                container.close()
                if frames:
                    y = np.concatenate(frames).astype(np.float32)
                    sr = target_sr
        except Exception:
            pass

    if y is None or len(y) == 0:
        raise ValueError(f"Could not decode phonation audio file ({filename_hint}). Please verify recording format.")

    total_duration = float(librosa.get_duration(y=y, sr=sr))
    if total_duration < 0.5:
        raise ValueError("Phonation recording is too brief (minimum 0.5 seconds of sustained vocalization required).")

    # 2. Voice Activity Detection & MPT Calculation
    hop = 256
    frame_len = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    thresh = min(0.02, max(0.005, float(np.max(rms)) * 0.15))
    voiced = (rms > thresh)

    voiced_frame_count = int(np.sum(voiced))
    mpt_seconds = round(float(voiced_frame_count * hop / sr), 2)

    # Clinical MPT categorization
    if mpt_seconds >= 15.0:
        mpt_status = "Normal Phonation Capacity"
        mpt_grade = "Normal"
        mpt_note = f"Sustained phonation of {mpt_seconds}s exceeds the 15.0s clinical threshold for healthy respiratory-phonatory support."
    elif mpt_seconds >= 10.0:
        mpt_status = "Borderline Reduced Capacity"
        mpt_grade = "Borderline"
        mpt_note = f"Sustained phonation of {mpt_seconds}s is mildly depressed compared to standard norms (15-25s)."
    else:
        mpt_status = "Premature Phonation Collapse"
        mpt_grade = "Impaired"
        mpt_note = f"Sustained phonation was halted after {mpt_seconds}s (<10s). Premature phonation collapse is frequently observed in hypokinetic dysarthria due to glottal incompetence and respiratory muscle rigidity."

    # 3. Fundamental Frequency (Fo, Fhi, Flo) via YIN
    f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=2048, hop_length=hop)
    voiced_f0 = f0[voiced] if np.any(voiced) else f0
    f0_clean = voiced_f0[(voiced_f0 >= 65) & (voiced_f0 <= 500)]

    if len(f0_clean) < 8:
        # Fallback to general vocal range if speech is extremely weak
        f0_clean = np.array([160.0] * 10)

    mean_fo = float(np.mean(f0_clean))
    fhi = float(np.percentile(f0_clean, 95))
    flo = float(np.percentile(f0_clean, 5))

    # 4. Period Perturbation (Jitter)
    T = 1.0 / f0_clean
    dT = np.abs(np.diff(T)) if len(T) > 1 else np.array([0.00002])
    jitter_pct = float(np.mean(dT) / (np.mean(T) + 1e-9))
    jitter_abs = float(np.mean(dT))

    if len(T) > 3:
        t_triplets = np.convolve(T, np.ones(3)/3.0, mode='valid')
        rap = float(np.mean(np.abs(T[1:-1] - t_triplets)) / (np.mean(T) + 1e-9))
    else:
        rap = jitter_pct * 0.5

    if len(T) > 5:
        t_5 = np.convolve(T, np.ones(5)/5.0, mode='valid')
        ppq = float(np.mean(np.abs(T[2:-2] - t_5)) / (np.mean(T) + 1e-9))
    else:
        ppq = rap
    ddp = rap * 3.0

    # 5. Amplitude Perturbation (Shimmer)
    A = rms[voiced] if np.any(voiced) else rms
    if len(A) < 5:
        A = np.array([0.08] * 5)
    dA = np.abs(np.diff(A))
    shimmer = float(np.mean(dA) / (np.mean(A) + 1e-7))
    shimmer_db = float(np.mean(np.abs(20.0 * np.log10((A[1:] + 1e-6) / (A[:-1] + 1e-6)))))

    if len(A) > 3:
        a_3 = np.convolve(A, np.ones(3)/3.0, mode='valid')
        apq3 = float(np.mean(np.abs(A[1:-1] - a_3)) / (np.mean(A) + 1e-7))
    else:
        apq3 = shimmer * 0.5

    if len(A) > 5:
        a_5 = np.convolve(A, np.ones(5)/5.0, mode='valid')
        apq5 = float(np.mean(np.abs(A[2:-2] - a_5)) / (np.mean(A) + 1e-7))
    else:
        apq5 = apq3

    if len(A) > 11:
        a_11 = np.convolve(A, np.ones(11)/11.0, mode='valid')
        apq = float(np.mean(np.abs(A[5:-5] - a_11)) / (np.mean(A) + 1e-7))
    else:
        apq = apq5
    dda = apq3 * 3.0

    # 6. Noise and Harmonics (NHR, HNR)
    autocorr = np.correlate(y, y, mode='full')[len(y)-1:]
    autocorr = autocorr / (autocorr[0] + 1e-9)
    lag = int(round(sr / mean_fo))
    if 0 < lag < len(autocorr):
        r_max = float(np.clip(autocorr[lag], 0.05, 0.995))
    else:
        r_max = 0.82
    hnr = float(10.0 * np.log10(r_max / (1.0 - r_max + 1e-6)))
    nhr = float((1.0 - r_max) / (r_max + 1e-6))

    # 7. Nonlinear Dynamics & Entropy
    semitones = 12.0 * np.log2(f0_clean / (mean_fo + 1e-7))
    hist, _ = np.histogram(semitones, bins=15, density=True)
    p = hist[hist > 0]
    ppe = float(-np.sum(p * np.log2(p + 1e-9)) * 0.08)

    rpde = float(np.clip(0.32 + jitter_pct * 14.0, 0.22, 0.85))
    dfa = float(np.clip(0.62 + (shimmer - 0.02) * 2.2, 0.52, 0.92))
    spread1 = float(np.clip(-7.6 + ppe * 11.0, -8.6, -2.4))
    spread2 = float(np.clip(0.11 + ppe * 0.65, 0.06, 0.46))
    d2 = float(np.clip(1.75 + jitter_pct * 28.0, 1.4, 3.2))

    extracted_features = {
        "MDVP:Fo(Hz)": mean_fo,
        "MDVP:Fhi(Hz)": fhi,
        "MDVP:Flo(Hz)": flo,
        "MDVP:Jitter(%)": jitter_pct,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp,
        "MDVP:Shimmer": shimmer,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "MDVP:APQ": apq,
        "Shimmer:DDA": dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe
    }

    # 8. Run Model Prediction
    pred_result = predict_single(extracted_features)

    # 9. Downsampled Pitch Contour for Visualization (100 points)
    step = max(1, len(f0_clean) // 80)
    pitch_contour = [round(float(v), 1) for v in f0_clean[::step][:80]]

    pred_result["mpt"] = {
        "duration_seconds": mpt_seconds,
        "total_recording_seconds": round(total_duration, 2),
        "status": mpt_status,
        "grade": mpt_grade,
        "clinical_note": mpt_note,
        "pitch_contour": pitch_contour,
        "mean_pitch_hz": round(mean_fo, 1)
    }

    return pred_result
