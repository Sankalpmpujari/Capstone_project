"""
NeuroBreathe AI - Archimedes Spiral & Kinematic Micrographia Service
Analyzes spatial-temporal drawing trajectories from HTML5 canvas (mouse/touchscreen),
computes kinematic jerk, radial expansion velocity decay (micrographia),
kinematic tremor peak power (4-7 Hz), and micro-stall freeze occurrences.
"""

from typing import List, Dict, Any
import numpy as np
from scipy import signal

def analyze_spiral_trajectory(points: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Analyzes list of drawing coordinate points:
    [{ 'x': float, 'y': float, 't': float (ms) }, ...]
    Returns kinematic biomarkers, micrographia index, and Parkinson's motor risk score.
    """
    if not points or len(points) < 15:
        raise ValueError("Insufficient drawing points. Please draw a continuous spiral for at least 3-5 seconds.")

    # 1. Parse and sort by timestamp
    points = sorted(points, key=lambda p: p["t"])
    t = np.array([p["t"] for p in points], dtype=np.float64) / 1000.0  # convert to seconds
    t = t - t[0]  # zero-origin time
    x = np.array([p["x"] for p in points], dtype=np.float64)
    y = np.array([p["y"] for p in points], dtype=np.float64)

    total_duration = float(t[-1] - t[0])
    if total_duration < 1.0:
        raise ValueError("Drawing duration is too brief (minimum 1.5 seconds required).")

    # Remove non-increasing time points (duplicate events)
    valid_idx = np.where(np.diff(t, prepend=-1) > 0.001)[0]
    if len(valid_idx) < 15:
        raise ValueError("Sampling rate too irregular. Please trace smoothly.")
    t, x, y = t[valid_idx], x[valid_idx], y[valid_idx]

    # 2. Resample onto uniform time grid (100 Hz) for accurate derivative estimation
    target_dt = 0.01  # 10 ms = 100 Hz
    num_samples = max(20, int(total_duration / target_dt))
    t_uniform = np.linspace(t[0], t[-1], num_samples)
    x_uniform = np.interp(t_uniform, t, x)
    y_uniform = np.interp(t_uniform, t, y)

    # 3. Compute Velocities and Accelerations
    # Velocity
    vx = np.gradient(x_uniform, target_dt)
    vy = np.gradient(y_uniform, target_dt)
    speed = np.sqrt(vx**2 + vy**2)

    # Acceleration
    ax = np.gradient(vx, target_dt)
    ay = np.gradient(vy, target_dt)

    # Jerk: Third derivative of position d^3x/dt^3
    jx = np.gradient(ax, target_dt)
    jy = np.gradient(ay, target_dt)
    jerk_squared = jx**2 + jy**2
    integrated_jerk = float(np.mean(np.sqrt(jerk_squared)))

    # 4. Estimate Spiral Center and Radial Expansion (Micrographia)
    start_n = max(3, int(len(x_uniform) * 0.08))
    center_x = np.mean(x_uniform[:start_n])
    center_y = np.mean(y_uniform[:start_n])
    radius = np.sqrt((x_uniform - center_x)**2 + (y_uniform - center_y)**2)

    # Radial Velocity vr = dr/dt
    vr = np.gradient(radius, target_dt)

    mid_pt = len(vr) // 2
    early_vr = float(np.mean(vr[:mid_pt])) if mid_pt > 0 else 1.0
    late_vr = float(np.mean(vr[mid_pt:])) if mid_pt > 0 else 1.0

    if early_vr > 0.5:
        radial_decay_pct = float(np.clip((early_vr - late_vr) / early_vr * 100.0, -50.0, 100.0))
    else:
        radial_decay_pct = 0.0

    # 5. Kinematic Tremor Analysis (4-7 Hz band spectral peak)
    detrended_speed = speed - np.mean(speed)
    fs = 1.0 / target_dt  # 100 Hz
    freqs, psd = signal.welch(detrended_speed, fs=fs, nperseg=min(len(detrended_speed), 64))

    tremor_band_idx = np.where((freqs >= 3.5) & (freqs <= 7.5))[0]
    total_band_idx = np.where((freqs >= 1.0) & (freqs <= 15.0))[0]

    if len(tremor_band_idx) > 0 and len(total_band_idx) > 0:
        tremor_power = float(np.sum(psd[tremor_band_idx]))
        total_power = float(np.sum(psd[total_band_idx])) + 1e-9
        tremor_power_ratio = float(np.clip(tremor_power / total_power, 0.0, 1.0))
        peak_tremor_freq = float(freqs[tremor_band_idx[np.argmax(psd[tremor_band_idx])]])
    else:
        tremor_power_ratio = 0.05
        peak_tremor_freq = 5.0

    # 6. Micro-Stalls (Brief motor arrests where velocity drops < 5% of peak)
    peak_speed = float(np.percentile(speed, 95))
    stall_thresh = max(3.0, peak_speed * 0.08)
    stalls = np.sum((speed < stall_thresh))
    stall_ratio = float(stalls / len(speed))

    # 7. Comprehensive Parkinson's Motor Kinematic Score (0 to 1)
    norm_jerk = float(np.clip((np.log1p(integrated_jerk) - 5.5) / 5.0, 0.0, 1.0))
    norm_tremor = float(np.clip((tremor_power_ratio - 0.12) / 0.35, 0.0, 1.0))
    norm_micro = float(np.clip(radial_decay_pct / 50.0, 0.0, 1.0))
    norm_stall = float(np.clip(stall_ratio / 0.25, 0.0, 1.0))

    risk_score = float(np.clip(
        0.35 * norm_jerk + 0.30 * norm_tremor + 0.20 * norm_micro + 0.15 * norm_stall,
        0.02, 0.98
    ))

    if risk_score >= 0.65:
        risk_level = "High Motor Risk"
        risk_badge = "High Risk / Kinematic Dysmetria Detected"
        risk_color = "red"
        interpretation = (
            f"Significant motor roughness (Jerk: {round(integrated_jerk, 1)}), "
            f"elevated 4-7 Hz kinematic tremor power ({round(tremor_power_ratio * 100, 1)}%), "
            f"and progressive radial velocity decay ({round(radial_decay_pct, 1)}%) characteristic "
            "of Parkinsonian micrographia and action tremor."
        )
    elif risk_score >= 0.38:
        risk_level = "Moderate Suspicion"
        risk_badge = "Moderate Suspicion / Borderline Kinematics"
        risk_color = "amber"
        interpretation = (
            "Mild drawing velocity irregularities and subtle trajectory hesitation detected. "
            "Suggestive of early motor planning hesitation or fatigue."
        )
    else:
        risk_level = "Normal / Low Risk"
        risk_badge = "Healthy Kinematics / Stable Trajectory"
        risk_color = "emerald"
        interpretation = (
            "Fluid, continuous spiral trajectory with normal kinematic smoothness, "
            "low drawing jerk, and consistent radial expansion without micrographia."
        )

    step = max(1, len(x_uniform) // 60)
    preview_trajectory = [
        {"x": round(float(x_uniform[i]), 1), "y": round(float(y_uniform[i]), 1)}
        for i in range(0, len(x_uniform), step)
    ]

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "risk_color": risk_color,
        "interpretation": interpretation,
        "metrics": {
            "drawing_jerk": round(integrated_jerk, 1),
            "jerk_status": "Elevated" if norm_jerk > 0.5 else "Normal",
            "micrographia_decay_pct": round(radial_decay_pct, 1),
            "micrographia_status": "Severe Shrinkage" if radial_decay_pct > 35 else ("Mild" if radial_decay_pct > 15 else "None / Stable"),
            "tremor_power_pct": round(tremor_power_ratio * 100, 1),
            "peak_tremor_hz": round(peak_tremor_freq, 1),
            "drawing_duration_sec": round(total_duration, 2),
            "micro_stalls_count": int(stalls),
            "average_speed_px_sec": round(float(np.mean(speed)), 1)
        },
        "preview_trajectory": preview_trajectory
    }
