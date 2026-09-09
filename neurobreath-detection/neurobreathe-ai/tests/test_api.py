"""
NeuroBreathe AI - Automated Test Suite
Verifies API endpoints, ML model pipelines, and audio feature extraction.
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.main import app
from backend.models_loader import models

def test_all_endpoints():
    print("Running automated integration tests with lifespan context...")
    with TestClient(app) as client:
        # 1. Health
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["models"]["parkinson"]["ready"] is True
        assert data["models"]["respiratory"]["ready"] is True
        print("[PASSED] Health endpoint verified - both models online.")

        # 2. Parkinson Presets
        res = client.get("/api/parkinson/presets")
        assert res.status_code == 200
        presets = res.json().get("presets", [])
        assert len(presets) >= 4
        print(f"[PASSED] Parkinson presets returned {len(presets)} cases.")

        # 3. Parkinson Healthy Prediction
        h_features = {
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
            "Shimmer:APQ5": 0.0068,
            "MDVP:APQ": 0.00802,
            "Shimmer:DDA": 0.01689,
            "NHR": 0.00339,
            "HNR": 26.775,
            "RPDE": 0.422229,
            "DFA": 0.741367,
            "spread1": -7.3483,
            "spread2": 0.177551,
            "D2": 1.743867,
            "PPE": 0.085569
        }
        res = client.post("/api/parkinson/predict", json={"features": h_features})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["data"]["status"] == 0
        assert data["data"]["diagnosis"] == "Healthy Control"
        assert data["data"]["probability_healthy"] > 0.8
        print(f"[PASSED] Healthy Parkinson control correctly diagnosed (Confidence: {data['data']['probability_healthy']}).")

        # 4. Parkinson Positive Case Prediction
        pd_features = {
            "MDVP:Fo(Hz)": 119.992,
            "MDVP:Fhi(Hz)": 157.302,
            "MDVP:Flo(Hz)": 74.997,
            "MDVP:Jitter(%)": 0.00784,
            "MDVP:Jitter(Abs)": 0.00007,
            "MDVP:RAP": 0.0037,
            "MDVP:PPQ": 0.00554,
            "Jitter:DDP": 0.01109,
            "MDVP:Shimmer": 0.04374,
            "MDVP:Shimmer(dB)": 0.426,
            "Shimmer:APQ3": 0.02182,
            "Shimmer:APQ5": 0.0313,
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
        res = client.post("/api/parkinson/predict", json={"features": pd_features})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["data"]["status"] == 1
        assert data["data"]["diagnosis"] == "Parkinson's Disease"
        assert data["data"]["probability_parkinson"] > 0.8
        print(f"[PASSED] Parkinson patient correctly diagnosed (Confidence: {data['data']['probability_parkinson']}).")

        # 5. Respiratory Samples List
        res = client.get("/api/respiratory/samples")
        assert res.status_code == 200
        data = res.json()
        assert len(data["samples"]) > 0
        print(f"[PASSED] Found {len(data['samples'])} active clinical lung sound samples.")

        # 6. Respiratory COPD Sample Prediction
        res = client.post("/api/respiratory/predict-sample/104_copd")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["data"]["prediction"] == "COPD"
        assert data["data"]["probabilities"]["COPD"] > 0.8
        print(f"[PASSED] Patient 104 correctly classified as COPD (Confidence: {data['data']['probabilities']['COPD']}).")

        # 7. Respiratory URTI/Non-COPD Sample Prediction
        res = client.post("/api/respiratory/predict-sample/101_urti")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["data"]["prediction"] == "Not_COPD"
        assert data["data"]["probabilities"]["Not_COPD"] > 0.8
        print(f"[PASSED] Patient 101 correctly classified as Non-COPD (Confidence: {data['data']['probabilities']['Not_COPD']}).")

        # 8. Parkinson Batch CSV Prediction
        batch_csv_content = (
            "name,MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE\n"
            "Patient_Healthy,197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.097,0.00563,0.0068,0.00802,0.01689,0.00339,26.775,0.422229,0.741367,-7.3483,0.177551,1.743867,0.085569\n"
            "Patient_Parkinson,119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654\n"
        )
        res = client.post(
            "/api/parkinson/predict-batch",
            files={"file": ("test_batch.csv", batch_csv_content, "text/csv")}
        )
        assert res.status_code == 200
        batch_data = res.json()
        assert batch_data["status"] == "success"
        assert batch_data["data"]["summary"]["total_patients"] == 2
        assert batch_data["data"]["summary"]["parkinson_detected"] == 1
        assert batch_data["data"]["summary"]["healthy_detected"] == 1
        print("[PASSED] Batch CSV prediction endpoint verified (1 healthy, 1 parkinson).")

        # 9. Live Respiratory Audio Upload (.wav)
        sample_resp_path = ROOT_DIR / "neurobreathe-ai" / "ml_training" / "data" / "respiratory" / "audio_and_txt_files" / "101_1b1_Al_sc_Meditron.wav"
        if sample_resp_path.exists():
            with open(sample_resp_path, "rb") as f:
                res = client.post(
                    "/api/respiratory/predict-audio",
                    files={"file": ("recorded_lung_sound.wav", f.read(), "audio/wav")}
                )
            assert res.status_code == 200
            resp_audio_data = res.json()
            assert resp_audio_data["status"] == "success"
            assert "probabilities" in resp_audio_data["data"]
            print(f"[PASSED] Live respiratory audio upload verified ({resp_audio_data['data']['diagnosis']}).")

        # 10. Parkinson Phonation & MPT Live Audio Assessment
        import io, struct
        import numpy as np
        sr = 22050
        dur = 2.5
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # Synthesize sustained /a/ at 180 Hz
        y = (0.5 * np.sin(2 * np.pi * 180.0 * t) + 0.01 * np.random.randn(len(t))).astype(np.float32)
        int_samples = (y * 32767).astype(np.int16).tobytes()

        hdr = bytearray()
        hdr.extend(b'RIFF')
        hdr.extend(struct.pack('<I', 36 + len(int_samples)))
        hdr.extend(b'WAVEfmt ')
        hdr.extend(struct.pack('<I', 16))
        hdr.extend(struct.pack('<H', 1))
        hdr.extend(struct.pack('<H', 1))
        hdr.extend(struct.pack('<I', sr))
        hdr.extend(struct.pack('<I', sr * 2))
        hdr.extend(struct.pack('<H', 2))
        hdr.extend(struct.pack('<H', 16))
        hdr.extend(b'data')
        hdr.extend(struct.pack('<I', len(int_samples)))
        synthetic_wav = bytes(hdr) + int_samples

        res = client.post(
            "/api/parkinson/predict-audio",
            files={"file": ("mpt_phonation.wav", synthetic_wav, "audio/wav")}
        )
        assert res.status_code == 200
        park_audio_data = res.json()
        assert park_audio_data["status"] == "success"
        assert "mpt" in park_audio_data["data"]
        assert park_audio_data["data"]["mpt"]["duration_seconds"] > 1.0
        assert "biomarker_analysis" in park_audio_data["data"]
        print(f"[PASSED] Parkinson phonation MPT assessment verified (MPT: {park_audio_data['data']['mpt']['duration_seconds']}s, Diagnosis: {park_audio_data['data']['diagnosis']}).")

        # 11. Fast Breathing Live Microphone Audio Screening (High Risk)
        sr = 4000
        dur = 8.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # 30 BPM: 1 breath cycle every 2.0s -> frequency 0.5 Hz
        fast_envelope = (0.5 * (1.0 + np.sin(2 * np.pi * (30.0 / 60.0) * t)))**2
        fast_noise = (fast_envelope * (np.random.randn(len(t)) * 0.4)).astype(np.float32)
        fast_samples = (np.clip(fast_noise, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        hdr_fast = bytearray()
        hdr_fast.extend(b'RIFF')
        hdr_fast.extend(struct.pack('<I', 36 + len(fast_samples)))
        hdr_fast.extend(b'WAVEfmt ')
        hdr_fast.extend(struct.pack('<I', 16))
        hdr_fast.extend(struct.pack('<H', 1))
        hdr_fast.extend(struct.pack('<H', 1))
        hdr_fast.extend(struct.pack('<I', sr))
        hdr_fast.extend(struct.pack('<I', sr * 2))
        hdr_fast.extend(struct.pack('<H', 2))
        hdr_fast.extend(struct.pack('<H', 16))
        hdr_fast.extend(b'data')
        hdr_fast.extend(struct.pack('<I', len(fast_samples)))
        fast_wav = bytes(hdr_fast) + fast_samples

        res = client.post(
            "/api/respiratory/predict-audio",
            data={"is_live_mic": "true"},
            files={"file": ("recorded_lung_sound.wav", fast_wav, "audio/wav")}
        )
        assert res.status_code == 200
        fast_data = res.json()
        assert fast_data["status"] == "success"
        assert fast_data["data"]["risk_color"] == "red"
        assert fast_data["data"]["respiratory_rate"]["bpm"] >= 22.0
        assert fast_data["data"]["respiratory_rate"]["is_fast"] is True
        print(f"[PASSED] Fast breathing test verified -> BPM: {fast_data['data']['respiratory_rate']['bpm']}, Risk: {fast_data['data']['risk_level']}.")

        # 12. Normal / Slow Breathing Live Microphone Audio Screening (Low Risk / Healthy)
        # 14 BPM: 1 breath cycle every 4.3s -> frequency 0.233 Hz
        slow_envelope = (0.5 * (1.0 + np.sin(2 * np.pi * (14.0 / 60.0) * t)))**2
        slow_noise = (slow_envelope * (np.random.randn(len(t)) * 0.4)).astype(np.float32)
        slow_samples = (np.clip(slow_noise, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        hdr_slow = bytearray()
        hdr_slow.extend(b'RIFF')
        hdr_slow.extend(struct.pack('<I', 36 + len(slow_samples)))
        hdr_slow.extend(b'WAVEfmt ')
        hdr_slow.extend(struct.pack('<I', 16))
        hdr_slow.extend(struct.pack('<H', 1))
        hdr_slow.extend(struct.pack('<H', 1))
        hdr_slow.extend(struct.pack('<I', sr))
        hdr_slow.extend(struct.pack('<I', sr * 2))
        hdr_slow.extend(struct.pack('<H', 2))
        hdr_slow.extend(struct.pack('<H', 16))
        hdr_slow.extend(b'data')
        hdr_slow.extend(struct.pack('<I', len(slow_samples)))
        slow_wav = bytes(hdr_slow) + slow_samples

        res = client.post(
            "/api/respiratory/predict-audio",
            data={"is_live_mic": "true"},
            files={"file": ("recorded_lung_sound.wav", slow_wav, "audio/wav")}
        )
        assert res.status_code == 200
        slow_data = res.json()
        assert slow_data["status"] == "success"
        assert slow_data["data"]["risk_color"] == "emerald"
        assert slow_data["data"]["respiratory_rate"]["bpm"] < 22.0
        assert slow_data["data"]["respiratory_rate"]["is_fast"] is False
        print(f"[PASSED] Normal/Slow breathing test verified -> BPM: {slow_data['data']['respiratory_rate']['bpm']}, Risk: {slow_data['data']['risk_level']}.")

        # 13. Frontend HTML Delivery
        res = client.get("/")
        assert res.status_code == 200
        assert "NeuroBreathe" in res.text
        print("[PASSED] Frontend index.html verified.")

        # 14. Digital Archimedes Spiral & Micrographia Kinematic Analysis
        import math
        healthy_points = []
        cx, cy = 250.0, 200.0
        for i in range(150):
            theta = (i / 150.0) * (6.0 * math.pi)
            r = 3.0 + 3.2 * theta
            healthy_points.append({
                "x": cx + r * math.cos(theta),
                "y": cy + r * math.sin(theta),
                "t": float(i * 30.0)  # 30 ms interval = 4.5s
            })

        res = client.post(
            "/api/parkinson/test-spiral",
            json={"points": healthy_points}
        )
        assert res.status_code == 200
        spiral_data = res.json()
        assert spiral_data["status"] == "success"
        assert spiral_data["data"]["risk_color"] == "emerald"
        assert "drawing_jerk" in spiral_data["data"]["metrics"]
        print(f"[PASSED] Archimedes Spiral kinematic test verified (Risk: {spiral_data['data']['risk_level']}, Jerk: {spiral_data['data']['metrics']['drawing_jerk']}).")

        # 15. Virtual Dynamic Acoustic Spirometry Test (FEV1 / FVC)
        spiro_sr = 16000
        spiro_dur = 3.0
        t_spiro = np.linspace(0, spiro_dur, int(spiro_sr * spiro_dur), endpoint=False)
        # Fast attack and rapid normal physiological decay
        flow_env = np.where(t_spiro < 0.15, (t_spiro / 0.15), np.exp(-(t_spiro - 0.15) * 3.2))
        spiro_noise = (flow_env * (np.random.randn(len(t_spiro)) * 0.6)).astype(np.float32)
        spiro_samples = (np.clip(spiro_noise, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        hdr_spiro = bytearray()
        hdr_spiro.extend(b'RIFF')
        hdr_spiro.extend(struct.pack('<I', 36 + len(spiro_samples)))
        hdr_spiro.extend(b'WAVEfmt ')
        hdr_spiro.extend(struct.pack('<I', 16))
        hdr_spiro.extend(struct.pack('<H', 1))
        hdr_spiro.extend(struct.pack('<H', 1))
        hdr_spiro.extend(struct.pack('<I', spiro_sr))
        hdr_spiro.extend(struct.pack('<I', spiro_sr * 2))
        hdr_spiro.extend(struct.pack('<H', 2))
        hdr_spiro.extend(struct.pack('<H', 16))
        hdr_spiro.extend(b'data')
        hdr_spiro.extend(struct.pack('<I', len(spiro_samples)))
        spiro_wav = bytes(hdr_spiro) + spiro_samples

        res = client.post(
            "/api/respiratory/test-spirometry",
            files={"file": ("candle_blow.wav", spiro_wav, "audio/wav")}
        )
        assert res.status_code == 200
        spiro_data = res.json()
        assert spiro_data["status"] == "success"
        assert spiro_data["data"]["fev1_liters"] > 1.0
        assert spiro_data["data"]["fvc_liters"] > 2.0
        assert spiro_data["data"]["fev1_fvc_percent"] >= 65.0
        assert len(spiro_data["data"]["flow_volume_curve"]) > 5
        print(f"[PASSED] Virtual Acoustic Spirometry test verified (FEV1/FVC: {spiro_data['data']['fev1_fvc_percent']}%, PEF: {spiro_data['data']['pef_lps']} L/s, Grade: {spiro_data['data']['obstruction_grade']}).")

        # 16. Contactless Webcam rPPG Cardiopulmonary Coupling Test (RSA)
        rppg_samples = []
        fs = 30.0
        phase = 0.0
        for i in range(450):  # 15s at 30 fps
            t_sec = i / fs
            resp_mod = math.sin(2 * math.pi * (15.0 / 60.0) * t_sec)  # 15 BPM
            inst_hr = 72.0 + 12.0 * resp_mod
            phase += 2 * math.pi * (inst_hr / 60.0) / fs
            pulse_mod = math.sin(phase)  # ~72 BPM
            val = 140.0 + 4.0 * pulse_mod + 2.5 * resp_mod
            rppg_samples.append({"t": round(t_sec * 1000.0), "val": round(val, 2)})

        res = client.post(
            "/api/respiratory/test-rppg",
            json={"samples": rppg_samples}
        )
        assert res.status_code == 200
        rppg_data = res.json()
        assert rppg_data["status"] == "success"
        assert 60.0 <= rppg_data["data"]["heart_rate_bpm"] <= 85.0
        assert 10.0 <= rppg_data["data"]["respiratory_rate_bpm"] <= 20.0
        assert rppg_data["data"]["ei_ratio"] >= 1.10
        # 17. Spirometry Presets Listing and Loading
        res = client.get("/api/respiratory/spirometry-presets")
        assert res.status_code == 200
        spiro_presets = res.json().get("presets", [])
        assert len(spiro_presets) >= 3
        print(f"[PASSED] Found {len(spiro_presets)} standardized clinical spirometry presets.")

        res_preset = client.post("/api/respiratory/spirometry-preset/normal")
        assert res_preset.status_code == 200
        preset_data = res_preset.json()["data"]
        assert preset_data["fev1_liters"] >= 3.0
        assert "what_this_means" in preset_data
        print(f"[PASSED] Spirometry preset 'normal' loaded (FEV1: {preset_data['fev1_liters']}L, Status: {preset_data['simple_status']}).")

        # 18. Composite Respiratory Unified Evaluation
        res_comp_resp = client.post(
            "/api/respiratory/composite-predict",
            json={"spirometry": preset_data, "auscultation": {"prediction": "Not_COPD", "risk_color": "emerald"}}
        )
        assert res_comp_resp.status_code == 200
        comp_resp_data = res_comp_resp.json()
        assert comp_resp_data["status"] == "success"
        assert comp_resp_data["risk_color"] == "emerald"
        assert len(comp_resp_data["care_tips"]["breathing_exercises"]) > 0
        print(f"[PASSED] Composite Respiratory Assessment verified ({comp_resp_data['overall_status']}).")

        # 19. Composite Parkinson Unified Evaluation
        res_comp_park = client.post(
            "/api/parkinson/composite-predict",
            json={"vocal": {"status": 0, "diagnosis": "Healthy Control", "risk_color": "emerald"}}
        )
        assert res_comp_park.status_code == 200
        comp_park_data = res_comp_park.json()
        assert comp_park_data["status"] == "success"
        assert comp_park_data["risk_color"] == "emerald"
        assert len(comp_park_data["care_tips"]["voice_exercises"]) > 0
        print(f"[PASSED] Composite Parkinson Assessment verified ({comp_park_data['overall_status']}).")

    print("\nALL 19 COMPREHENSIVE INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all_endpoints()

