# NeuroBreathe AI — Clinical Diagnostic Platform

NeuroBreathe AI is a unified full-stack biomedical machine learning application combining two pre-trained diagnostic pipelines:
1. **Parkinson's Disease Voice Biomarker Screening**: An ensemble XGBoost classifier utilizing 22 phonation acoustic features (jitter, shimmer, fundamental frequency, noise harmonics, and nonlinear fractal dynamics like RPDE, DFA, and PPE) to screen for neurological dysphonia.
2. **Respiratory Acoustic Diagnostic AI**: A LightGBM classifier with an automated 60-feature acoustic extraction pipeline (multi-band MFCCs, chroma STFT, spectral contrast, zero-crossing rate, RMS energy, spectral centroid, bandwidth, and rolloff) to detect Chronic Obstructive Pulmonary Disease (COPD) from electronic stethoscope and microphone lung sound recordings.

---

## Key Features

- **Production-Grade FastAPI Backend**: High-performance asynchronous API endpoints for single-patient prediction, batch CSV screening, clinical presets, and audio streaming.
- **Modern Medical Frontend**: Responsive, dark-themed clinical dashboard built with HTML5, Tailwind CSS, Lucide icons, and Chart.js. Zero Node.js or npm dependencies required.
- **Live Stethoscope / Microphone Recording**: Real-time in-browser audio capture with an HTML5 Web Audio API oscilloscope waveform visualizer.
- **1-Click Clinical Sound Library**: Preloaded with authentic ICBHI lung sound recordings (COPD, URTI, Healthy) with in-browser playback and 1-click analysis.
- **Batch Cohort CSV Processing**: Upload multi-patient datasets for instant cohort risk stratification and download enriched diagnostic CSVs.
- **Biomarker Explainability**: Reference normal ranges and deviation analysis indicating which specific vocal or acoustic biomarkers contributed to the risk score.

---

## System Architecture

```
neurobreathe-ai/
├── backend/                        # FastAPI application backend
│   ├── config.py                   # Path resolution and server settings
│   ├── main.py                     # App entry point, CORS, static mounting
│   ├── models_loader.py            # Singleton model manager
│   ├── routes/
│   │   ├── health.py               # /api/health diagnostic status
│   │   ├── parkinson.py            # /api/parkinson/* endpoints
│   │   └── respiratory.py          # /api/respiratory/* endpoints
│   └── services/
│       ├── parkinson_service.py    # Scaling, feature selection, XGBoost
│       └── respiratory_service.py  # Librosa 60-feat extraction, LightGBM
│
├── frontend/                       # Modern Clinical UI (Zero-npm SPA)
│   ├── index.html                  # Responsive UI layout & tabs
│   ├── styles.css                  # Custom styling & animations
│   └── app.js                      # Web Audio, Chart.js, API client
│
├── tests/
│   └── test_api.py                 # Automated integration test suite
│
├── run_app.py                      # One-click Python launcher with auto-browser
├── start.bat                       # Windows 1-click launch batch script
└── README.md                       # Documentation
```

---

## Quickstart & Launching

### Option 1: 1-Click Launch (Windows)
Double-click `start.bat`. It will automatically select the dedicated Python environment, launch the server, and open `http://127.0.0.1:8000` in your default browser.

### Option 2: Python Command Line
Activate the environment and run `run_app.py`:

```bash
# Using the dedicated virtual environment
neurobreathe-ai\ml_training\venv\Scripts\python.exe run_app.py

# Or using standard python
python run_app.py
```

The application will bind to `http://127.0.0.1:8000` and automatically open your web browser.

---

## Running Automated Tests

Run the full integration test suite to verify all endpoints, pipelines, and models:

```bash
neurobreathe-ai\ml_training\venv\Scripts\python.exe tests/test_api.py
```

All 9 tests will execute:
- System health status and model online flags
- Parkinson clinical presets retrieval
- Healthy control phonation prediction
- Parkinson positive patient prediction
- Clinical respiratory lung sound library listing
- COPD patient 104 audio classification
- URTI patient 101 audio classification
- Multi-patient batch CSV cohort processing
- Frontend index delivery

---

## API Reference

### Health & Diagnostics
- `GET /api/health`: Returns overall system status, model readiness, and dataset counts.

### Parkinson's Voice Biomarkers
- `GET /api/parkinson/presets`: Returns 4 pre-configured patient cases (Healthy female/male, Moderate/Severe Parkinson's).
- `GET /api/parkinson/features-info`: Returns reference standards, units, and descriptions for all 22 biomarkers.
- `POST /api/parkinson/predict`: Accepts JSON body `{"features": { ... }}` with 22 features, returning classification (`status`: 0 or 1), probability scores, risk tier, and biomarker deviations.
- `POST /api/parkinson/predict-batch`: Accepts a multipart `.csv` file with patient rows, returning cohort summary statistics and per-patient predictions.

### Respiratory Acoustic Analysis
- `GET /api/respiratory/samples`: Returns metadata on clinical lung sound recordings available on the server.
- `GET /api/respiratory/sample-audio/{sample_id}`: Streams audio for in-browser playback.
- `POST /api/respiratory/predict-sample/{sample_id}`: Runs 60-feature extraction and LightGBM inference on a clinical sample.
- `POST /api/respiratory/predict-audio`: Accepts uploaded `.wav` / `.mp3` or recorded microphone audio blob, extracts 60 features, and returns COPD classification, probabilities, waveform envelope, and MFCC summary.

---

## Model Pipeline Details

### 1. Parkinson's Disease Model (XGBoost)
- **Dataset**: Oxford Parkinson's Voice Dataset (Max Little et al.)
- **Preprocessing**: `StandardScaler` fitted on 22 acoustic features
- **Feature Selection**: `SelectFromModel` with median threshold on an initial XGBoost selector
- **Final Estimator**: `XGBClassifier` (300 estimators, max_depth=4, learning_rate=0.05)
- **Evaluation**: 5-Fold Stratified Cross-Validation (>94% accuracy, ROC-AUC > 0.95)

### 2. Respiratory Sound Model (LightGBM)
- **Dataset**: ICBHI 2017 Respiratory Sound Database
- **Sampling**: Resampled to 4,000 Hz target
- **Features Extracted (60)**:
  - 13 MFCC Means + 13 MFCC Standard Deviations (26)
  - 12 Chroma STFT Means + 12 Chroma STFT Standard Deviations (24)
  - 5 Multi-band Spectral Contrast Means (5)
  - Zero Crossing Rate Mean (1)
  - RMS Energy Mean (1)
  - Spectral Centroid Mean (1)
  - Spectral Bandwidth Mean (1)
  - Spectral Rolloff Mean (1)
- **Final Estimator**: `LGBMClassifier` (400 estimators, class_weight="balanced")
- **Evaluation**: Stratified Group 5-Fold Cross-Validation grouped at patient ID
