import os
import numpy as np
import pandas as pd
import librosa

AUDIO_DIR = "data/respiratory/audio_and_txt_files"
DIAGNOSIS_CSV = "data/respiratory/patient_diagnosis.csv"
OUTPUT_CSV = "data/respiratory/extracted_features.csv"


def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=4000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # fmin/n_bands lowered to fit within Nyquist for sr=4000 (lung sounds are low-frequency anyway)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50.0, n_bands=4)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
        spec_contrast.mean(axis=1),
        zcr.mean(axis=1), rmse.mean(axis=1),
        spec_centroid.mean(axis=1), spec_bandwidth.mean(axis=1), rolloff.mean(axis=1),
    ])


def main():
    diagnosis_df = pd.read_csv(DIAGNOSIS_CSV, names=["patient_id", "diagnosis"])
    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"Found {len(wav_files)} wav files")

    rows = []
    for i, fname in enumerate(wav_files):
        patient_id = int(fname.split("_")[0])
        match = diagnosis_df[diagnosis_df["patient_id"] == patient_id]
        if match.empty:
            continue
        diagnosis = match["diagnosis"].values[0]

        try:
            feats = extract_features(os.path.join(AUDIO_DIR, fname))
        except Exception as e:
            print("Skipped", fname, "->", e)
            continue

        row = {"patient_id": patient_id, "diagnosis": diagnosis}
        row.update({f"f{j}": v for j, v in enumerate(feats)})
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(wav_files)}")

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(OUTPUT_CSV, index=False)
    print("Saved:", OUTPUT_CSV, "shape:", feature_df.shape)


if __name__ == "__main__":
    main()