import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchaudio
import soundfile as sf
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer

BASE_DATA = r"D:\фокусы\исследования\antispoof\data\2019\LA"
PROTOCOL_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_cm_protocols")
TRAIN_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_train")
DEV_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_dev")

PROTOCOL_TRAIN = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
PROTOCOL_DEV = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

SAMPLE_RATE = 16000
N_MFCC = 20
HOP_LENGTH = 160
WIN_LENGTH = 400

def extract_mfcc_stats(file_path):
    waveform_np, sr = sf.read(file_path)
    waveform = torch.from_numpy(waveform_np.astype(np.float32))
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": 400, "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH}
    )
    mfcc = mfcc_transform(waveform).cpu().numpy()
    stats_list = []
    for c in range(N_MFCC):
        coef = mfcc[c, :]
        if coef.size == 0:
            coef = np.zeros(1)
        stats = [
            np.mean(coef), np.std(coef),
            np.min(coef), np.max(coef),
            np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)
        ]
        try:
            from scipy.stats import skew, kurtosis
            stats.append(skew(coef))
            stats.append(kurtosis(coef))
        except ImportError:
            pass
        stats_list.extend(stats)
    return np.array(stats_list)

def load_data(protocol_file, audio_dir):
    X, y = [], []
    with open(protocol_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label = parts[4]
            label = 0 if label == 'bonafide' else 1
            audio_path = os.path.join(audio_dir, "flac", file_id + '.flac')
            if not os.path.exists(audio_path):
                if idx < 3:
                    print(f"Missing: {audio_path}")
                continue
            feats = extract_mfcc_stats(audio_path)
            X.append(feats)
            y.append(label)
            if idx < 5:
                print(f"Loaded: {audio_path} -> feature length {len(feats)}")
    print(f"Total loaded: {len(X)}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Loading train...")
    X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR)
    print("Loading dev...")
    X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                               num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_dev)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    print(f"EER on dev set: {eer*100:.2f}%")
