import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchaudio
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from utils.metrics import compute_eer

# Пути
BASE_DATA = r"D:\фокусы\исследования\antispoof\data\2019\LA"
PROTOCOL_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_cm_protocols")
TRAIN_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_train")
DEV_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_dev")

PROTOCOL_TRAIN = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
PROTOCOL_DEV = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

def extract_lfcc(file_path, n_mfcc=20):
    try:
        waveform_np, sr = sf.read(file_path)
        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
        mfcc = transform(waveform)          # (n_mfcc, time)
        feat = mfcc.mean(dim=1)             # (n_mfcc,)
        feat = feat.cpu().numpy().flatten() # гарантированно 1D
        if feat.shape[0] != n_mfcc:
            # если меньше — дополним нулями, если больше — обрежем
            if feat.shape[0] < n_mfcc:
                feat = np.pad(feat, (0, n_mfcc - feat.shape[0]))
            else:
                feat = feat[:n_mfcc]
        return feat
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(n_mfcc)  # возвращаем нули, чтобы не ломать загрузку

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
                if idx < 5:  # покажем первые 5 пропущенных
                    print(f"Missing: {audio_path}")
                continue
            feat = extract_lfcc(audio_path)
            X.append(feat)
            y.append(label)
            if idx < 5:
                print(f"Loaded: {audio_path} -> feat shape {feat.shape}")
    print(f"Total loaded from {protocol_file}: {len(X)}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Loading train...")
    X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR)
    print(f"Train samples: {len(X_train)}")
    print("Loading dev...")
    X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR)
    print(f"Dev samples: {len(X_dev)}")

    if len(X_train) == 0 or len(X_dev) == 0:
        print("No data loaded. Check paths and protocol parsing.")
        sys.exit(1)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds_proba = clf.predict_proba(X_dev)[:, 1]
    eer, thresh = compute_eer(preds_proba, y_dev)
    print(f"EER on dev set: {eer*100:.2f}%")