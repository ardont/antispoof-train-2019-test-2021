import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import lightgbm as lgb
import librosa
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer
from utils.lfcc import extract_lfcc

# ---------------------------
# Конфигурация
# ---------------------------
# Динамически определяем путь в workspace, иначе откатываемся к D:\
LOCAL_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "2019", "LA"))
if os.path.exists(LOCAL_DATA):
    BASE_DATA = LOCAL_DATA
else:
    BASE_DATA = r"D:\фокусы\исследования\antispoof\data\2019\LA"

PROTOCOL_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_cm_protocols")
TRAIN_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_train")
DEV_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_dev")

PROTOCOL_TRAIN = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
PROTOCOL_DEV = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

SAMPLE_RATE = 16000
N_LFCC = 20
N_FILTERS = 128

# ---------------------------
# Извлечение статистик из LFCC
# ---------------------------
def extract_lfcc_stats(file_path):
    try:
        waveform_np, sr = sf.read(file_path)
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)

        if sr != SAMPLE_RATE:
            waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Используем наш кастомный LFCC экстрактор
        lfcc = extract_lfcc(waveform_np, sr=SAMPLE_RATE, n_lfcc=N_LFCC, n_filters=N_FILTERS)

        # Статистики для каждого коэффициента
        stats_list = []
        for c in range(N_LFCC):
            coef = lfcc[c, :]
            if coef.size == 0:
                coef = np.zeros(1)
            stats = [
                np.mean(coef),
                np.std(coef),
                np.min(coef),
                np.max(coef),
                np.percentile(coef, 25),
                np.percentile(coef, 50),
                np.percentile(coef, 75)
            ]
            # Асимметрия и эксцесс
            try:
                from scipy.stats import skew, kurtosis
                stats.append(skew(coef))
                stats.append(kurtosis(coef))
            except ImportError:
                pass
            stats_list.extend(stats)
        return np.array(stats_list)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Возвращаем нули в случае ошибки (20 коэффициентов * 9 статистик = 180)
        return np.zeros(N_LFCC * 9)

# ---------------------------
# Загрузка данных
# ---------------------------
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
            feats = extract_lfcc_stats(audio_path)
            X.append(feats)
            y.append(label)
            if idx < 5:
                print(f"Loaded: {audio_path} -> feature vector length {len(feats)}")
    print(f"Total loaded from {protocol_file}: {len(X)}")
    return np.array(X), np.array(y)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("Loading train data...")
    X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR)
    print(f"Train samples: {len(X_train)}")
    print("Loading dev data...")
    X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR)
    print(f"Dev samples: {len(X_dev)}")

    if len(X_train) == 0 or len(X_dev) == 0:
        print("No data loaded. Check paths and protocol parsing.")
        sys.exit(1)

    # Нормализация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)

    # Обучение LightGBM
    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    preds_proba = model.predict_proba(X_dev)[:, 1]
    eer, thresh = compute_eer(preds_proba, y_dev)
    print(f"EER on dev set: {eer*100:.2f}%")