import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import pickle
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer

# ---------------------------
# Конфигурация
# ---------------------------
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
N_MFCC = 30   # 30 коэффициентов
HOP_LENGTH = 160
WIN_LENGTH = 400

# ---------------------------
# Извлечение признаков (MFCC + Delta + Delta-Delta)
# ---------------------------
def extract_mfcc_delta_stats(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        # Добавим дельты и дельты-дельты для каждого коэффициента
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=WIN_LENGTH, hop_length=HOP_LENGTH)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Объединяем все признаки
        mfcc_full = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Вычисляем 7 статистик для каждого признака
        stats_list = []
        for c in range(mfcc_full.shape[0]):
            coef = mfcc_full[c, :]
            if coef.size == 0:
                coef = np.zeros(1)
            stats = [
                np.mean(coef), np.std(coef),
                np.min(coef), np.max(coef),
                np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)
            ]
            stats_list.extend(stats)
        return np.array(stats_list)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(N_MFCC * 3 * 7)  # 630 признаков

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
                continue
            feats = extract_mfcc_delta_stats(audio_path)
            X.append(feats)
            y.append(label)
            if idx % 2000 == 0:
                print(f"Processed {idx} files...")
    return np.array(X), np.array(y)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    cache_file = "xgboost_delta_cache.pkl"
    if os.path.exists(cache_file):
        print("Loading cached data...")
        with open(cache_file, 'rb') as f:
            X_train, y_train, X_dev, y_dev = pickle.load(f)
    else:
        print("Loading train data...")
        X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR)
        print("Loading dev data...")
        X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR)
        
        # Сохраняем в кэш
        with open(cache_file, 'wb') as f:
            pickle.dump((X_train, y_train, X_dev, y_dev), f)
        print("Data cached successfully.")

    print(f"Train size: {X_train.shape}, Dev size: {X_dev.shape}")

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    # Обучение XGBoost
    print("\nTraining XGBoost with MFCC + Delta + Delta2...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train_scaled, y_train)

    # Оценка
    preds_proba = xgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, thresh = compute_eer(preds_proba, y_dev)
    print(f"\nEER on dev set: {eer*100:.2f}%")

    # Сохраняем модель
    model_path = "xgboost_delta_model.json"
    xgb_model.save_model(model_path)
    print(f"Model saved to {model_path}")
