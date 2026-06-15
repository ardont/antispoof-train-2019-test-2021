import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import time
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from utils.metrics import compute_eer

# Импортируем функцию extract_mfcc_stats из предыдущего скрипта (или скопируем)
# Для надёжности скопируем её сюда же

import librosa

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
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                n_fft=WIN_LENGTH, hop_length=HOP_LENGTH)
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
                continue
            feats = extract_mfcc_stats(audio_path)
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

# Загрузка или кэширование данных
cache_file = "data_cache.pkl"
if os.path.exists(cache_file):
    print("Loading cached data...")
    with open(cache_file, 'rb') as f:
        X_train, y_train, X_dev, y_dev = pickle.load(f)
else:
    print("Loading train...")
    X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR)
    print("Loading dev...")
    X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR)
    with open(cache_file, 'wb') as f:
        pickle.dump((X_train, y_train, X_dev, y_dev), f)
    print("Data cached.")

# Нормализация (одинаковая для всех моделей)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)

results = {}

# 1. LightGBM (базовый, для проверки)
print("Training LightGBM...")
start = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                               num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbose=-1)
lgb_model.fit(X_train_scaled, y_train)
preds = lgb_model.predict_proba(X_dev_scaled)[:, 1]
eer, _ = compute_eer(preds, y_dev)
results['LightGBM (MFCC20+stats)'] = eer
print(f"EER: {eer*100:.2f}% (time {time.time()-start:.1f}s)")

# 2. XGBoost
print("Training XGBoost...")
start = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
preds = xgb_model.predict_proba(X_dev_scaled)[:, 1]
eer, _ = compute_eer(preds, y_dev)
results['XGBoost (MFCC20+stats)'] = eer
print(f"EER: {eer*100:.2f}% (time {time.time()-start:.1f}s)")

# 3. MLP (нейросеть)
print("Training MLP...")
start = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
                    max_iter=50, early_stopping=True, validation_fraction=0.1,
                    random_state=42, verbose=False)
mlp.fit(X_train_scaled, y_train)
preds = mlp.predict_proba(X_dev_scaled)[:, 1]
eer, _ = compute_eer(preds, y_dev)
results['MLP (MFCC20+stats)'] = eer
print(f"EER: {eer*100:.2f}% (time {time.time()-start:.1f}s)")

# Сохраняем результаты
with open("results_night.txt", "w") as f:
    for name, eer in results.items():
        f.write(f"{name}: {eer*100:.2f}%\n")
print("\nResults saved to results_night.txt")