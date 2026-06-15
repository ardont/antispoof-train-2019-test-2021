import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import soundfile as sf
import pickle
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from catboost import CatBoostClassifier
from utils.metrics import compute_eer
from concurrent.futures import ThreadPoolExecutor

# ---------------------------
# Конфигурация
# ---------------------------
LOCAL_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "2021"))
if os.path.exists(LOCAL_DATA):
    BASE_DATA = LOCAL_DATA
else:
    BASE_DATA = r"D:\фокусы\исследования\antispoof\data\2021"

METADATA_FILE = os.path.join(BASE_DATA, "keys", "LA", "CM", "trial_metadata.txt")
AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2021_LA_eval", "flac")

SAMPLE_RATE = 16000
N_MFCC = 30
HOP_LENGTH = 160
WIN_LENGTH = 400

# ---------------------------
# Извлечение признаков
# ---------------------------
def extract_mfcc_delta_stats(file_path):
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        # MFCC + Delta + Delta-Delta
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=WIN_LENGTH, hop_length=HOP_LENGTH)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_full = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # CMS (Cepstral Mean Subtraction) для устойчивости к каналу
        mean = np.mean(mfcc_full, axis=1, keepdims=True)
        mfcc_full = mfcc_full - mean
        
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
        return None

def process_file(line):
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    file_id = parts[1]
    label = parts[5] # 'spoof' or 'bonafide'
    label_num = 0 if label == 'bonafide' else 1
    
    audio_path = os.path.join(AUDIO_DIR, file_id + '.flac')
    if not os.path.exists(audio_path):
        return None
        
    feats = extract_mfcc_delta_stats(audio_path)
    if feats is None:
        return None
    return feats, label_num

# ---------------------------
# Загрузка и оценка
# ---------------------------
if __name__ == "__main__":
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file not found: {METADATA_FILE}")
        sys.exit(1)
        
    # Считываем строки метаданных
    with open(METADATA_FILE, 'r') as f:
        lines = f.readlines()
        
    # Возможность запуска на подмножестве для быстрой проверки (например, первые 15000 файлов)
    # По умолчанию обрабатываем всё, если не передан флаг --subset
    run_subset = False
    subset_size = 15000
    if len(sys.argv) > 1 and sys.argv[1] == "--subset":
        run_subset = True
        print(f"Running on a subset of {subset_size} files...")
        lines = lines[:subset_size]
        
    cache_file = "eval_2021_cache.pkl" if not run_subset else "eval_2021_subset_cache.pkl"
    
    if os.path.exists(cache_file):
        print(f"Loading cached 2021 features from {cache_file}...")
        with open(cache_file, 'rb') as f:
            X_eval, y_eval = pickle.load(f)
    else:
        print(f"Extracting features from 2021 eval files (total lines to process: {len(lines)})...")
        X_eval, y_eval = [], []
        
        # Используем ThreadPoolExecutor для ускорения IO-bound чтения файлов и librosa
        max_workers = min(16, os.cpu_count() * 2)
        print(f"Using {max_workers} worker threads...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_file, lines)
            
            for idx, res in enumerate(results):
                if res is not None:
                    X_eval.append(res[0])
                    y_eval.append(res[1])
                if idx > 0 and idx % 2000 == 0:
                    print(f"Processed {idx}/{len(lines)} lines... Extracted {len(X_eval)} features.")
                    
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        
        with open(cache_file, 'wb') as f:
            pickle.dump((X_eval, y_eval), f)
        print("Features extracted and cached successfully.")

    print(f"Evaluation set shape: X={X_eval.shape}, y={y_eval.shape}")
    
    # Загружаем скейлер
    if not os.path.exists("scaler.pkl"):
        print("Scaler scaler.pkl not found! Please run train_augmented.py first.")
        sys.exit(1)
        
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    X_eval_scaled = scaler.transform(X_eval)

    # 1. Оценка XGBoost (augmented)
    if os.path.exists("xgb_model_augmented.json"):
        print("\n--- Evaluating XGBoost (augmented) ---")
        model = xgb.XGBClassifier()
        model.load_model("xgb_model_augmented.json")
        preds = model.predict_proba(X_eval_scaled)[:, 1]
        eer, _ = compute_eer(preds, y_eval)
        print(f"ASVspoof 2021 EER: {eer*100:.2f}%")
        
    # 2. Оценка LightGBM (augmented)
    if os.path.exists("lgb_model_augmented.pkl"):
        print("\n--- Evaluating LightGBM (augmented) ---")
        with open("lgb_model_augmented.pkl", "rb") as f:
            model = pickle.load(f)
        preds = model.predict_proba(X_eval_scaled)[:, 1]
        eer, _ = compute_eer(preds, y_eval)
        print(f"ASVspoof 2021 EER: {eer*100:.2f}%")

    # 3. Оценка CatBoost (augmented)
    if os.path.exists("cat_model_augmented.cbm"):
        print("\n--- Evaluating CatBoost (augmented) ---")
        model = CatBoostClassifier()
        model.load_model("cat_model_augmented.cbm")
        preds = model.predict_proba(X_eval_scaled)[:, 1]
        eer, _ = compute_eer(preds, y_eval)
        print(f"ASVspoof 2021 EER: {eer*100:.2f}%")
