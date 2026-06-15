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
from utils.lfcc import extract_lfcc
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
N_LFCC = 20
N_FILTERS = 128

# ---------------------------
# Извлечение признаков
# ---------------------------
def extract_features_from_waveform(y, sr):
    lfcc = extract_lfcc(y, sr=sr, n_lfcc=N_LFCC, n_filters=N_FILTERS)
    lfcc_delta = librosa.feature.delta(lfcc)
    lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
    
    lfcc_full = np.vstack([lfcc, lfcc_delta, lfcc_delta2])
    
    # CMS (Cepstral Mean Subtraction) для устойчивости к каналу
    mean = np.mean(lfcc_full, axis=1, keepdims=True)
    lfcc_full = lfcc_full - mean
    
    stats_list = []
    for c in range(lfcc_full.shape[0]):
        coef = lfcc_full[c, :]
        if coef.size == 0:
            coef = np.zeros(1)
        stats = [
            np.mean(coef), np.std(coef),
            np.min(coef), np.max(coef),
            np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)
        ]
        stats_list.extend(stats)
    return np.array(stats_list)

def process_file(line):
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    file_id = parts[1]
    label = parts[5]
    label_num = 0 if label == 'bonafide' else 1
    
    audio_path = os.path.join(AUDIO_DIR, file_id + '.flac')
    if not os.path.exists(audio_path):
        return None
        
    try:
        waveform_np, sr = sf.read(audio_path)
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)
            
        if sr != SAMPLE_RATE:
            waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        feats = extract_features_from_waveform(waveform_np, sr)
        return feats, label_num
    except Exception as e:
        return None

# ---------------------------
# Загрузка и оценка
# ---------------------------
if __name__ == "__main__":
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file not found: {METADATA_FILE}")
        sys.exit(1)
        
    with open(METADATA_FILE, 'r') as f:
        lines = f.readlines()
        
    run_subset = False
    subset_size = 15000
    if len(sys.argv) > 1 and sys.argv[1] == "--subset":
        run_subset = True
        print(f"Running LFCC evaluation on a subset of {subset_size} files...")
        lines = lines[:subset_size]
        
    cache_file = "eval_lfcc_2021_cache.pkl" if not run_subset else "eval_lfcc_2021_subset_cache.pkl"
    
    if os.path.exists(cache_file):
        print(f"Loading cached LFCC 2021 features from {cache_file}...")
        with open(cache_file, 'rb') as f:
            X_eval, y_eval = pickle.load(f)
    else:
        print(f"Extracting LFCC features from 2021 eval files (total lines: {len(lines)})...")
        X_eval, y_eval = [], []
        
        # Используем ProcessPoolExecutor для параллельной обработки признаков на всех ядрах CPU
        from concurrent.futures import ProcessPoolExecutor
        max_workers = os.cpu_count()
        print(f"Using {max_workers} worker processes for multi-processing LFCC feature extraction...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_file, lines)
            
            for idx, res in enumerate(results):
                if res is not None:
                    X_eval.append(res[0])
                    y_eval.append(res[1])
                if idx > 0 and idx % 5000 == 0:
                    print(f"Processed {idx}/{len(lines)} lines... Extracted {len(X_eval)} features.")
                    
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        
        with open(cache_file, 'wb') as f:
            pickle.dump((X_eval, y_eval), f)
        print("LFCC features extracted and cached successfully.")

    print(f"Evaluation set shape (LFCC): X={X_eval.shape}, y={y_eval.shape}")
    
    # Загружаем скейлер
    if not os.path.exists("scaler_lfcc.pkl"):
        print("LFCC Scaler scaler_lfcc.pkl not found! Please run train_lfcc_augmented.py first.")
        sys.exit(1)
        
    with open("scaler_lfcc.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    X_eval_scaled = scaler.transform(X_eval)

    # 1. Оценка XGBoost (LFCC)
    if os.path.exists("xgb_model_lfcc_augmented.json"):
        print("\n--- Evaluating XGBoost (augmented LFCC) ---")
        model = xgb.XGBClassifier()
        model.load_model("xgb_model_lfcc_augmented.json")
        preds = model.predict_proba(X_eval_scaled)[:, 1]
        eer, _ = compute_eer(preds, y_eval)
        print(f"ASVspoof 2021 EER (LFCC): {eer*100:.2f}%")
        
    # 2. Оценка LightGBM (LFCC)
    if os.path.exists("lgb_model_lfcc_augmented.pkl"):
        print("\n--- Evaluating LightGBM (augmented LFCC) ---")
        with open("lgb_model_lfcc_augmented.pkl", "rb") as f:
            model = pickle.load(f)
        preds = model.predict_proba(X_eval_scaled)[:, 1]
        eer, _ = compute_eer(preds, y_eval) # fix potential name typo
        print(f"ASVspoof 2021 EER (LFCC): {eer*100:.2f}%")

    # 3. Оценка CatBoost (LFCC)
    if os.path.exists("cat_model_lfcc_augmented.cbm"):
        print("\n--- Evaluating CatBoost (augmented LFCC) ---")
        model = CatBoostClassifier()
        model.load_model("cat_model_lfcc_augmented.cbm")
        preds = model.predict_proba(X_eval_scaled)[:, 1]
        eer, _ = compute_eer(preds, y_eval)
        print(f"ASVspoof 2021 EER (LFCC): {eer*100:.2f}%")
