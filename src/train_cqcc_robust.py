import sys
import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import pickle
import warnings
import time

warnings.filterwarnings("ignore")

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer
from utils.augmentations import augment_audio
from utils.cqcc import extract_cqcc

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print("[INFO] NVIDIA GPU detected via PyTorch. Enabling GPU training.")
            return True
    except ImportError:
        pass
    return False

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
LOCAL_DATA = os.path.abspath(os.path.join(project_root, "data", "2019", "LA"))
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
HOP_LENGTH = 160
FMIN = 300
N_BINS = 42
BINS_PER_OCTAVE = 12

def extract_robust_cqcc_features(y, sr):
    """
    Extracts CQCC features restricted to the telephone band and applies CMS.
    """
    feats = extract_cqcc(y, sr=sr, n_cqcc=20, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    
    feats_delta = librosa.feature.delta(feats)
    feats_delta2 = librosa.feature.delta(feats, order=2)
    feats_full = np.vstack([feats, feats_delta, feats_delta2])
    
    # CMS
    mean = np.mean(feats_full, axis=1, keepdims=True)
    feats_full = feats_full - mean
    
    stats_list = []
    for c in range(feats_full.shape[0]):
        coef = feats_full[c, :]
        if coef.size == 0:
            coef = np.zeros(1)
        stats = [
            np.mean(coef), np.std(coef),
            np.min(coef), np.max(coef),
            np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)
        ]
        stats_list.extend(stats)
        
    return np.array(stats_list)

def extract_from_file(file_path, augment=False):
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
            
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        # 1. Original features
        feats_orig = extract_robust_cqcc_features(y, sr)
        
        if not augment:
            return [feats_orig]
            
        # 2. Augmented features
        y_aug = augment_audio(y, sr)
        feats_aug = extract_robust_cqcc_features(y_aug, sr)
        
        return [feats_orig, feats_aug]
    except Exception as e:
        # Fallback to zeros on error
        n_feats = 20 * 3 * 7
        zeros = np.zeros(n_feats)
        if augment:
            return [zeros, zeros]
        return [zeros]

from concurrent.futures import ProcessPoolExecutor

def process_line_wrapper(args):
    line, audio_dir, augment = args
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    file_id = parts[1]
    label = parts[4]
    label_num = 0 if label == 'bonafide' else 1
    audio_path = os.path.join(audio_dir, "flac", file_id + '.flac')
    if not os.path.exists(audio_path):
        return None
        
    feats_list = extract_from_file(audio_path, augment=augment)
    return feats_list, label_num

def load_data(protocol_file, audio_dir, augment=False, max_files=None):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
        
    if max_files is not None and len(lines) > max_files:
        np.random.seed(42)
        np.random.shuffle(lines)
        lines = lines[:max_files]
        
    total_files = len(lines)
    print(f"Loading {total_files} files from {protocol_file} (augment={augment})...")
    
    X, y = [], []
    max_workers = os.cpu_count()
    print(f"Using {max_workers} CPU worker processes...")
    
    args_list = [(line, audio_dir, augment) for line in lines]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_line_wrapper, args_list)
        for idx, res in enumerate(results):
            if res is not None:
                feats_list, label_num = res
                for feats in feats_list:
                    X.append(feats)
                    y.append(label_num)
            if idx > 0 and idx % 2000 == 0:
                print(f"Processed {idx}/{total_files} files...")
                
    return np.array(X), np.array(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CQCC Classifier on ASVspoof 2019")
    parser.add_argument("--subset", action="store_true", help="Run in subset mode for fast verification")
    args = parser.parse_args()
    
    run_subset = args.subset
    use_gpu = check_gpu()
    
    if run_subset:
        cache_file = "robust_cqcc_cache_subset.pkl"
        print("--- RUNNING CQCC IN SUBSET MODE ---")
    else:
        cache_file = "robust_cqcc_cache.pkl"
        print("--- RUNNING CQCC IN FULL MODE ---")
        
    if os.path.exists(cache_file):
        print(f"Loading cached robust data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            X_train, y_train, X_dev, y_dev = pickle.load(f)
    else:
        if run_subset:
            X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR, augment=True, max_files=3000)
            X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR, augment=False, max_files=1000)
        else:
            X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR, augment=True)
            X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR, augment=False)
            
        with open(cache_file, 'wb') as f:
            pickle.dump((X_train, y_train, X_dev, y_dev), f)
        print(f"Robust CQCC data cached to {cache_file}.")

    print(f"Train shape: {X_train.shape}, Dev shape: {X_dev.shape}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    scaler_file = "scaler_cqcc_robust.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Robust scaler saved to {scaler_file}")
    
    # Train LGBM
    print("\nTraining LightGBM on robust CQCC...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_dev_scaled, y_dev)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    preds = lgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    print(f"LGBM Dev EER: {eer*100:.2f}%")
    with open("lgb_model_cqcc_robust.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
        
    # Train XGBoost
    print("\nTraining XGBoost on robust CQCC...")
    xgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
        'use_label_encoder': False
    }
    if use_gpu:
        xgb_params['device'] = 'cuda'
        xgb_params['tree_method'] = 'hist'
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_dev_scaled, y_dev)], verbose=False)
    preds = xgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    print(f"XGBoost Dev EER: {eer*100:.2f}%")
    xgb_model.save_model("xgb_model_cqcc_robust.json")
    
    # Train CatBoost
    print("\nTraining CatBoost on robust CQCC...")
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'early_stopping_rounds': 50,
        'random_seed': 42,
        'verbose': False
    }
    if use_gpu:
        cat_params['task_type'] = 'GPU'
    else:
        cat_params['thread_count'] = -1
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train_scaled, y_train, eval_set=[(X_dev_scaled, y_dev)], verbose=False)
    preds = cat_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    print(f"CatBoost Dev EER: {eer*100:.2f}%")
    cat_model.save_model("cat_model_cqcc_robust.cbm")
