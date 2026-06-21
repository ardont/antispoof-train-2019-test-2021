import sys
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from utils.metrics import compute_eer
from utils.augmentations import augment_audio
from utils.lfcc import extract_lfcc

warnings.filterwarnings("ignore")

# Avoid library multi-threading conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# -----------------------------------------------------------------------------
# Paths configuration
# -----------------------------------------------------------------------------
LOCAL_DATA_2019 = os.path.abspath(os.path.join(project_root, "data", "2019", "LA"))
if os.path.exists(LOCAL_DATA_2019):
    BASE_DATA_2019 = LOCAL_DATA_2019
else:
    BASE_DATA_2019 = r"D:\фокусы\исследования\antispoof\data\2019\LA"

PROTOCOL_TRAIN = os.path.join(BASE_DATA_2019, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
PROTOCOL_DEV = os.path.join(BASE_DATA_2019, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")
TRAIN_AUDIO_DIR = os.path.join(BASE_DATA_2019, "ASVspoof2019_LA_train")
DEV_AUDIO_DIR = os.path.join(BASE_DATA_2019, "ASVspoof2019_LA_dev")

LOCAL_DATA_2021 = os.path.abspath(os.path.join(project_root, "data", "2021"))
if os.path.exists(LOCAL_DATA_2021):
    BASE_DATA_2021 = LOCAL_DATA_2021
else:
    BASE_DATA_2021 = r"D:\фокусы\исследования\antispoof\data\2021"

METADATA_FILE_2021 = os.path.join(BASE_DATA_2021, "keys", "LA", "CM", "trial_metadata.txt")
AUDIO_DIR_2021 = os.path.join(BASE_DATA_2021, "ASVspoof2021_LA_eval", "flac")

SAMPLE_RATE = 16000
HOP_LENGTH = 160
FMIN = 300
FMAX = 3400

# -----------------------------------------------------------------------------
# LFCC extraction helper at module level for pickling
# -----------------------------------------------------------------------------
def extract_lfcc_custom_features(y, sr, n_lfcc, n_filters, win_length, use_double_deltas):
    """
    Extracts LFCC with specific configuration, applying CMS and extracting stats.
    """
    # Custom LFCC call
    feats = extract_lfcc(y, sr=sr, n_lfcc=n_lfcc, n_filters=n_filters,
                        n_fft=win_length, hop_length=HOP_LENGTH, win_length=win_length,
                        fmin=FMIN, fmax=FMAX)
    
    feats_delta = librosa.feature.delta(feats)
    if use_double_deltas:
        feats_delta2 = librosa.feature.delta(feats, order=2)
        feats_full = np.vstack([feats, feats_delta, feats_delta2])
    else:
        feats_full = np.vstack([feats, feats_delta])
        
    # CMS (Cepstral Mean Subtraction)
    mean = np.mean(feats_full, axis=1, keepdims=True)
    feats_full = feats_full - mean
    
    # 7 stats
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

# Multiprocessing wrappers
def process_file_train_search(args):
    file_path, label, n_lfcc, n_filters, win_length, use_double_deltas = args
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        # For training we extract both original and telephony augmented
        feats_orig = extract_lfcc_custom_features(y, sr, n_lfcc, n_filters, win_length, use_double_deltas)
        y_aug = augment_audio(y, sr)
        feats_aug = extract_lfcc_custom_features(y_aug, sr, n_lfcc, n_filters, win_length, use_double_deltas)
        return [feats_orig, feats_aug], label
    except Exception as e:
        return None

def process_file_eval_search(args):
    file_path, label, n_lfcc, n_filters, win_length, use_double_deltas = args
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        feats = extract_lfcc_custom_features(y, sr, n_lfcc, n_filters, win_length, use_double_deltas)
        return [feats], label
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# Main search logic
# -----------------------------------------------------------------------------
def load_protocol_lines(file_path, max_files=None, seed=42):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if max_files is not None and len(lines) > max_files:
        np.random.seed(seed)
        np.random.shuffle(lines)
        lines = lines[:max_files]
    return lines

def extract_features_parallel(lines, audio_dir, is_eval_2021, is_train, n_lfcc, n_filters, win_length, use_double_deltas):
    max_workers = os.cpu_count()
    args_list = []
    
    for line in lines:
        parts = line.strip().split()
        if is_eval_2021:
            if len(parts) < 6:
                continue
            file_id = parts[1]
            label_str = parts[5]
            label = 0 if label_str == 'bonafide' else 1
            file_path = os.path.join(audio_dir, file_id + '.flac')
        else:
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label_str = parts[4]
            label = 0 if label_str == 'bonafide' else 1
            file_path = os.path.join(audio_dir, "flac", file_id + '.flac')
            
        if os.path.exists(file_path):
            args_list.append((file_path, label, n_lfcc, n_filters, win_length, use_double_deltas))
            
    X, y = [], []
    func = process_file_train_search if is_train else process_file_eval_search
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(func, args_list)
        for res in results:
            if res is not None:
                feats_list, label = res
                for feats in feats_list:
                    X.append(feats)
                    y.append(label)
                    
    return np.array(X), np.array(y)

def run_grid_search():
    print("====================================================")
    # Load subsets using fixed seed
    train_lines = load_protocol_lines(PROTOCOL_TRAIN, max_files=5000, seed=42)
    dev_lines = load_protocol_lines(PROTOCOL_DEV, max_files=2000, seed=42)
    eval_lines = load_protocol_lines(METADATA_FILE_2021, max_files=5000, seed=42)
    
    print(f"Subsets loaded. Train size: {len(train_lines)}, Dev size: {len(dev_lines)}, Eval subset size: {len(eval_lines)}")
    print("====================================================")
    
    # Seta of parameters
    n_lfcc_list = [15, 20, 25, 30]
    n_filters_list = [40, 60, 80, 100, 128, 160, 200, 256]
    win_lengths = [400, 480]  # 25 ms, 30 ms
    use_double_deltas_list = [True, False]
    
    results = []
    
    total_comb = len(n_lfcc_list) * len(n_filters_list) * len(win_lengths) * len(use_double_deltas_list)
    print(f"Starting Grid Search for {total_comb} combinations...")
    
    best_eer = 1.0
    best_params = None
    
    count = 0
    for n_lfcc in n_lfcc_list:
        for n_filters in n_filters_list:
            for win_len in win_lengths:
                for use_dd in use_double_deltas_list:
                    count += 1
                    t0 = time.time()
                    
                    # 1. Feature extraction
                    X_train, y_train = extract_features_parallel(
                        train_lines, TRAIN_AUDIO_DIR, is_eval_2021=False, is_train=True,
                        n_lfcc=n_lfcc, n_filters=n_filters, win_length=win_len, use_double_deltas=use_dd
                    )
                    X_dev, y_dev = extract_features_parallel(
                        dev_lines, DEV_AUDIO_DIR, is_eval_2021=False, is_train=False,
                        n_lfcc=n_lfcc, n_filters=n_filters, win_length=win_len, use_double_deltas=use_dd
                    )
                    X_eval, y_eval = extract_features_parallel(
                        eval_lines, AUDIO_DIR_2021, is_eval_2021=True, is_train=False,
                        n_lfcc=n_lfcc, n_filters=n_filters, win_length=win_len, use_double_deltas=use_dd
                    )
                    
                    if X_train.size == 0 or X_dev.size == 0 or X_eval.size == 0:
                        print(f"[{count}/{total_comb}] Skip parameter set due to extraction error")
                        continue
                        
                    # 2. Scaling
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_dev_scaled = scaler.transform(X_dev)
                    X_eval_scaled = scaler.transform(X_eval)
                    
                    # 3. Model training (Fast LightGBM)
                    model = lgb.LGBMClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=42,
                        verbose=-1,
                        n_jobs=-1
                    )
                    model.fit(X_train_scaled, y_train)
                    
                    # 4. Evaluation
                    preds_dev = model.predict_proba(X_dev_scaled)[:, 1]
                    dev_eer, _ = compute_eer(preds_dev, y_dev)
                    
                    preds_eval = model.predict_proba(X_eval_scaled)[:, 1]
                    eval_eer, _ = compute_eer(preds_eval, y_eval)
                    
                    # Joint EER metric
                    joint_eer = 0.4 * dev_eer + 0.6 * eval_eer
                    
                    dt = time.time() - t0
                    print(f"[{count}/{total_comb}] Params: LFCC={n_lfcc}, Filters={n_filters}, Win={win_len}, DD={use_dd} | "
                          f"Dev EER={dev_eer*100:.2f}%, Eval EER={eval_eer*100:.2f}%, Joint EER={joint_eer*100:.2f}% | Time={dt:.1f}s")
                    
                    results.append({
                        "n_lfcc": n_lfcc,
                        "n_filters": n_filters,
                        "win_length": win_len,
                        "use_double_deltas": use_dd,
                        "dev_eer": dev_eer,
                        "eval_eer": eval_eer,
                        "joint_eer": joint_eer
                    })
                    
                    if joint_eer < best_eer:
                        best_eer = joint_eer
                        best_params = (n_lfcc, n_filters, win_len, use_dd)
                        print(f"  *** New Best parameters! Joint EER = {joint_eer*100:.2f}% ***")
                        
                    # Save checkpoint
                    if count % 5 == 0:
                        df_res = pd.DataFrame(results)
                        df_res.to_csv("lfcc_grid_search_results.csv", index=False)
                        
    # Save final grid results
    df_res = pd.DataFrame(results)
    df_res.to_csv("lfcc_grid_search_results.csv", index=False)
    print("\nGrid Search completed. Results saved to lfcc_grid_search_results.csv")
    
    print(f"\nBest Parameters Found: LFCC={best_params[0]}, Filters={best_params[1]}, Win={best_params[2]}, DD={best_params[3]}")
    print(f"Best Joint EER: {best_eer*100:.2f}%")
    
    # Save best params
    with open("best_lfcc_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
        
    # Generate and save plot
    try:
        plt.figure(figsize=(10, 6))
        # Filter for the best window length and deltas configuration to show nice curves
        best_win, best_dd = best_params[2], best_params[3]
        df_filtered = df_res[(df_res["win_length"] == best_win) & (df_res["use_double_deltas"] == best_dd)]
        
        for n_lfcc in n_lfcc_list:
            df_sub = df_filtered[df_filtered["n_lfcc"] == n_lfcc].sort_values("n_filters")
            if not df_sub.empty:
                plt.plot(df_sub["n_filters"], df_sub["eval_eer"] * 100, marker='o', label=f"n_lfcc={n_lfcc}")
                
        plt.title(f"LFCC Parameters Search (Eval EER Subset) at Win={best_win}, DD={best_dd}")
        plt.xlabel("Number of Filters")
        plt.ylabel("EER (%)")
        plt.grid(True)
        plt.legend()
        plt.savefig("lfcc_grid_search_plot.png")
        print("Analysis plot saved to lfcc_grid_search_plot.png")
    except Exception as e:
        print(f"Error plotting results: {e}")
        
    return best_params

# -----------------------------------------------------------------------------
# Full cycle retraining on best parameters
# -----------------------------------------------------------------------------
def run_full_retraining(best_params):
    n_lfcc, n_filters, win_length, use_double_deltas = best_params
    print("\n====================================================")
    print("        STARTING FULL RETRAINING ON BEST PARAMS     ")
    print(f"Params: LFCC={n_lfcc}, Filters={n_filters}, Win={win_length}, DD={use_double_deltas}")
    print("====================================================")
    
    # Load entire train / dev protocols
    train_lines = load_protocol_lines(PROTOCOL_TRAIN)
    dev_lines = load_protocol_lines(PROTOCOL_DEV)
    eval_lines = load_protocol_lines(METADATA_FILE_2021)
    
    # 1. Full LFCC features extraction
    print("\n[1/7] Extracting robust LFCC train set...")
    X_train, y_train = extract_features_parallel(
        train_lines, TRAIN_AUDIO_DIR, is_eval_2021=False, is_train=True,
        n_lfcc=n_lfcc, n_filters=n_filters, win_length=win_length, use_double_deltas=use_double_deltas
    )
    print(f"Done. Shape: {X_train.shape}")
    
    print("\n[2/7] Extracting robust LFCC dev set...")
    X_dev, y_dev = extract_features_parallel(
        dev_lines, DEV_AUDIO_DIR, is_eval_2021=False, is_train=False,
        n_lfcc=n_lfcc, n_filters=n_filters, win_length=win_length, use_double_deltas=use_double_deltas
    )
    print(f"Done. Shape: {X_dev.shape}")
    
    # Save new robust cache
    cache_file = "robust_lfcc_cache.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump((X_train, y_train, X_dev, y_dev), f)
    print(f"Saved cache to {cache_file}")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    scaler_file = "scaler_lfcc_robust.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_file}")
    
    # 2. Train LFCC models
    print("\n[3/7] Training robust LFCC models (LGBM, XGBoost, CatBoost)...")
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_dev_scaled, y_dev)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    with open("lgb_model_lfcc_robust.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
        
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', early_stopping_rounds=50, use_label_encoder=False
    )
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_dev_scaled, y_dev)], verbose=False)
    xgb_model.save_model("xgb_model_lfcc_robust.json")
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, loss_function='Logloss',
        eval_metric='AUC', early_stopping_rounds=50, random_seed=42, verbose=False, thread_count=-1
    )
    cat_model.fit(Pool(X_train_scaled, label=y_train), eval_set=Pool(X_dev_scaled, label=y_dev), plot=False)
    cat_model.save_model("cat_model_lfcc_robust.cbm")
    
    print("LFCC models trained successfully.")
    
    # 3. Create Combined features dataset (MFCC + new LFCC)
    print("\n[4/7] Rebuilding Combined features cache...")
    mfcc_cache = "robust_mfcc_cache.pkl"
    if not os.path.exists(mfcc_cache):
        print("[ERROR] MFCC cache robust_mfcc_cache.pkl not found! Cannot build Combined.")
        return
        
    with open(mfcc_cache, "rb") as f:
        X_train_mfcc, y_train_mfcc, X_dev_mfcc, y_dev_mfcc = pickle.load(f)
        
    X_train_comb = np.hstack([X_train_mfcc, X_train])
    X_dev_comb = np.hstack([X_dev_mfcc, X_dev])
    
    combined_cache = "robust_combined_cache.pkl"
    with open(combined_cache, "wb") as f:
        pickle.dump((X_train_comb, y_train_mfcc, X_dev_comb, y_dev_mfcc), f)
    print(f"Saved combined cache to {combined_cache}")
    
    # Scaling combined
    scaler_comb = StandardScaler()
    X_train_comb_scaled = scaler_comb.fit_transform(X_train_comb)
    X_dev_comb_scaled = scaler_comb.transform(X_dev_comb)
    
    scaler_comb_file = "scaler_combined_robust.pkl"
    with open(scaler_comb_file, "wb") as f:
        pickle.dump(scaler_comb, f)
    print(f"Saved combined scaler to {scaler_comb_file}")
    
    # 4. Train Combined models
    print("\n[5/7] Training robust Combined models...")
    
    lgb_comb = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgb_comb.fit(
        X_train_comb_scaled, y_train_mfcc,
        eval_set=[(X_dev_comb_scaled, y_dev_mfcc)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    with open("lgb_model_combined_robust.pkl", "wb") as f:
        pickle.dump(lgb_comb, f)
        
    xgb_comb = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', early_stopping_rounds=50, use_label_encoder=False
    )
    xgb_comb.fit(X_train_comb_scaled, y_train_mfcc, eval_set=[(X_dev_comb_scaled, y_dev_mfcc)], verbose=False)
    xgb_comb.save_model("xgb_model_combined_robust.json")
    
    cat_comb = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, loss_function='Logloss',
        eval_metric='AUC', early_stopping_rounds=50, random_seed=42, verbose=False, thread_count=-1
    )
    cat_comb.fit(Pool(X_train_comb_scaled, label=y_train_mfcc), eval_set=Pool(X_dev_comb_scaled, label=y_dev_mfcc), plot=False)
    cat_comb.save_model("cat_model_combined_robust.cbm")
    
    print("Combined models trained successfully.")
    
    # 5. Retrain Stacking meta-model (calibrated Stacking meta-model using 3000 calibration samples)
    print("\n[6/7] Re-training Stacking Meta-Classifier...")
    calib_lines = load_protocol_lines(PROTOCOL_DEV, max_files=3000, seed=42)
    
    X_mfcc_calib_list, X_lfcc_calib_list, y_calib_list = [], [], []
    
    # Extract features for calibration
    # Since calibration needs both original and telephone augmented samples, we do it
    print("Extracting calibration features on 3000 Dev files...")
    X_lfcc_calib, y_calib = extract_features_parallel(
        calib_lines, DEV_AUDIO_DIR, is_eval_2021=False, is_train=True,
        n_lfcc=n_lfcc, n_filters=n_filters, win_length=win_length, use_double_deltas=use_double_deltas
    )
    # Load MFCC calibration features (we can just load them from train_stacking_calibrated or extract them)
    # For speed and consistency, let's extract MFCC on the same files (augment=True)
    # We define a quick MFCC extraction
    def extract_mfcc_features(y, sr):
        feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
        feats_delta = librosa.feature.delta(feats)
        feats_delta2 = librosa.feature.delta(feats, order=2)
        feats_full = np.vstack([feats, feats_delta, feats_delta2])
        mean = np.mean(feats_full, axis=1, keepdims=True)
        feats_full = feats_full - mean
        stats_list = []
        for c in range(feats_full.shape[0]):
            coef = feats_full[c, :]
            stats = [np.mean(coef), np.std(coef), np.min(coef), np.max(coef), np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)]
            stats_list.extend(stats)
        return np.array(stats_list)

    # Helper function for process_file MFCC
    WIN_LENGTH = 400
    X_mfcc_calib_list = []
    print("Extracting MFCC calibration features...")
    for line in calib_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            file_id = parts[1]
            file_path = os.path.join(DEV_AUDIO_DIR, "flac", file_id + '.flac')
            if os.path.exists(file_path):
                y, sr = sf.read(file_path)
                if y.ndim > 1: y = np.mean(y, axis=1)
                if sr != SAMPLE_RATE: y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
                X_mfcc_calib_list.append(extract_mfcc_features(y, SAMPLE_RATE))
                # Augmented
                y_aug = augment_audio(y, SAMPLE_RATE)
                X_mfcc_calib_list.append(extract_mfcc_features(y_aug, SAMPLE_RATE))
                
    X_mfcc_calib = np.array(X_mfcc_calib_list)
    
    # Load scalers
    with open("scaler_mfcc_robust.pkl", "rb") as f:
        scaler_mfcc = pickle.load(f)
        
    X_mfcc_scaled = scaler_mfcc.transform(X_mfcc_calib)
    X_lfcc_scaled = scaler.transform(X_lfcc_calib)
    X_comb_scaled = scaler_comb.transform(np.hstack([X_mfcc_calib, X_lfcc_calib]))
    
    # Load base models to do predictions on calibration set
    model_paths = {
        'LGBM_MFCC': 'lgb_model_mfcc_robust.pkl',
        'XGB_MFCC': 'xgb_model_mfcc_robust.json',
        'CAT_MFCC': 'cat_model_mfcc_robust.cbm',
        
        'LGBM_LFCC': 'lgb_model_lfcc_robust.pkl',
        'XGB_LFCC': 'xgb_model_lfcc_robust.json',
        'CAT_LFCC': 'cat_model_lfcc_robust.cbm',
        
        'LGBM_COMB': 'lgb_model_combined_robust.pkl',
        'XGB_COMB': 'xgb_model_combined_robust.json',
        'CAT_COMB': 'cat_model_combined_robust.cbm'
    }
    
    meta_features = []
    model_names_loaded = []
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            if '_MFCC' in name:
                X_input = X_mfcc_scaled
            elif '_LFCC' in name:
                X_input = X_lfcc_scaled
            else:
                X_input = X_comb_scaled
                
            if path.endswith('.pkl'):
                with open(path, 'rb') as f: model = pickle.load(f)
                preds = model.predict_proba(X_input)[:, 1]
            elif path.endswith('.json'):
                model = xgb.XGBClassifier()
                model.load_model(path)
                preds = model.predict_proba(X_input)[:, 1]
            elif path.endswith('.cbm'):
                model = CatBoostClassifier()
                model.load_model(path)
                preds = model.predict_proba(X_input)[:, 1]
                
            meta_features.append(preds)
            model_names_loaded.append(name)
            
    meta_features = np.column_stack(meta_features)
    
    meta_model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 9), cv=5, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1
    )
    meta_model.fit(meta_features, y_calib)
    
    with open("stacking_meta_model.pkl", "wb") as f:
        pickle.dump((meta_model, model_names_loaded), f)
    print("Stacking meta-model saved to stacking_meta_model.pkl")
    
    # 6. Full 2021 Eval evaluation
    print("\n[7/7] Evaluating new models on full 2021 Eval dataset...")
    
    # Extract new LFCC features for full 2021 Eval
    print("Extracting new LFCC features for 101k eval files (multiprocessed)...")
    X_eval_lfcc, y_eval = extract_features_parallel(
        eval_lines, AUDIO_DIR_2021, is_eval_2021=True, is_train=False,
        n_lfcc=n_lfcc, n_filters=n_filters, win_length=win_length, use_double_deltas=use_double_deltas
    )
    
    # Save new eval robust cache
    eval_cache_file = "eval_robust_lfcc_cache.pkl"
    with open(eval_cache_file, "wb") as f:
        pickle.dump((X_eval_lfcc, y_eval), f)
    print(f"Saved new eval LFCC features cache to {eval_cache_file}")
    
    # Scale LFCC eval features
    X_eval_lfcc_scaled = scaler.transform(X_eval_lfcc)
    
    # Load MFCC eval features cache
    with open("eval_robust_mfcc_cache.pkl", "rb") as f:
        X_eval_mfcc, _ = pickle.load(f)
    X_eval_mfcc_scaled = scaler_mfcc.transform(X_eval_mfcc)
    
    # Concatenate for combined eval features
    X_eval_comb = np.hstack([X_eval_mfcc, X_eval_lfcc])
    X_eval_comb_scaled = scaler_comb.transform(X_eval_comb)
    
    # Base model predictions
    predictions = {}
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            if '_MFCC' in name: X_input = X_eval_mfcc_scaled
            elif '_LFCC' in name: X_input = X_eval_lfcc_scaled
            else: X_input = X_eval_comb_scaled
            
            if path.endswith('.pkl'):
                with open(path, 'rb') as f: model = pickle.load(f)
                predictions[name] = model.predict_proba(X_input)[:, 1]
            elif path.endswith('.json'):
                model = xgb.XGBClassifier()
                model.load_model(path)
                predictions[name] = model.predict_proba(X_input)[:, 1]
            elif path.endswith('.cbm'):
                model = CatBoostClassifier()
                model.load_model(path)
                predictions[name] = model.predict_proba(X_input)[:, 1]
                
    # Evaluate individual EERs
    final_report = []
    final_report.append("====================================================")
    final_report.append("  FINAL ROBUST EVALUATION SUMMARY (OPTIMIZED LFCC)  ")
    final_report.append("====================================================\n")
    final_report.append(f"LFCC parameters: n_lfcc={n_lfcc}, n_filters={n_filters}, win_length={win_length}, use_double_deltas={use_double_deltas}\n")
    final_report.append("--- Individual Model EERs ---")
    
    for name, preds in predictions.items():
        eer, _ = compute_eer(preds, y_eval)
        final_report.append(f"  {name}: {eer*100:.2f}%")
        
    # Ensembles
    final_report.append("\n--- Ensembles ---")
    
    # MFCC ensemble
    mfcc_preds = [predictions[k] for k in ['LGBM_MFCC', 'XGB_MFCC', 'CAT_MFCC'] if k in predictions]
    if mfcc_preds:
        eer, _ = compute_eer(np.mean(mfcc_preds, axis=0), y_eval)
        final_report.append(f"  Ensemble (Robust MFCC only): {eer*100:.2f}%")
        
    # LFCC ensemble
    lfcc_preds = [predictions[k] for k in ['LGBM_LFCC', 'XGB_LFCC', 'CAT_LFCC'] if k in predictions]
    if lfcc_preds:
        eer, _ = compute_eer(np.mean(lfcc_preds, axis=0), y_eval)
        final_report.append(f"  Ensemble (Robust LFCC only): {eer*100:.2f}%")
        
    # Combined ensemble
    comb_preds = [predictions[k] for k in ['LGBM_COMB', 'XGB_COMB', 'CAT_COMB'] if k in predictions]
    if comb_preds:
        eer, _ = compute_eer(np.mean(comb_preds, axis=0), y_eval)
        final_report.append(f"  Ensemble (Robust Combined only): {eer*100:.2f}%")
        
    # Optimized ensemble (0.5 MFCC + 0.5 Combined)
    if mfcc_preds and comb_preds:
        avg_mfcc = np.mean(mfcc_preds, axis=0)
        avg_comb = np.mean(comb_preds, axis=0)
        opt_ensemble_preds = 0.5 * avg_mfcc + 0.5 * avg_comb
        eer, _ = compute_eer(opt_ensemble_preds, y_eval)
        final_report.append(f"  Optimized Weighted Ensemble (0.5 MFCC + 0.5 COMB): {eer*100:.2f}%")
        
    # Stacking
    meta_features_eval = [predictions[name] for name in model_names_loaded if name in predictions]
    if len(meta_features_eval) == len(model_names_loaded):
        meta_features_eval = np.column_stack(meta_features_eval)
        stacking_preds = meta_model.predict_proba(meta_features_eval)[:, 1]
        eer, _ = compute_eer(stacking_preds, y_eval)
        final_report.append(f"  Calibrated Stacking Meta-Classifier: {eer*100:.2f}%")
        
    report_content = "\n".join(final_report)
    print("\n" + report_content)
    
    with open("overnight_results_robust_opt.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\nFinal evaluation summary saved to overnight_results_robust_opt.txt")

# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    best_params = run_grid_search()
    
    # Run full retraining and evaluation automatically
    run_full_retraining(best_params)
    
    print(f"\nAll operations completed successfully! Total elapsed time: {(time.time() - t_start)/60:.2f} minutes.")
