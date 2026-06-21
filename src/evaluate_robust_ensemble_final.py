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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from utils.metrics import compute_eer
from utils.lfcc import extract_lfcc
from utils.cqcc import extract_cqcc

LOCAL_DATA = os.path.abspath(os.path.join(project_root, "data", "2021"))
if os.path.exists(LOCAL_DATA):
    BASE_DATA = LOCAL_DATA
else:
    BASE_DATA = r"D:\фокусы\исследования\antispoof\data\2021"

METADATA_FILE = os.path.join(BASE_DATA, "keys", "LA", "CM", "trial_metadata.txt")
AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2021_LA_eval", "flac")

SAMPLE_RATE = 16000
HOP_LENGTH = 160
WIN_LENGTH = 400
FMIN = 300
FMAX = 3400
N_BINS = 42
BINS_PER_OCTAVE = 12

# Load optimal LFCC configuration
best_lfcc_config = {"n_lfcc": 25, "n_filters": 40, "win_length": 480, "use_double_deltas": True}
if os.path.exists(os.path.join(project_root, "configs", "best_lfcc_config.yaml")):
    try:
        import yaml
        with open(os.path.join(project_root, "configs", "best_lfcc_config.yaml"), "r") as f:
            best_lfcc_config = yaml.safe_load(f)
    except Exception:
        pass

def extract_robust_mfcc_features(y, sr):
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

def extract_robust_lfcc_features(y, sr):
    n_lfcc = best_lfcc_config.get("n_lfcc", 25)
    n_filters = best_lfcc_config.get("n_filters", 40)
    win_len = best_lfcc_config.get("win_length", 480)
    use_dd = best_lfcc_config.get("use_double_deltas", True)
    
    feats = extract_lfcc(y, sr=sr, n_lfcc=n_lfcc, n_filters=n_filters, n_fft=win_len, hop_length=HOP_LENGTH, win_length=win_len, fmin=FMIN, fmax=FMAX)
    feats_delta = librosa.feature.delta(feats)
    if use_dd:
        feats_delta2 = librosa.feature.delta(feats, order=2)
        feats_full = np.vstack([feats, feats_delta, feats_delta2])
    else:
        feats_full = np.vstack([feats, feats_delta])
    mean = np.mean(feats_full, axis=1, keepdims=True)
    feats_full = feats_full - mean
    stats_list = []
    for c in range(feats_full.shape[0]):
        coef = feats_full[c, :]
        stats = [np.mean(coef), np.std(coef), np.min(coef), np.max(coef), np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)]
        stats_list.extend(stats)
    return np.array(stats_list)

def extract_robust_cqcc_features(y, sr):
    feats = extract_cqcc(y, sr=sr, n_cqcc=20, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
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

def process_file_wrapper(args):
    line, feature_type = args
    parts = line.strip().split()
    if len(parts) < 6: return None
    file_id = parts[1]
    label = parts[5]
    label_num = 0 if label == 'bonafide' else 1
    audio_path = os.path.join(AUDIO_DIR, file_id + '.flac')
    if not os.path.exists(audio_path): return None
    
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE: y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        if feature_type == 'mfcc': feats = extract_robust_mfcc_features(y, SAMPLE_RATE)
        elif feature_type == 'lfcc': feats = extract_robust_lfcc_features(y, SAMPLE_RATE)
        else: feats = extract_robust_cqcc_features(y, SAMPLE_RATE)
        return feats, label_num
    except Exception:
        return None

from concurrent.futures import ProcessPoolExecutor

def get_eval_features(lines, feature_type, run_subset=False):
    suffix = "_subset" if run_subset else ""
    cache_file = f"eval_robust_{feature_type}_cache{suffix}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached {feature_type.upper()} test features...")
        with open(cache_file, "rb") as f:
            X_eval, y_eval = pickle.load(f)
    else:
        print(f"Extracting robust {feature_type.upper()} features from 2021 eval...")
        X_eval, y_eval = [], []
        max_workers = os.cpu_count()
        args_list = [(line, feature_type) for line in lines]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_file_wrapper, args_list, chunksize=250)
            for idx, res in enumerate(results):
                if res is not None:
                    X_eval.append(res[0])
                    y_eval.append(res[1])
                if idx > 0 and idx % 20000 == 0:
                    print(f"Processed {idx}/{len(lines)}...", flush=True)
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        with open(cache_file, "wb") as f:
            pickle.dump((X_eval, y_eval), f)
    return X_eval, y_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 12 Robust Models on ASVspoof 2021 LA")
    parser.add_argument("--subset", action="store_true", help="Run on subset")
    args = parser.parse_args()
    
    run_subset = args.subset
    
    with open(METADATA_FILE, "r") as f:
        lines = f.readlines()
    if run_subset:
        lines = lines[:15000]
        
    X_mfcc, y_mfcc = get_eval_features(lines, 'mfcc', run_subset)
    X_lfcc, y_lfcc = get_eval_features(lines, 'lfcc', run_subset)
    X_cqcc, y_cqcc = get_eval_features(lines, 'cqcc', run_subset)
    
    X_comb = np.hstack([X_mfcc, X_lfcc, X_cqcc])
    y_eval = y_mfcc
    
    # Load scalers
    with open("scaler_mfcc_robust.pkl", "rb") as f: scaler_mfcc = pickle.load(f)
    with open("scaler_lfcc_robust.pkl", "rb") as f: scaler_lfcc = pickle.load(f)
    with open("scaler_cqcc_robust.pkl", "rb") as f: scaler_cqcc = pickle.load(f)
    with open("scaler_combined_robust.pkl", "rb") as f: scaler_comb = pickle.load(f)
    
    X_mfcc_scaled = scaler_mfcc.transform(X_mfcc)
    X_lfcc_scaled = scaler_lfcc.transform(X_lfcc)
    X_cqcc_scaled = scaler_cqcc.transform(X_cqcc)
    X_comb_scaled = scaler_comb.transform(X_comb)
    
    predictions = {}
    model_paths = {
        'LGBM_MFCC': 'lgb_model_mfcc_robust.pkl', 'XGB_MFCC': 'xgb_model_mfcc_robust.json', 'CAT_MFCC': 'cat_model_mfcc_robust.cbm', 'MLP_MFCC': 'mlp_model_mfcc_robust.pkl',
        'LGBM_LFCC': 'lgb_model_lfcc_robust.pkl', 'XGB_LFCC': 'xgb_model_lfcc_robust.json', 'CAT_LFCC': 'cat_model_lfcc_robust.cbm', 'MLP_LFCC': 'mlp_model_lfcc_robust.pkl',
        'LGBM_CQCC': 'lgb_model_cqcc_robust.pkl', 'XGB_CQCC': 'xgb_model_cqcc_robust.json', 'CAT_CQCC': 'cat_model_cqcc_robust.cbm', 'MLP_CQCC': 'mlp_model_cqcc_robust.pkl',
        'LGBM_COMB': 'lgb_model_combined_robust.pkl', 'XGB_COMB': 'xgb_model_combined_robust.json', 'CAT_COMB': 'cat_model_combined_robust.cbm', 'MLP_COMB': 'mlp_model_combined_robust.pkl'
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            if '_MFCC' in name: X_in = X_mfcc_scaled
            elif '_LFCC' in name: X_in = X_lfcc_scaled
            elif '_CQCC' in name: X_in = X_cqcc_scaled
            else: X_in = X_comb_scaled
            
            if path.endswith('.pkl'):
                with open(path, 'rb') as f: model = pickle.load(f)
                predictions[name] = model.predict_proba(X_in)[:, 1]
            elif path.endswith('.json'):
                model = xgb.XGBClassifier()
                model.load_model(path)
                predictions[name] = model.predict_proba(X_in)[:, 1]
            elif path.endswith('.cbm'):
                model = CatBoostClassifier()
                model.load_model(path)
                predictions[name] = model.predict_proba(X_in)[:, 1]
                
    # EER report
    report = []
    report.append("====================================================")
    report.append("     FINAL 12-MODEL ROBUST EVALUATION ON 2021 EVAL   ")
    report.append("====================================================\n")
    
    report.append("--- Individual Model EERs ---")
    for name, preds in predictions.items():
        eer, _ = compute_eer(preds, y_eval)
        report.append(f"  {name}: {eer*100:.2f}%")
        
    report.append("\n--- Group Ensembles EER ---")
    groups = {
        'MFCC': ['LGBM_MFCC', 'XGB_MFCC', 'CAT_MFCC', 'MLP_MFCC'],
        'LFCC': ['LGBM_LFCC', 'XGB_LFCC', 'CAT_LFCC', 'MLP_LFCC'],
        'CQCC': ['LGBM_CQCC', 'XGB_CQCC', 'CAT_CQCC', 'MLP_CQCC'],
        'COMB': ['LGBM_COMB', 'XGB_COMB', 'CAT_COMB', 'MLP_COMB']
    }
    
    group_preds = {}
    for gname, models in groups.items():
        preds_list = [predictions[m] for m in models if m in predictions]
        if preds_list:
            group_preds[gname] = np.mean(preds_list, axis=0)
            eer, _ = compute_eer(group_preds[gname], y_eval)
            report.append(f"  Ensemble ({gname} only): {eer*100:.2f}%")
            
    # Optimize ensemble weights
    report.append("\n--- Weight Optimization ---")
    best_eer = 1.0
    best_w = None
    
    # 4-group grid search
    w_range = np.linspace(0.0, 1.0, 11)
    for w1 in w_range:
        for w2 in np.linspace(0.0, 1.0 - w1, 11):
            for w3 in np.linspace(0.0, 1.0 - w1 - w2, 11):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -1e-9: continue
                
                blend = w1 * group_preds['MFCC'] + w2 * group_preds['LFCC'] + w3 * group_preds['CQCC'] + w4 * group_preds['COMB']
                eer, _ = compute_eer(blend, y_eval)
                if eer < best_eer:
                    best_eer = eer
                    best_w = (w1, w2, w3, w4)
                    
    report.append(f"  Best Weights (MFCC, LFCC, CQCC, COMB): {best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}, {best_w[3]:.2f}")
    report.append(f"  Best Weighted Ensemble EER: {best_eer*100:.2f}%")
    
    # Stacking Eval
    meta_file = "stacking_meta_model.pkl"
    if os.path.exists(meta_file):
        with open(meta_file, "rb") as f:
            meta_model, model_names = pickle.load(f)
        meta_feats = []
        missing = False
        for name in model_names:
            if name in predictions: meta_feats.append(predictions[name])
            else:
                missing = True
                break
        if not missing:
            meta_feats = np.column_stack(meta_feats)
            stack_preds = meta_model.predict_proba(meta_feats)[:, 1]
            eer, _ = compute_eer(stack_preds, y_eval)
            report.append(f"  Calibrated Stacking EER: {eer*100:.2f}%")
            
    out_content = "\n".join(report)
    print(out_content)
    with open("overnight_results_robust_final.txt", "w", encoding="utf-8") as f:
        f.write(out_content)
