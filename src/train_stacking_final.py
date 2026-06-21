import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import pickle
import warnings
import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer
from utils.augmentations import augment_audio
from utils.lfcc import extract_lfcc
from utils.cqcc import extract_cqcc

warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

LOCAL_DATA = os.path.abspath(os.path.join(project_root, "data", "2019", "LA"))
if os.path.exists(LOCAL_DATA):
    BASE_DATA = LOCAL_DATA
else:
    BASE_DATA = r"D:\фокусы\исследования\antispoof\data\2019\LA"

PROTOCOL_DEV = os.path.join(BASE_DATA, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")
DEV_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_dev")

SAMPLE_RATE = 16000
HOP_LENGTH = 160
WIN_LENGTH = 400
FMIN = 300
FMAX = 3400
N_BINS = 42
BINS_PER_OCTAVE = 12

# Optimal LFCC params
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

def process_file(line):
    parts = line.strip().split()
    if len(parts) < 5: return None
    file_id = parts[1]
    label = parts[4]
    label_num = 0 if label == 'bonafide' else 1
    
    audio_path = os.path.join(DEV_AUDIO_DIR, "flac", file_id + '.flac')
    if not os.path.exists(audio_path): return None
    
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE: y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        mfcc_orig = extract_robust_mfcc_features(y, sr)
        lfcc_orig = extract_robust_lfcc_features(y, sr)
        cqcc_orig = extract_robust_cqcc_features(y, sr)
        
        y_aug = augment_audio(y, sr)
        mfcc_aug = extract_robust_mfcc_features(y_aug, sr)
        lfcc_aug = extract_robust_lfcc_features(y_aug, sr)
        cqcc_aug = extract_robust_cqcc_features(y_aug, sr)
        
        return (mfcc_orig, lfcc_orig, cqcc_orig, mfcc_aug, lfcc_aug, cqcc_aug, label_num)
    except Exception:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Final Stacking Meta-Classifier")
    parser.add_argument("--num-samples", type=int, default=3000, help="Number of files")
    args = parser.parse_args()
    
    with open(PROTOCOL_DEV, "r") as f:
        lines = f.readlines()
        
    np.random.seed(42)
    np.random.shuffle(lines)
    lines_subset = lines[:args.num_samples]
    
    print(f"Processing {len(lines_subset)} files for Calibration Stacking...")
    
    X_mfcc_list, X_lfcc_list, X_cqcc_list, y_list = [], [], [], []
    max_workers = os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_file, lines_subset)
        for res in results:
            if res is not None:
                mfcc_orig, lfcc_orig, cqcc_orig, mfcc_aug, lfcc_aug, cqcc_aug, label = res
                X_mfcc_list.append(mfcc_orig); X_lfcc_list.append(lfcc_orig); X_cqcc_list.append(cqcc_orig); y_list.append(label)
                X_mfcc_list.append(mfcc_aug); X_lfcc_list.append(lfcc_aug); X_cqcc_list.append(cqcc_aug); y_list.append(label)
                
    X_mfcc = np.array(X_mfcc_list)
    X_lfcc = np.array(X_lfcc_list)
    X_cqcc = np.array(X_cqcc_list)
    X_comb = np.hstack([X_mfcc, X_lfcc, X_cqcc])
    y_calib = np.array(y_list)
    
    print("Loading scalers...")
    with open("scaler_mfcc_robust.pkl", "rb") as f: scaler_mfcc = pickle.load(f)
    with open("scaler_lfcc_robust.pkl", "rb") as f: scaler_lfcc = pickle.load(f)
    with open("scaler_cqcc_robust.pkl", "rb") as f: scaler_cqcc = pickle.load(f)
    with open("scaler_combined_robust.pkl", "rb") as f: scaler_comb = pickle.load(f)
    
    X_mfcc_scaled = scaler_mfcc.transform(X_mfcc)
    X_lfcc_scaled = scaler_lfcc.transform(X_lfcc)
    X_cqcc_scaled = scaler_cqcc.transform(X_cqcc)
    X_comb_scaled = scaler_comb.transform(X_comb)
    
    model_paths = {
        'LGBM_MFCC': 'lgb_model_mfcc_robust.pkl', 'XGB_MFCC': 'xgb_model_mfcc_robust.json', 'CAT_MFCC': 'cat_model_mfcc_robust.cbm', 'MLP_MFCC': 'mlp_model_mfcc_robust.pkl',
        'LGBM_LFCC': 'lgb_model_lfcc_robust.pkl', 'XGB_LFCC': 'xgb_model_lfcc_robust.json', 'CAT_LFCC': 'cat_model_lfcc_robust.cbm', 'MLP_LFCC': 'mlp_model_lfcc_robust.pkl',
        'LGBM_CQCC': 'lgb_model_cqcc_robust.pkl', 'XGB_CQCC': 'xgb_model_cqcc_robust.json', 'CAT_CQCC': 'cat_model_cqcc_robust.cbm', 'MLP_CQCC': 'mlp_model_cqcc_robust.pkl',
        'LGBM_COMB': 'lgb_model_combined_robust.pkl', 'XGB_COMB': 'xgb_model_combined_robust.json', 'CAT_COMB': 'cat_model_combined_robust.cbm', 'MLP_COMB': 'mlp_model_combined_robust.pkl'
    }
    
    meta_features = []
    model_names_loaded = []
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            if '_MFCC' in name: X_input = X_mfcc_scaled
            elif '_LFCC' in name: X_input = X_lfcc_scaled
            elif '_CQCC' in name: X_input = X_cqcc_scaled
            else: X_input = X_comb_scaled
            
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
        
    print(f"Stacking training complete. Saved meta model with {len(model_names_loaded)} base models.")
