import sys
import os

# Добавляем корень проекта через абсолютный путь
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
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

warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Конфигурация путей
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

def extract_robust_features(y, sr, feature_type='mfcc'):
    if feature_type == 'mfcc':
        feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30,
                                    n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
                                    fmin=FMIN, fmax=FMAX)
    else:
        feats = extract_lfcc(y, sr=sr, n_lfcc=20, n_filters=128,
                             n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
                             fmin=FMIN, fmax=FMAX)
        
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

def process_file(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    file_id = parts[1]
    label = parts[4]
    label_num = 0 if label == 'bonafide' else 1
    
    audio_path = os.path.join(DEV_AUDIO_DIR, "flac", file_id + '.flac')
    if not os.path.exists(audio_path):
        return None
        
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        # Извлекаем оригинальные признаки
        mfcc_orig = extract_robust_features(y, sr, 'mfcc')
        lfcc_orig = extract_robust_features(y, sr, 'lfcc')
        
        # Применяем телефонные искажения и извлекаем аугментированные признаки
        y_aug = augment_audio(y, sr)
        mfcc_aug = extract_robust_features(y_aug, sr, 'mfcc')
        lfcc_aug = extract_robust_features(y_aug, sr, 'lfcc')
        
        return (mfcc_orig, lfcc_orig, mfcc_aug, lfcc_aug, label_num)
    except Exception as e:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stacking Meta-Classifier with Calibration")
    parser.add_argument("--num-samples", type=int, default=3000, help="Number of Dev files to process for calibration")
    parser.add_argument("--exclude-lfcc", action="store_true", help="Exclude pure LFCC models from stacking")
    args = parser.parse_args()
    
    num_samples = args.num_samples
    exclude_lfcc = args.exclude_lfcc
    
    if not os.path.exists(PROTOCOL_DEV):
        print(f"[ERROR] Dev protocol file not found at {PROTOCOL_DEV}")
        sys.exit(1)
        
    print(f"Reading dev protocol {PROTOCOL_DEV}...")
    with open(PROTOCOL_DEV, "r") as f:
        lines = f.readlines()
        
    # Случайный выбор файлов
    np.random.seed(42)
    np.random.shuffle(lines)
    lines_subset = lines[:num_samples]
    
    print(f"Processing {len(lines_subset)} files for Calibration...")
    
    X_mfcc_list = []
    X_lfcc_list = []
    y_list = []
    
    max_workers = os.cpu_count()
    print(f"Running extraction with {max_workers} processes...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_file, lines_subset)
        
        for idx, res in enumerate(results):
            if res is not None:
                mfcc_orig, lfcc_orig, mfcc_aug, lfcc_aug, label = res
                
                # Добавляем чистый пример
                X_mfcc_list.append(mfcc_orig)
                X_lfcc_list.append(lfcc_orig)
                y_list.append(label)
                
                # Добавляем телефонный пример
                X_mfcc_list.append(mfcc_aug)
                X_lfcc_list.append(lfcc_aug)
                y_list.append(label)
                
            if idx > 0 and idx % 500 == 0:
                print(f"Processed {idx}/{len(lines_subset)} files...")
                
    X_mfcc = np.array(X_mfcc_list)
    X_lfcc = np.array(X_lfcc_list)
    X_comb = np.hstack([X_mfcc, X_lfcc])
    y_calib = np.array(y_list)
    
    print(f"\nExtracted calibration data shape: {X_mfcc.shape}")
    print(f"Bonafide vs Spoof count: {np.bincount(y_calib)}")
    
    # Загружаем скейлеры
    print("\nLoading scalers...")
    with open("scaler_mfcc_robust.pkl", "rb") as f:
        scaler_mfcc = pickle.load(f)
    with open("scaler_lfcc_robust.pkl", "rb") as f:
        scaler_lfcc = pickle.load(f)
    with open("scaler_combined_robust.pkl", "rb") as f:
        scaler_comb = pickle.load(f)
        
    X_mfcc_scaled = scaler_mfcc.transform(X_mfcc)
    X_lfcc_scaled = scaler_lfcc.transform(X_lfcc)
    X_comb_scaled = scaler_comb.transform(X_comb)
    
    # Названия моделей
    model_paths = {
        'LGBM_MFCC': 'lgb_model_mfcc_robust.pkl',
        'XGB_MFCC': 'xgb_model_mfcc_robust.json',
        'CAT_MFCC': 'cat_model_mfcc_robust.cbm',
        'MLP_MFCC': 'mlp_model_mfcc_robust.pkl',
        
        'LGBM_LFCC': 'lgb_model_lfcc_robust.pkl',
        'XGB_LFCC': 'xgb_model_lfcc_robust.json',
        'CAT_LFCC': 'cat_model_lfcc_robust.cbm',
        'MLP_LFCC': 'mlp_model_lfcc_robust.pkl',
        
        'LGBM_COMB': 'lgb_model_combined_robust.pkl',
        'XGB_COMB': 'xgb_model_combined_robust.json',
        'CAT_COMB': 'cat_model_combined_robust.cbm',
        'MLP_COMB': 'mlp_model_combined_robust.pkl'
    }
    
    if exclude_lfcc:
        print("[INFO] Excluding LFCC models from stacking...")
        model_paths = {k: v for k, v in model_paths.items() if '_LFCC' not in k}
        
    print("\n--- Generating Meta-Features (predictions of base models) ---")
    meta_features = []
    model_names_loaded = []
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"[WARNING] Model file {path} not found. Skipping {name}...")
            continue
            
        print(f"Generating predictions for {name}...")
        
        if '_MFCC' in name:
            X_input = X_mfcc_scaled
        elif '_LFCC' in name:
            X_input = X_lfcc_scaled
        else:
            X_input = X_comb_scaled
            
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                model = pickle.load(f)
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
    print(f"\nMeta-features shape: {meta_features.shape}")
    
    # Обучаем мета-классификатор с кросс-валидацией
    print("\nTraining Calibrated Meta-Classifier (Logistic Regression CV)...")
    meta_model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 9),
        cv=5,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    meta_model.fit(meta_features, y_calib)
    
    print(f"Optimized regularization strength C: {meta_model.C_[0]:.4f}")
    
    meta_preds = meta_model.predict_proba(meta_features)[:, 1]
    eer, _ = compute_eer(meta_preds, y_calib)
    print(f"Meta-Classifier Calibration Set EER: {eer*100:.2f}%")
    
    # Сохраняем мета-модель
    meta_model_file = "stacking_meta_model.pkl"
    with open(meta_model_file, "wb") as f:
        pickle.dump((meta_model, model_names_loaded), f)
        
    print(f"Calibrated Stacking meta-model saved to {meta_model_file}")
    
    print("\nCalibrated Meta-Model Coefficients:")
    for name, coef in zip(model_names_loaded, meta_model.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {meta_model.intercept_[0]:.4f}")
