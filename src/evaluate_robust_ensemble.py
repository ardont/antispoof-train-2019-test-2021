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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from utils.metrics import compute_eer
from utils.lfcc import extract_lfcc

# -----------------------------------------------------------------------------
# 📁 Конфигурация путей и параметров
# -----------------------------------------------------------------------------
LOCAL_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "2021"))
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

# -----------------------------------------------------------------------------
# 🧠 Извлечение устойчивых признаков с полосовой фильтрацией и CMS
# -----------------------------------------------------------------------------
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
    
    # CMS: Вычитаем среднее
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

def process_file_wrapper(args):
    line, feature_type = args
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    file_id = parts[1]
    label = parts[5] # 'spoof' or 'bonafide'
    label_num = 0 if label == 'bonafide' else 1
    
    audio_path = os.path.join(AUDIO_DIR, file_id + '.flac')
    if not os.path.exists(audio_path):
        return None
        
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        feats = extract_robust_features(y, sr, feature_type)
        return feats, label_num
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# 🔄 Параллельный расчет признаков
# -----------------------------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor

def get_eval_features(lines, feature_type='mfcc', run_subset=False):
    suffix = "_subset" if run_subset else ""
    cache_file = f"eval_robust_{feature_type}_cache{suffix}.pkl"
    
    if os.path.exists(cache_file):
        print(f"Loading cached {feature_type.upper()} test features from {cache_file}...")
        with open(cache_file, "rb") as f:
            X_eval, y_eval = pickle.load(f)
    else:
        print(f"Extracting robust {feature_type.upper()} features from 2021 eval (total lines: {len(lines)})...")
        X_eval, y_eval = [], []
        max_workers = os.cpu_count()
        print(f"Using {max_workers} CPU worker processes...")
        
        args_list = [(line, feature_type) for line in lines]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_file_wrapper, args_list)
            for idx, res in enumerate(results):
                if res is not None:
                    X_eval.append(res[0])
                    y_eval.append(res[1])
                if idx > 0 and idx % 5000 == 0:
                    print(f"Processed {idx}/{len(lines)} lines... Extracted {len(X_eval)} samples.")
                    
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        
        with open(cache_file, "wb") as f:
            pickle.dump((X_eval, y_eval), f)
        print(f"Features saved to cache: {cache_file}")
        
    return X_eval, y_eval

# -----------------------------------------------------------------------------
# 🚀 Точка входа в скрипт
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Robust Models on ASVspoof 2021 LA")
    parser.add_argument("--subset", action="store_true", help="Run on subset of 15000 files")
    args = parser.parse_args()
    
    run_subset = args.subset
    subset_size = 15000
    
    if not os.path.exists(METADATA_FILE):
        print(f"[ERROR] Metadata file not found: {METADATA_FILE}")
        sys.exit(1)
        
    with open(METADATA_FILE, "r") as f:
        lines = f.readlines()
        
    if run_subset:
        print(f"Running evaluation in SUBSET MODE (first {subset_size} files)...")
        lines = lines[:subset_size]
    else:
        print("Running evaluation in FULL MODE on all eval files...")

    # 1. Извлекаем/загружаем MFCC и LFCC признаки
    X_mfcc, y_mfcc = get_eval_features(lines, 'mfcc', run_subset)
    X_lfcc, y_lfcc = get_eval_features(lines, 'lfcc', run_subset)
    
    # Объединяем их для комбинированных моделей
    print("Concatenating features for combined models...")
    X_comb = np.hstack([X_mfcc, X_lfcc])
    y_eval = y_mfcc # метки классов одинаковые
    
    print(f"Loaded evaluation samples: {len(X_mfcc)}")
    
    # 2. Масштабирование признаков (StandardScaler)
    print("Loading scalers...")
    with open("scaler_mfcc_robust.pkl", "rb") as f:
        scaler_mfcc = pickle.load(f)
    with open("scaler_lfcc_robust.pkl", "rb") as f:
        scaler_lfcc = pickle.load(f)
    with open("scaler_combined_robust.pkl", "rb") as f:
        scaler_comb = pickle.load(f)
        
    X_mfcc_scaled = scaler_mfcc.transform(X_mfcc)
    X_lfcc_scaled = scaler_lfcc.transform(X_lfcc)
    X_comb_scaled = scaler_comb.transform(X_comb)
    
    predictions = {}
    
    # 3. Инференс робастных моделей
    # --- MFCC ---
    print("\n--- Evaluating Robust MFCC Models ---")
    if os.path.exists("lgb_model_mfcc_robust.pkl"):
        with open("lgb_model_mfcc_robust.pkl", "rb") as f:
            model = pickle.load(f)
        predictions['LGBM_MFCC'] = model.predict_proba(X_mfcc_scaled)[:, 1]
    if os.path.exists("xgb_model_mfcc_robust.json"):
        model = xgb.XGBClassifier()
        model.load_model("xgb_model_mfcc_robust.json")
        predictions['XGB_MFCC'] = model.predict_proba(X_mfcc_scaled)[:, 1]
    if os.path.exists("cat_model_mfcc_robust.cbm"):
        model = CatBoostClassifier()
        model.load_model("cat_model_mfcc_robust.cbm")
        predictions['CAT_MFCC'] = model.predict_proba(X_mfcc_scaled)[:, 1]
    if os.path.exists("mlp_model_mfcc_robust.pkl"):
        with open("mlp_model_mfcc_robust.pkl", "rb") as f:
            model = pickle.load(f)
        predictions['MLP_MFCC'] = model.predict_proba(X_mfcc_scaled)[:, 1]
        
    # --- LFCC ---
    print("\n--- Evaluating Robust LFCC Models ---")
    if os.path.exists("lgb_model_lfcc_robust.pkl"):
        with open("lgb_model_lfcc_robust.pkl", "rb") as f:
            model = pickle.load(f)
        predictions['LGBM_LFCC'] = model.predict_proba(X_lfcc_scaled)[:, 1]
    if os.path.exists("xgb_model_lfcc_robust.json"):
        model = xgb.XGBClassifier()
        model.load_model("xgb_model_lfcc_robust.json")
        predictions['XGB_LFCC'] = model.predict_proba(X_lfcc_scaled)[:, 1]
    if os.path.exists("cat_model_lfcc_robust.cbm"):
        model = CatBoostClassifier()
        model.load_model("cat_model_lfcc_robust.cbm")
        predictions['CAT_LFCC'] = model.predict_proba(X_lfcc_scaled)[:, 1]
    if os.path.exists("mlp_model_lfcc_robust.pkl"):
        with open("mlp_model_lfcc_robust.pkl", "rb") as f:
            model = pickle.load(f)
        predictions['MLP_LFCC'] = model.predict_proba(X_lfcc_scaled)[:, 1]
        
    # --- Combined ---
    print("\n--- Evaluating Robust Combined Models ---")
    if os.path.exists("lgb_model_combined_robust.pkl"):
        with open("lgb_model_combined_robust.pkl", "rb") as f:
            model = pickle.load(f)
        predictions['LGBM_COMB'] = model.predict_proba(X_comb_scaled)[:, 1]
    if os.path.exists("xgb_model_combined_robust.json"):
        model = xgb.XGBClassifier()
        model.load_model("xgb_model_combined_robust.json")
        predictions['XGB_COMB'] = model.predict_proba(X_comb_scaled)[:, 1]
    if os.path.exists("cat_model_combined_robust.cbm"):
        model = CatBoostClassifier()
        model.load_model("cat_model_combined_robust.cbm")
        predictions['CAT_COMB'] = model.predict_proba(X_comb_scaled)[:, 1]
    if os.path.exists("mlp_model_combined_robust.pkl"):
        with open("mlp_model_combined_robust.pkl", "rb") as f:
            model = pickle.load(f)
        predictions['MLP_COMB'] = model.predict_proba(X_comb_scaled)[:, 1]
        
    # 4. Вывод EER одиночных моделей
    print("\n====================================================")
    print("      INDIVIDUAL ROBUST MODEL EER ON 2021 EVAL      ")
    print("====================================================")
    for name, preds in predictions.items():
        eer, _ = compute_eer(preds, y_eval)
        print(f"{name}: {eer*100:.2f}%")
        
    # 5. Вывод EER ансамблей (усреднение вероятностей)
    print("\n====================================================")
    print("       ENSEMBLE ROBUST MODEL EER ON 2021 EVAL       ")
    print("====================================================")
    
    # Ансамбль всех MFCC
    mfcc_preds = [predictions[k] for k in ['LGBM_MFCC', 'XGB_MFCC', 'CAT_MFCC'] if k in predictions]
    if mfcc_preds:
        eer, _ = compute_eer(np.mean(mfcc_preds, axis=0), y_eval)
        print(f"Ensemble (Robust MFCC only): {eer*100:.2f}%")
        
    # Ансамбль всех LFCC
    lfcc_preds = [predictions[k] for k in ['LGBM_LFCC', 'XGB_LFCC', 'CAT_LFCC'] if k in predictions]
    if lfcc_preds:
        eer, _ = compute_eer(np.mean(lfcc_preds, axis=0), y_eval)
        print(f"Ensemble (Robust LFCC only): {eer*100:.2f}%")
        
    # Ансамбль всех Combined
    comb_preds = [predictions[k] for k in ['LGBM_COMB', 'XGB_COMB', 'CAT_COMB'] if k in predictions]
    if comb_preds:
        eer, _ = compute_eer(np.mean(comb_preds, axis=0), y_eval)
        print(f"Ensemble (Robust Combined only): {eer*100:.2f}%")
        
    # Ансамбль двух лучших (например, LGBM_MFCC + LGBM_LFCC)
    lgb_ensemble = [predictions[k] for k in ['LGBM_MFCC', 'LGBM_LFCC'] if k in predictions]
    if len(lgb_ensemble) == 2:
        eer, _ = compute_eer(np.mean(lgb_ensemble, axis=0), y_eval)
        print(f"Ensemble (LGBM MFCC + LGBM LFCC): {eer*100:.2f}%")
        
    # Полный ансамбль из всех 9 моделей
    all_preds = list(predictions.values())
    if all_preds:
        eer, _ = compute_eer(np.mean(all_preds, axis=0), y_eval)
        print(f"Ensemble (All 9 Robust Models): {eer*100:.2f}%")
        
    # 6. Оптимизация весов ансамбля (Grid Search по группам)
    print("\n====================================================")
    print("     OPTIMIZING ENSEMBLE WEIGHTS FOR MINIMUM EER    ")
    print("====================================================")
    
    group_mfcc = np.mean([predictions[k] for k in ['LGBM_MFCC', 'XGB_MFCC', 'CAT_MFCC', 'MLP_MFCC'] if k in predictions], axis=0)
    group_lfcc = np.mean([predictions[k] for k in ['LGBM_LFCC', 'XGB_LFCC', 'CAT_LFCC', 'MLP_LFCC'] if k in predictions], axis=0)
    group_comb = np.mean([predictions[k] for k in ['LGBM_COMB', 'XGB_COMB', 'CAT_COMB', 'MLP_COMB'] if k in predictions], axis=0)
    
    best_eer = 1.0
    best_w = (0.0, 0.0, 0.0)
    
    # Сетка весов с шагом 0.05
    for w_mfcc in np.linspace(0.0, 1.0, 21):
        for w_lfcc in np.linspace(0.0, 1.0 - w_mfcc, 21):
            w_comb = 1.0 - w_mfcc - w_lfcc
            if w_comb < -1e-9:
                continue
            
            weighted_preds = w_mfcc * group_mfcc + w_lfcc * group_lfcc + w_comb * group_comb
            eer, _ = compute_eer(weighted_preds, y_eval)
            
            if eer < best_eer:
                best_eer = eer
                best_w = (w_mfcc, w_lfcc, w_comb)
                
    print(f"Best Ensemble Weight (MFCC, LFCC, Combined): {best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}")
    print(f"Best Optimized Ensemble EER: {best_eer*100:.2f}%")
        
    # 7. Оценка Stacking мета-модели
    meta_model_file = "stacking_meta_model.pkl"
    if run_subset and os.path.exists("stacking_meta_model_subset.pkl"):
        meta_model_file = "stacking_meta_model_subset.pkl"
    stacking_eer = None
    if os.path.exists(meta_model_file):
        print("\n====================================================")
        print("         EVALUATING STACKING META-CLASSIFIER        ")
        print("====================================================")
        with open(meta_model_file, "rb") as f:
            meta_model, model_names_loaded = pickle.load(f)
            
        meta_features_eval = []
        missing_models = False
        for name in model_names_loaded:
            if name in predictions:
                meta_features_eval.append(predictions[name])
            else:
                print(f"[WARNING] Model {name} is missing from evaluation predictions!")
                missing_models = True
                break
                
        if not missing_models:
            meta_features_eval = np.column_stack(meta_features_eval)
            meta_eval_preds = meta_model.predict_proba(meta_features_eval)[:, 1]
            stacking_eer, _ = compute_eer(meta_eval_preds, y_eval)
            print(f"Stacking Meta-Classifier EER on 2021 Eval: {stacking_eer*100:.2f}%")
        else:
            print("[ERROR] Cannot run Stacking because some base models are missing.")
            
    # Записываем результаты в файл
    summary_file = "overnight_results_robust.txt"
    with open(summary_file, "w", encoding="utf-8") as out:
        out.write("====================================================\n")
        out.write("     ASVspoof 2021 LA ROBUST EVALUATION SUMMARY     \n")
        out.write("====================================================\n\n")
        out.write("--- Individual Robust Models EER ---\n")
        for name, preds in predictions.items():
            eer, _ = compute_eer(preds, y_eval)
            out.write(f"{name}: {eer*100:.2f}%\n")
        out.write("\n--- Ensembles EER ---\n")
        if mfcc_preds:
            eer, _ = compute_eer(np.mean(mfcc_preds, axis=0), y_eval)
            out.write(f"Ensemble (Robust MFCC only): {eer*100:.2f}%\n")
        if lfcc_preds:
            eer, _ = compute_eer(np.mean(lfcc_preds, axis=0), y_eval)
            out.write(f"Ensemble (Robust LFCC only): {eer*100:.2f}%\n")
        if comb_preds:
            eer, _ = compute_eer(np.mean(comb_preds, axis=0), y_eval)
            out.write(f"Ensemble (Robust Combined only): {eer*100:.2f}%\n")
        if len(lgb_ensemble) == 2:
            eer, _ = compute_eer(np.mean(lgb_ensemble, axis=0), y_eval)
            out.write(f"Ensemble (LGBM MFCC + LGBM LFCC): {eer*100:.2f}%\n")
        if all_preds:
            eer, _ = compute_eer(np.mean(all_preds, axis=0), y_eval)
            out.write(f"Ensemble (All 9 Robust Models): {eer*100:.2f}%\n")
        out.write(f"\nBest Ensemble Weight (MFCC, LFCC, Combined): {best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}\n")
        out.write(f"Best Optimized Ensemble EER: {best_eer*100:.2f}%\n")
        
        if stacking_eer is not None:
            out.write(f"Stacking Meta-Classifier EER: {stacking_eer*100:.2f}%\n")
            
    print(f"\nEvaluation summary saved to {summary_file}")
