import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier
from utils.metrics import compute_eer

if __name__ == "__main__":
    run_subset = len(sys.argv) > 1 and sys.argv[1] == "--subset"
    if run_subset:
        mfcc_cache = "eval_2021_subset_cache.pkl"
        lfcc_cache = "eval_lfcc_2021_subset_cache.pkl"
        print("Evaluating ensemble on SUBSET...")
    else:
        mfcc_cache = "eval_2021_cache.pkl"
        lfcc_cache = "eval_lfcc_2021_cache.pkl"
        print("Evaluating ensemble on FULL 2021 set...")
        
    if not os.path.exists(mfcc_cache) or not os.path.exists(lfcc_cache):
        print(f"Required feature caches not found! Make sure both evaluate_2021.py and evaluate_lfcc_2021.py have run.")
        sys.exit(1)
        
    print("Loading cached features...")
    with open(mfcc_cache, 'rb') as f:
        X_mfcc, y_mfcc = pickle.load(f)
        
    with open(lfcc_cache, 'rb') as f:
        X_lfcc, y_lfcc = pickle.load(f)
        
    # Проверяем равенство меток (чтобы убедиться в совпадении порядка файлов)
    if not np.array_equal(y_mfcc, y_lfcc):
        print("Error: Target labels between MFCC and LFCC caches do not match!")
        sys.exit(1)
        
    y_eval = y_mfcc
    print(f"Loaded {len(y_eval)} evaluation samples.")
    
    # 2. Нормализация признаков
    with open("scaler.pkl", "rb") as f:
        scaler_mfcc = pickle.load(f)
    X_mfcc_scaled = scaler_mfcc.transform(X_mfcc)
    
    with open("scaler_lfcc.pkl", "rb") as f:
        scaler_lfcc = pickle.load(f)
    X_lfcc_scaled = scaler_lfcc.transform(X_lfcc)
    
    # 3. Собираем предсказания моделей
    preds_dict = {}
    
    # --- MFCC модели ---
    # LightGBM MFCC
    if os.path.exists("lgb_model_augmented.pkl"):
        with open("lgb_model_augmented.pkl", "rb") as f:
            model = pickle.load(f)
        preds_dict['LGBM_MFCC'] = model.predict_proba(X_mfcc_scaled)[:, 1]
        
    # XGBoost MFCC
    if os.path.exists("xgb_model_augmented.json"):
        model = xgb.XGBClassifier()
        model.load_model("xgb_model_augmented.json")
        preds_dict['XGB_MFCC'] = model.predict_proba(X_mfcc_scaled)[:, 1]
        
    # CatBoost MFCC
    if os.path.exists("cat_model_augmented.cbm"):
        model = CatBoostClassifier()
        model.load_model("cat_model_augmented.cbm")
        preds_dict['CAT_MFCC'] = model.predict_proba(X_mfcc_scaled)[:, 1]
        
    # --- LFCC модели ---
    # LightGBM LFCC
    if os.path.exists("lgb_model_lfcc_augmented.pkl"):
        with open("lgb_model_lfcc_augmented.pkl", "rb") as f:
            model = pickle.load(f)
        preds_dict['LGBM_LFCC'] = model.predict_proba(X_lfcc_scaled)[:, 1]
        
    # XGBoost LFCC
    if os.path.exists("xgb_model_lfcc_augmented.json"):
        model = xgb.XGBClassifier()
        model.load_model("xgb_model_lfcc_augmented.json")
        preds_dict['XGB_LFCC'] = model.predict_proba(X_lfcc_scaled)[:, 1]
        
    # CatBoost LFCC
    if os.path.exists("cat_model_lfcc_augmented.cbm"):
        model = CatBoostClassifier()
        model.load_model("cat_model_lfcc_augmented.cbm")
        preds_dict['CAT_LFCC'] = model.predict_proba(X_lfcc_scaled)[:, 1]

    # 4. Оценка индивидуальных моделей для сверки
    print("\n=== Individual Models EER on ASVspoof 2021 ===")
    for name, preds in preds_dict.items():
        eer, _ = compute_eer(preds, y_eval)
        print(f"{name}: {eer*100:.2f}%")
        
    # 5. Различные варианты ансамблей
    print("\n=== Ensembles EER on ASVspoof 2021 ===")
    
    # Ансамбль 1: Все модели MFCC
    mfcc_preds = [preds_dict[k] for k in ['LGBM_MFCC', 'XGB_MFCC', 'CAT_MFCC'] if k in preds_dict]
    if mfcc_preds:
        avg_mfcc = np.mean(mfcc_preds, axis=0)
        eer_mfcc, _ = compute_eer(avg_mfcc, y_eval)
        print(f"Ensemble (All MFCC Models): {eer_mfcc*100:.2f}%")
        
    # Ансамбль 2: Все модели LFCC
    lfcc_preds = [preds_dict[k] for k in ['LGBM_LFCC', 'XGB_LFCC', 'CAT_LFCC'] if k in preds_dict]
    if lfcc_preds:
        avg_lfcc = np.mean(lfcc_preds, axis=0)
        eer_lfcc, _ = compute_eer(avg_lfcc, y_eval)
        print(f"Ensemble (All LFCC Models): {eer_lfcc*100:.2f}%")
        
    # Ансамбль 3: Лучшая MFCC (LGBM) + Лучшая LFCC (LGBM)
    if 'LGBM_MFCC' in preds_dict and 'LGBM_LFCC' in preds_dict:
        avg_best_two = (preds_dict['LGBM_MFCC'] + preds_dict['LGBM_LFCC']) / 2.0
        eer_best_two, _ = compute_eer(avg_best_two, y_eval)
        print(f"Ensemble (LGBM MFCC + LGBM LFCC): {eer_best_two*100:.2f}%")

    # Ансамбль 4: Вообще все 6 моделей
    all_preds = list(preds_dict.values())
    if len(all_preds) > 0:
        avg_all = np.mean(all_preds, axis=0)
        eer_all, _ = compute_eer(avg_all, y_eval)
        print(f"Ensemble (All 6 Models): {eer_all*100:.2f}%")
