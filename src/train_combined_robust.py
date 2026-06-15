import sys
import os
import argparse
import numpy as np
import pickle
import warnings
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer

# -----------------------------------------------------------------------------
# 🚀 Точка входа в скрипт
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Robust Feature Training")
    parser.add_argument("--subset", action="store_true", help="Run in subset mode for fast verification")
    args = parser.parse_args()
    
    run_subset = args.subset
    
    # Имена файлов кэша
    if run_subset:
        mfcc_cache = "robust_mfcc_cache_subset.pkl"
        lfcc_cache = "robust_lfcc_cache_subset.pkl"
        combined_cache = "robust_combined_cache_subset.pkl"
        print("--- RUNNING COMBINED IN SUBSET MODE ---")
    else:
        mfcc_cache = "robust_mfcc_cache.pkl"
        lfcc_cache = "robust_lfcc_cache.pkl"
        combined_cache = "robust_combined_cache.pkl"
        print("--- RUNNING COMBINED IN FULL TRAINING MODE ---")
        
    # Проверяем, есть ли готовый комбинированный кэш
    if os.path.exists(combined_cache):
        print(f"Loading combined features from cache: {combined_cache}")
        with open(combined_cache, "rb") as f:
            X_train, y_train, X_dev, y_dev = pickle.load(f)
    else:
        # Проверяем наличие отдельных кэшей
        if not os.path.exists(mfcc_cache) or not os.path.exists(lfcc_cache):
            print("[ERROR] Не найдены кэш-файлы MFCC или LFCC. Пожалуйста, запустите сначала:")
            print(f"  python src/train_robust.py --feature mfcc {'--subset' if run_subset else ''}")
            print(f"  python src/train_robust.py --feature lfcc {'--subset' if run_subset else ''}")
            sys.exit(1)
            
        print("Loading separate MFCC and LFCC caches...")
        with open(mfcc_cache, "rb") as f:
            X_train_mfcc, y_train_mfcc, X_dev_mfcc, y_dev_mfcc = pickle.load(f)
        with open(lfcc_cache, "rb") as f:
            X_train_lfcc, y_train_lfcc, X_dev_lfcc, y_dev_lfcc = pickle.load(f)
            
        print("Concatenating features (Early Fusion)...")
        # Объединяем вдоль оси признаков (столбцы)
        X_train = np.hstack([X_train_mfcc, X_train_lfcc])
        X_dev = np.hstack([X_dev_mfcc, X_dev_lfcc])
        
        y_train = y_train_mfcc
        y_dev = y_dev_mfcc
        
        # Кэшируем объединенные данные
        with open(combined_cache, "wb") as f:
            pickle.dump((X_train, y_train, X_dev, y_dev), f)
        print(f"Combined features saved to cache: {combined_cache}")

    print(f"Combined Train size: {X_train.shape}, Dev size: {X_dev.shape}")
    
    # -----------------------------------------------------------------------------
    # ⚖️ Стандартизация признаков (StandardScaler)
    # -----------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    scaler_file = "scaler_combined_robust.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Robust combined scaler saved to {scaler_file}")
    
    results = {}
    
    # -----------------------------------------------------------------------------
    # 🌲 1. Обучение LightGBM
    # -----------------------------------------------------------------------------
    print("\nTraining LightGBM on combined features...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_dev_scaled, y_dev)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    preds = lgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    results['LightGBM (combined)'] = eer
    print(f"LGBM 2019 Dev EER: {eer*100:.2f}%")
    
    with open("lgb_model_combined_robust.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
        
    # -----------------------------------------------------------------------------
    # 🌲 2. Обучение XGBoost
    # -----------------------------------------------------------------------------
    print("\nTraining XGBoost on combined features...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50,
        use_label_encoder=False
    )
    
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_dev_scaled, y_dev)],
        verbose=False
    )
    
    preds = xgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    results['XGBoost (combined)'] = eer
    print(f"XGBoost 2019 Dev EER: {eer*100:.2f}%")
    
    xgb_model.save_model("xgb_model_combined_robust.json")
    
    # -----------------------------------------------------------------------------
    # 🌲 3. Обучение CatBoost
    # -----------------------------------------------------------------------------
    print("\nTraining CatBoost on combined features...")
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
        thread_count=-1
    )
    
    train_pool = Pool(X_train_scaled, label=y_train)
    eval_pool = Pool(X_dev_scaled, label=y_dev)
    cat_model.fit(train_pool, eval_set=eval_pool, plot=False)
    
    preds = cat_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    results['CatBoost (combined)'] = eer
    print(f"CatBoost 2019 Dev EER: {eer*100:.2f}%")
    
    cat_model.save_model("cat_model_combined_robust.cbm")
    
    print("\n--- Summary Combined Dev 2019 EER ---")
    for name, eer_val in results.items():
        print(f"{name}: {eer_val*100:.2f}%")
