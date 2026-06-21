import sys
import os
import argparse
import numpy as np
import pickle
import warnings
import time

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-Way Combined Robust Feature Training")
    parser.add_argument("--subset", action="store_true", help="Run in subset mode")
    args = parser.parse_args()
    
    run_subset = args.subset
    
    if run_subset:
        mfcc_cache = "robust_mfcc_cache_subset.pkl"
        lfcc_cache = "robust_lfcc_cache_subset.pkl"
        cqcc_cache = "robust_cqcc_cache_subset.pkl"
        combined_cache = "robust_combined_cache_subset.pkl"
        print("--- RUNNING COMBINED FINAL IN SUBSET MODE ---")
    else:
        mfcc_cache = "robust_mfcc_cache.pkl"
        lfcc_cache = "robust_lfcc_cache.pkl"
        cqcc_cache = "robust_cqcc_cache.pkl"
        combined_cache = "robust_combined_cache.pkl"
        print("--- RUNNING COMBINED FINAL IN FULL TRAINING MODE ---")
        
    if not (os.path.exists(mfcc_cache) and os.path.exists(lfcc_cache) and os.path.exists(cqcc_cache)):
        print("[ERROR] Missing cache files for MFCC, LFCC, or CQCC.")
        sys.exit(1)
        
    print("Loading caches...")
    with open(mfcc_cache, "rb") as f: X_train_mfcc, y_train_mfcc, X_dev_mfcc, y_dev_mfcc = pickle.load(f)
    with open(lfcc_cache, "rb") as f: X_train_lfcc, y_train_lfcc, X_dev_lfcc, y_dev_lfcc = pickle.load(f)
    with open(cqcc_cache, "rb") as f: X_train_cqcc, y_train_cqcc, X_dev_cqcc, y_dev_cqcc = pickle.load(f)
    
    print("Concatenating features (Early Fusion: MFCC + LFCC + CQCC)...")
    X_train = np.hstack([X_train_mfcc, X_train_lfcc, X_train_cqcc])
    X_dev = np.hstack([X_dev_mfcc, X_dev_lfcc, X_dev_cqcc])
    y_train = y_train_mfcc
    y_dev = y_dev_mfcc
    
    # Save combined cache
    with open(combined_cache, "wb") as f:
        pickle.dump((X_train, y_train, X_dev, y_dev), f)
    print(f"Saved 3-way combined cache to {combined_cache}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    scaler_file = "scaler_combined_robust.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Combined scaler saved to {scaler_file}")
    
    # Train LGBM
    print("\nTraining LightGBM on 3-way combined features...")
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
    print(f"LGBM Combined Dev EER: {eer*100:.2f}%")
    with open("lgb_model_combined_robust.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
        
    # Train XGBoost
    print("\nTraining XGBoost on 3-way combined features...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', early_stopping_rounds=50, use_label_encoder=False
    )
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_dev_scaled, y_dev)], verbose=False)
    preds = xgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    print(f"XGBoost Combined Dev EER: {eer*100:.2f}%")
    xgb_model.save_model("xgb_model_combined_robust.json")
    
    # Train CatBoost
    print("\nTraining CatBoost on 3-way combined features...")
    cat_model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, loss_function='Logloss',
        eval_metric='AUC', early_stopping_rounds=50, random_seed=42, verbose=False, thread_count=-1
    )
    cat_model.fit(X_train_scaled, y_train, eval_set=[(X_dev_scaled, y_dev)], verbose=False)
    preds = cat_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    print(f"CatBoost Combined Dev EER: {eer*100:.2f}%")
    cat_model.save_model("cat_model_combined_robust.cbm")
