import sys
import os
import argparse
import numpy as np
import pickle
import warnings

# Устанавливаем корень проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.metrics import compute_eer

warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Проверяем и устанавливаем optuna при необходимости
try:
    import optuna
except ImportError:
    print("Installing optuna...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

def objective_lgb(trial, X_train, y_train, X_dev, y_dev):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    preds = model.predict_proba(X_dev)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    return eer

def objective_xgb(trial, X_train, y_train, X_dev, y_dev):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        early_stopping_rounds=30,
        verbose=False
    )
    
    preds = model.predict_proba(X_dev)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    return eer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Robust Models using Optuna")
    parser.add_argument("--feature", type=str, default="combined", choices=["mfcc", "lfcc", "combined"])
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()
    
    feature_type = args.feature.lower()
    
    cache_file = f"robust_{feature_type}_cache.pkl"
    if not os.path.exists(cache_file):
        print(f"[ERROR] Cache file {cache_file} not found. Train baseline first.")
        sys.exit(1)
        
    print(f"Loading features from cache: {cache_file}...")
    with open(cache_file, "rb") as f:
        X_train, y_train, X_dev, y_dev = pickle.load(f)
        
    # Масштабируем признаки
    print("Scaling features...")
    scaler_file = f"scaler_{feature_type}_robust.pkl"
    if os.path.exists(scaler_file):
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        X_train_scaled = scaler.transform(X_train)
        X_dev_scaled = scaler.transform(X_dev)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_dev_scaled = scaler.transform(X_dev)
        
    # Уменьшаем выборку для ускорения тюнинга (берем 15000 случайных сэмплов для обучения)
    if X_train_scaled.shape[0] > 15000:
        print("Subsampling training set to 15,000 files for faster optimization...")
        np.random.seed(42)
        indices = np.random.choice(X_train_scaled.shape[0], 15000, replace=False)
        X_train_sub = X_train_scaled[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train_scaled
        y_train_sub = y_train
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # 1. Тюнинг LightGBM
    print(f"\n--- Optimizing LightGBM on robust {feature_type.upper()} ---")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lambda trial: objective_lgb(trial, X_train_sub, y_train_sub, X_dev_scaled, y_dev), n_trials=args.trials)
    
    print("Best LightGBM Trial:")
    print(f"  Value (EER): {study_lgb.best_value*100:.2f}%")
    print("  Params: ")
    for k, v in study_lgb.best_params.items():
        print(f"    {k}: {v}")
        
    # Обучаем итоговую модель LightGBM с лучшими параметрами на полной выборке
    print("\nTraining final LightGBM with best params...")
    best_params_lgb = study_lgb.best_params
    best_params_lgb['random_state'] = 42
    best_params_lgb['verbose'] = -1
    
    final_lgb = lgb.LGBMClassifier(**best_params_lgb)
    final_lgb.fit(
        X_train_scaled, y_train,
        eval_set=[(X_dev_scaled, y_dev)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    preds_lgb = final_lgb.predict_proba(X_dev_scaled)[:, 1]
    eer_lgb, _ = compute_eer(preds_lgb, y_dev)
    print(f"Final LightGBM Dev EER: {eer_lgb*100:.2f}%")
    
    # Сохраняем лучшую модель
    with open(f"lgb_model_{feature_type}_robust.pkl", "wb") as f:
        pickle.dump(final_lgb, f)
    print(f"Saved optimized LightGBM to lgb_model_{feature_type}_robust.pkl")
    
    # 2. Тюнинг XGBoost
    print(f"\n--- Optimizing XGBoost on robust {feature_type.upper()} ---")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train_sub, y_train_sub, X_dev_scaled, y_dev), n_trials=args.trials)
    
    print("Best XGBoost Trial:")
    print(f"  Value (EER): {study_xgb.best_value*100:.2f}%")
    print("  Params: ")
    for k, v in study_xgb.best_params.items():
        print(f"    {k}: {v}")
        
    # Обучаем итоговую модель XGBoost с лучшими параметрами на полной выборке
    print("\nTraining final XGBoost with best params...")
    best_params_xgb = study_xgb.best_params
    best_params_xgb['random_state'] = 42
    best_params_xgb['eval_metric'] = 'logloss'
    best_params_xgb['use_label_encoder'] = False
    
    final_xgb = xgb.XGBClassifier(**best_params_xgb)
    final_xgb.fit(
        X_train_scaled, y_train,
        eval_set=[(X_dev_scaled, y_dev)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    preds_xgb = final_xgb.predict_proba(X_dev_scaled)[:, 1]
    eer_xgb, _ = compute_eer(preds_xgb, y_dev)
    print(f"Final XGBoost Dev EER: {eer_xgb*100:.2f}%")
    
    # Сохраняем лучшую модель
    final_xgb.save_model(f"xgb_model_{feature_type}_robust.json")
    print(f"Saved optimized XGBoost to xgb_model_{feature_type}_robust.json")
