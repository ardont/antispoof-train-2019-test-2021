import sys
import os

# Добавляем корень проекта через абсолютный путь, чтобы избежать багов с кириллицей в Windows
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import pickle
import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer

warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stacking Meta-Classifier")
    parser.add_argument("--subset", action="store_true", help="Run in subset mode for fast verification")
    parser.add_argument("--exclude-lfcc", action="store_true", help="Exclude LFCC base models to avoid domain shift overfitting")
    args = parser.parse_args()
    
    run_subset = args.subset
    exclude_lfcc = args.exclude_lfcc
    
    # Имена файлов кэша
    if run_subset:
        mfcc_cache = "robust_mfcc_cache_subset.pkl"
        lfcc_cache = "robust_lfcc_cache_subset.pkl"
        combined_cache = "robust_combined_cache_subset.pkl"
        print("--- RUNNING STACKING IN SUBSET MODE ---")
    else:
        mfcc_cache = "robust_mfcc_cache.pkl"
        lfcc_cache = "robust_lfcc_cache.pkl"
        combined_cache = "robust_combined_cache.pkl"
        print("--- RUNNING STACKING IN FULL MODE ---")
        
    if not (os.path.exists(mfcc_cache) and os.path.exists(lfcc_cache) and os.path.exists(combined_cache)):
        print("[ERROR] Не найдены кэш-файлы. Пожалуйста, убедитесь, что вы обучили все модели.")
        sys.exit(1)
        
    print("Loading Dev caches...")
    with open(mfcc_cache, "rb") as f:
        _, _, X_dev_mfcc, y_dev_mfcc = pickle.load(f)
    with open(lfcc_cache, "rb") as f:
        _, _, X_dev_lfcc, y_dev_lfcc = pickle.load(f)
    with open(combined_cache, "rb") as f:
        _, _, X_dev_comb, y_dev_comb = pickle.load(f)
        
    # Загружаем скейлеры
    print("Loading scalers...")
    with open("scaler_mfcc_robust.pkl", "rb") as f:
        scaler_mfcc = pickle.load(f)
    with open("scaler_lfcc_robust.pkl", "rb") as f:
        scaler_lfcc = pickle.load(f)
    with open("scaler_combined_robust.pkl", "rb") as f:
        scaler_comb = pickle.load(f)
        
    X_mfcc_scaled = scaler_mfcc.transform(X_dev_mfcc)
    X_lfcc_scaled = scaler_lfcc.transform(X_dev_lfcc)
    X_comb_scaled = scaler_comb.transform(X_dev_comb)
    
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
    
    print("\n--- Generating Meta-Features (predictions of base models on Dev 2019) ---")
    meta_features = []
    model_names_loaded = []
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"[WARNING] Model file {path} not found. Skipping {name}...")
            continue
            
        print(f"Generating predictions for {name}...")
        
        # Определяем тип признаков для модели
        if '_MFCC' in name:
            X_input = X_mfcc_scaled
        elif '_LFCC' in name:
            X_input = X_lfcc_scaled
        else:
            X_input = X_comb_scaled
            
        # Загружаем модель и делаем предсказание
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
    
    # Метки классов
    y_dev = y_dev_mfcc
    
    # Обучаем мета-классификатор
    # Используем кросс-валидационную логистическую регрессию для выбора силы регуляризации C
    print("\nTraining Meta-Classifier (Logistic Regression CV)...")
    meta_model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 9),
        cv=5,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    meta_model.fit(meta_features, y_dev)
    
    print(f"Optimized regularization strength C: {meta_model.C_[0]:.4f}")
    
    # Оценка качества мета-классификатора на Dev 2019
    meta_preds = meta_model.predict_proba(meta_features)[:, 1]
    eer, _ = compute_eer(meta_preds, y_dev)
    print(f"Meta-Classifier Dev 2019 EER: {eer*100:.2f}%")
    
    # Сохраняем мета-модель и список моделей
    suffix = "_subset" if run_subset else ""
    meta_model_file = f"stacking_meta_model{suffix}.pkl"
    with open(meta_model_file, "wb") as f:
        pickle.dump((meta_model, model_names_loaded), f)
        
    print(f"Stacking meta-model saved to {meta_model_file}")
    
    # Выведем коэффициенты мета-модели, чтобы увидеть вклад базовых классификаторов
    print("\nMeta-Model Coefficients (Feature Importances):")
    for name, coef in zip(model_names_loaded, meta_model.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {meta_model.intercept_[0]:.4f}")
