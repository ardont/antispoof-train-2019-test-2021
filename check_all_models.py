import os
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier

model_paths = {
    'LGBM_MFCC': 'lgb_model_mfcc_robust.pkl', 'XGB_MFCC': 'xgb_model_mfcc_robust.json', 'CAT_MFCC': 'cat_model_mfcc_robust.cbm', 'MLP_MFCC': 'mlp_model_mfcc_robust.pkl',
    'LGBM_LFCC': 'lgb_model_lfcc_robust.pkl', 'XGB_LFCC': 'xgb_model_lfcc_robust.json', 'CAT_LFCC': 'cat_model_lfcc_robust.cbm', 'MLP_LFCC': 'mlp_model_lfcc_robust.pkl',
    'LGBM_CQCC': 'lgb_model_cqcc_robust.pkl', 'XGB_CQCC': 'xgb_model_cqcc_robust.json', 'CAT_CQCC': 'cat_model_cqcc_robust.cbm', 'MLP_CQCC': 'mlp_model_cqcc_robust.pkl',
    'LGBM_COMB': 'lgb_model_combined_robust.pkl', 'XGB_COMB': 'xgb_model_combined_robust.json', 'CAT_COMB': 'cat_model_combined_robust.cbm', 'MLP_COMB': 'mlp_model_combined_robust.pkl'
}

corrupt = []
missing = []

for name, path in model_paths.items():
    if not os.path.exists(path):
        print(f"{name} ({path}): MISSING")
        missing.append(path)
        continue
    try:
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                pickle.load(f)
        elif path.endswith('.json'):
            m = xgb.XGBClassifier()
            m.load_model(path)
        elif path.endswith('.cbm'):
            m = CatBoostClassifier()
            m.load_model(path)
        print(f"{name} ({path}): OK")
    except Exception as e:
        print(f"{name} ({path}): CORRUPT ({e})")
        corrupt.append(path)

print("\n--- Summary ---")
print(f"Total models checked: {len(model_paths)}")
print(f"Missing models: {len(missing)} {missing}")
print(f"Corrupt models: {len(corrupt)} {corrupt}")
