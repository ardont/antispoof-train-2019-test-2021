import sys
import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import pickle
import warnings
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

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

def process_file_wrapper_custom(args):
    file_id, label_num, ext, audio_dir, feature_type = args
    
    if ext:
        audio_path = os.path.join(audio_dir, file_id + ext)
    else:
        audio_path = os.path.join(audio_dir, file_id + '.flac')
        if not os.path.exists(audio_path):
            audio_path = os.path.join(audio_dir, file_id + '.wav')
            
    if not os.path.exists(audio_path):
        return None
        
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE: y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        if feature_type == 'mfcc': feats = extract_robust_mfcc_features(y, SAMPLE_RATE)
        elif feature_type == 'lfcc': feats = extract_robust_lfcc_features(y, SAMPLE_RATE)
        else: feats = extract_robust_cqcc_features(y, SAMPLE_RATE)
        return file_id, label_num, feats
    except Exception:
        return None

def find_metadata_file(data_dir, audio_dir):
    candidates = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.txt', '.csv', '.tsv')) and not file.startswith('custom_'):
                path = os.path.join(root, file)
                candidates.append(path)
                
    # Сортируем кандидатов: приоритет файлам с "CM" в пути и "keys", штраф для "ASV"
    def sort_key(path):
        score = 0
        if "CM" in path: score -= 10
        if "keys" in path.lower(): score -= 5
        if "ASV" in path: score += 10
        return score
        
    candidates.sort(key=sort_key)
    
    for path in candidates:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                valid_count = 0
                for _ in range(15):
                    line = f.readline()
                    if not line: break
                    parsed = parse_metadata_line(line)
                    if parsed:
                        file_id, label_num, ext = parsed
                        if ext:
                            exists = os.path.exists(os.path.join(audio_dir, file_id + ext))
                        else:
                            exists = os.path.exists(os.path.join(audio_dir, file_id + '.flac')) or os.path.exists(os.path.join(audio_dir, file_id + '.wav'))
                        if exists:
                            valid_count += 1
                if valid_count > 0:
                    return path
        except Exception:
            pass
    return None

def find_audio_dir(data_dir):
    # Try to find a directory containing several .flac or .wav files
    for root, dirs, files in os.walk(data_dir):
        audio_files = [f for f in files if f.lower().endswith(('.flac', '.wav'))]
        if len(audio_files) > 5:
            return root
    for root, dirs, files in os.walk(data_dir):
        audio_files = [f for f in files if f.lower().endswith(('.flac', '.wav'))]
        if len(audio_files) > 0:
            return root
    return None

def parse_metadata_line(line):
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    if len(parts) >= 6:
        # ASVspoof format: LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
        file_id = parts[1]
        label = parts[5]
    else:
        # Simple format: file_id label
        file_id = parts[0]
        label = parts[-1]
        
    file_name, ext = os.path.splitext(file_id)
    if ext.lower() in ['.flac', '.wav', '.mp3', '.ogg']:
        file_id = file_name
    else:
        ext = None
        
    label_num = 0 if label.lower() in ['bonafide', 'genuine', '0'] else 1
    return file_id, label_num, ext

def get_custom_features(files_info, audio_dir, feature_type, data_dir, run_subset=False):
    suffix = "_subset" if run_subset else ""
    cache_file = os.path.join(data_dir, f"cache_robust_{feature_type}{suffix}.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached custom {feature_type.upper()} features from cache file...")
        with open(cache_file, "rb") as f:
            X_feats, y_labels, file_ids = pickle.load(f)
    else:
        print(f"Extracting robust {feature_type.upper()} features...")
        X_feats, y_labels, file_ids = [], [], []
        max_workers = os.cpu_count()
        args_list = [(info[0], info[1], info[2], audio_dir, feature_type) for info in files_info]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_file_wrapper_custom, args_list, chunksize=100)
            for idx, res in enumerate(results):
                if res is not None:
                    file_ids.append(res[0])
                    y_labels.append(res[1])
                    X_feats.append(res[2])
                if idx > 0 and idx % 1000 == 0:
                    print(f"  Processed {idx}/{len(files_info)}...")
                    
        X_feats = np.array(X_feats)
        y_labels = np.array(y_labels)
        if len(X_feats) > 0:
            with open(cache_file, "wb") as f:
                pickle.dump((X_feats, y_labels, file_ids), f)
            
    return X_feats, y_labels, file_ids

def main():
    parser = argparse.ArgumentParser(description="Evaluate Robust Ensemble on Custom Dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory under data/ containing the custom dataset")
    parser.add_argument("--metadata", type=str, default=None, help="Optional path to metadata file")
    parser.add_argument("--audio-dir", type=str, default=None, help="Optional path to audio folder")
    parser.add_argument("--subset", action="store_true", help="Run only on first 1000 samples for checking")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        # check if it is under data/
        alt_dir = os.path.join(project_root, "data", data_dir)
        if os.path.exists(alt_dir):
            data_dir = alt_dir
        else:
            print(f"[ERROR] Custom dataset directory not found: {data_dir}")
            sys.exit(1)
            
    print(f"====================================================")
    print(f"  Custom Dataset Evaluation: {os.path.basename(data_dir)}")
    print(f"====================================================")
    
    # 1. Resolve Audio Dir
    audio_dir = args.audio_dir
    if not audio_dir:
        audio_dir = find_audio_dir(data_dir)
    if not audio_dir or not os.path.exists(audio_dir):
        print(f"[ERROR] Could not find any audio folder with .flac or .wav files in: {data_dir}")
        sys.exit(1)
    print(f"[INFO] Resolved Audio Directory: {audio_dir}")
    
    # 2. Resolve Metadata
    metadata_file = args.metadata
    if not metadata_file:
        metadata_file = find_metadata_file(data_dir, audio_dir)
        
    has_labels = False
    files_info = [] # list of tuples: (file_id, label, ext)
    
    if metadata_file and os.path.exists(metadata_file):
        print(f"[INFO] Resolved Metadata/Keys File: {metadata_file}")
        has_labels = True
        with open(metadata_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parsed = parse_metadata_line(line)
                if parsed:
                    files_info.append(parsed)
    else:
        print(f"[WARNING] No metadata file with labels found. Operating in unlabelled mode.")
        print(f"[INFO] Scanning all audio files directly from the audio directory...")
        for file in os.listdir(audio_dir):
            if file.lower().endswith(('.flac', '.wav')):
                file_name, ext = os.path.splitext(file)
                files_info.append((file_name, -1, ext))
                
    if not files_info:
        print(f"[ERROR] No audio files to evaluate.")
        sys.exit(1)
        
    print(f"[INFO] Found {len(files_info)} files for evaluation.")
    if args.subset:
        files_info = files_info[:1000]
        print(f"[INFO] Subset mode active. Evaluating on first {len(files_info)} samples.")
        
    # 3. Extract Features
    X_mfcc, y_eval, file_ids = get_custom_features(files_info, audio_dir, 'mfcc', data_dir, args.subset)
    X_lfcc, _, _ = get_custom_features(files_info, audio_dir, 'lfcc', data_dir, args.subset)
    X_cqcc, _, _ = get_custom_features(files_info, audio_dir, 'cqcc', data_dir, args.subset)
    
    if len(X_mfcc) == 0:
        print(f"[ERROR] Feature extraction failed or yielded 0 files.")
        sys.exit(1)
        
    X_comb = np.hstack([X_mfcc, X_lfcc, X_cqcc])
    
    # 4. Load scalers
    scalers_paths = {
        'scaler_mfcc_robust.pkl': 'scaler_mfcc_robust.pkl',
        'scaler_lfcc_robust.pkl': 'scaler_lfcc_robust.pkl',
        'scaler_cqcc_robust.pkl': 'scaler_cqcc_robust.pkl',
        'scaler_combined_robust.pkl': 'scaler_combined_robust.pkl'
    }
    
    scalers = {}
    for name, path in scalers_paths.items():
        full_path = os.path.join(project_root, path)
        if not os.path.exists(full_path):
            print(f"[ERROR] Scaler not found: {full_path}. Have you trained the models?")
            sys.exit(1)
        with open(full_path, "rb") as f:
            scalers[name] = pickle.load(f)
            
    X_mfcc_scaled = scalers['scaler_mfcc_robust.pkl'].transform(X_mfcc)
    X_lfcc_scaled = scalers['scaler_lfcc_robust.pkl'].transform(X_lfcc)
    X_cqcc_scaled = scalers['scaler_cqcc_robust.pkl'].transform(X_cqcc)
    X_comb_scaled = scalers['scaler_combined_robust.pkl'].transform(X_comb)
    
    # 5. Load models and predict
    predictions = {}
    model_paths = {
        'LGBM_MFCC': 'lgb_model_mfcc_robust.pkl', 'XGB_MFCC': 'xgb_model_mfcc_robust.json', 'CAT_MFCC': 'cat_model_mfcc_robust.cbm', 'MLP_MFCC': 'mlp_model_mfcc_robust.pkl',
        'LGBM_LFCC': 'lgb_model_lfcc_robust.pkl', 'XGB_LFCC': 'xgb_model_lfcc_robust.json', 'CAT_LFCC': 'cat_model_lfcc_robust.cbm', 'MLP_LFCC': 'mlp_model_lfcc_robust.pkl',
        'LGBM_CQCC': 'lgb_model_cqcc_robust.pkl', 'XGB_CQCC': 'xgb_model_cqcc_robust.json', 'CAT_CQCC': 'cat_model_cqcc_robust.cbm', 'MLP_CQCC': 'mlp_model_cqcc_robust.pkl',
        'LGBM_COMB': 'lgb_model_combined_robust.pkl', 'XGB_COMB': 'xgb_model_combined_robust.json', 'CAT_COMB': 'cat_model_combined_robust.cbm', 'MLP_COMB': 'mlp_model_combined_robust.pkl'
    }
    
    for name, path in model_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            if '_MFCC' in name: X_in = X_mfcc_scaled
            elif '_LFCC' in name: X_in = X_lfcc_scaled
            elif '_CQCC' in name: X_in = X_cqcc_scaled
            else: X_in = X_comb_scaled
            
            print(f"Running predictions with {name}...")
            if path.endswith('.pkl'):
                with open(full_path, 'rb') as f: model = pickle.load(f)
                predictions[name] = model.predict_proba(X_in)[:, 1]
            elif path.endswith('.json'):
                model = xgb.XGBClassifier()
                model.load_model(full_path)
                predictions[name] = model.predict_proba(X_in)[:, 1]
            elif path.endswith('.cbm'):
                model = CatBoostClassifier()
                model.load_model(full_path)
                predictions[name] = model.predict_proba(X_in)[:, 1]
        else:
            print(f"[WARNING] Model weights file not found: {full_path}")
            
    # 6. Group Ensembles
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
            
    # 7. Weighted Ensemble (13% EER optimal weights)
    # Weights: MFCC = 0.30, LFCC = 0.00, CQCC = 0.14, COMB = 0.56
    w_mfcc, w_lfcc, w_cqcc, w_comb = 0.30, 0.00, 0.14, 0.56
    weighted_preds = (w_mfcc * group_preds.get('MFCC', 0.0) +
                      w_lfcc * group_preds.get('LFCC', 0.0) +
                      w_cqcc * group_preds.get('CQCC', 0.0) +
                      w_comb * group_preds.get('COMB', 0.0))
    predictions['Weighted_Ensemble'] = weighted_preds
    
    # 8. Calibrated Stacking Eval
    meta_file = os.path.join(project_root, "stacking_meta_model.pkl")
    if os.path.exists(meta_file):
        print("Running Calibrated Stacking Meta-Classifier...")
        with open(meta_file, "rb") as f:
            meta_model, model_names = pickle.load(f)
        meta_feats = []
        missing = False
        for name in model_names:
            if name in predictions:
                meta_feats.append(predictions[name])
            else:
                missing = True
                break
        if not missing:
            meta_feats = np.column_stack(meta_feats)
            predictions['Calibrated_Stacking'] = meta_model.predict_proba(meta_feats)[:, 1]
        else:
            print("[WARNING] Skipping Stacking (some models are missing in the current weights).")
            
    # 9. Output report
    report = []
    report.append("====================================================")
    report.append(f"     ROBUST ENSEMBLE EVALUATION ON CUSTOM DATASET   ")
    report.append(f"     Dataset: {os.path.basename(data_dir)}")
    report.append("====================================================\n")
    
    # Check if we have both classes present in y_eval
    can_compute_eer = has_labels and len(np.unique(y_eval)) >= 2
    
    if can_compute_eer:
        report.append("--- Individual Model EERs ---")
        for name, preds in predictions.items():
            if name not in ['Weighted_Ensemble', 'Calibrated_Stacking']:
                eer, _ = compute_eer(preds, y_eval)
                report.append(f"  {name}: {eer*100:.2f}%")
                
        report.append("\n--- Group Ensembles EER ---")
        for gname, preds in group_preds.items():
            eer, _ = compute_eer(preds, y_eval)
            report.append(f"  Ensemble ({gname} only): {eer*100:.2f}%")
            
        report.append("\n--- Weighted / Stacked Ensembles ---")
        eer_weighted, _ = compute_eer(predictions['Weighted_Ensemble'], y_eval)
        report.append(f"  Weighted Ensemble (MFCC 0.30, LFCC 0.00, CQCC 0.14, COMB 0.56) EER: {eer_weighted*100:.2f}%")
        
        if 'Calibrated_Stacking' in predictions:
            eer_stack, _ = compute_eer(predictions['Calibrated_Stacking'], y_eval)
            report.append(f"  Calibrated Stacking EER: {eer_stack*100:.2f}%")
    else:
        if has_labels:
            report.append("[WARNING] Cannot compute EER because the dataset/subset contains only one class.")
            num_bonafide = np.sum(y_eval == 0)
            num_spoof = np.sum(y_eval == 1)
            report.append(f"  Total samples evaluated: {len(y_eval)} (bonafide: {num_bonafide}, spoof: {num_spoof})")
        else:
            report.append("[INFO] Labels are not available, skipping EER computation.")
        report.append("[INFO] Saving predictions for all audio files.")
        
    out_content = "\n".join(report)
    print(out_content)
    
    # Save text report
    report_file = os.path.join(data_dir, "custom_evaluation_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(out_content)
    print(f"[SUCCESS] Text report saved to {report_file}")
    
    # Save CSV with predictions
    df_dict = {'file_id': file_ids}
    if has_labels:
        df_dict['label'] = y_eval
        
    for name, preds in predictions.items():
        df_dict[name] = preds
        
    df = pd.DataFrame(df_dict)
    csv_file = os.path.join(data_dir, "custom_evaluation_scores.csv")
    df.to_csv(csv_file, index=False)
    print(f"[SUCCESS] CSV predictions saved to {csv_file}")

if __name__ == "__main__":
    main()
