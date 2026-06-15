import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ограничиваем количество внутренних потоков библиотек для избежания конфликтов и зависаний
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import librosa
import soundfile as sf
import pickle
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer
from utils.augmentations import augment_audio
from utils.lfcc import extract_lfcc

# ---------------------------
# Конфигурация
# ---------------------------
LOCAL_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "2019", "LA"))
if os.path.exists(LOCAL_DATA):
    BASE_DATA = LOCAL_DATA
else:
    BASE_DATA = r"D:\фокусы\исследования\antispoof\data\2019\LA"

PROTOCOL_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_cm_protocols")
TRAIN_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_train")
DEV_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_dev")

PROTOCOL_TRAIN = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
PROTOCOL_DEV = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

SAMPLE_RATE = 16000
N_LFCC = 20
N_FILTERS = 128

# ---------------------------
# Извлечение признаков (LFCC + Delta + Delta-Delta)
# ---------------------------
def extract_features_from_waveform(y, sr):
    # Извлечение LFCC
    lfcc = extract_lfcc(y, sr=sr, n_lfcc=N_LFCC, n_filters=N_FILTERS)
    lfcc_delta = librosa.feature.delta(lfcc)
    lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
    
    # Объединяем
    lfcc_full = np.vstack([lfcc, lfcc_delta, lfcc_delta2])
    
    # CMS (Cepstral Mean Subtraction) для устойчивости к каналу связи
    mean = np.mean(lfcc_full, axis=1, keepdims=True)
    lfcc_full = lfcc_full - mean
    
    # 7 статистик по оси времени
    stats_list = []
    for c in range(lfcc_full.shape[0]):
        coef = lfcc_full[c, :]
        if coef.size == 0:
            coef = np.zeros(1)
        stats = [
            np.mean(coef), np.std(coef),
            np.min(coef), np.max(coef),
            np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)
        ]
        stats_list.extend(stats)
    return np.array(stats_list)

def extract_from_file(file_path, augment=False):
    try:
        waveform_np, sr = sf.read(file_path)
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)
            
        if sr != SAMPLE_RATE:
            waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        # 1. Оригинальные признаки
        feats_orig = extract_features_from_waveform(waveform_np, sr)
        
        if not augment:
            return [feats_orig]
            
        # 2. Аугментированные признаки
        waveform_aug = augment_audio(waveform_np, sr)
        feats_aug = extract_features_from_waveform(waveform_aug, sr)
        
        return [feats_orig, feats_aug]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Возвращаем нули в случае ошибки (20 коэффициентов * 3 * 7 статистик = 420 признаков)
        zeros = np.zeros(N_LFCC * 3 * 7)
        if augment:
            return [zeros, zeros]
        return [zeros]

# ---------------------------
# Загрузка данных
# ---------------------------
from concurrent.futures import ProcessPoolExecutor

def process_line_wrapper(args):
    line, audio_dir, augment = args
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    file_id = parts[1]
    label = parts[4]
    label_num = 0 if label == 'bonafide' else 1
    audio_path = os.path.join(audio_dir, "flac", file_id + '.flac')
    if not os.path.exists(audio_path):
        return None
        
    feats_list = extract_from_file(audio_path, augment=augment)
    return feats_list, label_num

def load_data(protocol_file, audio_dir, augment=False, max_files=None):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
        
    if max_files is not None and len(lines) > max_files:
        np.random.seed(42)
        np.random.shuffle(lines)
        lines = lines[:max_files]
        
    total_files = len(lines)
    print(f"Loading {total_files} files from {protocol_file} (augment={augment})...")
    
    X, y = [], []
    max_workers = os.cpu_count()
    print(f"Using {max_workers} CPU worker processes for feature extraction...")
    
    args_list = [(line, audio_dir, augment) for line in lines]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_line_wrapper, args_list)
        
        for idx, res in enumerate(results):
            if res is not None:
                feats_list, label_num = res
                for feats in feats_list:
                    X.append(feats)
                    y.append(label_num)
            if idx > 0 and idx % 2000 == 0:
                print(f"Processed {idx}/{total_files} files (current X shape: {len(X)})...")
                
    return np.array(X), np.array(y)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    run_subset = len(sys.argv) > 1 and sys.argv[1] == "--subset"
    
    if run_subset:
        cache_file = "augmented_lfcc_data_cache_subset.pkl"
        print("Running LFCC in subset mode for fast verification...")
    else:
        cache_file = "augmented_lfcc_data_cache.pkl"
        print("Running LFCC in full training mode...")
        
    if os.path.exists(cache_file):
        print(f"Loading cached LFCC data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            X_train, y_train, X_dev, y_dev = pickle.load(f)
    else:
        if run_subset:
            print("Extracting LFCC features from train set subset (with augmentations)...")
            X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR, augment=True, max_files=3000)
            
            print("Extracting LFCC features from dev set subset (without augmentations)...")
            X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR, augment=False, max_files=1000)
        else:
            print("Extracting LFCC features from train set (with augmentations)...")
            X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR, augment=True)
            
            print("Extracting LFCC features from dev set (without augmentations)...")
            X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR, augment=False)
        
        # Кэшируем
        with open(cache_file, 'wb') as f:
            pickle.dump((X_train, y_train, X_dev, y_dev), f)
        print(f"LFCC data successfully cached to {cache_file}.")

    print(f"Train size: {X_train.shape}, Dev size: {X_dev.shape}")

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    # Сохраняем скейлер
    with open("scaler_lfcc.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("LFCC Scaler saved to scaler_lfcc.pkl")

    results = {}

    # 1. Обучение LightGBM
    print("\nTraining LightGBM on augmented LFCC data...")
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
    lgb_model.fit(X_train_scaled, y_train)
    preds = lgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    results['LightGBM (augmented LFCC)'] = eer
    print(f"LGBM Dev EER (LFCC): {eer*100:.2f}%")
    
    # Сохраняем LightGBM
    with open("lgb_model_lfcc_augmented.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    # 2. Обучение XGBoost
    print("\nTraining XGBoost on augmented LFCC data...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train_scaled, y_train)
    preds = xgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    results['XGBoost (augmented LFCC)'] = eer
    print(f"XGBoost Dev EER (LFCC): {eer*100:.2f}%")
    
    # Сохраняем XGBoost
    xgb_model.save_model("xgb_model_lfcc_augmented.json")

    # 3. Обучение CatBoost
    print("\nTraining CatBoost on augmented LFCC data...")
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
    results['CatBoost (augmented LFCC)'] = eer
    print(f"CatBoost Dev EER (LFCC): {eer*100:.2f}%")
    
    # Сохраняем CatBoost
    cat_model.save_model("cat_model_lfcc_augmented.cbm")

    # Сравнение
    print("\n--- Summary Dev EER (LFCC) ---")
    for name, eer_val in results.items():
        print(f"{name}: {eer_val*100:.2f}%")
