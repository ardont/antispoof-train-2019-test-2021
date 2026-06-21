import sys
import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import pickle
import warnings
import time

# Отключаем предупреждения библиотек
warnings.filterwarnings("ignore")

# Добавляем корневой путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer
from utils.augmentations import augment_audio
from utils.lfcc import extract_lfcc

# Ограничиваем многопоточность внутри библиотек для стабильности
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print("[INFO] NVIDIA GPU detected via PyTorch. Enabling GPU training.")
            return True
    except ImportError:
        pass
    return False

# -----------------------------------------------------------------------------
# 📁 Конфигурация путей и параметров
# -----------------------------------------------------------------------------
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
HOP_LENGTH = 160
WIN_LENGTH = 400

# Телефония срезает низкие и высокие частоты
FMIN = 300
FMAX = 3400

# -----------------------------------------------------------------------------
# 🧠 Извлечение устойчивых признаков с полосовой фильтрацией и CMS
# -----------------------------------------------------------------------------
def extract_robust_features(y, sr, feature_type='mfcc'):
    """
    Извлекает признаки (MFCC или LFCC) только в частотной полосе телефона (300-3400 Гц)
    и применяет CMS (Cepstral Mean Subtraction) для компенсации сдвига домена.
    """
    if feature_type == 'mfcc':
        # n_mfcc = 30
        # Ограничиваем спектр fmin и fmax внутри librosa.feature.mfcc
        feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30,
                                    n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
                                    fmin=FMIN, fmax=FMAX)
    else:
        # n_lfcc = 20
        # Наш кастомный LFCC теперь поддерживает fmin и fmax
        feats = extract_lfcc(y, sr=sr, n_lfcc=20, n_filters=128,
                             n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
                             fmin=FMIN, fmax=FMAX)
        
    # Вычисляем дельты (первую и вторую производные по времени)
    feats_delta = librosa.feature.delta(feats)
    feats_delta2 = librosa.feature.delta(feats, order=2)
    
    # Объединяем статические коэффициенты с их динамикой
    feats_full = np.vstack([feats, feats_delta, feats_delta2])
    
    # CMS: Вычитаем среднее по времени для компенсации АЧХ телефонного канала
    mean = np.mean(feats_full, axis=1, keepdims=True)
    feats_full = feats_full - mean
    
    # Извлекаем 7 устойчивых статистик по времени
    stats_list = []
    for c in range(feats_full.shape[0]):
        coef = feats_full[c, :]
        if coef.size == 0:
            coef = np.zeros(1)
        stats = [
            np.mean(coef), np.std(coef),  # mean будет около 0 из-за CMS, но std важен
            np.min(coef), np.max(coef),
            np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)
        ]
        stats_list.extend(stats)
        
    return np.array(stats_list)

def extract_from_file(file_path, feature_type='mfcc', augment=False):
    """
    Загружает аудиофайл и извлекает для него оригинальные и (опционально) аугментированные фичи.
    """
    try:
        # Чтение аудио с помощью soundfile для совместимости и скорости
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1) # Стерео в моно
            
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            
        # 1. Извлечение оригинальных признаков
        feats_orig = extract_robust_features(y, sr, feature_type)
        
        if not augment:
            return [feats_orig]
            
        # 2. Применение цепочки кодек-аугментаций и извлечение робастных признаков
        y_aug = augment_audio(y, sr)
        feats_aug = extract_robust_features(y_aug, sr, feature_type)
        
        return [feats_orig, feats_aug]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Заглушка из нулей при ошибках
        n_feats = (30 * 3 * 7) if feature_type == 'mfcc' else (20 * 3 * 7)
        zeros = np.zeros(n_feats)
        if augment:
            return [zeros, zeros]
        return [zeros]

# -----------------------------------------------------------------------------
# 🔄 Параллельный загрузчик данных
# -----------------------------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor

def process_line_wrapper(args):
    line, audio_dir, feature_type, augment = args
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    file_id = parts[1]
    label = parts[4]
    label_num = 0 if label == 'bonafide' else 1
    audio_path = os.path.join(audio_dir, "flac", file_id + '.flac')
    if not os.path.exists(audio_path):
        return None
        
    feats_list = extract_from_file(audio_path, feature_type=feature_type, augment=augment)
    return feats_list, label_num

def load_data(protocol_file, audio_dir, feature_type='mfcc', augment=False, max_files=None):
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
    print(f"Using {max_workers} CPU worker processes for extraction...")
    
    args_list = [(line, audio_dir, feature_type, augment) for line in lines]

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

# -----------------------------------------------------------------------------
# 🚀 Точка входа в скрипт
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Classifier Training for ASVspoof 2019")
    parser.main_args = parser.add_argument("--feature", type=str, default="mfcc", choices=["mfcc", "lfcc"],
                        help="Type of features to extract: mfcc or lfcc")
    parser.add_argument("--subset", action="store_true", help="Run in subset mode for fast verification")
    args = parser.parse_args()
    
    feature_type = args.feature.lower()
    run_subset = args.subset
    use_gpu = check_gpu()
    
    # Задаем имена новых робастных кешей
    if run_subset:
        cache_file = f"robust_{feature_type}_cache_subset.pkl"
        print(f"--- RUNNING IN SUBSET MODE for {feature_type.upper()} ---")
    else:
        cache_file = f"robust_{feature_type}_cache.pkl"
        print(f"--- RUNNING IN FULL TRAINING MODE for {feature_type.upper()} ---")
        
    if os.path.exists(cache_file):
        print(f"Loading cached robust data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            X_train, y_train, X_dev, y_dev = pickle.load(f)
    else:
        if run_subset:
            print(f"Extracting robust {feature_type} train set (with telephony augmentations)...")
            X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR, feature_type=feature_type, augment=True, max_files=3000)
            
            print(f"Extracting robust {feature_type} dev set (clean)...")
            X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR, feature_type=feature_type, augment=False, max_files=1000)
        else:
            print(f"Extracting robust {feature_type} train set (with telephony augmentations)...")
            X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR, feature_type=feature_type, augment=True)
            
            print(f"Extracting robust {feature_type} dev set (clean)...")
            X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR, feature_type=feature_type, augment=False)
            
        with open(cache_file, 'wb') as f:
            pickle.dump((X_train, y_train, X_dev, y_dev), f)
        print(f"Robust {feature_type} data cached to {cache_file}.")

    print(f"Loaded Train size: {X_train.shape}, Dev size: {X_dev.shape}")
    
    # -----------------------------------------------------------------------------
    # ⚖️ Стандартизация признаков (StandardScaler)
    # -----------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    scaler_file = f"scaler_{feature_type}_robust.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Robust scaler saved to {scaler_file}")
    
    results = {}
    
    # -----------------------------------------------------------------------------
    # 🌲 1. Обучение LightGBM
    # -----------------------------------------------------------------------------
    print(f"\nTraining LightGBM on robust {feature_type} features...")
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
    results['LightGBM (robust)'] = eer
    print(f"LGBM 2019 Dev EER: {eer*100:.2f}%")
    
    with open(f"lgb_model_{feature_type}_robust.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
        
    # -----------------------------------------------------------------------------
    # 🌲 2. Обучение XGBoost
    # -----------------------------------------------------------------------------
    print(f"\nTraining XGBoost on robust {feature_type} features...")
    xgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
        'use_label_encoder': False
    }
    if use_gpu:
        xgb_params['device'] = 'cuda'
        xgb_params['tree_method'] = 'hist'
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_dev_scaled, y_dev)],
        verbose=False
    )
    
    preds = xgb_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    results['XGBoost (robust)'] = eer
    print(f"XGBoost 2019 Dev EER: {eer*100:.2f}%")
    
    xgb_model.save_model(f"xgb_model_{feature_type}_robust.json")
    
    # -----------------------------------------------------------------------------
    # 🌲 3. Обучение CatBoost
    # -----------------------------------------------------------------------------
    print(f"\nTraining CatBoost on robust {feature_type} features...")
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'early_stopping_rounds': 50,
        'random_seed': 42,
        'verbose': 100
    }
    if use_gpu:
        cat_params['task_type'] = 'GPU'
    else:
        cat_params['thread_count'] = -1
    cat_model = CatBoostClassifier(**cat_params)
    
    train_pool = Pool(X_train_scaled, label=y_train)
    eval_pool = Pool(X_dev_scaled, label=y_dev)
    cat_model.fit(train_pool, eval_set=eval_pool, plot=False)
    
    preds = cat_model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    results['CatBoost (robust)'] = eer
    print(f"CatBoost 2019 Dev EER: {eer*100:.2f}%")
    
    cat_model.save_model(f"cat_model_{feature_type}_robust.cbm")
    
    print("\n--- Summary Dev 2019 EER ---")
    for name, eer_val in results.items():
        print(f"{name}: {eer_val*100:.2f}%")
