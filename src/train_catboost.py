import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import pickle
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

# ---------- Функция для расчёта EER ----------
def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, None

# ---------- Параметры ----------
BASE_DATA = r"D:\фокусы\исследования\antispoof\data\2019\LA"
PROTOCOL_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_cm_protocols")
TRAIN_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_train")
DEV_AUDIO_DIR = os.path.join(BASE_DATA, "ASVspoof2019_LA_dev")

PROTOCOL_TRAIN = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
PROTOCOL_DEV = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

SAMPLE_RATE = 16000
N_MFCC = 30   # Увеличили количество коэффициентов
HOP_LENGTH = 160
WIN_LENGTH = 400

# ---------- Извлечение признаков ----------
def extract_mfcc_stats(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        # Добавим дельты и дельты-дельты для каждого коэффициента
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=WIN_LENGTH, hop_length=HOP_LENGTH)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Объединяем все признаки
        mfcc_full = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Вычисляем статистики для каждого признака (по оси времени)
        stats_list = []
        for c in range(mfcc_full.shape[0]):
            coef = mfcc_full[c, :]
            if coef.size == 0:
                coef = np.zeros(1)
            stats = [
                np.mean(coef), np.std(coef),
                np.min(coef), np.max(coef),
                np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)
            ]
            stats_list.extend(stats)
        return np.array(stats_list)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(N_MFCC * 3 * 7)  # 30 коэффициентов * 3 варианта * 7 статистик = 630 признаков

# ---------- Загрузка данных ----------
def load_data(protocol_file, audio_dir):
    X, y = [], []
    with open(protocol_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label = parts[4]
            label = 0 if label == 'bonafide' else 1
            audio_path = os.path.join(audio_dir, "flac", file_id + '.flac')
            if not os.path.exists(audio_path):
                continue
            feats = extract_mfcc_stats(audio_path)
            X.append(feats)
            y.append(label)
            if idx % 1000 == 0:
                print(f"Processed {idx} files...")
    return np.array(X), np.array(y)

# ---------- Основная часть ----------
if __name__ == "__main__":
    # Загружаем данные (с кешированием)
    cache_file = "catboost_data_cache.pkl"
    if os.path.exists(cache_file):
        print("Loading cached data...")
        with open(cache_file, 'rb') as f:
            X_train, y_train, X_dev, y_dev = pickle.load(f)
    else:
        print("Loading train data...")
        X_train, y_train = load_data(PROTOCOL_TRAIN, TRAIN_AUDIO_DIR)
        print("Loading dev data...")
        X_dev, y_dev = load_data(PROTOCOL_DEV, DEV_AUDIO_DIR)
        with open(cache_file, 'wb') as f:
            pickle.dump((X_train, y_train, X_dev, y_dev), f)
        print("Data cached.")

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    # Обучение CatBoost
    print("\nTraining CatBoost...")
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',          # Можно использовать 'AUC' или 'Accuracy'
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
        thread_count=-1
    )
    
    # Используем Pool для удобства и валидации
    train_pool = Pool(X_train_scaled, label=y_train)
    eval_pool = Pool(X_dev_scaled, label=y_dev)
    
    model.fit(train_pool, eval_set=eval_pool, plot=False)
    
    # Оценка
    preds_proba = model.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds_proba, y_dev)
    print(f"\nEER on dev set: {eer*100:.2f}%")
    
    # Сохраняем модель
    model.save_model("catboost_model.cbm")
    print("Model saved as catboost_model.cbm")