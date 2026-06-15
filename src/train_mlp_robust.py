import sys
import os
import argparse
import pickle
import warnings
import time
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_eer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP Classifier on Robust Features")
    parser.add_argument("--feature", type=str, default="mfcc", choices=["mfcc", "lfcc", "combined"],
                        help="Feature type to use: mfcc, lfcc or combined")
    parser.add_argument("--subset", action="store_true", help="Run in subset mode for fast verification")
    args = parser.parse_args()
    
    feature_type = args.feature.lower()
    run_subset = args.subset
    
    # Имена файлов кэша
    if run_subset:
        cache_file = f"robust_{feature_type}_cache_subset.pkl"
        print(f"--- RUNNING MLP IN SUBSET MODE FOR {feature_type.upper()} ---")
    else:
        cache_file = f"robust_{feature_type}_cache.pkl"
        print(f"--- RUNNING MLP IN FULL MODE FOR {feature_type.upper()} ---")
        
    if not os.path.exists(cache_file):
        print(f"[ERROR] Cache file not found: {cache_file}")
        print("Пожалуйста, сначала запустите train_robust.py или train_combined_robust.py для генерации кэша.")
        sys.exit(1)
        
    print(f"Loading cached robust data from {cache_file}...")
    with open(cache_file, 'rb') as f:
        X_train, y_train, X_dev, y_dev = pickle.load(f)
        
    print(f"Loaded Train size: {X_train.shape}, Dev size: {X_dev.shape}")
    
    # ⚖️ Стандартизация признаков (используем существующий скейлер или обучаем заново)
    scaler_file = f"scaler_{feature_type}_robust.pkl"
    if os.path.exists(scaler_file):
        print(f"Loading existing scaler from {scaler_file}...")
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        X_train_scaled = scaler.transform(X_train)
        X_dev_scaled = scaler.transform(X_dev)
    else:
        print(f"Fitting new scaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_dev_scaled = scaler.transform(X_dev)
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
            
    # 🧠 Настройка MLP-нейросети
    print(f"\nTraining Multi-Layer Perceptron (MLP) on {feature_type.upper()}...")
    
    # Используем глубокую структуру для нелинейных границ решений
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.001,           # L2 регуляризация для предотвращения переообучения
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,   # Валидационный сет внутри обучения
        validation_fraction=0.1,
        random_state=42,
        verbose=True
    )
    
    start_time = time.time()
    mlp.fit(X_train_scaled, y_train)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s.")
    
    # Метрика на Dev 2019
    preds = mlp.predict_proba(X_dev_scaled)[:, 1]
    eer, _ = compute_eer(preds, y_dev)
    print(f"MLP {feature_type.upper()} 2019 Dev EER: {eer*100:.2f}%")
    
    # Сохраняем модель
    model_file = f"mlp_model_{feature_type}_robust.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(mlp, f)
    print(f"MLP model saved to {model_file}")
