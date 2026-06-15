import os
import sys
import subprocess
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Добавляем корневой путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Пытаемся импортировать тяжелые библиотеки для полноценного инференса
try:
    import librosa
    import soundfile as sf
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import pickle
    
    # Импортируем наши экстракторы признаков
    from utils.lfcc import extract_lfcc
    from utils.augmentations import augment_audio
    
    HEAVY_MODE_SUPPORTED = True
except ImportError:
    HEAVY_MODE_SUPPORTED = False

# Определение директории моделей
MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Проверяем наличие обычных (baseline/augmented) моделей
MFCC_MODELS_EXIST = (
    os.path.exists(os.path.join(MODELS_DIR, "lgb_model_augmented.pkl")) and
    os.path.exists(os.path.join(MODELS_DIR, "xgb_model_augmented.json")) and
    os.path.exists(os.path.join(MODELS_DIR, "cat_model_augmented.cbm"))
)
LFCC_MODELS_EXIST = (
    os.path.exists(os.path.join(MODELS_DIR, "lgb_model_lfcc_augmented.pkl")) and
    os.path.exists(os.path.join(MODELS_DIR, "xgb_model_lfcc_augmented.json")) and
    os.path.exists(os.path.join(MODELS_DIR, "cat_model_lfcc_augmented.cbm"))
)

# Проверяем наличие новых робастных моделей (телефонная полоса + CMS)
ROBUST_MFCC_EXIST = (
    os.path.exists(os.path.join(MODELS_DIR, "lgb_model_mfcc_robust.pkl")) and
    os.path.exists(os.path.join(MODELS_DIR, "xgb_model_mfcc_robust.json")) and
    os.path.exists(os.path.join(MODELS_DIR, "cat_model_mfcc_robust.cbm"))
)
ROBUST_LFCC_EXIST = (
    os.path.exists(os.path.join(MODELS_DIR, "lgb_model_lfcc_robust.pkl")) and
    os.path.exists(os.path.join(MODELS_DIR, "xgb_model_lfcc_robust.json")) and
    os.path.exists(os.path.join(MODELS_DIR, "cat_model_lfcc_robust.cbm"))
)
ROBUST_COMBINED_EXIST = (
    os.path.exists(os.path.join(MODELS_DIR, "lgb_model_combined_robust.pkl")) and
    os.path.exists(os.path.join(MODELS_DIR, "xgb_model_combined_robust.json")) and
    os.path.exists(os.path.join(MODELS_DIR, "cat_model_combined_robust.cbm"))
)

MODELS_LOADED = HEAVY_MODE_SUPPORTED and (
    MFCC_MODELS_EXIST or LFCC_MODELS_EXIST or 
    ROBUST_MFCC_EXIST or ROBUST_LFCC_EXIST or ROBUST_COMBINED_EXIST
)

# Настройка интерфейса Streamlit
st.set_page_config(
    page_title="Vercel App | ASVspoof Anti-Spoofing Admin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация под Vercel (темная премиальная тема, Outfit/JetBrains Mono шрифты, градиенты, ховеры)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    /* Базовые стили */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #050507 !important;
        color: #f4f4f5 !important;
    }
    
    /* Сайдбар */
    [data-testid="stSidebar"] {
        background-color: #09090b !important;
        border-right: 1px solid #1e1e24 !important;
    }
    
    /* Хедер */
    .header-box {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0) 100%);
        padding: 40px 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: left;
    }
    .header-box h1 {
        font-weight: 800;
        letter-spacing: -0.04em;
        font-size: 2.8em;
        background: linear-gradient(to right, #ffffff, #a3a3a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    /* Vercel-Style Карточки */
    .vercel-card {
        background: rgba(18, 18, 22, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
    }
    .vercel-card:hover {
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), 0 0 1px 1px rgba(255, 255, 255, 0.1) inset;
        transform: translateY(-2px);
    }
    .vercel-title {
        font-size: 1.25em;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge {
        font-size: 0.75em;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        background-color: rgba(255, 255, 255, 0.1);
        color: #a1a1aa;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .badge-robust {
        background-color: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Светодиоды статуса */
    .dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .dot-green {
        background-color: #10b981;
        box-shadow: 0 0 12px #10b981;
        animation: pulse-green 2s infinite;
    }
    .dot-orange {
        background-color: #f59e0b;
        box-shadow: 0 0 12px #f59e0b;
        animation: pulse-orange 2s infinite;
    }
    .dot-grey {
        background-color: #71717a;
    }
    
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-orange {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(245, 158, 11, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }
    
    .status-text {
        font-size: 0.9em;
        font-weight: 600;
    }
    .status-ready { color: #10b981; }
    .status-missing { color: #71717a; }
    .status-building { color: #f59e0b; }
    
    .metadata-row {
        color: #a1a1aa;
        font-size: 0.88em;
        margin-top: 6px;
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        padding-bottom: 4px;
    }
    .metadata-row b {
        color: #e4e4e7;
    }
    
    /* Кнопки Vercel */
    .stButton>button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 6px !important;
        border: 1px solid #ffffff !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95em !important;
        transition: all 0.2s ease !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(255,255,255,0.1);
    }
    .stButton>button:hover {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-color: #3f3f46 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.8);
    }
    
    /* Терминал */
    .terminal-box {
        background-color: #050507;
        border: 1px solid #1e1e24;
        border-radius: 8px;
        padding: 20px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        color: #10b981;
        height: 350px;
        overflow-y: auto;
        font-size: 0.88em;
        line-height: 1.6;
        box-shadow: inset 0 4px 20px rgba(0,0,0,0.6);
    }
    
    /* Изменение дефолтных элементов Streamlit */
    div[data-baseweb="tab-list"] {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        gap: 24px !important;
    }
    button[data-baseweb="tab"] {
        font-size: 1.05em !important;
        font-weight: 500 !important;
        padding: 12px 4px !important;
        color: #71717a !important;
        background-color: transparent !important;
        border: none !important;
    }
    button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown("""
<div class="header-box">
    <h1>🎙️ ASVspoof Audio Spoofing Detector</h1>
    <p style="color: #a1a1aa; margin: 10px 0 0 0; font-size: 1.15em; font-weight: 300;">
        Профессиональная система верификации голоса против дипфейков на основе признаков MFCC/LFCC и бустинг-ансамбля.
    </p>
</div>
""", unsafe_allow_html=True)

# Сайдбар: Системный статус
st.sidebar.markdown("<h2 style='font-size:1.5em; letter-spacing:-0.02em;'>⚡ System Status</h2>", unsafe_allow_html=True)

if HEAVY_MODE_SUPPORTED:
    st.sidebar.markdown("<div style='margin-bottom:15px;'><span class='dot dot-green'></span><span class='status-ready status-text'>Heavy Mode Active</span></div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='margin-bottom:15px;'><span class='dot dot-orange'></span><span class='status-building status-text'>Light Mode (Read Only)</span></div>", unsafe_allow_html=True)
    st.sidebar.warning("Установите librosa, soundfile, xgboost, lightgbm и catboost для полного инференса.")

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-size:1.1em; color:#a1a1aa;'>Target Metrics (2021 Eval)</h3>", unsafe_allow_html=True)
st.sidebar.metric(label="Robust Ensemble EER", value="11.52%", delta="-7.8% vs Baseline")
st.sidebar.metric(label="Baseline Ensemble EER", value="19.32%", delta="Reference")
st.sidebar.info("Цель проекта — снизить EER ниже 5% на датасете ASVspoof 2021 LA с помощью доменной адаптации.")

# Разделение по вкладкам в стиле Vercel
tab1, tab2, tab3 = st.tabs(["📈 Журнал экспериментов", "🎙️ Тестер аудиофайлов", "⚙️ Панель развертывания"])

# -----------------------------------------------------------------------------
# TAB 1: Журнал экспериментов
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📈 Результаты тестирования гипотез")
    st.write(
        "Ниже приведена таблица сравнения различных подходов. Обучение велось на чистом наборе **ASVspoof 2019 LA**, "
        "а оценка — на тяжелом тестовом наборе **ASVspoof 2021 LA** (содержащем реальные телефонные каналы и сжатие)."
    )
    
    # Данные экспериментов
    exp_data = {
        "Эксперимент": [
            "Exp 1: MFCC Baseline (Full)", 
            "Exp 2: LFCC Baseline (Full)", 
            "Exp 3: Baseline Ensemble (6 models)", 
            "Exp 4: CMS Normalization (Subset)", 
            "Exp 5: Robust Ensemble (MFCC+LFCC Subset)"
        ],
        "Частоты (Гц)": ["0 - 8000", "0 - 8000", "0 - 8000", "0 - 8000", "300 - 3400"],
        "Нормализация": ["Нет", "Нет", "Нет", "CMS", "CMS"],
        "Аугментация": ["Нет", "Нет", "Нет", "Нет", "Telephony (G.711/722)"],
        "EER 2019 Dev (%)": [5.42, 0.31, 0.31, 2.50, 6.20],
        "EER 2021 Eval (%)": [17.78, 21.71, 19.32, 14.97, 11.52]
    }
    df_exp = pd.DataFrame(exp_data)
    st.dataframe(df_exp, use_container_width=True)
    
    st.markdown("#### Сравнение EER на полной выборке ASVspoof 2021 LA")
    
    col_chart, col_desc = st.columns([3, 2])
    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Фирменные цвета Vercel/Indigo
        colors = ['#3f3f46', '#3f3f46', '#71717a', '#6366f1', '#10b981']
        bars = ax.barh(df_exp["Эксперимент"], df_exp["EER 2021 Eval (%)"], color=colors, height=0.55)
        ax.set_xlabel("Equal Error Rate (EER, %)", color='#a1a1aa', fontsize=10)
        ax.axvline(5.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Целевой EER (5.0%)")
        ax.legend(facecolor='#050507', edgecolor='none', labelcolor='white')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.4, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                    va='center', ha='left', color='#ffffff', fontweight='bold', fontsize=9)
                    
        fig.patch.set_facecolor('#050507')
        ax.set_facecolor('#09090b')
        ax.spines['bottom'].set_color('#1e1e24')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('#1e1e24')
        ax.tick_params(colors='#a1a1aa', labelsize=9)
        ax.xaxis.label.set_color('#a1a1aa')
        ax.title.set_color('#ffffff')
        st.pyplot(fig)

    with col_desc:
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 20px; border-radius: 10px;">
            <h4 style="margin-top:0; color:#ffffff; font-size:1.15em;">🔑 Ключевые выводы доменной адаптации:</h4>
            <ul style="padding-left: 18px; color: #a1a1aa; font-size: 0.92em; line-height: 1.6;">
                <li style="margin-bottom: 8px;"><b>Доменный сдвиг:</b> Чистые модели переобучаются на высокие частоты (>3400 Гц). В реальных звонках 2021 года этих частот нет, из-за чего EER достигал 21%.</li>
                <li style="margin-bottom: 8px;"><b>CMS Нормализация:</b> Вычитание среднего значения кепстральных коэффициентов нейтрализует аддитивные искажения АЧХ каналов связи, снижая ошибку до 14.97%.</li>
                <li style="margin-bottom: 8px;"><b>Частотный срез (300-3400 Гц):</b> Ограничение диапазона заставляет классификатор искать признаки только там, где они физически могут пройти сквозь кодеки.</li>
                <li><b>Телефония-аугментация:</b> Искусственное сжатие u-law/A-law и ресемплинг 8 кГц во время обучения снизили EER до <b>11.52%</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: Тестер аудиофайлов
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🎙️ Тестирование аудиофайла в реальном времени")
    
    if not HEAVY_MODE_SUPPORTED:
        st.warning("⚠️ Раздел отключен: отсутствуют библиотеки librosa / soundfile / xgboost / catboost.")
    elif not MODELS_LOADED:
        st.warning("⚠️ Модели не обнаружены в корневой директории. Перейдите во вкладку «Панель развертывания» для их сборки.")
    else:
        st.write("Загрузите аудиофайл в формате WAV или FLAC или выберите тестовый пример из базы данных ASVspoof.")
        
        # Загрузка моделей в память (кэшированная)
        @st.cache_resource
        def load_detection_models():
            models = {}
            # Загрузка робастных моделей
            try:
                if ROBUST_MFCC_EXIST:
                    with open(os.path.join(MODELS_DIR, "lgb_model_mfcc_robust.pkl"), "rb") as f:
                        models['lgb_mfcc'] = pickle.load(f)
                    models['xgb_mfcc'] = xgb.XGBClassifier()
                    models['xgb_mfcc'].load_model(os.path.join(MODELS_DIR, "xgb_model_mfcc_robust.json"))
                    models['cat_mfcc'] = CatBoostClassifier()
                    models['cat_mfcc'].load_model(os.path.join(MODELS_DIR, "cat_model_mfcc_robust.cbm"))
                    with open(os.path.join(MODELS_DIR, "scaler_mfcc_robust.pkl"), "rb") as f:
                        models['scaler_mfcc'] = pickle.load(f)
                    models['has_mfcc'] = True
                elif MFCC_MODELS_EXIST:
                    with open(os.path.join(MODELS_DIR, "lgb_model_augmented.pkl"), "rb") as f:
                        models['lgb_mfcc'] = pickle.load(f)
                    models['xgb_mfcc'] = xgb.XGBClassifier()
                    models['xgb_mfcc'].load_model(os.path.join(MODELS_DIR, "xgb_model_augmented.json"))
                    models['cat_mfcc'] = CatBoostClassifier()
                    models['cat_mfcc'].load_model(os.path.join(MODELS_DIR, "cat_model_augmented.cbm"))
                    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
                        models['scaler_mfcc'] = pickle.load(f)
                    models['has_mfcc'] = True
                else:
                    models['has_mfcc'] = False
            except Exception as e:
                st.error(f"Ошибка загрузки MFCC моделей: {e}")
                models['has_mfcc'] = False
                
            try:
                if ROBUST_LFCC_EXIST:
                    with open(os.path.join(MODELS_DIR, "lgb_model_lfcc_robust.pkl"), "rb") as f:
                        models['lgb_lfcc'] = pickle.load(f)
                    models['xgb_lfcc'] = xgb.XGBClassifier()
                    models['xgb_lfcc'].load_model(os.path.join(MODELS_DIR, "xgb_model_lfcc_robust.json"))
                    models['cat_lfcc'] = CatBoostClassifier()
                    models['cat_lfcc'].load_model(os.path.join(MODELS_DIR, "cat_model_lfcc_robust.cbm"))
                    with open(os.path.join(MODELS_DIR, "scaler_lfcc_robust.pkl"), "rb") as f:
                        models['scaler_lfcc'] = pickle.load(f)
                    models['has_lfcc'] = True
                elif LFCC_MODELS_EXIST:
                    with open(os.path.join(MODELS_DIR, "lgb_model_lfcc_augmented.pkl"), "rb") as f:
                        models['lgb_lfcc'] = pickle.load(f)
                    models['xgb_lfcc'] = xgb.XGBClassifier()
                    models['xgb_lfcc'].load_model(os.path.join(MODELS_DIR, "xgb_model_lfcc_augmented.json"))
                    models['cat_lfcc'] = CatBoostClassifier()
                    models['cat_lfcc'].load_model(os.path.join(MODELS_DIR, "cat_model_lfcc_augmented.cbm"))
                    with open(os.path.join(MODELS_DIR, "scaler_lfcc.pkl"), "rb") as f:
                        models['scaler_lfcc'] = pickle.load(f)
                    models['has_lfcc'] = True
                else:
                    models['has_lfcc'] = False
            except Exception as e:
                st.error(f"Ошибка загрузки LFCC моделей: {e}")
                models['has_lfcc'] = False

            try:
                if ROBUST_COMBINED_EXIST:
                    with open(os.path.join(MODELS_DIR, "lgb_model_combined_robust.pkl"), "rb") as f:
                        models['lgb_comb'] = pickle.load(f)
                    models['xgb_comb'] = xgb.XGBClassifier()
                    models['xgb_comb'].load_model(os.path.join(MODELS_DIR, "xgb_model_combined_robust.json"))
                    models['cat_comb'] = CatBoostClassifier()
                    models['cat_comb'].load_model(os.path.join(MODELS_DIR, "cat_model_combined_robust.cbm"))
                    with open(os.path.join(MODELS_DIR, "scaler_combined_robust.pkl"), "rb") as f:
                        models['scaler_comb'] = pickle.load(f)
                    models['has_combined'] = True
                else:
                    models['has_combined'] = False
            except Exception as e:
                st.error(f"Ошибка загрузки Combined моделей: {e}")
                models['has_combined'] = False
                
            return models
            
        models = load_detection_models()
        
        # Извлечение статистик
        def extract_stats_from_audio(y, sr, feature_type='mfcc'):
            is_robust = ROBUST_MFCC_EXIST or ROBUST_LFCC_EXIST or ROBUST_COMBINED_EXIST
            fmin_val = 300 if is_robust else 0
            fmax_val = 3400 if is_robust else None
            
            if feature_type == 'mfcc':
                if is_robust:
                    feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30, n_fft=400, hop_length=160, fmin=fmin_val, fmax=fmax_val)
                else:
                    feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30, n_fft=400, hop_length=160)
                feats_delta = librosa.feature.delta(feats)
                feats_delta2 = librosa.feature.delta(feats, order=2)
                feats_full = np.vstack([feats, feats_delta, feats_delta2])
            else:
                feats = extract_lfcc(y, sr=sr, n_lfcc=20, n_filters=128, fmin=fmin_val, fmax=fmax_val)
                feats_delta = librosa.feature.delta(feats)
                feats_delta2 = librosa.feature.delta(feats, order=2)
                feats_full = np.vstack([feats, feats_delta, feats_delta2])
                
            # CMS
            mean = np.mean(feats_full, axis=1, keepdims=True)
            feats_full = feats_full - mean
            
            # Извлекаем 7 статистик
            stats_list = []
            for c in range(feats_full.shape[0]):
                coef = feats_full[c, :]
                if coef.size == 0:
                    coef = np.zeros(1)
                stats = [
                    np.mean(coef), np.std(coef),
                    np.min(coef), np.max(coef),
                    np.percentile(coef, 25), np.percentile(coef, 50), np.percentile(coef, 75)
                ]
                stats_list.extend(stats)
            return np.array(stats_list).reshape(1, -1)
            
        uploaded_file = st.file_uploader("Выберите аудиофайл...", type=["flac", "wav"])
        
        # Примеры для быстрого тестирования
        st.markdown("💡 **Или выберите тестовый пример:**")
        example_cols = st.columns(4)
        selected_example = None
        
        examples_paths = []
        # Ищем в стандартных путях
        search_dirs = [
            os.path.join(MODELS_DIR, "data", "2019", "LA", "ASVspoof2019_LA_dev", "flac"),
            os.path.join(MODELS_DIR, "data", "2019", "LA", "ASVspoof2019_LA_train", "flac")
        ]
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith('.flac')]
                if len(files) > 0:
                    examples_paths = files[:4]
                    break
                    
        for i, example_col in enumerate(example_cols):
            if i < len(examples_paths):
                basename = os.path.basename(examples_paths[i])
                if example_col.button(f"📄 Пример {i+1} ({basename})"):
                    selected_example = examples_paths[i]
                    
        # Обработка выбранного аудио
        audio_to_process = None
        if uploaded_file is not None:
            audio_to_process = uploaded_file
            st.audio(uploaded_file)
        elif selected_example is not None:
            audio_to_process = selected_example
            st.audio(selected_example)
            st.info(f"Выбран пример: `{os.path.basename(selected_example)}`")
            
        if audio_to_process is not None:
            with st.spinner("Загрузка и отрисовка аудиоволны..."):
                try:
                    # Чтение файла
                    y, sr = sf.read(audio_to_process)
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)
                    if sr != 16000:
                        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                        sr = 16000
                    
                    # Отрисовка Waveform в стиле Vercel (индиго-синий градиент)
                    st.markdown("#### 🌊 Waveform (Форма звуковой волны)")
                    fig, ax = plt.subplots(figsize=(10, 2.2))
                    time_axis = np.linspace(0, len(y) / sr, num=len(y))
                    ax.plot(time_axis, y, color='#6366f1', alpha=0.8, linewidth=1)
                    ax.fill_between(time_axis, y, color='#6366f1', alpha=0.15)
                    ax.set_facecolor('#050507')
                    fig.patch.set_facecolor('#050507')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Извлечение фичей
                    mfcc_feats = extract_stats_from_audio(y, sr, 'mfcc')
                    lfcc_feats = extract_stats_from_audio(y, sr, 'lfcc')
                    
                    predictions = {}
                    
                    # Предсказания MFCC моделей
                    if models.get('has_mfcc', False):
                        mfcc_scaled = models['scaler_mfcc'].transform(mfcc_feats)
                        predictions['LGBM_MFCC'] = models['lgb_mfcc'].predict_proba(mfcc_scaled)[0, 1]
                        predictions['XGB_MFCC'] = models['xgb_mfcc'].predict_proba(mfcc_scaled)[0, 1]
                        predictions['CAT_MFCC'] = models['cat_mfcc'].predict_proba(mfcc_scaled)[0, 1]
                        
                    # Предсказания LFCC моделей
                    if models.get('has_lfcc', False):
                        lfcc_scaled = models['scaler_lfcc'].transform(lfcc_feats)
                        predictions['LGBM_LFCC'] = models['lgb_lfcc'].predict_proba(lfcc_scaled)[0, 1]
                        predictions['XGB_LFCC'] = models['xgb_lfcc'].predict_proba(lfcc_scaled)[0, 1]
                        predictions['CAT_LFCC'] = models['cat_lfcc'].predict_proba(lfcc_scaled)[0, 1]

                    # Предсказания Combined моделей
                    if models.get('has_combined', False):
                        comb_feats = np.hstack([mfcc_feats, lfcc_feats])
                        comb_scaled = models['scaler_comb'].transform(comb_feats)
                        predictions['LGBM_COMB'] = models['lgb_comb'].predict_proba(comb_scaled)[0, 1]
                        predictions['XGB_COMB'] = models['xgb_comb'].predict_proba(comb_scaled)[0, 1]
                        predictions['CAT_COMB'] = models['cat_comb'].predict_proba(comb_scaled)[0, 1]
                        
                    all_scores = list(predictions.values())
                    ensemble_score = np.mean(all_scores) if len(all_scores) > 0 else 0.0
                    
                    st.markdown("---")
                    st.markdown("### 🧬 Результат анализа классификатора")
                    
                    col_verdict, col_chart = st.columns([2, 3])
                    
                    with col_verdict:
                        is_spoof = ensemble_score > 0.5
                        verdict_class = "🚨 ВЕРДИКТ: ПОДДЕЛКА (SPOOF)" if is_spoof else "✅ ВЕРДИКТ: ОРИГИНАЛ (BONA FIDE)"
                        verdict_color = "#ef4444" if is_spoof else "#10b981"
                        
                        st.markdown(f"""
                        <div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 25px; border-radius: 12px; text-align: center;">
                            <h3 style="color: {verdict_color}; margin-top: 0; font-size: 1.4em; font-weight:800;">{verdict_class}</h3>
                            <p style="color: #a1a1aa; font-size: 0.95em;">Ансамбль взвесил оценки всех активных моделей бустинга.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Градиентный спидометр / индикатор
                        st.markdown(f"""
                        <div style="margin-top: 25px; margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.85em; margin-bottom: 6px; color: #a1a1aa;">
                                <span>Human (0.0)</span>
                                <span>Deepfake (1.0)</span>
                            </div>
                            <div style="height: 10px; width: 100%; background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%); border-radius: 5px; position: relative;">
                                <div style="position: absolute; left: calc({ensemble_score * 100}% - 6px); top: -3px; height: 16px; width: 12px; background: white; border-radius: 3px; box-shadow: 0 2px 8px rgba(0,0,0,0.6); border: 1.5px solid #050507; transition: left 0.3s ease;"></div>
                            </div>
                            <div style="text-align: center; margin-top: 12px; font-weight: 700; font-size: 1.1em; color: {verdict_color}">
                                Вероятность спуфинга: {ensemble_score*100:.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_chart:
                        st.markdown("<p style='font-weight:600; font-size:0.95em; color:#e4e4e7; margin-bottom:12px;'>Детализация вероятностей по моделям (ближе к 1.0 = синтез):</p>", unsafe_allow_html=True)
                        df_scores = pd.DataFrame({
                            "Модель": list(predictions.keys()),
                            "Оценка": list(predictions.values())
                        })
                        
                        # Отрисовка красивых баров через matplotlib в темном стиле
                        fig, ax = plt.subplots(figsize=(8, 4))
                        bar_colors = ['#ef4444' if v > 0.5 else '#10b981' for v in df_scores["Оценка"]]
                        y_pos = np.arange(len(df_scores))
                        bars = ax.barh(y_pos, df_scores["Оценка"], color=bar_colors, height=0.5)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(df_scores["Модель"])
                        ax.set_xlim(0, 1.0)
                        
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                                    va='center', ha='left', color='#ffffff', fontweight='bold', fontsize=8)
                                    
                        fig.patch.set_facecolor('#050507')
                        ax.set_facecolor('#09090b')
                        ax.spines['bottom'].set_color('#1e1e24')
                        ax.spines['top'].set_color('none')
                        ax.spines['right'].set_color('none')
                        ax.spines['left'].set_color('#1e1e24')
                        ax.tick_params(colors='#a1a1aa', labelsize=8)
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Произошла ошибка при обработке файла: {e}")

# -----------------------------------------------------------------------------
# TAB 3: Панель развертывания
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### ⚙️ Мониторинг развертывания моделей (Deployments)")
    st.write("Каждая карточка отражает статус обучения и метаданные конкретного семейства классификаторов на диске.")
    
    # Вспомогательная функция для генерации карточки Vercel
    def render_vercel_card(title, exist_flag, main_file, scaler_file, is_robust_style=True):
        status_dot = "dot-green" if exist_flag else "dot-grey"
        status_lbl = "Ready" if exist_flag else "Not Deployed"
        status_cls = "status-ready" if exist_flag else "status-missing"
        badge_cls = "badge badge-robust" if is_robust_style else "badge"
        badge_lbl = "ROBUST (CMS)" if is_robust_style else "BASELINE"
        
        mod_time = "—"
        size_str = "—"
        
        target_file_path = os.path.join(MODELS_DIR, main_file)
        if os.path.exists(target_file_path):
            t_stamp = os.path.getmtime(target_file_path)
            mod_time = time.strftime('%d.%m.%Y %H:%M:%S', time.localtime(t_stamp))
            f_size = os.path.getsize(target_file_path) / (1024 * 1024)
            size_str = f"{f_size:.2f} MB"
            
        st.markdown(f"""
        <div class="vercel-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div class="vercel-title">
                    <span>🎙️ {title}</span>
                    <span class="{badge_cls}">{badge_lbl}</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span class="dot {status_dot}"></span>
                    <span class="{status_cls} status-text">{status_lbl}</span>
                </div>
            </div>
            <div class="metadata-row"><span>Scaler:</span> <b>{scaler_file}</b></div>
            <div class="metadata-row"><span>Active File:</span> <b>{main_file}</b></div>
            <div class="metadata-row"><span>Last Modified:</span> <b>{mod_time}</b></div>
            <div class="metadata-row"><span>Artifact Size:</span> <b>{size_str}</b></div>
        </div>
        """, unsafe_allow_html=True)

    # Карточки Robust моделей
    st.markdown("#### Робастные модели доменной адаптации")
    card_col1, card_col2, card_col3 = st.columns(3)
    with card_col1:
        render_vercel_card("Robust Combined Ensemble", ROBUST_COMBINED_EXIST, "xgb_model_combined_robust.json", "scaler_combined_robust.pkl")
    with card_col2:
        render_vercel_card("Robust MFCC Classifier", ROBUST_MFCC_EXIST, "xgb_model_mfcc_robust.json", "scaler_mfcc_robust.pkl")
    with card_col3:
        render_vercel_card("Robust LFCC Classifier", ROBUST_LFCC_EXIST, "xgb_model_lfcc_robust.json", "scaler_lfcc_robust.pkl")
        
    # Карточки Baseline моделей
    st.markdown("#### Базовые модели (Legacy / Full Band)")
    card_col4, card_col5 = st.columns(2)
    with card_col4:
        render_vercel_card("Baseline MFCC Classifier", MFCC_MODELS_EXIST, "xgb_model_augmented.json", "scaler.pkl", is_robust_style=False)
    with card_col5:
        render_vercel_card("Baseline LFCC Classifier", LFCC_MODELS_EXIST, "xgb_model_lfcc_augmented.json", "scaler_lfcc.pkl", is_robust_style=False)
        
    st.markdown("---")
    st.markdown("### 🛠️ Панель сборки и переобучения (Build Trigger)")
    
    build_col1, build_col2 = st.columns([1, 2])
    with build_col1:
        st.write("Настройте параметры сборки пайплайна:")
        use_subset = st.checkbox("Быстрый режим (Subset, 3000 файлов)", value=True, 
                                 help="Если включено, обучение займет 2-3 минуты на подмножестве для верификации. Отключите для полного обучения на 25k файлах.")
        
        feature_choice = st.radio("Какой стек переобучить:", 
                                  ["Робастные MFCC модели", "Робастные LFCC модели", "Объединенные (Combined) модели"])
        
        trigger_btn = st.button("🚀 Запустить Сборку (Deploy System)")
        
    with build_col2:
        st.write("Логи сборки в реальном времени (Build Logs):")
        log_view = st.empty()
        log_view.markdown('<div class="terminal-box">Terminal idle. Ready for build trigger...</div>', unsafe_allow_html=True)
        
        if trigger_btn:
            subset_flag = ["--subset"] if use_subset else []
            
            # Определяем команду запуска
            if feature_choice == "Робастные MFCC модели":
                cmd = [sys.executable, "-u", "src/train_robust.py", "--feature", "mfcc"] + subset_flag
                log_view.markdown('<div class="terminal-box">● Initializing Robust MFCC pipeline...</div>', unsafe_allow_html=True)
            elif feature_choice == "Робастные LFCC модели":
                cmd = [sys.executable, "-u", "src/train_robust.py", "--feature", "lfcc"] + subset_flag
                log_view.markdown('<div class="terminal-box">● Initializing Robust LFCC pipeline...</div>', unsafe_allow_html=True)
            else:
                cmd = [sys.executable, "-u", "src/train_combined_robust.py"] + subset_flag
                log_view.markdown('<div class="terminal-box">● Initializing Combined Early Fusion pipeline...</div>', unsafe_allow_html=True)
                
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="ignore"
                )
                
                output_lines = []
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    output_lines.append(line.replace("\n", "<br>"))
                    # Ограничиваем лог последними 30 строками
                    log_html = "".join(output_lines[-30:])
                    log_view.markdown(f'<div class="terminal-box">● Build Output:<br>{log_html}</div>', unsafe_allow_html=True)
                    
                process.wait()
                if process.returncode == 0:
                    st.success("🎉 Сборка завершена успешно! Модели развернуты на диске.")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    log_view.markdown(f'<div class="terminal-box" style="color: #ef4444;">● Build failed with code {process.returncode}</div>', unsafe_allow_html=True)
            except Exception as e:
                log_view.markdown(f'<div class="terminal-box" style="color: #ef4444;">● Subprocess error: {str(e)}</div>', unsafe_allow_html=True)

    # Просмотр исходного кода в скрытых экспандерах
    st.markdown("---")
    st.markdown("### 🔍 Проверка исходного кода (Code Inspector)")
    
    with st.expander("Посмотреть исходный код train_robust.py"):
        if os.path.exists(os.path.join(MODELS_DIR, "src", "train_robust.py")):
            with open(os.path.join(MODELS_DIR, "src", "train_robust.py"), "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        else:
            st.error("Файл src/train_robust.py не найден.")
            
    with st.expander("Посмотреть исходный код train_combined_robust.py"):
        if os.path.exists(os.path.join(MODELS_DIR, "src", "train_combined_robust.py")):
            with open(os.path.join(MODELS_DIR, "src", "train_combined_robust.py"), "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        else:
            st.error("Файл src/train_combined_robust.py не найден.")
            
    with st.expander("Посмотреть исходный код evaluate_robust_ensemble.py"):
        if os.path.exists(os.path.join(MODELS_DIR, "src", "evaluate_robust_ensemble.py")):
            with open(os.path.join(MODELS_DIR, "src", "evaluate_robust_ensemble.py"), "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        else:
            st.error("Файл src/evaluate_robust_ensemble.py не найден.")
