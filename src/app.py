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

# Проверяем наличие обученных моделей
MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
MODELS_LOADED = HEAVY_MODE_SUPPORTED and (MFCC_MODELS_EXIST or LFCC_MODELS_EXIST)

# Настройка интерфейса Streamlit
st.set_page_config(
    page_title="ASVspoof Audio Spoofing Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация в премиальном темном стиле с CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        color: white;
    }
    .header-box {
        background: linear-gradient(135deg, #1f6feb 0%, #111e38 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        border: 1px solid #30363d;
    }
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .status-ok {
        color: #2ea043;
        font-weight: bold;
    }
    .status-warn {
        color: #d29922;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown("""
<div class="header-box">
    <h1 style="color: white; margin: 0;">🎙️ Мультимодальный Детектор Аудио-Подделок (Anti-Spoofing)</h1>
    <p style="color: #8b949e; margin: 10px 0 0 0; font-size: 1.1em;">
        Система верификации подлинности голоса на основе признаков MFCC/LFCC и ансамбля градиентного бустинга (LightGBM, XGBoost, CatBoost).
    </p>
</div>
""", unsafe_allow_html=True)

# Боковое меню / Статус системы
st.sidebar.markdown("## 📊 Статус системы")

if HEAVY_MODE_SUPPORTED:
    st.sidebar.markdown("🔧 **Режим:** <span class='status-ok'>Полнофункциональный (Heavy)</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("🔧 **Режим:** <span class='status-warn'>Легковесный (Light, только визуализация)</span>", unsafe_allow_html=True)
    st.sidebar.info("Для активации полного режима установите все зависимости из requirements.txt.")

# Статус моделей
if MFCC_MODELS_EXIST:
    st.sidebar.markdown("🔊 **MFCC Модели:** <span class='status-ok'>Загружены</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("🔊 **MFCC Модели:** <span class='status-warn'>Не найдены</span>", unsafe_allow_html=True)

if LFCC_MODELS_EXIST:
    st.sidebar.markdown("📶 **LFCC Модели:** <span class='status-ok'>Загружены</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("📶 **LFCC Модели:** <span class='status-warn'>Не найдены</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Текущие метрики (ASVspoof 2021)")
st.sidebar.metric(label="Лучший EER (MFCC XGBoost)", value="17.78%", delta="-30.8% vs Baseline")
st.sidebar.metric(label="Лучший EER (LFCC XGBoost)", value="21.71%", delta="-10.5% vs Baseline")
st.sidebar.metric(label="EER Ансамбля (6 моделей)", value="19.32%")

# Разделение по вкладкам
tab1, tab2, tab3 = st.tabs(["📊 Журнал экспериментов", "🎙️ Проверить аудиофайл", "⚙️ Админ-панель (Обучение)"])

# -----------------------------------------------------------------------------
# TAB 1: Журнал экспериментов
# -----------------------------------------------------------------------------
with tab1:
    st.header("📈 Результаты тестирования гипотез")
    st.write(
        "В процессе работы над проектом мы провели серию экспериментов по обучению на базе данных ASVspoof 2019 LA "
        "и тестированию на полной выборке ASVspoof 2021 LA с помехами канала связи."
    )
    
    # Данные экспериментов
    exp_data = {
        "Эксперимент": ["Exp 1 (MFCC Subset)", "Exp 2 (LFCC Subset)", "Exp 3 (MFCC Full)", "Exp 4 (LFCC Full)", "Exp 5 (Ensemble)", "Exp 6 (CMS Subset Test)"],
        "Дата": ["14.06.2026", "14.06.2026", "15.06.2026", "15.06.2026", "15.06.2026", "15.06.2026"],
        "Нормализация": ["Нет", "Нет", "Нет", "Нет", "Нет", "CMS (Mean Subtraction)"],
        "Модель": ["LightGBM", "LightGBM", "XGBoost", "XGBoost", "6-Model Ensemble", "LightGBM"],
        "EER 2019 Dev (%)": [7.61, 1.10, 5.42, 0.31, 0.31, 2.50],
        "EER 2021 Eval (%)": [15.45, 19.90, 17.78, 21.71, 19.32, 14.97]
    }
    df_exp = pd.DataFrame(exp_data)
    st.dataframe(df_exp, use_container_width=True)
    
    # Визуализация графиков EER
    st.markdown("### Сравнение EER на тестовой выборке ASVspoof 2021 LA")
    
    col1, col2 = st.columns(2)
    with col1:
        # График EER на тесте
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#1f6feb', '#1f6feb', '#238636', '#238636', '#da3633', '#8957e5']
        bars = ax.barh(df_exp["Эксперимент"], df_exp["EER 2021 Eval (%)"], color=colors, height=0.6)
        ax.set_xlabel("Equal Error Rate (EER, %)")
        ax.set_title("EER на полной выборке ASVspoof 2021 Eval (меньше = лучше)")
        ax.axvline(6.0, color="#d29922", linestyle="--", label="Целевой порог (6.0%)")
        ax.legend()
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                    va='center', ha='left', color='white', fontweight='bold')
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#161b22')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('#30363d')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        st.pyplot(fig)

    with col2:
        st.markdown("""
        #### 🔍 Основные инсайты исследований:
        
        1. **Generalization Gap (Разрыв генерализации):**
           Модели, обученные на чистых данных ASVspoof 2019, показывают фантастические результаты на чистом валидационном множестве (например, **LFCC XGBoost выдает EER 0.31%**). Однако при переносе на 2021 год с реальными искажениями каналов (кодеки A-law/U-law, VoIP-сжатие, реверберация) EER падал до **17.78% - 21.71%**.
           
        2. **Успех нормализации CMS (Cepstral Mean Subtraction):**
           Внедрение **CMS** (вычитание среднего значения спектральных коэффициентов для каждого аудиофайла по отдельности) позволило нейтрализовать постоянное частотное искажение телефонного канала. На подмножестве 2021 Eval это позволило обрушить EER с **48.64% до 14.97%**!
           
        3. **Разница между MFCC и LFCC:**
           * **MFCC** лучше улавливает общую форму огибающей и более устойчив к шумам канала связи.
           * **LFCC** имеет линейную шкалу частот, лучше локализует артефакты вокодера/синтеза, но крайне чувствителен к искажениям. Нормализация CMS является для него обязательной.
        """)

# -----------------------------------------------------------------------------
# TAB 2: Тестер аудиофайлов
# -----------------------------------------------------------------------------
with tab2:
    st.header("🎙️ Проверка аудио на подлинность (Deepfake / Spoof Detection)")
    
    if not HEAVY_MODE_SUPPORTED:
        st.warning("⚠️ Данный раздел не поддерживается в легковесном режиме, так как не установлены библиотеки librosa / soundfile / xgboost / catboost.")
    elif not MODELS_LOADED:
        st.warning("⚠️ Не найдены бинарные файлы обученных моделей. Пожалуйста, запустите обучение во вкладке «Админ-панель» или поместите файлы моделей в корневую директорию проекта.")
    else:
        st.write("Загрузите звуковой файл в формате FLAC или WAV, чтобы проверить его на наличие признаков синтеза или клонирования голоса.")
        
        # Загрузка моделей в память (кэшированная)
        @st.cache_resource
        def load_detection_models():
            models = {}
            # Загрузка MFCC моделей
            try:
                if MFCC_MODELS_EXIST:
                    with open(os.path.join(MODELS_DIR, "lgb_model_augmented.pkl"), "rb") as f:
                        models['lgb_mfcc'] = pickle.load(f)
                    models['xgb_mfcc'] = xgb.XGBClassifier()
                    models['xgb_mfcc'].load_model(os.path.join(MODELS_DIR, "xgb_model_augmented.json"))
                    models['cat_mfcc'] = CatBoostClassifier()
                    models['cat_mfcc'].load_model(os.path.join(MODELS_DIR, "cat_model_augmented.cbm"))
                    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
                        models['scaler_mfcc'] = pickle.load(f)
            except Exception as e:
                st.error(f"Ошибка загрузки MFCC моделей: {e}")
                
            # Загрузка LFCC моделей
            try:
                if LFCC_MODELS_EXIST:
                    with open(os.path.join(MODELS_DIR, "lgb_model_lfcc_augmented.pkl"), "rb") as f:
                        models['lgb_lfcc'] = pickle.load(f)
                    models['xgb_lfcc'] = xgb.XGBClassifier()
                    models['xgb_lfcc'].load_model(os.path.join(MODELS_DIR, "xgb_model_lfcc_augmented.json"))
                    models['cat_lfcc'] = CatBoostClassifier()
                    models['cat_lfcc'].load_model(os.path.join(MODELS_DIR, "cat_model_lfcc_augmented.cbm"))
                    with open(os.path.join(MODELS_DIR, "scaler_lfcc.pkl"), "rb") as f:
                        models['scaler_lfcc'] = pickle.load(f)
            except Exception as e:
                st.error(f"Ошибка загрузки LFCC моделей: {e}")
                
            return models
            
        models = load_detection_models()
        
        # Вспомогательная функция извлечения признаков
        def extract_stats_from_audio(y, sr, feature_type='mfcc'):
            if feature_type == 'mfcc':
                feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30, n_fft=400, hop_length=160)
                # Вычисление Delta и Delta-Delta
                feats_delta = librosa.feature.delta(feats)
                feats_delta2 = librosa.feature.delta(feats, order=2)
                feats_full = np.vstack([feats, feats_delta, feats_delta2])
            else:
                feats = extract_lfcc(y, sr=sr, n_lfcc=20, n_filters=128)
                feats_delta = librosa.feature.delta(feats)
                feats_delta2 = librosa.feature.delta(feats, order=2)
                feats_full = np.vstack([feats, feats_delta, feats_delta2])
                
            # Применяем CMS (Cepstral Mean Subtraction) для борьбы с шумом канала
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
        
        # Готовые примеры файлов для быстрого клика
        st.markdown("💡 **Или выберите один из тестовых примеров:**")
        example_cols = st.columns(4)
        selected_example = None
        
        # Ищем несколько flac-файлов в папках данных
        examples_paths = []
        if os.path.exists(os.path.join(MODELS_DIR, "data", "2019", "LA", "ASVspoof2019_LA_dev", "flac")):
            example_dir = os.path.join(MODELS_DIR, "data", "2019", "LA", "ASVspoof2019_LA_dev", "flac")
            examples_paths = [os.path.join(example_dir, f) for f in os.listdir(example_dir)[:4]]
            
        for i, example_col in enumerate(example_cols):
            if i < len(examples_paths):
                basename = os.path.basename(examples_paths[i])
                if example_col.button(f"📄 Пример {i+1} ({basename[:10]}...)"):
                    selected_example = examples_paths[i]
                    
        # Обработка аудиофайла
        audio_to_process = None
        if uploaded_file is not None:
            audio_to_process = uploaded_file
            st.audio(uploaded_file)
        elif selected_example is not None:
            audio_to_process = selected_example
            st.audio(selected_example)
            st.info(f"Выбран пример: `{os.path.basename(selected_example)}`")
            
        if audio_to_process is not None:
            with st.spinner("Извлечение признаков (MFCC + LFCC с нормализацией CMS) и классификация..."):
                try:
                    # Чтение аудио
                    y, sr = sf.read(audio_to_process)
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)
                    if sr != 16000:
                        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                        sr = 16000
                        
                    # Извлекаем признаки
                    mfcc_feats = extract_stats_from_audio(y, sr, 'mfcc')
                    lfcc_feats = extract_stats_from_audio(y, sr, 'lfcc')
                    
                    predictions = {}
                    
                    # Предсказания MFCC моделей
                    if MFCC_MODELS_EXIST:
                        mfcc_scaled = models['scaler_mfcc'].transform(mfcc_feats)
                        predictions['LGBM_MFCC'] = models['lgb_mfcc'].predict_proba(mfcc_scaled)[0, 1]
                        predictions['XGB_MFCC'] = models['xgb_mfcc'].predict_proba(mfcc_scaled)[0, 1]
                        predictions['CAT_MFCC'] = models['cat_mfcc'].predict_proba(mfcc_scaled)[0, 1]
                        
                    # Предсказания LFCC моделей
                    if LFCC_MODELS_EXIST:
                        lfcc_scaled = models['scaler_lfcc'].transform(lfcc_feats)
                        predictions['LGBM_LFCC'] = models['lgb_lfcc'].predict_proba(lfcc_scaled)[0, 1]
                        predictions['XGB_LFCC'] = models['xgb_lfcc'].predict_proba(lfcc_scaled)[0, 1]
                        predictions['CAT_LFCC'] = models['cat_lfcc'].predict_proba(lfcc_scaled)[0, 1]
                        
                    # Расчет усредненного ансамбля
                    all_scores = list(predictions.values())
                    ensemble_score = np.mean(all_scores)
                    
                    # Вывод результатов
                    st.markdown("---")
                    st.markdown("### 🧬 Результат анализа:")
                    
                    col_res1, col_res2 = st.columns([1, 2])
                    
                    with col_res1:
                        # Финальный вердикт
                        # ASVspoof вычисляет EER по порогу, но в целом порог около 0.5 для классификации
                        if ensemble_score > 0.5:
                            st.error("🚨 ВЕРДИКТ: ПОДДЕЛКА (SPOOF)")
                            st.write(f"Вероятность синтеза/клонирования: **{ensemble_score*100:.1f}%**")
                        else:
                            st.success("✅ ВЕРДИКТ: ОРИГИНАЛ (BONA FIDE)")
                            st.write(f"Вероятность оригинальности: **{(1-ensemble_score)*100:.1f}%**")
                            
                    with col_res2:
                        # Показываем скоры всех 6 моделей
                        st.write("**Детализация оценок по моделям (ближе к 1.0 = синтез, ближе к 0.0 = человек):**")
                        df_scores = pd.DataFrame({
                            "Модель": list(predictions.keys()),
                            "Оценка": list(predictions.values())
                        })
                        st.bar_chart(df_scores.set_index("Модель"))
                        
                except Exception as e:
                    st.error(f"Произошла ошибка при обработке файла: {e}")

# -----------------------------------------------------------------------------
# TAB 3: Админ-панель (Обучение)
# -----------------------------------------------------------------------------
with tab3:
    st.header("⚙️ Верификация и запуск переобучения моделей")
    st.write(
        "Вы можете использовать эту панель для верификации честности процесса обучения. "
        "Обучение моделей производится строго на выборках **ASVspoof 2019 LA** (train + dev)."
    )
    
    col_code1, col_code2 = st.columns(2)
    with col_code1:
        st.markdown("### Скрипт обучения MFCC моделей")
        if os.path.exists(os.path.join(MODELS_DIR, "src", "train_augmented.py")):
            with open(os.path.join(MODELS_DIR, "src", "train_augmented.py"), "r", encoding="utf-8") as f:
                code_mfcc = f.read()
            st.code(code_mfcc[:1500] + "\n\n# ... Код обрезан для удобства просмотра ...", language="python")
        else:
            st.error("Файл src/train_augmented.py не найден.")
            
    with col_code2:
        st.markdown("### Скрипт обучения LFCC моделей")
        if os.path.exists(os.path.join(MODELS_DIR, "src", "train_lfcc_augmented.py")):
            with open(os.path.join(MODELS_DIR, "src", "train_lfcc_augmented.py"), "r", encoding="utf-8") as f:
                code_lfcc = f.read()
            st.code(code_lfcc[:1500] + "\n\n# ... Код обрезан для удобства просмотра ...", language="python")
        else:
            st.error("Файл src/train_lfcc_augmented.py не найден.")
            
    st.markdown("---")
    st.markdown("### 🚀 Запуск полного цикла обучения моделей")
    st.write(
        "Внимание: Запуск обучения требует наличия распакованного датасета ASVspoof 2019 в директории `data/2019/LA`. "
        "Обучение выполняется в параллельных процессах, чтобы задействовать все ядра вашего CPU."
    )
    
    btn_col1, btn_col2 = st.columns(2)
    
    # Плейсхолдер для логов
    log_placeholder = st.empty()
    
    if btn_col1.button("🏃 Запустить обучение MFCC моделей"):
        log_placeholder.info("Запуск `src/train_augmented.py`...")
        try:
            # Запускаем скрипт и транслируем вывод
            process = subprocess.Popen(
                [sys.executable, "-u", "src/train_augmented.py"],
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
                output_lines.append(line)
                # Держим последние 25 строк
                log_placeholder.code("".join(output_lines[-25:]), language="text")
                
            process.wait()
            if process.returncode == 0:
                log_placeholder.success("Обучение MFCC моделей успешно завершено! Создан файл моделей и скейлер.")
                # Очищаем кэш ресурсов для обновления моделей
                st.cache_resource.clear()
            else:
                log_placeholder.error(f"Скрипт завершился с кодом ошибки {process.returncode}")
        except Exception as e:
            log_placeholder.error(f"Не удалось запустить скрипт: {e}")
            
    if btn_col2.button("🏃 Запустить обучение LFCC моделей"):
        log_placeholder.info("Запуск `src/train_lfcc_augmented.py`...")
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "src/train_lfcc_augmented.py"],
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
                output_lines.append(line)
                log_placeholder.code("".join(output_lines[-25:]), language="text")
                
            process.wait()
            if process.returncode == 0:
                log_placeholder.success("Обучение LFCC моделей успешно завершено!")
                st.cache_resource.clear()
            else:
                log_placeholder.error(f"Скрипт завершился с кодом ошибки {process.returncode}")
        except Exception as e:
            log_placeholder.error(f"Не удалось запустить скрипт: {e}")
