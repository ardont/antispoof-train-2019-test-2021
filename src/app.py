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

def get_git_commits():
    try:
        result = subprocess.run(
            ["git", "log", "-n", "5", "--pretty=format:%h|%an|%ar|%s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=MODELS_DIR
        )
        if result.returncode == 0:
            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|")
                    if len(parts) == 4:
                        commits.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "date": parts[2],
                            "message": parts[3]
                        })
            return commits
    except Exception:
        pass
    return []


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
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Выбор темы в сайдбаре
theme_mode = st.sidebar.selectbox("🎨 Тема интерфейса (Theme Mode)", ["Темная (Soft Dark)", "Светлая (Soft Light)"], index=0)

if theme_mode == "Темная (Soft Dark)":
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700;800&family=Geist+Mono:wght@300;400;500;600;700&display=swap');
        
        /* Базовые стили */
        html, body, [class*="css"], .stApp {
            font-family: 'Geist', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0b0c0e !important;
            color: #f3f4f6 !important;
        }
        
        /* Принудительная читаемость текстов в Streamlit */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
            color: #f3f4f6 !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
            color: #ffffff !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
        }
        
        /* Системные лейблы и тексты виджетов Streamlit */
        label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span, [data-testid="stWidgetLabel"], .stFileUploader label {
            color: #f3f4f6 !important;
            font-weight: 500 !important;
        }
        
        /* Тексты для радиокнопок, чекбоксов и подписей */
        [data-testid="stRadio"] label, [data-testid="stRadio"] p, [data-testid="stRadio"] span,
        [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p, [data-testid="stCheckbox"] span {
            color: #f3f4f6 !important;
        }
        
        /* Сайдбар */
        [data-testid="stSidebar"] {
            background-color: #111216 !important;
            border-right: 1px solid #22242a !important;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: #f3f4f6 !important;
        }
        [data-testid="stSidebar"] .stMarkdown p {
            color: #9ca3af !important;
        }
        
        /* Стилизация метрик Streamlit */
        div[data-testid="stMetric"] {
            background-color: #16171d !important;
            border: 1px solid #22242a !important;
            padding: 16px !important;
            border-radius: 6px !important;
            margin-bottom: 12px !important;
        }
        div[data-testid="stMetricLabel"] > div {
            color: #9ca3af !important;
            font-size: 0.85em !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricValue"] > div {
            color: #ffffff !important;
            font-size: 1.8em !important;
            font-weight: 700 !important;
        }
        
        /* Стилизация алертов/предупреждений */
        div[data-testid="stAlert"] {
            background-color: #16171d !important;
            border: 1px solid #22242a !important;
            border-radius: 6px !important;
        }
        div[data-testid="stAlert"] * {
            color: #f3f4f6 !important;
        }
        
        /* Vercel Nav Bar */
        .vercel-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #22242a;
            padding: 16px 0px;
            margin-bottom: 24px;
            background-color: #0b0c0e;
        }
        .vercel-nav-left {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }
        .vercel-triangle {
            font-size: 1.1em;
            color: #ffffff;
            font-weight: 800;
        }
        .vercel-breadcrumb-divider {
            color: #3f4450;
            font-weight: 300;
        }
        .vercel-project-owner {
            color: #9ca3af !important;
            font-weight: 400;
        }
        .vercel-project-name {
            color: #ffffff !important;
            font-weight: 500;
        }
        .vercel-badge {
            font-size: 0.75em;
            padding: 2px 8px;
            border-radius: 9999px;
            font-weight: 500;
            border: 1px solid #22242a;
            background-color: #111216;
            color: #9ca3af !important;
        }
        .vercel-badge-prod {
            border-color: #3b82f6;
            color: #3b82f6 !important;
            background-color: rgba(59, 130, 246, 0.1);
        }
        .vercel-badge-robust {
            border-color: #10b981;
            color: #10b981 !important;
            background-color: rgba(16, 185, 129, 0.1);
        }
        .vercel-nav-right {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
            color: #9ca3af !important;
        }
        
        /* Vercel Header Box */
        .vercel-header {
            padding: 0px 0px 24px 0px;
            margin-bottom: 32px;
            border-bottom: 1px solid #22242a;
        }
        .vercel-title {
            font-size: 2.4em;
            font-weight: 800;
            letter-spacing: -0.05em;
            color: #ffffff !important;
            margin: 0;
        }
        .vercel-subtitle {
            font-size: 1.05em;
            font-weight: 300;
            color: #9ca3af !important;
            margin: 8px 0 0 0;
            line-height: 1.5;
        }
        
        /* Vercel-Style Карточки */
        .vercel-card {
            background: #16171d;
            border: 1px solid #22242a;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            transition: border-color 0.15s ease, background-color 0.15s ease;
        }
        .vercel-card:hover {
            border-color: #2a2c35;
            background-color: #1b1c24;
        }
        .vercel-title-card {
            font-size: 1.1em;
            font-weight: 600;
            color: #ffffff !important;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Светодиоды статуса */
        .dot {
            height: 8px;
            width: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .dot-green {
            background-color: #10b981;
        }
        .dot-orange {
            background-color: #f59e0b;
        }
        .dot-grey {
            background-color: #3f4450;
        }
        
        .status-text {
            font-size: 0.85em;
            font-weight: 500;
        }
        .status-ready { color: #10b981 !important; }
        .status-missing { color: #9ca3af !important; }
        .status-building { color: #f59e0b !important; }
        
        .metadata-row {
            color: #9ca3af !important;
            font-size: 0.85em;
            margin-top: 8px;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #22242a;
            padding-bottom: 6px;
            font-family: 'Geist Mono', monospace;
        }
        .metadata-row b {
            color: #ffffff !important;
            font-weight: 400;
        }
        
        /* Кнопки Vercel */
        .stButton>button {
            background-color: #ffffff !important;
            color: #0b0c0e !important;
            border-radius: 6px !important;
            border: 1px solid #ffffff !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            font-size: 0.9em !important;
            transition: all 0.15s ease !important;
            width: 100%;
            box-shadow: none !important;
        }
        .stButton>button:hover {
            background-color: #0b0c0e !important;
            color: #ffffff !important;
            border-color: #2a2c35 !important;
        }
        
        /* Терминал Vercel */
        .terminal-box {
            background-color: #111216;
            border: 1px solid #22242a;
            border-radius: 6px;
            padding: 16px;
            font-family: 'Geist Mono', monospace;
            color: #f3f4f6 !important;
            height: 350px;
            overflow-y: auto;
            font-size: 0.85em;
            line-height: 1.6;
        }
        
        /* Изменение дефолтных элементов Streamlit */
        div[data-baseweb="tab-list"] {
            border-bottom: 1px solid #22242a !important;
            gap: 20px !important;
        }
        button[data-baseweb="tab"] {
            font-size: 0.95em !important;
            font-weight: 400 !important;
            padding: 10px 4px !important;
            color: #9ca3af !important;
            background-color: transparent !important;
            border: none !important;
        }
        button[aria-selected="true"] {
            color: #ffffff !important;
            border-bottom: 2px solid #ffffff !important;
        }
        
        /* Input styling */
        input, select, textarea, div[role="listbox"], [data-baseweb="select"] {
            background-color: #111216 !important;
            color: #f3f4f6 !important;
            border: 1px solid #22242a !important;
            border-radius: 6px !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #111216 !important;
            color: #f3f4f6 !important;
        }
        
        /* Стилизация статических HTML-таблиц под Vercel */
        table {
            width: 100% !important;
            border-collapse: collapse !important;
            border: 1px solid #22242a !important;
            background-color: #16171d !important;
            margin-bottom: 24px !important;
            border-radius: 6px !important;
            overflow: hidden !important;
        }
        th {
            background-color: #111216 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.90em !important;
            padding: 12px 16px !important;
            border-bottom: 1px solid #22242a !important;
            text-align: left !important;
        }
        td {
            color: #e5e7eb !important;
            font-size: 0.90em !important;
            padding: 12px 16px !important;
            border-bottom: 1px solid #22242a !important;
            background-color: #16171d !important;
            text-align: left !important;
        }
        tr:hover td {
            color: #ffffff !important;
            background-color: #1b1c24 !important;
        }
        
        /* Дополнительные стили для темной темы */
        .vercel-commit-message {
            color: #ffffff !important;
        }
        .vercel-commit-hash {
            background-color: #111216 !important;
            border: 1px solid #22242a !important;
            color: #f3f4f6 !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: 'Geist Mono', monospace;
        }
        .vercel-secondary-text {
            color: #9ca3af !important;
        }
        .vercel-list {
            color: #9ca3af !important;
        }
        [data-testid="stRadio"] *, [data-testid="stCheckbox"] * {
            color: #f3f4f6 !important;
        }
        [data-testid="stFileUploader"] {
            background-color: #16171d !important;
            border: 1px solid #22242a !important;
            border-radius: 6px !important;
            padding: 8px !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: #111216 !important;
            border: 1px dashed #22242a !important;
            border-radius: 4px !important;
        }
        [data-testid="stFileUploaderDropzone"] div {
            color: #f3f4f6 !important;
        }
        [data-testid="stFileUploaderDropzone"] span {
            color: #9ca3af !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700;800&family=Geist+Mono:wght@300;400;500;600;700&display=swap');
        
        /* Базовые стили */
        html, body, [class*="css"], .stApp {
            font-family: 'Geist', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #fafafa !important;
            color: #18181b !important;
        }
        
        /* Принудительная читаемость текстов в Streamlit */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
            color: #18181b !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
            color: #000000 !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
        }
        
        /* Системные лейблы и тексты виджетов Streamlit */
        label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span, [data-testid="stWidgetLabel"], .stFileUploader label {
            color: #18181b !important;
            font-weight: 500 !important;
        }
        
        /* Тексты для радиокнопок, чекбоксов и подписей */
        [data-testid="stRadio"] label, [data-testid="stRadio"] p, [data-testid="stRadio"] span,
        [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p, [data-testid="stCheckbox"] span {
            color: #18181b !important;
        }
        
        /* Сайдбар */
        [data-testid="stSidebar"] {
            background-color: #f4f4f5 !important;
            border-right: 1px solid #e4e4e7 !important;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: #18181b !important;
        }
        [data-testid="stSidebar"] .stMarkdown p {
            color: #52525b !important;
        }
        
        /* Стилизация метрик Streamlit */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #e4e4e7 !important;
            padding: 16px !important;
            border-radius: 6px !important;
            margin-bottom: 12px !important;
        }
        div[data-testid="stMetricLabel"] > div {
            color: #52525b !important;
            font-size: 0.85em !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricValue"] > div {
            color: #000000 !important;
            font-size: 1.8em !important;
            font-weight: 700 !important;
        }
        
        /* Стилизация алертов/предупреждений */
        div[data-testid="stAlert"] {
            background-color: #ffffff !important;
            border: 1px solid #e4e4e7 !important;
            border-radius: 6px !important;
        }
        div[data-testid="stAlert"] * {
            color: #18181b !important;
        }
        
        /* Vercel Nav Bar */
        .vercel-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e4e4e7;
            padding: 16px 0px;
            margin-bottom: 24px;
            background-color: #fafafa;
        }
        .vercel-nav-left {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }
        .vercel-triangle {
            font-size: 1.1em;
            color: #000000;
            font-weight: 800;
        }
        .vercel-breadcrumb-divider {
            color: #a1a1aa;
            font-weight: 300;
        }
        .vercel-project-owner {
            color: #52525b !important;
            font-weight: 400;
        }
        .vercel-project-name {
            color: #000000 !important;
            font-weight: 500;
        }
        .vercel-badge {
            font-size: 0.75em;
            padding: 2px 8px;
            border-radius: 9999px;
            font-weight: 500;
            border: 1px solid #e4e4e7;
            background-color: #f4f4f5;
            color: #52525b !important;
        }
        .vercel-badge-prod {
            border-color: #3b82f6;
            color: #3b82f6 !important;
            background-color: rgba(59, 130, 246, 0.05);
        }
        .vercel-badge-robust {
            border-color: #10b981;
            color: #10b981 !important;
            background-color: rgba(16, 185, 129, 0.05);
        }
        .vercel-nav-right {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
            color: #52525b !important;
        }
        
        /* Vercel Header Box */
        .vercel-header {
            padding: 0px 0px 24px 0px;
            margin-bottom: 32px;
            border-bottom: 1px solid #e4e4e7;
        }
        .vercel-title {
            font-size: 2.4em;
            font-weight: 800;
            letter-spacing: -0.05em;
            color: #000000 !important;
            margin: 0;
        }
        .vercel-subtitle {
            font-size: 1.05em;
            font-weight: 300;
            color: #52525b !important;
            margin: 8px 0 0 0;
            line-height: 1.5;
        }
        
        /* Vercel-Style Карточки */
        .vercel-card {
            background: #ffffff;
            border: 1px solid #e4e4e7;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            transition: border-color 0.15s ease, background-color 0.15s ease;
        }
        .vercel-card:hover {
            border-color: #a1a1aa;
            background-color: #f4f4f5;
        }
        .vercel-title-card {
            font-size: 1.1em;
            font-weight: 600;
            color: #000000 !important;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Светодиоды статуса */
        .dot {
            height: 8px;
            width: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .dot-green {
            background-color: #10b981;
        }
        .dot-orange {
            background-color: #f59e0b;
        }
        .dot-grey {
            background-color: #a1a1aa;
        }
        
        .status-text {
            font-size: 0.85em;
            font-weight: 500;
        }
        .status-ready { color: #10b981 !important; }
        .status-missing { color: #52525b !important; }
        .status-building { color: #f59e0b !important; }
        
        .metadata-row {
            color: #52525b !important;
            font-size: 0.85em;
            margin-top: 8px;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #e4e4e7;
            padding-bottom: 6px;
            font-family: 'Geist Mono', monospace;
        }
        .metadata-row b {
            color: #000000 !important;
            font-weight: 400;
        }
        
        /* Кнопки Vercel */
        .stButton>button {
            background-color: #18181b !important;
            color: #ffffff !important;
            border-radius: 6px !important;
            border: 1px solid #18181b !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            font-size: 0.9em !important;
            transition: all 0.15s ease !important;
            width: 100%;
            box-shadow: none !important;
        }
        .stButton>button:hover {
            background-color: #ffffff !important;
            color: #18181b !important;
            border-color: #a1a1aa !important;
        }
        
        /* Терминал Vercel - терминал оставляем темным для профессионального вида */
        .terminal-box {
            background-color: #18181b;
            border: 1px solid #18181b;
            border-radius: 6px;
            padding: 16px;
            font-family: 'Geist Mono', monospace;
            color: #f3f4f6 !important;
            height: 350px;
            overflow-y: auto;
            font-size: 0.85em;
            line-height: 1.6;
        }
        
        /* Изменение дефолтных элементов Streamlit */
        div[data-baseweb="tab-list"] {
            border-bottom: 1px solid #e4e4e7 !important;
            gap: 20px !important;
        }
        button[data-baseweb="tab"] {
            font-size: 0.95em !important;
            font-weight: 400 !important;
            padding: 10px 4px !important;
            color: #52525b !important;
            background-color: transparent !important;
            border: none !important;
        }
        button[aria-selected="true"] {
            color: #000000 !important;
            border-bottom: 2px solid #000000 !important;
        }
        
        /* Input styling */
        input, select, textarea, div[role="listbox"], [data-baseweb="select"] {
            background-color: #ffffff !important;
            color: #18181b !important;
            border: 1px solid #d4d4d8 !important;
            border-radius: 6px !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #18181b !important;
        }
        
        /* Стилизация статических HTML-таблиц под Vercel */
        table {
            width: 100% !important;
            border-collapse: collapse !important;
            border: 1px solid #e4e4e7 !important;
            background-color: #ffffff !important;
            margin-bottom: 24px !important;
            border-radius: 6px !important;
            overflow: hidden !important;
        }
        th {
            background-color: #f4f4f5 !important;
            color: #000000 !important;
            font-weight: 600 !important;
            font-size: 0.90em !important;
            padding: 12px 16px !important;
            border-bottom: 1px solid #e4e4e7 !important;
            text-align: left !important;
        }
        td {
            color: #27272a !important;
            font-size: 0.90em !important;
            padding: 12px 16px !important;
            border-bottom: 1px solid #e4e4e7 !important;
            background-color: #ffffff !important;
            text-align: left !important;
        }
        tr:hover td {
            color: #000000 !important;
            background-color: #f4f4f5 !important;
        }
        
        /* Дополнительные стили для светлой темы */
        .vercel-commit-message {
            color: #000000 !important;
        }
        .vercel-commit-hash {
            background-color: #f4f4f5 !important;
            border: 1px solid #e4e4e7 !important;
            color: #18181b !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: 'Geist Mono', monospace;
        }
        .vercel-secondary-text {
            color: #52525b !important;
        }
        .vercel-list {
            color: #52525b !important;
        }
        [data-testid="stRadio"] *, [data-testid="stCheckbox"] * {
            color: #18181b !important;
        }
        [data-testid="stFileUploader"] {
            background-color: #ffffff !important;
            border: 1px solid #e4e4e7 !important;
            border-radius: 6px !important;
            padding: 8px !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: #fafafa !important;
            border: 1px dashed #d4d4d8 !important;
            border-radius: 4px !important;
        }
        [data-testid="stFileUploaderDropzone"] div {
            color: #18181b !important;
        }
        [data-testid="stFileUploaderDropzone"] span {
            color: #52525b !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Верхняя панель навигации Vercel
st.markdown("""
<div class="vercel-nav">
    <div class="vercel-nav-left">
        <span class="vercel-triangle">▲</span>
        <span class="vercel-breadcrumb-divider">/</span>
        <span class="vercel-project-owner">ardont</span>
        <span class="vercel-breadcrumb-divider">/</span>
        <span class="vercel-project-name">antispoof-detector</span>
        <span class="vercel-badge vercel-badge-prod">Production</span>
    </div>
    <div class="vercel-nav-right">
        <span class="vercel-status-dot dot-green"></span>
        <span class="vercel-status-text" style="color: #10b981;">Active</span>
    </div>
</div>
<div class="vercel-header">
    <h1 class="vercel-title">ASVspoof Audio Spoofing Detector</h1>
    <p class="vercel-subtitle">
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
    
    exp_data = {
        "Эксперимент": [
            "Exp 1: MFCC Baseline (Full)", 
            "Exp 2: LFCC Baseline (Full)", 
            "Exp 3: Baseline Ensemble (6 models)", 
            "Exp 4: CMS Normalization (Subset)", 
            "Exp 5: Robust Ensemble (Subset)",
            "Exp 6: Robust Stacking (Optuna tuned)",
            "Exp 7: Optimized Weighted Ensemble (Optuna)"
        ],
        "Частоты (Гц)": ["0 - 8000", "0 - 8000", "0 - 8000", "0 - 8000", "300 - 3400", "300 - 3400", "300 - 3400"],
        "Нормализация": ["Нет", "Нет", "Нет", "CMS", "CMS", "CMS", "CMS"],
        "Аугментация": ["Нет", "Нет", "Нет", "Нет", "Telephony (G.711/722)", "Telephony", "Telephony"],
        "EER 2019 Dev (%)": [5.42, 0.31, 0.31, 2.50, 6.20, 10.42, 6.63],
        "EER 2021 Eval (%)": [17.78, 21.71, 19.32, 14.97, 11.52, 9.03, 8.64]
    }
    df_exp = pd.DataFrame(exp_data)
    st.table(df_exp)
    
    st.markdown("#### Сравнение EER на полной выборке ASVspoof 2021 LA")
    
    col_chart, col_desc = st.columns([3, 2])
    with col_chart:
        is_dark = (theme_mode == "Темная (Soft Dark)")
        fig_bg = '#16171d' if is_dark else '#ffffff'
        axis_color = '#9ca3af' if is_dark else '#52525b'
        border_color = '#22242a' if is_dark else '#e4e4e7'
        legend_bg = '#111216' if is_dark else '#f4f4f5'
        legend_text = 'white' if is_dark else 'black'
        bar_text_color = '#f3f4f6' if is_dark else '#18181b'
        
        if is_dark:
            colors = ['#22242a', '#22242a', '#3f4450', '#5c6370', '#8a8f98', '#10b981', '#3b82f6']
        else:
            colors = ['#e4e4e7', '#e4e4e7', '#a1a1aa', '#71717a', '#52525b', '#10b981', '#0070f3']
            
        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = ax.barh(df_exp["Эксперимент"], df_exp["EER 2021 Eval (%)"], color=colors, height=0.55)
        ax.set_xlabel("Equal Error Rate (EER, %)", color=axis_color, fontsize=10)
        ax.axvline(5.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Целевой EER (5.0%)")
        ax.legend(facecolor=legend_bg, edgecolor=border_color, labelcolor=legend_text)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.4, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                    va='center', ha='left', color=bar_text_color, fontweight='bold', fontsize=9)
                    
        fig.patch.set_facecolor(fig_bg)
        ax.set_facecolor(fig_bg)
        ax.spines['bottom'].set_color(border_color)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color(border_color)
        ax.tick_params(colors=axis_color, labelsize=9)
        ax.xaxis.label.set_color(axis_color)
        ax.title.set_color(bar_text_color)
        st.pyplot(fig)
        
    with col_desc:
        st.markdown(f"""
        <div class="vercel-card" style="padding: 20px;">
            <h4 class="vercel-title-card" style="margin-top:0; font-size:1.15em; letter-spacing:-0.02em;">🔑 Ключевые выводы доменной адаптации:</h4>
            <ul class="vercel-list" style="padding-left: 18px; font-size: 0.92em; line-height: 1.6; margin-bottom: 0;">
                <li style="margin-bottom: 8px;"><b>Доменный сдвиг:</b> Чистые модели переобучаются на высокие частоты (>3400 Гц). В реальных звонках 2021 года этих частот нет, из-за чего EER достигал 21%.</li>
                <li style="margin-bottom: 8px;"><b>CMS Нормализация:</b> Вычитание среднего кепстрального спектра нейтрализует аддитивные искажения АЧХ каналов связи, снижая ошибку до 14.97%.</li>
                <li style="margin-bottom: 8px;"><b>Частотный срез (300-3400 Гц):</b> Ограничение частот заставляет бустинги искать паттерны там, где звук проходит сквозь кодеки телефонии.</li>
                <li style="margin-bottom: 8px;"><b>Телефония-аугментация + Optuna:</b> Подбор гиперпараметров снизил ошибку EER с 11.52% до <b>8.64%</b> на взвешенном ансамбле.</li>
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
                    
                    is_dark = (theme_mode == "Темная (Soft Dark)")
                    wave_bg = '#0b0c0e' if is_dark else '#fafafa'
                    wave_color = '#ffffff' if is_dark else '#18181b'
                    
                    # Отрисовка Waveform в стиле Vercel (минималистичный монохром)
                    st.markdown("#### 🌊 Waveform (Форма звуковой волны)")
                    fig, ax = plt.subplots(figsize=(10, 2.2))
                    time_axis = np.linspace(0, len(y) / sr, num=len(y))
                    ax.plot(time_axis, y, color=wave_color, alpha=0.9, linewidth=1)
                    ax.fill_between(time_axis, y, color=wave_color, alpha=0.1)
                    ax.set_facecolor(wave_bg)
                    fig.patch.set_facecolor(wave_bg)
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
                        
                        is_dark = (theme_mode == "Темная (Soft Dark)")
                        slider_bg = '#22242a' if is_dark else '#e4e4e7'
                        slider_handle = '#ffffff' if is_dark else '#000000'
                        secondary_text_color = '#888888' if is_dark else '#6b7280'
                        
                        st.markdown(f"""
                        <div class="vercel-card" style="padding: 25px; text-align: center;">
                            <h3 style="color: {verdict_color}; margin-top: 0; font-size: 1.4em; font-weight:800;">{verdict_class}</h3>
                            <p style="color: {secondary_text_color}; font-size: 0.95em; margin-bottom: 0;">Ансамбль взвесил оценки всех активных моделей бустинга.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Градиентный спидометр / индикатор в стиле Vercel (плоский серый бэкграунд слайдера)
                        st.markdown(f"""
                        <div style="margin-top: 25px; margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; font-weight: 500; font-size: 0.85em; margin-bottom: 6px; color: {secondary_text_color};">
                                <span>Human (0.0)</span>
                                <span>Deepfake (1.0)</span>
                            </div>
                            <div style="height: 6px; width: 100%; background: {slider_bg}; border-radius: 3px; position: relative;">
                                <div style="position: absolute; left: calc({ensemble_score * 100}% - 4px); top: -3px; height: 12px; width: 8px; background: {slider_handle}; border-radius: 2px; transition: left 0.3s ease;"></div>
                            </div>
                            <div style="text-align: center; margin-top: 12px; font-weight: 700; font-size: 1.1em; color: {verdict_color}">
                                Вероятность спуфинга: {ensemble_score*100:.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_chart:
                        fig_bg = '#16171d' if is_dark else '#ffffff'
                        axis_color = '#9ca3af' if is_dark else '#52525b'
                        border_color = '#22242a' if is_dark else '#e4e4e7'
                        text_color = '#ffffff' if is_dark else '#18181b'
                        
                        st.markdown(f"<p style='font-weight:600; font-size:0.95em; color:{text_color}; margin-bottom:12px;'>Детализация вероятностей по моделям (ближе к 1.0 = синтез):</p>", unsafe_allow_html=True)
                        df_scores = pd.DataFrame({
                            "Модель": list(predictions.keys()),
                            "Оценка": list(predictions.values())
                        })
                        
                        # Отрисовка красивых баров через matplotlib в динамическом стиле
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
                                    va='center', ha='left', color=text_color, fontweight='500', fontsize=8)
                                    
                        fig.patch.set_facecolor(fig_bg)
                        ax.set_facecolor(fig_bg)
                        ax.spines['bottom'].set_color(border_color)
                        ax.spines['top'].set_color('none')
                        ax.spines['right'].set_color('none')
                        ax.spines['left'].set_color(border_color)
                        ax.tick_params(colors=axis_color, labelsize=8)
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
        badge_cls = "badge-robust" if is_robust_style else ""
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
                <div class="vercel-title-card">
                    <span>🎙️ {title}</span>
                    <span class="vercel-badge {badge_cls}">{badge_lbl}</span>
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
    # История деплоев из коммитов Git
    st.markdown("#### 🚀 История развертываний (Git Deployments)")
    commits = get_git_commits()
    if commits:
        for commit in commits:
            st.markdown(f"""
            <div class="vercel-card" style="margin-bottom: 12px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="vercel-commit-message" style="font-weight: 600;">{commit['message']}</div>
                        <div class="vercel-secondary-text" style="font-size: 0.85em; margin-top: 4px; font-family: 'Geist Mono', monospace;">
                            <code class="vercel-commit-hash">{commit['hash']}</code>
                            &nbsp;•&nbsp; {commit['author']} &nbsp;•&nbsp; {commit['date']}
                        </div>
                    </div>
                    <div>
                        <span class="vercel-badge vercel-badge-prod" style="border-color: #10b981; color: #10b981; background-color: rgba(16, 185, 129, 0.1);">Ready</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Нет доступной истории коммитов Git.")
        
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
        log_view.markdown('<div class="terminal-box"><span style="color: #444;">[SYSTEM]</span> Terminal idle. Ready for build trigger...</div>', unsafe_allow_html=True)
        
        if trigger_btn:
            subset_flag = ["--subset"] if use_subset else []
            t_start = time.strftime('%H:%M:%S')
            
            # Определяем команду запуска
            if feature_choice == "Робастные MFCC модели":
                cmd = [sys.executable, "-u", "src/train_robust.py", "--feature", "mfcc"] + subset_flag
                log_view.markdown(f'<div class="terminal-box"><span style="color: #888;">[{t_start}]</span> <span style="color: #0070f3;">●</span> Initializing Robust MFCC pipeline...</div>', unsafe_allow_html=True)
            elif feature_choice == "Робастные LFCC модели":
                cmd = [sys.executable, "-u", "src/train_robust.py", "--feature", "lfcc"] + subset_flag
                log_view.markdown(f'<div class="terminal-box"><span style="color: #888;">[{t_start}]</span> <span style="color: #0070f3;">●</span> Initializing Robust LFCC pipeline...</div>', unsafe_allow_html=True)
            else:
                cmd = [sys.executable, "-u", "src/train_combined_robust.py"] + subset_flag
                log_view.markdown(f'<div class="terminal-box"><span style="color: #888;">[{t_start}]</span> <span style="color: #0070f3;">●</span> Initializing Combined Early Fusion pipeline...</div>', unsafe_allow_html=True)
                
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
                    t_curr = time.strftime('%H:%M:%S')
                    # Очищаем перевод строки и добавляем таймстамп
                    clean_line = line.strip()
                    if clean_line:
                        output_lines.append(f'<span style="color: #444;">[{t_curr}]</span> {clean_line}')
                    
                    # Ограничиваем лог последними 30 строками
                    log_html = "<br>".join(output_lines[-30:])
                    log_view.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
                    
                process.wait()
                t_end = time.strftime('%H:%M:%S')
                if process.returncode == 0:
                    st.success("🎉 Сборка завершена успешно! Модели развернуты на диске.")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    log_view.markdown(f'<div class="terminal-box"><span style="color: #444;">[{t_end}]</span> <span style="color: #ef4444;">●</span> Build failed with code {process.returncode}</div>', unsafe_allow_html=True)
            except Exception as e:
                t_err = time.strftime('%H:%M:%S')
                log_view.markdown(f'<div class="terminal-box"><span style="color: #444;">[{t_err}]</span> <span style="color: #ef4444;">●</span> Subprocess error: {str(e)}</div>', unsafe_allow_html=True)

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
