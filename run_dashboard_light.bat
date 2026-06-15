@echo off
echo ========================================================
echo   ASVspoof Project: Running Dashboard (LIGHTWEIGHT MODE)
echo ========================================================
echo [INFO] В этом режиме устанавливаются только библиотеки отображения результатов (без ML/DL).

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден! Установите Python и добавьте его в PATH.
    pause
    exit /b 1
)

if not exist venv (
    echo [INFO] Создаем виртуальное окружение venv...
    python -m venv venv
)

call venv\Scripts\activate
echo [INFO] Обновляем pip и устанавливаем легковесные библиотеки (Streamlit, Pandas, Matplotlib)...
python -m pip install --upgrade pip
pip install streamlit pandas matplotlib

echo [INFO] Запуск дашборда...
streamlit run src/app.py
pause
