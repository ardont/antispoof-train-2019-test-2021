@echo off
echo ========================================================
echo   ASVspoof Project: Running Dashboard (HEAVYWEIGHT MODE)
echo ========================================================

if not exist venv (
    echo [ERROR] Виртуальное окружение не найдено! Запустите сначала 'install.bat'.
    pause
    exit /b 1
)

call venv\Scripts\activate
echo [INFO] Запуск дашборда с полной поддержкой моделей...
streamlit run src/app.py
pause
