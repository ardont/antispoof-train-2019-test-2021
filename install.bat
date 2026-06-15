@echo off
echo ========================================================
echo       ASVspoof Project: Installing Requirements
echo ========================================================

:: Проверяем наличие Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден! Установите Python 3.9+ и добавьте его в PATH.
    pause
    exit /b 1
)

:: Создаем виртуальное окружение, если его нет
if not exist venv (
    echo [INFO] Создаем виртуальное окружение venv...
    python -m venv venv
)

:: Активируем venv и обновляем pip
echo [INFO] Активируем виртуальное окружение и обновляем pip...
call venv\Scripts\activate
python -m pip install --upgrade pip

:: Устанавливаем зависимости
echo [INFO] Установка зависимостей из requirements.txt...
pip install -r requirements.txt

echo ========================================================
echo [SUCCESS] Все зависимости успешно установлены!
echo Запустите 'run_dashboard_heavy.bat' для запуска детектора.
echo ========================================================
pause
