@echo off
echo ========================================================
echo       ASVspoof: Installing Dependencies and Setup
echo ========================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo [INFO] Creating Python virtual environment [venv]...
    python -m venv venv
)

:: Activate virtual environment and upgrade pip
echo [INFO] Activating virtual environment and upgrading pip...
call venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel

:: Install dependencies from requirements.txt
echo [INFO] Installing required packages from requirements.txt...
pip install -r requirements.txt

echo ========================================================
echo [SUCCESS] Dependencies installed successfully!
echo.
echo Use:
echo  - 'train_robust_ensemble.bat' to train the model from scratch.
echo  - 'evaluate_robust_ensemble.bat' to run the evaluation.
echo ========================================================
pause
