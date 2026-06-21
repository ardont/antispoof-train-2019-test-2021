@echo off
echo ========================================================
echo       ASVspoof: Installing GPU-enabled Dependencies
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
    echo [INFO] Creating Python virtual environment (venv)...
    python -m venv venv
)

:: Activate virtual environment and upgrade pip
echo [INFO] Activating virtual environment and upgrading pip...
call venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel

:: Install CUDA-enabled PyTorch (CUDA 12.1)
echo [INFO] Installing CUDA-enabled PyTorch (cu121)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install other requirements
echo [INFO] Installing remaining packages from requirements.txt...
pip install -r requirements.txt

echo ========================================================
echo [SUCCESS] GPU Dependencies installed successfully!
echo.
echo Use:
echo  - 'train_robust_ensemble.bat' to run training (will auto-detect GPU).
echo  - 'docker_train.bat' to run training in Docker (with GPU support).
echo ========================================================
pause
