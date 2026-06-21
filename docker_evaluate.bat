@echo off
echo ========================================================
echo       ASVspoof: Evaluating Robust Ensemble in Docker
echo ========================================================
echo.

:: Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running or not installed! Please start Docker Desktop first.
    pause
    exit /b 1
)

:: Build the Docker image
echo [1/3] Building Docker image 'asvspoof_exp'...
docker build -t asvspoof_exp .
if %errorlevel% neq 0 (
    echo [ERROR] Docker image build failed!
    exit /b 1
)

:: Unpack data inside Docker
echo [2/3] Preparing and checking evaluation data in Docker...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python scripts/prepare_data.py --eval
if %errorlevel% neq 0 (
    echo [ERROR] Evaluation data preparation failed inside Docker!
    exit /b 1
)

:: Run evaluation inside Docker
echo [3/3] Running robust ensemble evaluation inside container...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/evaluate_robust_ensemble_final.py %*

echo.
echo ========================================================
echo Evaluation completed inside Docker. Summary results saved on host.
echo ========================================================
pause
