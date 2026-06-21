@echo off
echo ========================================================
echo       ASVspoof: Training Robust Ensemble in Docker
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

:: Check command arguments for subset mode
set SUBSET_FLAG=
set NUM_SAMPLES=3000
set RUN_TUNING=yes

if "%1"=="--subset" (
    set SUBSET_FLAG=--subset
    set NUM_SAMPLES=500
    set RUN_TUNING=no
    echo [INFO] Running in SUBSET mode (quick check).
    echo [INFO] Optuna sweeps will be skipped for speed.
)

:: Unpack data inside Docker
echo [2/3] Preparing and checking training data in Docker...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python scripts/prepare_data.py --train
if %errorlevel% neq 0 (
    echo [ERROR] Training data preparation failed inside Docker!
    exit /b 1
)

:: Train models inside Docker
echo [3/3] Training robust models inside container...
echo Training MFCC models...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_robust.py --feature mfcc %SUBSET_FLAG%

echo Training LFCC models...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_robust.py --feature lfcc %SUBSET_FLAG%

echo Training CQCC models...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_cqcc_robust.py %SUBSET_FLAG%

echo Training MLPs...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_mlp_robust.py --feature mfcc %SUBSET_FLAG%
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_mlp_robust.py --feature lfcc %SUBSET_FLAG%
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_mlp_robust.py --feature cqcc %SUBSET_FLAG%

echo Training Fusion & Combined models...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_combined_final.py %SUBSET_FLAG%
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_mlp_robust.py --feature combined %SUBSET_FLAG%

:: Tuning step inside Docker
if "%RUN_TUNING%"=="yes" (
    echo Running Optuna hyperparameter sweeps inside container...
    docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/tune_optuna_robust.py --feature mfcc --trials 30 --subsample-size 15000
    docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/tune_optuna_robust.py --feature lfcc --trials 30 --subsample-size 15000
    docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/tune_optuna_robust.py --feature cqcc --trials 30 --subsample-size 15000
    docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/tune_optuna_robust.py --feature combined --trials 30 --subsample-size 15000
)

echo Training Stacking meta-classifier...
docker run --rm -v "%cd%:/workspace" asvspoof_exp python src/train_stacking_final.py --num-samples %NUM_SAMPLES%

echo ========================================================
echo [SUCCESS] Robust Ensemble training in Docker completed successfully!
echo Weights and scalers have been written back to your local folder.
echo ========================================================
pause
