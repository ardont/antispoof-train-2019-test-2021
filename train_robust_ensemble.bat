@echo off
echo ========================================================
echo       ASVspoof: Training Robust Ensemble from Scratch
echo ========================================================
echo.

:: Ensure venv is active or exists
if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment 'venv' not found! Please run 'install_deps.bat' first.
    pause
    exit /b 1
)

call venv\Scripts\activate

:: Check command arguments
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

:: Unpack data if needed
echo [1/6] Preparing and checking training data...
if not exist data\2019\LA.zip (
    if not exist data\2019\LA\ASVspoof2019_LA_train (
        echo [WARNING] Training data archive 'data\2019\LA.zip' not found!
        echo [INFO] Running 'download_data.bat' to download it...
        call download_data.bat
    )
)
python scripts/prepare_data.py --train
if %errorlevel% neq 0 (
    echo [ERROR] Training data extraction failed!
    exit /b 1
)

:: Train base models (MFCC, LFCC, CQCC)
echo [2/6] Training base classifiers (LGBM, XGBoost, CatBoost)...
echo Training MFCC models...
python src/train_robust.py --feature mfcc %SUBSET_FLAG%
echo Training LFCC models...
python src/train_robust.py --feature lfcc %SUBSET_FLAG%
echo Training CQCC models...
python src/train_cqcc_robust.py %SUBSET_FLAG%

:: Train Multi-Layer Perceptrons
echo [3/6] Training Multi-Layer Perceptrons (MLPs)...
python src/train_mlp_robust.py --feature mfcc %SUBSET_FLAG%
python src/train_mlp_robust.py --feature lfcc %SUBSET_FLAG%
python src/train_mlp_robust.py --feature cqcc %SUBSET_FLAG%

:: Train Early Fusion Combined classifiers
echo [4/6] Training combined early fusion classifiers and MLP...
python src/train_combined_final.py %SUBSET_FLAG%
python src/train_mlp_robust.py --feature combined %SUBSET_FLAG%

:: Optionally run hyperparameter sweeps (Tuning)
if "%RUN_TUNING%"=="yes" (
    echo [5/6] Running hyperparameter sweeps using Optuna (30 trials per feature group)...
    echo Tuning MFCC...
    python src/tune_optuna_robust.py --feature mfcc --trials 30 --subsample-size 15000
    echo Tuning LFCC...
    python src/tune_optuna_robust.py --feature lfcc --trials 30 --subsample-size 15000
    echo Tuning CQCC...
    python src/tune_optuna_robust.py --feature cqcc --trials 30 --subsample-size 15000
    echo Tuning Combined...
    python src/tune_optuna_robust.py --feature combined --trials 30 --subsample-size 15000
) else (
    echo [5/6] Skipping Optuna hyperparameter sweeps (subset mode/skipped).
)

:: Train Calibration Stacking
echo [6/6] Training Stacking meta-classifier...
python src/train_stacking_final.py --num-samples %NUM_SAMPLES%

echo ========================================================
echo [SUCCESS] Robust Ensemble training completed successfully!
echo Models and scalers are saved in the project root.
echo ========================================================
pause
