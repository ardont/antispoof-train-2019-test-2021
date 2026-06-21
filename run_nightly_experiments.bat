@echo off
echo ====================================================
echo     ASVSPOOF 2019-2021 HEAVY NIGHTLY EXPERIMENTS
echo ====================================================
echo Start time: %date% %time%
echo.

:: Ensure virtual environment is present
if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment venv not found! Please create it.
    exit /b 1
)

echo [1/8] Extracting robust CQCC features and training base classifiers...
.\venv\Scripts\python.exe src/train_cqcc_robust.py > train_cqcc_run.log 2>&1
echo Completed CQCC base training. Log saved to train_cqcc_run.log.
echo Current time: %time%
echo.

echo [2/8] Training Multi-Layer Perceptrons (MLPs) on MFCC, LFCC, and CQCC...
.\venv\Scripts\python.exe src/train_mlp_robust.py --feature mfcc > train_mlp_mfcc.log 2>&1
.\venv\Scripts\python.exe src/train_mlp_robust.py --feature lfcc > train_mlp_lfcc.log 2>&1
.\venv\Scripts\python.exe src/train_mlp_robust.py --feature cqcc > train_mlp_cqcc.log 2>&1
echo Completed base MLP models training.
echo Current time: %time%
echo.

echo [3/8] Rebuilding 3-Way Combined features (Early Fusion) and training combined classifiers...
.\venv\Scripts\python.exe src/train_combined_final.py > train_combined_final.log 2>&1
echo Completed 3-Way Combined base models training. Log saved to train_combined_final.log.
echo Current time: %time%
echo.

echo [4/8] Training MLP on 3-Way Combined features...
.\venv\Scripts\python.exe src/train_mlp_robust.py --feature combined > train_mlp_combined.log 2>&1
echo Completed Combined MLP training. Log saved to train_mlp_combined.log.
echo Current time: %time%
echo.

echo [5/8] Running Optuna deep hyperparameter sweeps (100 trials, 30k subsample)...
echo Tuning MFCC...
.\venv\Scripts\python.exe src/tune_optuna_robust.py --feature mfcc --trials 100 --subsample-size 30000 > optuna_mfcc.log 2>&1
echo Tuning LFCC...
.\venv\Scripts\python.exe src/tune_optuna_robust.py --feature lfcc --trials 100 --subsample-size 30000 > optuna_lfcc.log 2>&1
echo Tuning CQCC...
.\venv\Scripts\python.exe src/tune_optuna_robust.py --feature cqcc --trials 100 --subsample-size 30000 > optuna_cqcc.log 2>&1
echo Tuning Combined...
.\venv\Scripts\python.exe src/tune_optuna_robust.py --feature combined --trials 100 --subsample-size 30000 > optuna_combined.log 2>&1
echo Completed all Optuna hyperparameter sweeps.
echo Current time: %time%
echo.

echo [6/8] Re-training Calibrated Stacking meta-model with all 12 optimized models...
.\venv\Scripts\python.exe src/train_stacking_final.py --num-samples 3000 > train_stacking_final.log 2>&1
echo Completed Stacking training. Log saved to train_stacking_final.log.
echo Current time: %time%
echo.

echo [7/8] Running final 12-model Evaluation on 101k ASVspoof 2021 Eval...
.\venv\Scripts\python.exe src/evaluate_robust_ensemble_final.py > evaluate_final_full.log 2>&1
echo Completed final full evaluation. Log saved to evaluate_final_full.log.
echo Current time: %time%
echo.

echo [8/8] Summary of final results:
if exist overnight_results_robust_final.txt (
    type overnight_results_robust_final.txt
) else (
    echo [ERROR] Final summary report overnight_results_robust_final.txt was not generated!
)
echo.

echo ====================================================
echo     ALL NIGHTLY EXPERIMENTS COMPLETED SUCCESSFULLY!
echo ====================================================
echo End time: %date% %time%
