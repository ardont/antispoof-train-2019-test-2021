@echo off
echo ====================================================
echo     ASVSPOOF 2019-2021 NIGHTLY EXPERIMENTS RUNNER
echo ====================================================
echo Start time: %date% %time%
echo.

:: Убеждаемся, что виртуальное окружение активно
if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment venv not found! Please create it.
    exit /b 1
)

echo [1/5] Running FULL EVALUATION of current ensemble on 101k ASVspoof 2021 Eval...
.\venv\Scripts\python.exe src/evaluate_robust_ensemble.py > evaluation_ensemble_full.log 2>&1
echo Completed full evaluation. Log saved to evaluation_ensemble_full.log.
echo Current time: %time%
echo.

echo [2/5] Running Optuna hyperparameter tuning for COMBINED features (30 trials)...
.\venv\Scripts\python.exe src/tune_optuna_robust.py --feature combined --trials 30 > optuna_combined_tuning.log 2>&1
echo Completed Optuna Combined tuning. Log saved to optuna_combined_tuning.log.
echo Current time: %time%
echo.

echo [3/5] Running Optuna hyperparameter tuning for MFCC features (30 trials)...
.\venv\Scripts\python.exe src/tune_optuna_robust.py --feature mfcc --trials 30 > optuna_mfcc_tuning.log 2>&1
echo Completed Optuna MFCC tuning. Log saved to optuna_mfcc_tuning.log.
echo Current time: %time%
echo.

echo [4/5] Re-training Calibrated Stacking meta-model using newly tuned base models...
.\venv\Scripts\python.exe src/train_stacking_calibrated.py --num-samples 3000 --exclude-lfcc > train_stacking_calibrated_new.log 2>&1
echo Completed Calibrated Stacking training. Log saved to train_stacking_calibrated_new.log.
echo Current time: %time%
echo.

echo [5/5] Re-evaluating optimized ensemble and stacking on 2021 Eval Subset...
.\venv\Scripts\python.exe src/evaluate_robust_ensemble.py --subset > evaluation_ensemble_subset_after_tuning.log 2>&1
echo Completed final validation. Log saved to evaluation_ensemble_subset_after_tuning.log.
echo.

echo ====================================================
echo     ALL NIGHTLY EXPERIMENTS COMPLETED SUCCESSFULLY!
echo ====================================================
echo End time: %date% %time%
pause
