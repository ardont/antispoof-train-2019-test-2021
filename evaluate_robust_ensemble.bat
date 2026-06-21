@echo off
echo ========================================================
echo       ASVspoof: Evaluating Robust Ensemble on 2021 Eval
echo ========================================================
echo.

:: Ensure venv is active or exists
if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment 'venv' not found! Please run 'install_deps.bat' first.
    pause
    exit /b 1
)

call venv\Scripts\activate

:: Unpack data if needed
echo [1/2] Preparing and checking evaluation data...
python scripts/prepare_data.py --eval
if %errorlevel% neq 0 (
    echo [ERROR] Evaluation data extraction failed!
    exit /b 1
)

:: Run evaluation
echo [2/2] Running 16-Model Robust Evaluation on 2021 Eval...
python src/evaluate_robust_ensemble_final.py %*

echo.
echo ========================================================
echo Evaluation finished. Summary results saved in overnight_results_robust_final.txt
echo ========================================================
pause
