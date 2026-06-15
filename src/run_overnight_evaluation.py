import os
import sys
import time
import subprocess

def run_script(cmd_args, log_file):
    print(f"Running: {' '.join(cmd_args)} ... output redirected to {log_file}")
    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd_args, stdout=f, stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0

if __name__ == "__main__":
    print("=== ASVspoof 2021 Overnight Evaluation Master Script ===")
    
    # 1. Ждем завершения обучения LFCC моделей (task-484)
    target_lfcc_model = "cat_model_lfcc_augmented.cbm"
    print(f"Waiting for full LFCC models to finish training (checking for {target_lfcc_model})...")
    
    # На всякий случай проверяем, обучался ли кэш признаков
    start_wait = time.time()
    while not os.path.exists(target_lfcc_model):
        time.sleep(30)
        elapsed = (time.time() - start_wait) / 60
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            print(f"Still waiting... {int(elapsed)} minutes elapsed.")
            
    print("LFCC models are ready! Starting full evaluations on 181,566 files...")
    
    # 2. Оценка MFCC моделей (полная выборка 2021)
    print("\n--- [1/3] Full MFCC Evaluation ---")
    mfcc_success = run_script([sys.executable, "-u", "src/evaluate_2021.py"], "evaluation_mfcc_full.log")
    if mfcc_success:
        print("MFCC Evaluation completed successfully.")
    else:
        print("Warning: MFCC Evaluation script returned non-zero code. Check evaluation_mfcc_full.log")
        
    # 3. Оценка LFCC моделей (полная выборка 2021)
    print("\n--- [2/3] Full LFCC Evaluation ---")
    lfcc_success = run_script([sys.executable, "-u", "src/evaluate_lfcc_2021.py"], "evaluation_lfcc_full.log")
    if lfcc_success:
        print("LFCC Evaluation completed successfully.")
    else:
        print("Warning: LFCC Evaluation script returned non-zero code. Check evaluation_lfcc_full.log")
        
    # 4. Оценка Ансамбля (полная выборка 2021)
    print("\n--- [3/3] Full Ensemble Evaluation ---")
    ensemble_success = run_script([sys.executable, "-u", "src/ensemble_evaluate.py"], "evaluation_ensemble_full.log")
    if ensemble_success:
        print("Ensemble Evaluation completed successfully.")
    else:
        print("Warning: Ensemble Evaluation script returned non-zero code. Check evaluation_ensemble_full.log")
        
    # 5. Сбор результатов в один файл
    print("\n=== Generating Summary ===")
    summary_path = "overnight_results.txt"
    
    with open(summary_path, "w", encoding="utf-8") as out:
        out.write("====================================================\n")
        out.write("        ASVspoof 2021 LA OVERNIGHT RESULTS SUMMARY  \n")
        out.write("====================================================\n\n")
        
        # Считываем лог MFCC
        if os.path.exists("evaluation_mfcc_full.log"):
            out.write("--- MFCC Models Evaluation ---\n")
            with open("evaluation_mfcc_full.log", "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                # Берем последние 15 строк
                out.write("".join(lines[-15:]))
            out.write("\n")
            
        # Считываем лог LFCC
        if os.path.exists("evaluation_lfcc_full.log"):
            out.write("--- LFCC Models Evaluation ---\n")
            with open("evaluation_lfcc_full.log", "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                out.write("".join(lines[-15:]))
            out.write("\n")
            
        # Считываем лог Ансамбля
        if os.path.exists("evaluation_ensemble_full.log"):
            out.write("--- Ensemble Evaluation ---\n")
            with open("evaluation_ensemble_full.log", "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                out.write("".join(lines[-25:]))
            out.write("\n")
            
    print(f"\nAll tasks completed! Summary written to: {summary_path}")
