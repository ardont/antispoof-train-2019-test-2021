import sys
import os
import subprocess
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_cmd(args, log_file):
    print(f"Running: {' '.join(args)}")
    t0 = time.time()
    # Explicitly use the virtual environment's python
    python_exe = os.path.join(project_root, "venv", "Scripts", "python.exe")
    cmd_args = [python_exe] + args
    
    with open(os.path.join(project_root, log_file), "w", encoding="utf-8") as f:
        res = subprocess.run(cmd_args, cwd=project_root, stdout=f, stderr=subprocess.STDOUT)
        
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Failed to run {' '.join(args)}. Return code: {res.returncode}. Time: {dt/60:.1f} min. Check {log_file} for details.")
        sys.exit(res.returncode)
    else:
        print(f"[SUCCESS] Completed {' '.join(args)}. Time: {dt/60:.1f} min.")

if __name__ == "__main__":
    print("====================================================")
    print("      RUNNING REMAINING STEPS (DISK SPACE FIXED)    ")
    print("====================================================")
    
    # 1. Train MLP on CQCC
    run_cmd(["src/train_mlp_robust.py", "--feature", "cqcc"], "train_mlp_cqcc_retry.log")
    
    # 2. Train MLP on Combined
    run_cmd(["src/train_mlp_robust.py", "--feature", "combined"], "train_mlp_combined_retry.log")
    
    # 3. Optuna tuning for Combined features
    run_cmd(["src/tune_optuna_robust.py", "--feature", "combined", "--trials", "100", "--subsample-size", "30000"], "optuna_combined_retry.log")
    
    # 4. Train final Stacking meta-model
    run_cmd(["src/train_stacking_final.py", "--num-samples", "3000"], "train_stacking_final_retry.log")
    
    # 5. Final 12-model evaluation
    run_cmd(["src/evaluate_robust_ensemble_final.py"], "evaluate_final_full_retry.log")
    
    print("\n====================================================")
    print("         ALL REMAINING STEPS COMPLETED!              ")
    print("====================================================")
    if os.path.exists(os.path.join(project_root, "overnight_results_robust_final.txt")):
        with open(os.path.join(project_root, "overnight_results_robust_final.txt"), "r") as f:
            print(f.read())
