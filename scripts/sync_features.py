import os
import sys
import shutil
import argparse
import glob

# Устанавливаем корень проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

PC_IP = "100.90.91.54"
# Сетевой путь к папке проекта на ПК (предполагаем, что папка расшарена под именем 'antispoof')
REMOTE_PROJECT_PATH = f"\\\\{PC_IP}\\antispoof"

def copy_file(src, dst):
    try:
        print(f"Copying {src} -> {dst}...")
        shutil.copy2(src, dst)
        print("  Success!")
    except Exception as e:
        print(f"[ERROR] Failed to copy {src}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync features and models between Laptop and Remote PC")
    parser.add_argument("--pc-path", type=str, default=REMOTE_PROJECT_PATH, help="Network path to the shared project folder on PC")
    
    # Действия по скачиванию (с ПК на ноутбук)
    parser.add_argument("--pull-features", action="store_true", help="Download cache files (*_cache*.pkl) from PC to Laptop")
    parser.add_argument("--pull-models", action="store_true", help="Download trained robust models from PC to Laptop")
    
    # Действия по закачиванию (с ноутбука на ПК)
    parser.add_argument("--push-features", action="store_true", help="Upload cache files (*_cache*.pkl) from Laptop to PC")
    parser.add_argument("--push-models", action="store_true", help="Upload trained robust models from Laptop to PC")
    
    args = parser.parse_args()
    
    remote_path = args.pc_path
    
    # Проверяем доступность сетевой папки
    if not os.path.exists(remote_path):
        print(f"[ERROR] Remote network path {remote_path} is not accessible.")
        print("Please make sure that:")
        print(f"  1. The remote PC (IP: {PC_IP}) is connected to Tailscale and turned on.")
        print("  2. The project folder on the PC is shared (Right-click folder -> Properties -> Sharing -> Share).")
        print("  3. Windows Network Discovery is enabled on both devices.")
        sys.exit(1)
        
    print(f"Network connection established with: {remote_path}\n")
    
    # --- PULL: С ПК на ноутбук ---
    if args.pull_features:
        print("--- Pulling features from PC to Laptop ---")
        # Ищем все файлы кэша на ПК
        search_pattern = os.path.join(remote_path, "*_cache*.pkl")
        files = glob.glob(search_pattern)
        if not files:
            print("No cache files found on PC.")
        for f in files:
            filename = os.path.basename(f)
            dest = os.path.join(project_root, filename)
            copy_file(f, dest)
            
    if args.pull_models:
        print("--- Pulling models from PC to Laptop ---")
        model_patterns = ["*_robust.pkl", "*_robust.json", "*_robust.cbm", "stacking_meta_model*.pkl"]
        found = False
        for pattern in model_patterns:
            search_pattern = os.path.join(remote_path, pattern)
            files = glob.glob(search_pattern)
            for f in files:
                found = True
                filename = os.path.basename(f)
                dest = os.path.join(project_root, filename)
                copy_file(f, dest)
        if not found:
            print("No robust models found on PC.")
            
    # --- PUSH: С ноутбука на ПК ---
    if args.push_features:
        print("--- Pushing features from Laptop to PC ---")
        search_pattern = os.path.join(project_root, "*_cache*.pkl")
        files = glob.glob(search_pattern)
        if not files:
            print("No cache files found on Laptop.")
        for f in files:
            filename = os.path.basename(f)
            dest = os.path.join(remote_path, filename)
            copy_file(f, dest)
            
    if args.push_models:
        print("--- Pushing models from Laptop to PC ---")
        model_patterns = ["*_robust.pkl", "*_robust.json", "*_robust.cbm", "stacking_meta_model*.pkl"]
        found = False
        for pattern in model_patterns:
            search_pattern = os.path.join(project_root, pattern)
            files = glob.glob(search_pattern)
            for f in files:
                found = True
                filename = os.path.basename(f)
                dest = os.path.join(remote_path, filename)
                copy_file(f, dest)
        if not found:
            print("No robust models found on Laptop.")
            
    print("\nSync operations completed.")
