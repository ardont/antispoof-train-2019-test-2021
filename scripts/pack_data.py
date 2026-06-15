import os
import sys
import zipfile
import argparse

# Устанавливаем корень проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

def zip_directory(folder_path, zip_handle, archive_prefix=""):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Вычисляем относительный путь для сохранения структуры папок в архиве
            relative_path = os.path.relpath(file_path, folder_path)
            arcname = os.path.join(archive_prefix, relative_path)
            zip_handle.write(file_path, arcname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack dataset and models for transfer to remote PC")
    parser.add_argument("--pack-2021", action="store_true", help="Pack ASVspoof 2021 LA Eval dataset")
    parser.add_argument("--pack-2019", action="store_true", help="Pack ASVspoof 2019 LA Train/Dev dataset")
    parser.add_argument("--pack-models", action="store_true", help="Pack trained robust models and scalers")
    parser.add_argument("--output", type=str, default="data_payload.zip", help="Output zip file name")
    args = parser.parse_args()
    
    output_path = os.path.join(project_root, args.output)
    
    print(f"Creating archive: {output_path}...")
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zip_file:
        # 1. Архивируем датасет 2021 LA (если выбран)
        if args.pack_2021:
            dir_2021 = os.path.join(project_root, "data", "2021")
            if os.path.exists(dir_2021):
                print("Packing ASVspoof 2021 dataset (this may take several minutes)...")
                zip_directory(dir_2021, zip_file, archive_prefix="data/2021")
            else:
                print(f"[WARNING] 2021 data folder not found at {dir_2021}. Skipping...")
                
        # 2. Архивируем датасет 2019 LA (если выбран)
        if args.pack_2019:
            dir_2019 = os.path.join(project_root, "data", "2019")
            if os.path.exists(dir_2019):
                print("Packing ASVspoof 2019 dataset (this may take several minutes)...")
                zip_directory(dir_2019, zip_file, archive_prefix="data/2019")
            else:
                print(f"[WARNING] 2019 data folder not found at {dir_2019}. Skipping...")
                
        # 3. Архивируем модели и скейлеры (если выбран)
        if args.pack_models:
            print("Packing trained robust models and scalers...")
            for file in os.listdir(project_root):
                if file.endswith("_robust.pkl") or file.endswith("_robust.json") or file.endswith("_robust.cbm"):
                    file_path = os.path.join(project_root, file)
                    print(f"  Adding model/scaler: {file}")
                    zip_file.write(file_path, os.path.join("models", file))
                    
    print(f"\nSuccessfully packed! Archive saved to {output_path}")
    print("You can now transfer this file via Tailscale Taildrop to your remote PC.")
