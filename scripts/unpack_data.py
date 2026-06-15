import os
import sys
import zipfile
import shutil
import argparse

# Устанавливаем корень проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack datasets and models transferred from laptop")
    parser.add_argument("--input", type=str, default="data_payload.zip", help="Input zip file name")
    args = parser.parse_args()
    
    zip_path = os.path.join(project_root, args.input)
    
    if not os.path.exists(zip_path):
        # Попробуем найти в папке Загрузок (Downloads) пользователя, так как Taildrop кидает туда
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        alt_path = os.path.join(downloads_dir, args.input)
        if os.path.exists(alt_path):
            zip_path = alt_path
        else:
            print(f"[ERROR] ZIP file not found at {zip_path} or {alt_path}")
            sys.exit(1)
            
    print(f"Unpacking archive: {zip_path} to {project_root}...")
    
    # Создаем временную папку для извлечения
    temp_extract_dir = os.path.join(project_root, "temp_extracted")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    print("Distributing files to correct directories...")
    
    # 1. Переносим data/2021 и data/2019
    data_extracted = os.path.join(temp_extract_dir, "data")
    if os.path.exists(data_extracted):
        for year in os.listdir(data_extracted):
            src_year_dir = os.path.join(data_extracted, year)
            dest_year_dir = os.path.join(project_root, "data", year)
            os.makedirs(os.path.dirname(dest_year_dir), exist_ok=True)
            
            # Если папка назначения уже есть, удалим её для чистой распаковки
            if os.path.exists(dest_year_dir):
                print(f"Removing old folder: {dest_year_dir}")
                shutil.rmtree(dest_year_dir)
                
            print(f"Moving dataset {year} to {dest_year_dir}...")
            shutil.move(src_year_dir, dest_year_dir)
            
    # 2. Переносим модели из папки models в корень
    models_extracted = os.path.join(temp_extract_dir, "models")
    if os.path.exists(models_extracted):
        print("Moving models to project root...")
        for file in os.listdir(models_extracted):
            src_file = os.path.join(models_extracted, file)
            dest_file = os.path.join(project_root, file)
            
            if os.path.exists(dest_file):
                os.remove(dest_file)
                
            shutil.move(src_file, dest_file)
            print(f"  Unpacked model: {file}")
            
    # Очищаем временную папку
    shutil.rmtree(temp_extract_dir)
    print("\nExtraction and distribution completed successfully!")
