import os
import sys
import zipfile
import tarfile
import argparse

# Set project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_zip(zip_path, dest_dir):
    print(f"Opening ZIP archive: {zip_path}")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        total_files = len(members)
        print(f"Extracting {total_files} files to {dest_dir}...")
        
        for idx, member in enumerate(members, 1):
            zip_ref.extract(member, dest_dir)
            if idx % 5000 == 0 or idx == total_files:
                print(f"  Extracted {idx}/{total_files} files ({(idx/total_files)*100:.1f}%)")

def extract_tar(tar_path, dest_dir):
    print(f"Opening TAR archive: {tar_path}")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        # Get members list
        print("Reading archive members...")
        members = tar_ref.getmembers()
        total_files = len(members)
        print(f"Extracting {total_files} files to {dest_dir}...")
        
        for idx, member in enumerate(members, 1):
            tar_ref.extract(member, dest_dir)
            if idx % 5000 == 0 or idx == total_files:
                print(f"  Extracted {idx}/{total_files} files ({(idx/total_files)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="ASVspoof Dataset Preparation Script")
    parser.add_argument("--train", action="store_true", help="Prepare 2019 training/dev data")
    parser.add_argument("--eval", action="store_true", help="Prepare 2021 evaluation data")
    args = parser.parse_args()
    
    # If no flags are passed, explain usage and exit
    if not args.train and not args.eval:
        print("Please specify action: --train or --eval (or both).")
        sys.exit(1)
        
    if args.train:
        train_zip = os.path.join(project_root, "data", "2019", "LA.zip")
        train_dest = os.path.join(project_root, "data", "2019")
        target_dir = os.path.join(project_root, "data", "2019", "LA", "ASVspoof2019_LA_train")
        
        if os.path.exists(target_dir):
            print(f"[INFO] 2019 Train data folder '{target_dir}' already exists. Skipping extraction.")
        elif not os.path.exists(train_zip):
            print(f"[ERROR] Training archive not found at: {train_zip}")
            sys.exit(1)
        else:
            print("--- Unpacking 2019 training/dev data ---")
            extract_zip(train_zip, train_dest)
            print("2019 Data preparation finished successfully!\n")
            
    if args.eval:
        eval_tar = os.path.join(project_root, "data", "2021", "ASVspoof2021_LA_eval.tar.gz")
        keys_tar = os.path.join(project_root, "data", "2021", "LA-keys-full.tar.gz")
        eval_dest = os.path.join(project_root, "data", "2021")
        
        target_eval_dir = os.path.join(project_root, "data", "2021", "ASVspoof2021_LA_eval", "flac")
        target_keys_dir = os.path.join(project_root, "data", "2021", "keys", "LA")
        
        # 1. Extract keys
        if os.path.exists(target_keys_dir):
            print(f"[INFO] 2021 keys folder '{target_keys_dir}' already exists. Skipping keys extraction.")
        elif not os.path.exists(keys_tar):
            print(f"[ERROR] Keys archive not found at: {keys_tar}")
            sys.exit(1)
        else:
            print("--- Unpacking 2021 keys ---")
            extract_tar(keys_tar, eval_dest)
            print("Keys preparation finished successfully!\n")
            
        # 2. Extract audio
        if os.path.exists(target_eval_dir):
            print(f"[INFO] 2021 Eval audio folder '{target_eval_dir}' already exists. Skipping audio extraction.")
        elif not os.path.exists(eval_tar):
            print(f"[ERROR] Evaluation audio archive not found at: {eval_tar}")
            sys.exit(1)
        else:
            print("--- Unpacking 2021 evaluation audio (this might take a few minutes) ---")
            extract_tar(eval_tar, eval_dest)
            print("2021 Evaluation audio preparation finished successfully!\n")

if __name__ == "__main__":
    main()
