import os
import sys
import zipfile
import tarfile
import argparse

def extract_zip(zip_path, dest_dir):
    print(f"[INFO] Opening ZIP archive: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        total_files = len(members)
        print(f"[INFO] Extracting {total_files} files to {dest_dir}...")
        for idx, member in enumerate(members, 1):
            zip_ref.extract(member, dest_dir)
            if idx % 5000 == 0 or idx == total_files:
                print(f"  Extracted {idx}/{total_files} files ({(idx/total_files)*100:.1f}%)")

def extract_tar(tar_path, dest_dir):
    print(f"[INFO] Opening TAR archive: {tar_path}")
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        print("[INFO] Reading archive members...")
        members = tar_ref.getmembers()
        total_files = len(members)
        print(f"[INFO] Extracting {total_files} files to {dest_dir}...")
        for idx, member in enumerate(members, 1):
            tar_ref.extract(member, dest_dir)
            if idx % 5000 == 0 or idx == total_files:
                print(f"  Extracted {idx}/{total_files} files ({(idx/total_files)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Unpack custom dataset archive")
    parser.add_argument("--dir", type=str, required=True, help="Directory of the custom dataset")
    args = parser.parse_args()
    
    data_dir = args.dir
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory does not exist: {data_dir}")
        sys.exit(1)
        
    # Check if there are already audio files unpacked (flac or wav)
    has_audio = False
    for root, dirs, files in os.walk(data_dir):
        if any(f.lower().endswith(('.flac', '.wav')) for f in files):
            has_audio = True
            break
            
    if has_audio:
        print(f"[INFO] Audio files (.flac or .wav) already exist in {data_dir}. Skipping extraction.")
        sys.exit(0)
        
    # Look for archives
    archive_found = None
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.zip', '.tar.gz', '.tgz', '.tar')):
            archive_found = os.path.join(data_dir, file)
            break
            
    if not archive_found:
        print(f"[WARNING] No archive (.zip, .tar.gz) found in {data_dir} and no unpacked audio files found.")
        print("[INFO] Please place your archive in this folder to allow automatic extraction.")
        sys.exit(2) # return 2 to signal no archive was found
        
    print(f"[INFO] Found archive: {archive_found}")
    if archive_found.lower().endswith('.zip'):
        extract_zip(archive_found, data_dir)
    else:
        extract_tar(archive_found, data_dir)
        
    print(f"[SUCCESS] Unpacking completed for {data_dir}")

if __name__ == "__main__":
    main()
