import pickle
import os

models = [f for f in os.listdir('.') if f.endswith('.pkl')]
corrupt = []
for f in models:
    try:
        with open(f, 'rb') as file:
            pickle.load(file)
        print(f"{f}: OK")
    except Exception as e:
        print(f"{f}: CORRUPT ({e})")
        corrupt.append(f)

print("\n--- Summary ---")
if corrupt:
    print(f"Found {len(corrupt)} corrupt files: {corrupt}")
else:
    print("All pickle files are OK!")
