import os
import zipfile
from pathlib import Path

def main():
    zip_path = Path("submission_code.zip")
    print(f"Creating project package {zip_path} for BTC review...")

    # Folders and files to include
    include_paths = [
        "src",
        "competition",
        "data",
        "input",
        "models/phobert-medical-ner",
        "main.py",
        "README.md",
        "requirements.txt"
    ]

    # Exclude patterns to keep zip clean (cache, logs, virtual environments, large output zips)
    exclude_extensions = {".pyc", ".pyo", ".log", ".zip", ".git"}
    exclude_folders = {"__pycache__", ".pytest_cache", ".git", ".agents"}

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path_str in include_paths:
            path = Path(path_str)
            if not path.exists():
                print(f"Warning: path {path_str} does not exist. Skipping.")
                continue

            if path.is_file():
                zf.write(path, arcname=path_str)
                print(f"Added file: {path_str}")
            elif path.is_dir():
                for root, dirs, files in os.walk(path):
                    # Filter out directories to ignore (including intermediate checkpoints)
                    dirs[:] = [d for d in dirs if d not in exclude_folders and not d.startswith("checkpoint-")]
                    
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.suffix in exclude_extensions:
                            continue
                        
                        # Write to zip using its relative path directly
                        zf.write(file_path, arcname=file_path.as_posix())
                print(f"Added directory: {path_str}")

    print(f"Successfully created project package at {zip_path}!")

if __name__ == "__main__":
    main()
