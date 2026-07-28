import os
from pathlib import Path

def main():
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    print("Checking Hugging Face cache dir:", cache_dir)
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            print(f"- {item.name}")
    else:
        print("Cache directory does not exist.")

if __name__ == "__main__":
    main()
