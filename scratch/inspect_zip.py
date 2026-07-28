import zipfile

def main():
    zip_path = r"C:\Users\Quang Huy\Downloads\input_turn2_vong1\output.zip"
    print(f"Opening zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        print(f"Total files inside zip: {len(namelist)}")
        print("First 10 files:")
        for name in namelist[:10]:
            print(f"  - {name}")

if __name__ == "__main__":
    main()
