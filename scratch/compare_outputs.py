import zipfile
import json
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    user_zip = r"C:\Users\Quang Huy\Downloads\input_turn2_vong1\output.zip"
    our_zip = r"c:\Users\Quang Huy\source\repos\Mediacal-AI-Agent\output.zip"

    with zipfile.ZipFile(user_zip, 'r') as z1, zipfile.ZipFile(our_zip, 'r') as z2:
        # Load output/1.json from both
        data1 = json.loads(z1.read("output/1.json").decode('utf-8'))
        data2 = json.loads(z2.read("output/1.json").decode('utf-8'))

        print(f"User's 1.json predicted entity count: {len(data1)}")
        print(f"Our 1.json predicted entity count: {len(data2)}")
        
        print("\nUser's first 3 entities:")
        for ent in data1[:3]:
            print(f"  - {ent.get('text')} ({ent.get('type')}) | candidates: {ent.get('candidates')} | assertions: {ent.get('assertions')}")

        print("\nOur first 3 entities:")
        for ent in data2[:3]:
            print(f"  - {ent.get('text')} ({ent.get('type')}) | candidates: {ent.get('candidates')} | assertions: {ent.get('assertions')}")

if __name__ == "__main__":
    main()
