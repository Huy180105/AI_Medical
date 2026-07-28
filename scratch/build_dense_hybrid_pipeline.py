import re
import sys
import json
from pathlib import Path

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from src.ranking.concept_database import get_comprehensive_concepts
    
    concepts = get_comprehensive_concepts()
    print(f"Loaded {len(concepts)} master concepts for dense hybrid pipeline!")

if __name__ == "__main__":
    main()
