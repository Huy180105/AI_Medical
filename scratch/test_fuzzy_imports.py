import sys
try:
    import Levenshtein
    print("Levenshtein is installed!")
except ImportError:
    print("Levenshtein is not installed.")

try:
    import rapidfuzz
    print("rapidfuzz is installed!")
except ImportError:
    print("rapidfuzz is not installed.")

try:
    from fuzzywuzzy import fuzz
    print("fuzzywuzzy is installed!")
except ImportError:
    print("fuzzywuzzy is not installed.")
