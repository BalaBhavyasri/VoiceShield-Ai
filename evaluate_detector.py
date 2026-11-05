import sys
from pathlib import Path

# Forward execution to evaluation/evaluate_detector.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.evaluate_detector import main

if __name__ == "__main__":
    main()
