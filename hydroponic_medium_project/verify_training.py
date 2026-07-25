"""
Trains all targets on the currently-selected data source and prints a short
confirmation. Run with:  python verify_training.py
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.data_utils import select_data_source
from src.modeling import train_all_targets

path, source = select_data_source(PROJECT_DIR)
print("Using data source:", source, "->", path)

if not path.exists():
    print(f"ERROR: {path} does not exist yet. Run the app once first, or place a CSV "
          f"at data/experimental_data.csv.")
    sys.exit(1)

df = pd.read_csv(path)
results = train_all_targets(df)
print("done", len(results))
for target in ("Bulk_Density", "Plant_Growth_Index"):
    if target in results:
        print(target, results[target]["best_model_name"])
    else:
        print(target, "-- not enough labeled rows to train yet")
