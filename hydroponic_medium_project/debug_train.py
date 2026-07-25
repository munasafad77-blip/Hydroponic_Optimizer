"""
Quick sanity-check script: trains all targets on whatever data file the app
would currently select, and prints the result for the Bulk_Density model.
Run with:  python debug_train.py
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
    print(f"ERROR: {path} does not exist yet. Run the app once (DOE + Data Entry) "
          f"or place a CSV at data/experimental_data.csv first.")
    sys.exit(1)

df = pd.read_csv(path)
print("rows", len(df))
print("columns", df.columns.tolist())
try:
    results = train_all_targets(df)
    print("targets", list(results.keys()))
    if "Bulk_Density" in results:
        print("bulk", results["Bulk_Density"]["best_model_name"])
        print("bulk metrics", results["Bulk_Density"]["best_metrics"])
except Exception:
    import traceback
    traceback.print_exc()
