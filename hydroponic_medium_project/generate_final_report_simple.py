"""
Trains all targets on data/experimental_data.csv specifically (the simple,
non-auto-selecting variant) and writes results/final_report.json.
Run with:  python generate_final_report_simple.py
"""
import sys
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.data_utils import ensure_project_dirs
from src.modeling import train_all_targets

dirs = ensure_project_dirs(PROJECT_DIR)
data_path = dirs["data"] / "experimental_data.csv"

if not data_path.exists():
    print(f"ERROR: {data_path} does not exist yet. Run the app once (Data Entry tab) first.")
    sys.exit(1)

df = pd.read_csv(data_path)
try:
    results = train_all_targets(df)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

summary = []
for target, bundle in results.items():
    summary.append({
        "target": target,
        "model": bundle["best_model_name"],
        "r2": bundle["best_metrics"].get("r2"),
        "rmse": bundle["best_metrics"].get("rmse"),
    })

report_path = dirs["results"] / "final_report.json"
report_path.write_text(json.dumps({"rows": len(df), "summary": summary}, indent=2), encoding="utf-8")
print("created", report_path)
