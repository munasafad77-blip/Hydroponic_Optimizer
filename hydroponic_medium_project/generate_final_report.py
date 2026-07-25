"""
Full report generator: auto-selects the best available data source (same
logic the app itself uses), trains all targets, persists that choice, and
writes results/final_report.json with a training summary.
Run with:  python generate_final_report.py
"""
import sys
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.data_utils import ensure_project_dirs, select_data_source, save_persisted_data_selection
from src.modeling import train_all_targets

dirs = ensure_project_dirs(PROJECT_DIR)
selected_path, selected_source = select_data_source(PROJECT_DIR)

if not selected_path.exists():
    print(f"ERROR: no data file found ({selected_path}). Run the app once (DOE + Data Entry) first.")
    sys.exit(1)

save_persisted_data_selection(PROJECT_DIR, selected_path, selected_source)

df = pd.read_csv(selected_path)
results = train_all_targets(df)

summary = []
for target, bundle in results.items():
    summary.append({
        "target": target,
        "model": bundle["best_model_name"],
        "r2": bundle["best_metrics"].get("r2"),
        "rmse": bundle["best_metrics"].get("rmse"),
    })

report_path = dirs["results"] / "final_report.json"
report_path.write_text(json.dumps({
    "selected_data_file": str(selected_path),
    "selected_source": selected_source,
    "rows": len(df),
    "columns": list(df.columns),
    "summary": summary,
}, indent=2), encoding="utf-8")

print("REPORT_PATH", report_path)
print("SELECTED", selected_path.name, selected_source)
print("SUMMARY", summary)
