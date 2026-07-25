"""End-to-end smoke test for app.py using Streamlit's AppTest (no browser needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent / "app.py")


def run_menu(menu_label):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception, f"Startup exception: {at.exception}"
    at.sidebar.radio[0].set_value(menu_label).run()
    assert not at.exception, f"{menu_label} tab load exception: {at.exception}"
    return at


print("=== 1) App startup ===")
at = AppTest.from_file(APP_PATH, default_timeout=120)
at.run()
assert not at.exception, f"Startup exception: {at.exception}"
print("OK - app starts without exceptions")

print("\n=== 2) DOE tab: generate a design ===")
at.sidebar.radio[0].set_value("DOE").run()
gen_btn = [b for b in at.button if b.label == "Generate DOE"][0]
gen_btn.click().run()
assert not at.exception, f"DOE generation exception: {at.exception}"
doe_file = Path(__file__).resolve().parent / "data" / "doe_design.csv"
assert doe_file.exists(), "doe_design.csv was not written"
import pandas as pd
doe_df = pd.read_csv(doe_file)
assert len(doe_df) > 0
print(f"OK - DOE generated with {len(doe_df)} runs, columns: {list(doe_df.columns)}")

print("\n=== 3) Modeling tab: train models on bootstrap sample data ===")
at.sidebar.radio[0].set_value("Modeling").run()
assert not at.exception, f"Modeling tab load exception: {at.exception}"
train_btn = [b for b in at.button if b.label == "Train Models"][0]
train_btn.click().run()
assert not at.exception, f"Training exception: {at.exception}"
print("OK - training completed without exception")

print("\n=== 4) Modeling tab: run optimization + report ===")
opt_btn = [b for b in at.button if b.label == "Run Optimization + Generate Report"][0]
opt_btn.click().run(timeout=180)
assert not at.exception, f"Optimization exception: {at.exception}"
ranked_file = Path(__file__).resolve().parent / "results" / "ranked_formulations.csv"
assert ranked_file.exists(), "ranked_formulations.csv was not written"
ranked_df = pd.read_csv(ranked_file)
assert len(ranked_df) > 0
print(f"OK - optimization completed, {len(ranked_df)} ranked formulations, best score={ranked_df['score'].iloc[0]:.3f}")

print("\n=== 5) Reports tab: generate figures ===")
at.sidebar.radio[0].set_value("Reports").run(timeout=60)
assert not at.exception, f"Reports tab exception: {at.exception}"
fig_dir = Path(__file__).resolve().parent / "figures"
expected_figs = [
    "mixture_triangle.png", "response_surface.png", "correlation_heatmap.png",
    "feature_importance.png", "optimization_progress.png", "radar_comparison.png",
]
for f in expected_figs:
    assert (fig_dir / f).exists(), f"{f} was not generated"
ranking_figs = list(fig_dir.glob("ranking_*.png"))
assert len(ranking_figs) >= 3, f"expected per-property ranking charts, found {len(ranking_figs)}"
print(f"OK - all {len(expected_figs)} general figures + {len(ranking_figs)} per-property ranking charts generated in figures/")

print("\n=== ALL CHECKS PASSED ===")
