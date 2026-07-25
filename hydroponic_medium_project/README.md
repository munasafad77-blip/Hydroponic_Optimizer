# Hydroponic Growing Medium Design & Optimization

## Handing this to someone non-technical (e.g. your supervisor)

They only ever need to touch ONE file. Nothing else in this folder needs to
be opened, and no code is ever shown on screen during normal use:

1. Unzip this folder anywhere.
2. **Windows:** double-click `START_APP.bat`.
   **Mac/Linux:** double-click `start_app.sh` (or run `./start_app.sh`).
3. A small window opens with a few plain-English status lines only --
   no code, no technical logs. The very first run installs everything
   automatically in the background (2-5 minutes, one-time only); every run
   after that is ready in a few seconds.
4. The app opens automatically in the browser. That window with the status
   lines should stay open in the background -- it's what's running the app.
   Closing it stops the app.
5. In the app: **Data Entry** to upload measured results, **Modeling** to
   train, **Optimization**/**Reports** to get the recommended formulation.

All the technical output that used to print to the screen (package
installation, server logs) is now saved quietly to `setup_log.txt` and
`app_log.txt` in this folder instead -- only useful if something needs
debugging, never shown during normal use.

For the full mathematical explanation of every number the app produces, see
`Hydroponic_Optimizer_Complete_Model_Manual.docx` (included alongside this project).

---

A working Streamlit application for designing, modeling, and optimizing ANY
multi-material hydroponic growing medium: define your own materials (any
number, any names), generate a lab Design of Experiments (or enter your own
exact ratios), enter measured results, train ML property models, and run a
genetic-algorithm search for the best-performing formulation -- with a
report you customize around the properties that matter to you.

## What this app does

| Tab | What it does |
|---|---|
| **DOE** | First define your materials (add/remove/rename -- any number, any names) in the Materials editor. Then generate a mixture (+ process-variable) experiment table using `manual` (a spreadsheet-style grid where you type in the exact ratio of every material for every experiment -- rows are auto-normalized to sum to 1), `simplex_lattice`, `simplex_centroid`, or an approximate `d_optimal` design |
| **Data Entry** | Upload a CSV of real laboratory measurements (composition + process settings + measured properties). The materials list updates automatically to match whatever columns your file contains |
| **Modeling** | Trains a predictive model for each measured property (Bulk_Density, Crushing_Strength, pH, ...), automatically picking the best-performing algorithm per property, with cross-validated accuracy reported |
| **Optimization** | Runs a genetic algorithm (through the trained models) to search for the formulation that best balances ALL properties at once, using Derringer desirability scoring -- works with however many materials you configured |
| **Reports** | Pick the hydroponic properties you personally care about from a checklist, then get: a composition-by-run chart, a response surface, a correlation heatmap, feature importance, optimization progress, a ranking chart for EACH property you picked, and a radar chart comparing your top formulations across all of them |

## Materials are fully yours to control

Nothing about the materials is hard-coded. Add a 3rd, 4th, or 5th material
in the DOE tab's Materials editor (e.g. Bentonite, OSA, a different lime
source), give them any names you want, and every other part of the app --
DOE generation, modeling, optimization, and every chart -- adapts
automatically. You can also skip algorithmic DOE entirely and just type in
the exact ratios you want to test yourself using the "manual" method's grid.


## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).

For headless/remote use (e.g. on a lab server):
```bash
streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```

### Windows quick start

```powershell
.\run_app.ps1
```
This script uses a local `.venv` if one exists, otherwise falls back to your
system Python. It works from any folder location (no hardcoded paths).

## Verify your installation

After `pip install -r requirements.txt`, you can confirm everything works on
your machine before showing it to anyone else:

```bash
python test_app_end_to_end.py
```

This drives the actual app (DOE generation -> training -> optimization ->
report figures) headlessly and prints `ALL CHECKS PASSED` if everything
works. It's the same check that was run before this project was handed off.

## First run — what to expect

On first launch the app auto-creates `data/experimental_data.csv` with 8
illustrative example rows (plausible values, NOT real measurements) purely so
every tab has something to show immediately. Go to **Modeling -> Train
Models** and you'll see it train successfully on this bootstrap sample.

**Before drawing any real conclusions, replace this with real laboratory
data**, either by:
- **Data Entry tab**: upload a CSV with the same column layout (see below), or
- placing a CSV named `experimental_data_training.csv` or
  `experimental_data_training_extended.csv` directly in `data/` (the app
  prefers these, largest first, over the small bootstrap sample).

## Expected data columns

| Type | Columns |
|---|---|
| Mixture (must sum to 1.0) | `SAC`, `Lime` |
| Process variables | `Temperature_C`, `Holding_Time_min`, `Heating_Rate_C_per_min`, `Pellet_Size_mm`, `Shell_Thickness_mm` |
| Measured responses | `Bulk_Density`, `Particle_Density`, `Open_Porosity`, `Water_Absorption`, `Crushing_Strength`, `pH`, `EC`, `Water_Retention`, `Plant_Growth_Index` |

Any subset of the response columns is fine — the Modeling tab trains a model
for every response column that has at least 3 filled-in values, and simply
skips the rest.

## Project structure

```
data/       experimental datasets and DOE files
results/    optimized/ranked formulations (CSV)
figures/    generated PNG figures
reports/    JSON summary reports (from the CLI scripts below)
models/     cached trained models + training summary
src/        core implementation (see below)
app.py      the Streamlit interface
```

### `src/` modules

- `data_utils.py` — project folders, data loading/saving, and remembering which data file was last selected
- `mixture_design.py` — the three DOE methods
- `modeling.py` — multi-model training, cross-validation, caching
- `optimization.py` — genetic-algorithm (DEAP) + Derringer desirability multi-response optimization
- `visualization.py` — all report figures

## Command-line scripts

These duplicate parts of the app's workflow for scripted/batch use — handy for
regenerating a report without opening the browser UI:

- `debug_train.py` — quick sanity check: trains all targets, prints the result
- `verify_training.py` — trains and prints a short confirmation
- `generate_final_report.py` — auto-selects the best available data file, trains, writes `results/final_report.json`
- `generate_final_report_simple.py` — same, but always uses `data/experimental_data.csv` specifically

All four now resolve paths relative to their own location
(`Path(__file__).resolve().parent`), so they work regardless of where the
project folder lives on disk — no editing required before handing this to
someone else.

## Method notes (for the thesis / report)

- **DOE**: `simplex_lattice` and `simplex_centroid` are the standard textbook
  mixture designs; `d_optimal` is an approximate D-optimal design built by
  greedy forward selection (Fedorov-style exchange) against a quadratic
  Scheffe mixture model. Process variables are added via a paired Latin
  Hypercube sample.
- **Modeling**: for each response, Random Forest, Gradient Boosting, Ridge
  regression, SVR (RBF), and — once at least 8 samples are available —
  XGBoost are all cross-validated (K-Fold, or Leave-One-Out below 10 samples),
  and the best-scoring model per response is kept automatically.
- **Optimization**: NSGA-style genetic search (DEAP), scoring each candidate
  formulation with a Derringer & Suich (1980) composite desirability function
  — the geometric mean of per-response desirabilities, so a formulation that
  is unacceptable on any single property scores 0 overall, not just "average".
- **Caveat**: model quality is only as good as the lab data behind it. Treat
  early results (before real data is entered) as a functional demonstration,
  not a scientific recommendation — the Modeling tab's reported R²/RMSE
  tell you how much to trust the current model.
