"""
src/data_utils.py
==================
Filesystem + data-selection helpers shared by app.py and the command-line
report/debug scripts, plus the user-editable materials list.

All paths are resolved relative to the project root that is passed in
(usually Path(__file__).resolve().parent from the calling script), so the
project is portable across machines -- nothing here is hardcoded to a
particular user or OS.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Fixed process-variable columns (firing settings) -- always the same 5.
PROCESS_COLUMNS = [
    "Temperature_C", "Holding_Time_min", "Heating_Rate_C_per_min",
    "Pellet_Size_mm", "Shell_Thickness_mm",
]

# Fixed measured-response columns (hydroponic properties) if present.
TARGET_COLUMNS = [
    "Bulk_Density", "Particle_Density", "Open_Porosity", "Water_Absorption",
    "Crushing_Strength", "pH", "EC", "Water_Retention", "Plant_Growth_Index",
]

# Non-data housekeeping columns that are never features or targets.
NON_DATA_COLUMNS = ["Run", "Method"]

DEFAULT_MATERIALS = ["SAC", "Lime"]

_SELECTION_FILE = "data_selection.json"
_MATERIALS_FILE = "materials_config.json"


def ensure_project_dirs(project_dir: Path) -> dict:
    """Create (if needed) and return the standard project sub-folders."""
    project_dir = Path(project_dir)
    dirs = {
        name: project_dir / name
        for name in ["data", "results", "figures", "reports", "models", "notebooks"]
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Materials (mixture components) -- fully user-editable, any number, any names
# ---------------------------------------------------------------------------
def load_materials(project_dir: Path) -> List[str]:
    """Read the current list of mixture materials the user has configured.
    Falls back to DEFAULT_MATERIALS if nothing has been saved yet."""
    dirs = ensure_project_dirs(project_dir)
    mat_file = dirs["models"] / _MATERIALS_FILE
    if mat_file.exists():
        try:
            payload = json.loads(mat_file.read_text(encoding="utf-8"))
            materials = [str(m).strip() for m in payload.get("materials", []) if str(m).strip()]
            if len(materials) >= 2:
                return materials
        except (json.JSONDecodeError, OSError):
            pass
    return list(DEFAULT_MATERIALS)


def save_materials(project_dir: Path, materials: List[str]) -> List[str]:
    """Persist the user's material list (deduplicated, order preserved)."""
    dirs = ensure_project_dirs(project_dir)
    seen = set()
    cleaned = []
    for m in materials:
        name = str(m).strip()
        if name and name not in seen:
            seen.add(name)
            cleaned.append(name)
    if len(cleaned) < 2:
        raise ValueError("At least 2 materials are required for a mixture.")
    mat_file = dirs["models"] / _MATERIALS_FILE
    mat_file.write_text(json.dumps({"materials": cleaned}, indent=2), encoding="utf-8")
    return cleaned


def infer_mixture_components(df: pd.DataFrame) -> List[str]:
    """
    Detect which columns of a loaded dataframe are mixture materials: every
    column that is not a known process variable, not a known target
    property, and not a housekeeping column. This lets modeling and
    optimization work with whatever materials are actually in the data,
    without needing to keep a second hard-coded list in sync.
    """
    excluded = set(PROCESS_COLUMNS) | set(TARGET_COLUMNS) | set(NON_DATA_COLUMNS)
    return [c for c in df.columns if c not in excluded]


# ---------------------------------------------------------------------------
# Experiment data I/O
# ---------------------------------------------------------------------------
def load_experiment_data(path: Path) -> pd.DataFrame:
    """Load an experimental CSV, raising a clear error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experimental data file not found: {path}")
    return pd.read_csv(path)


def save_experiment_data(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Split a raw experimental dataframe into:
      - X: the feature (mixture materials, whichever are present + process)
           columns actually present
      - y: dict of {target_name: Series} for every target column present
    Only numeric, non-empty columns are used. Mixture materials are detected
    dynamically via infer_mixture_components, so this works no matter how
    many materials the user has configured.
    """
    materials = infer_mixture_components(df)
    feature_cols = materials + [c for c in PROCESS_COLUMNS if c in df.columns]
    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = {}
    for t in target_cols:
        series = pd.to_numeric(df[t], errors="coerce")
        y[t] = series

    valid_rows = X.dropna().index
    X = X.loc[valid_rows]
    y = {t: s.loc[valid_rows] for t, s in y.items()}
    return X, y


def select_data_source(project_dir: Path) -> Tuple[Path, str]:
    """
    Choose which data file to use when nothing has been explicitly selected
    yet, preferring larger / more complete datasets over the small bootstrap
    sample:
      1. experimental_data_training_extended.csv
      2. experimental_data_training.csv
      3. experimental_data.csv (sample or last-uploaded)
    """
    dirs = ensure_project_dirs(project_dir)
    data_dir = dirs["data"]
    candidates = [
        ("experimental_data_training_extended.csv", "training_extended"),
        ("experimental_data_training.csv", "training"),
        ("experimental_data.csv", "sample_or_uploaded"),
    ]
    for fname, label in candidates:
        p = data_dir / fname
        if p.exists():
            return p, label
    return data_dir / "experimental_data.csv", "sample_or_uploaded"


def load_persisted_data_selection(project_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Read back the last data-source choice the user made in the app, if any."""
    dirs = ensure_project_dirs(project_dir)
    sel_file = dirs["models"] / _SELECTION_FILE
    if not sel_file.exists():
        return None, None
    try:
        payload = json.loads(sel_file.read_text(encoding="utf-8"))
        path = Path(payload["path"])
        source = payload.get("source")
        if path.exists():
            return path, source
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None, None


def save_persisted_data_selection(project_dir: Path, path: Path, source: str) -> None:
    dirs = ensure_project_dirs(project_dir)
    sel_file = dirs["models"] / _SELECTION_FILE
    sel_file.write_text(
        json.dumps({"path": str(Path(path)), "source": source}, indent=2), encoding="utf-8"
    )
