"""
src/mixture_design.py
======================
Design of Experiments (DOE) for a user-configurable mixture (any number of
materials, any names, summing to 1.0) combined with 5 process variables
(firing temperature, holding time, heating rate, pellet size, shell
thickness).

The material list itself is NOT hard-coded here -- it is read from
src/data_utils.py (load_materials / save_materials), which the user edits
from the Materials section of the DOE tab. Every function below accepts an
explicit `components` list so the same code works for 2 materials, 5
materials, or any other number the user configures.

Four DOE methods are supported, selectable from the Streamlit sidebar:

  - "manual"           : YOU enter the exact ratios you want tested, one row
                          per experiment, for every material -- a full
                          spreadsheet-style grid. Each row is normalized to
                          sum to 1 automatically. Use this when you already
                          know which points you want to run.
  - "simplex_lattice"   : evenly-spaced points across the composition space --
                          every combination that is a multiple of 1/levels.
  - "simplex_centroid"  : the compact classic design -- every pure
                          component, every binary (50/50) blend, and the
                          overall centroid.
  - "d_optimal"         : an approximate D-optimal design for a Scheffe
                          quadratic mixture model, built by greedy forward
                          selection (Fedorov-style exchange) from a large
                          random candidate pool.

Every mixture point is paired with one Latin-Hypercube-sampled combination of
process variables, so the returned DOE is a combined mixture-process design
ready to hand to the laboratory.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .data_utils import DEFAULT_MATERIALS, TARGET_COLUMNS, PROCESS_COLUMNS

# Realistic process-variable ranges (consistent with the laboratory protocol)
PROCESS_RANGES = {
    "Temperature_C": (1000.0, 1200.0),
    "Holding_Time_min": (10.0, 25.0),
    "Heating_Rate_C_per_min": (5.0, 15.0),
    "Pellet_Size_mm": (6.0, 10.0),
    "Shell_Thickness_mm": (0.8, 1.5),
}
assert list(PROCESS_RANGES.keys()) == PROCESS_COLUMNS, "PROCESS_RANGES keys must match data_utils.PROCESS_COLUMNS"


def custom_ratio_points(rows: List[Dict[str, float]], components: Sequence[str]) -> List[List[float]]:
    """
    Build mixture points directly from a list of {material: value} rows you
    provide (e.g. from the Streamlit data-editor grid). Values do not need
    to already sum to 1 -- each row is normalized automatically:
        x_i = value_i / sum(all values in that row)
    Rows where every value is 0 or missing are skipped.
    """
    points = []
    for row in rows:
        raw = [max(0.0, float(row.get(c, 0.0) or 0.0)) for c in components]
        total = sum(raw)
        if total <= 0:
            continue
        points.append([v / total for v in raw])
    return points


def manual_points(sac_values: Sequence[float], components: Sequence[str] = DEFAULT_MATERIALS) -> List[List[float]]:
    """
    Backward-compatible helper for the 2-material case: build points from a
    list of fractions for components[0] (Lime is computed as 1 - value for
    components[1]).
    """
    points = []
    for v in sac_values:
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Fraction must be between 0 and 1 -- got {v}")
        points.append([v, 1.0 - v])
    return points


def _simplex_lattice(levels: int, components: Sequence[str]) -> List[List[float]]:
    """All fractions that are multiples of 1/levels and sum to 1, across n materials."""
    n = len(components)
    points = []
    for combo in itertools.product(range(levels + 1), repeat=n - 1):
        if sum(combo) > levels:
            continue
        last = levels - sum(combo)
        point = [c / levels for c in combo] + [last / levels]
        points.append(point)
    return points


def _simplex_centroid(components: Sequence[str]) -> List[List[float]]:
    """n pure vertices + all binary (50/50) blends + 1 overall centroid (n > 2 only)."""
    n = len(components)
    points = []
    for i in range(n):
        p = [0.0] * n
        p[i] = 1.0
        points.append(p)
    for i, j in itertools.combinations(range(n), 2):
        p = [0.0] * n
        p[i] = p[j] = 0.5
        points.append(p)
    if n > 2:
        points.append([1.0 / n] * n)
    return points


def _scheffe_design_row(x: np.ndarray) -> np.ndarray:
    """Quadratic Scheffe mixture model terms: every linear term + every pairwise product."""
    n = len(x)
    linear = list(x)
    pairwise = [x[i] * x[j] for i, j in itertools.combinations(range(n), 2)]
    return np.array(linear + pairwise)


def _d_optimal(n_points: int, components: Sequence[str], rng: np.random.Generator,
               pool_size: int = 5000) -> List[List[float]]:
    """
    Approximate D-optimal design via greedy forward selection: build a large
    random candidate pool on the simplex (Dirichlet sampling), then
    repeatedly add whichever remaining candidate increases det(X'X) the most
    for the quadratic Scheffe model matrix.
    """
    n = len(components)
    n_terms = n + len(list(itertools.combinations(range(n), 2)))
    candidates = rng.dirichlet([1.0] * n, size=pool_size)

    seed_points = np.array(_simplex_centroid(components))
    design_rows = [_scheffe_design_row(p) for p in seed_points]
    chosen_points = list(seed_points)
    remaining = list(range(pool_size))

    while len(chosen_points) < n_points and remaining:
        best_idx, best_det = None, -np.inf
        X_current = np.array(design_rows)
        for ridx in remaining[:400]:  # cap search per iteration for speed
            cand_row = _scheffe_design_row(candidates[ridx])
            X_try = np.vstack([X_current, cand_row])
            info_try = X_try.T @ X_try + np.eye(n_terms) * 1e-9
            det = np.linalg.det(info_try)
            if det > best_det:
                best_det, best_idx = det, ridx
        if best_idx is None:
            break
        chosen_points.append(candidates[best_idx])
        design_rows.append(_scheffe_design_row(candidates[best_idx]))
        remaining.remove(best_idx)

    return [list(p) for p in chosen_points[:n_points]]


def _latin_hypercube(n: int, ranges: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Simple Latin Hypercube sample over the given {name: (lo, hi)} ranges."""
    cols = {}
    for name, (lo, hi) in ranges.items():
        cut_points = (np.arange(n) + rng.uniform(size=n)) / n
        rng.shuffle(cut_points)
        cols[name] = lo + cut_points * (hi - lo)
    return pd.DataFrame(cols)


def build_doe(method: str = "manual", levels: int = 4, n_points: int = 12,
              manual_values: Sequence[float] = None,
              manual_rows: List[Dict[str, float]] = None,
              components: Sequence[str] = None, seed: int = 42) -> pd.DataFrame:
    """
    Build a combined mixture-process DOE.

    Parameters
    ----------
    method : "manual" | "simplex_lattice" | "simplex_centroid" | "d_optimal"
    levels : lattice degree, used only by "simplex_lattice"
    n_points : target design size, used only by "d_optimal"
    manual_values : list of fractions for a 2-material mixture (legacy path)
    manual_rows : list of {material: value} dicts, any number of materials
                  (preferred path for "manual" -- values are normalized to
                  sum to 1 per row automatically)
    components : the material names to use. Defaults to DEFAULT_MATERIALS
                 (["SAC", "Lime"]) if not given.
    """
    components = list(components) if components else list(DEFAULT_MATERIALS)
    if len(components) < 2:
        raise ValueError("At least 2 materials are required for a mixture design.")
    rng = np.random.default_rng(seed)

    if method == "manual":
        if manual_rows:
            mix_points = custom_ratio_points(manual_rows, components)
        elif manual_values:
            mix_points = manual_points(manual_values, components)
        else:
            raise ValueError("method='manual' requires manual_rows (or manual_values for 2 materials).")
        if not mix_points:
            raise ValueError("No valid mixture rows were provided (every row summed to 0).")
    elif method == "simplex_lattice":
        mix_points = _simplex_lattice(levels, components)
    elif method == "simplex_centroid":
        mix_points = _simplex_centroid(components)
    elif method == "d_optimal":
        mix_points = _d_optimal(n_points, components, rng)
    else:
        raise ValueError(f"Unknown DOE method: {method}")

    mix_df = pd.DataFrame(mix_points, columns=components)
    process_df = _latin_hypercube(len(mix_df), PROCESS_RANGES, rng)

    doe = pd.concat([mix_df.reset_index(drop=True), process_df.reset_index(drop=True)], axis=1)
    doe.insert(0, "Run", [f"R{i+1:02d}" for i in range(len(doe))])
    doe[components] = doe[components].round(4)
    for col in PROCESS_RANGES:
        doe[col] = doe[col].round(2)

    # blank columns for the lab to fill in
    for target in TARGET_COLUMNS:
        doe[target] = ""

    return doe
