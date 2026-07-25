"""
src/optimization.py
====================
Multi-response optimization: searches the mixture (SAC/Lime) + process
variable space for the formulation that best satisfies ALL trained property
targets simultaneously, using a genetic algorithm (DEAP) driving a composite
Derringer desirability score evaluated through the trained ML models from
src.modeling.

Derringer desirability functions
---------------------------------
Each predicted response y is converted to a desirability d in [0, 1]:
  - "maximize" responses (Crushing_Strength, Water_Retention, Plant_Growth_Index):
        d = 0                                   if y <= low
        d = ((y - low) / (high - low))          if low < y < high
        d = 1                                   if y >= high
  - "minimize" responses (Bulk_Density, EC):
        mirror image of the above (low value -> d=1)
  - "target" responses (pH -> aim for neutral ~6.5-7.5):
        triangular desirability peaking at the target range, falling off
        toward the low/high bounds

The overall desirability is the geometric mean of the individual d_i, which
is the standard Derringer & Suich (1980) formulation: if ANY single response
is at d=0 (completely unacceptable), the overall score is 0 regardless of how
good the other responses are.

Low/high bounds for each response are taken from the observed range in the
training data (with a small margin), since no fixed spec limits were given
elsewhere in the project.
"""
from __future__ import annotations

import random
from typing import Dict

import numpy as np
import pandas as pd
from deap import base, creator, tools

from .mixture_design import PROCESS_RANGES
from .data_utils import infer_mixture_components

RESPONSE_GOALS = {
    "Bulk_Density": "minimize",
    "Particle_Density": "minimize",
    "Open_Porosity": "maximize",
    "Water_Absorption": "maximize",
    "Crushing_Strength": "maximize",
    "pH": "target",          # aim for neutral
    "EC": "minimize",
    "Water_Retention": "maximize",
    "Plant_Growth_Index": "maximize",
}
PH_TARGET_RANGE = (6.5, 7.5)

N_GENERATIONS = 60
POP_SIZE = 80
N_RESULTS = 15

# ---------------------------------------------------------------------------
# DEAP setup (guarded so re-importing the module twice in one Streamlit
# session -- which reloads the module -- doesn't raise "class already exists")
# ---------------------------------------------------------------------------
if not hasattr(creator, "FitnessMaxDesirability"):
    creator.create("FitnessMaxDesirability", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMaxDesirability)


def _response_bounds(df: pd.DataFrame) -> Dict[str, tuple]:
    bounds = {}
    for target in RESPONSE_GOALS:
        if target in df.columns:
            series = pd.to_numeric(df[target], errors="coerce").dropna()
            if len(series) >= 2:
                lo, hi = float(series.min()), float(series.max())
                pad = (hi - lo) * 0.1 or abs(hi) * 0.1 or 1.0
                bounds[target] = (lo - pad, hi + pad)
    return bounds


def _desirability(value: float, goal: str, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    if goal == "maximize":
        d = (value - lo) / (hi - lo)
    elif goal == "minimize":
        d = (hi - value) / (hi - lo)
    else:  # "target" -- triangular, peaking inside PH_TARGET_RANGE
        t_lo, t_hi = PH_TARGET_RANGE
        if value < lo or value > hi:
            d = 0.0
        elif t_lo <= value <= t_hi:
            d = 1.0
        elif value < t_lo:
            d = (value - lo) / (t_lo - lo) if t_lo > lo else 1.0
        else:
            d = (hi - value) / (hi - t_hi) if hi > t_hi else 1.0
    return float(np.clip(d, 0.0, 1.0))


def _decode_individual(ind, components) -> Dict[str, float]:
    """Individual genome: len(components) raw mixture weights (normalized to sum 1) + 5 process vars in [0,1]."""
    n_mix = len(components)
    raw_mix = np.abs(np.array(ind[:n_mix])) + 1e-9
    mix = raw_mix / raw_mix.sum()
    values = {comp: mix[i] for i, comp in enumerate(components)}
    for i, (name, (lo, hi)) in enumerate(PROCESS_RANGES.items()):
        frac = float(np.clip(ind[n_mix + i], 0.0, 1.0))
        values[name] = lo + frac * (hi - lo)
    return values


def _predict_all(model_results: dict, feature_row: pd.DataFrame) -> Dict[str, float]:
    preds = {}
    for target, bundle in model_results.items():
        cols = bundle["feature_names"]
        x = feature_row.reindex(columns=cols, fill_value=0.0).values
        preds[target] = float(bundle["best_model"].predict(x)[0])
    return preds


def run_optimization(model_results: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Genetic-algorithm search for the formulations with the highest overall
    Derringer desirability, evaluated through the trained models.

    Returns a DataFrame of the top N_RESULTS distinct candidates, ranked by
    descending 'score' (0-1 overall desirability), with predicted responses.
    """
    bounds = _response_bounds(df)
    usable_targets = [t for t in model_results if t in bounds]
    if not usable_targets:
        raise ValueError(
            "No target has both a trained model AND enough historical data to set "
            "desirability bounds. Add more labeled rows and retrain."
        )

    components = infer_mixture_components(df)
    if len(components) < 2:
        raise ValueError("Could not detect at least 2 mixture materials in the training data.")

    feature_names = next(iter(model_results.values()))["feature_names"]

    def evaluate(ind):
        values = _decode_individual(ind, components)
        row = pd.DataFrame([values]).reindex(columns=feature_names, fill_value=0.0)
        preds = _predict_all(model_results, row)
        ds = [
            _desirability(preds[t], RESPONSE_GOALS.get(t, "maximize"), *bounds[t])
            for t in usable_targets
            if t in preds
        ]
        if not ds:
            return (0.0,)
        overall = float(np.prod(ds) ** (1.0 / len(ds)))  # geometric mean
        return (overall,)

    toolbox = base.Toolbox()
    n_genes = len(components) + len(PROCESS_RANGES)
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.15, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(42)
    pop = toolbox.population(n=POP_SIZE)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    progress = [max(ind.fitness.values[0] for ind in pop)]
    for _ in range(N_GENERATIONS):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mut in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mut)
                del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring
        progress.append(max(ind.fitness.values[0] for ind in pop))

    ranked_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)

    rows = []
    seen = set()
    for ind in ranked_pop:
        values = _decode_individual(ind, components)
        key = tuple(round(values[c], 3) for c in components)
        if key in seen:
            continue
        seen.add(key)
        row = pd.DataFrame([values]).reindex(columns=feature_names, fill_value=0.0)
        preds = _predict_all(model_results, row)
        record = {**values, **{f"Predicted_{k}": v for k, v in preds.items()}, "score": ind.fitness.values[0]}
        rows.append(record)
        if len(rows) >= N_RESULTS:
            break

    result_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    result_df.attrs["optimization_progress"] = progress
    return result_df
