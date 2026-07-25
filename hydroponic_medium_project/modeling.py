"""
src/modeling.py
================
Trains and caches predictive models for each measured response (target)
column found in the experimental data, and exposes the trained bundles for
use by src.optimization.

For every target, several candidate regressors are fit and cross-validated;
the best one (by CV R^2) is kept. This avoids hard-coding a single algorithm
that might not suit every response (e.g. Crushing_Strength might be highly
non-linear while pH might be nearly linear in composition).

Cross-validation strategy: K-Fold with k = min(5, n_samples) when there is
enough data, automatically falling back to Leave-One-Out for very small
datasets (fewer than 10 rows) so every sample is still used for validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:  # pragma: no cover - xgboost is in requirements.txt but degrade gracefully
    _HAS_XGB = False

from .data_utils import prepare_feature_matrix

CACHE_FILE = "training_cache.joblib"


def _candidate_models(n_samples: int) -> dict:
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=1, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=80, random_state=42, max_depth=3),
        "Ridge": Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "SVR": Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.05))]),
    }
    if _HAS_XGB and n_samples >= 8:
        models["XGBoost"] = XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.12, random_state=42, verbosity=0, n_jobs=-1
        )
    return models


def _cv_splitter(n_samples: int):
    if n_samples < 10:
        return LeaveOneOut()
    return KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)


def _data_hash(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def train_all_targets(df: pd.DataFrame) -> dict:
    """
    Train + cross-validate models for every response column present in df.

    Returns
    -------
    dict: {target_name: {
        "best_model_name": str,
        "best_model": fitted estimator (on ALL available rows for that target),
        "best_metrics": {"r2": float, "rmse": float},
        "all_metrics": {model_name: {"r2":.., "rmse":..}, ...},
        "feature_names": [...],
    }}
    """
    X_full, y_dict = prepare_feature_matrix(df)
    if X_full.empty or not y_dict:
        raise ValueError(
            "No usable feature/target columns found. Expected feature columns like "
            "SAC/Lime/Temperature_C/... and target columns like Bulk_Density/Crushing_Strength/..."
        )

    results = {}
    for target, y_full in y_dict.items():
        valid = y_full.dropna().index.intersection(X_full.index)
        X = X_full.loc[valid].values
        y = y_full.loc[valid].values
        n = len(y)
        if n < 3:
            continue  # not enough labeled rows to fit anything at all for this target

        splitter = _cv_splitter(n)
        all_metrics = {}
        fitted_by_name = {}

        for name, model in _candidate_models(n).items():
            try:
                preds = cross_val_predict(model, X, y, cv=splitter)
                r2 = r2_score(y, preds)
                rmse = float(np.sqrt(mean_squared_error(y, preds)))
                all_metrics[name] = {"r2": float(r2), "rmse": rmse}
                fitted_by_name[name] = model
            except Exception as exc:  # keep going even if one model fails for a given target
                all_metrics[name] = {"r2": None, "rmse": None, "error": str(exc)}

        scored = {k: v["r2"] for k, v in all_metrics.items() if v.get("r2") is not None}
        if not scored:
            continue
        best_name = max(scored, key=scored.get)
        best_model = _candidate_models(n)[best_name]
        best_model.fit(X, y)  # final refit on ALL labeled rows for this target

        results[target] = {
            "best_model_name": best_name,
            "best_model": best_model,
            "best_metrics": all_metrics[best_name],
            "all_metrics": all_metrics,
            "feature_names": list(X_full.columns),
            "n_samples": n,
        }

    if not results:
        raise ValueError(
            "Could not train any target -- every response column had fewer than 3 labeled rows. "
            "Enter more laboratory measurements before training."
        )
    return results


def save_training_artifacts(project_dir: Path, results: dict, data_path: Path, df: pd.DataFrame) -> Path:
    from .data_utils import ensure_project_dirs
    dirs = ensure_project_dirs(project_dir)
    cache_path = dirs["models"] / CACHE_FILE
    payload = {
        "data_path": str(data_path),
        "data_hash": _data_hash(df),
        "results": results,
    }
    joblib.dump(payload, cache_path)

    summary_path = dirs["models"] / "training_summary.json"
    summary = {
        target: {
            "model": bundle["best_model_name"],
            "r2": bundle["best_metrics"].get("r2"),
            "rmse": bundle["best_metrics"].get("rmse"),
            "n_samples": bundle["n_samples"],
        }
        for target, bundle in results.items()
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cache_path


def load_training_artifacts(project_dir: Path, data_path: Path, df: pd.DataFrame) -> Optional[dict]:
    from .data_utils import ensure_project_dirs
    dirs = ensure_project_dirs(project_dir)
    cache_path = dirs["models"] / CACHE_FILE
    if not cache_path.exists():
        return None
    try:
        payload = joblib.load(cache_path)
    except Exception:
        return None
    if payload.get("data_hash") != _data_hash(df):
        return None  # data changed since the cache was built -- force retraining
    return payload.get("results")
