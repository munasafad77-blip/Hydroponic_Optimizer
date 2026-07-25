"""Streamlit application for hydroponic medium DOE, modeling, optimization, and reporting."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.data_utils import (
    ensure_project_dirs,
    load_experiment_data,
    save_experiment_data,
    prepare_feature_matrix,
    select_data_source,
    load_persisted_data_selection,
    save_persisted_data_selection,
    load_materials,
    save_materials,
    infer_mixture_components,
    DEFAULT_MATERIALS,
    TARGET_COLUMNS,
)
from src.mixture_design import build_doe
from src.modeling import load_training_artifacts, save_training_artifacts, train_all_targets
from src.optimization import run_optimization
from src.visualization import (
    plot_mixture_triangle,
    plot_response_surface,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_optimization_progress,
    plot_property_ranking,
    plot_radar_comparison,
)

st.set_page_config(page_title="Hydroponic Medium Optimizer", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
DIRS = ensure_project_dirs(PROJECT_DIR)

if "materials" not in st.session_state:
    st.session_state["materials"] = load_materials(PROJECT_DIR)

sample_data_path = DIRS["data"] / "experimental_data.csv"
training_data_path = DIRS["data"] / "experimental_data_training.csv"
training_extended_data_path = DIRS["data"] / "experimental_data_training_extended.csv"


def _default_bootstrap_sample(materials):
    """Curated 8-row example for the default SAC/Lime pair; a procedurally
    generated (clearly synthetic) bootstrap for any other material list."""
    if materials == DEFAULT_MATERIALS:
        return pd.DataFrame([
            {"SAC": 0.65, "Lime": 0.35, "Temperature_C": 1100.0, "Holding_Time_min": 15.0, "Heating_Rate_C_per_min": 10.0, "Pellet_Size_mm": 8.0, "Shell_Thickness_mm": 1.0, "Bulk_Density": 1.35, "Particle_Density": 2.10, "Open_Porosity": 0.42, "Water_Absorption": 0.28, "Crushing_Strength": 36.2, "pH": 7.1, "EC": 0.45, "Water_Retention": 0.71, "Plant_Growth_Index": 0.82},
            {"SAC": 0.70, "Lime": 0.30, "Temperature_C": 1050.0, "Holding_Time_min": 12.0, "Heating_Rate_C_per_min": 12.0, "Pellet_Size_mm": 7.0, "Shell_Thickness_mm": 1.2, "Bulk_Density": 1.28, "Particle_Density": 2.02, "Open_Porosity": 0.45, "Water_Absorption": 0.24, "Crushing_Strength": 38.1, "pH": 6.9, "EC": 0.40, "Water_Retention": 0.74, "Plant_Growth_Index": 0.88},
            {"SAC": 0.60, "Lime": 0.40, "Temperature_C": 1150.0, "Holding_Time_min": 18.0, "Heating_Rate_C_per_min": 8.0, "Pellet_Size_mm": 9.0, "Shell_Thickness_mm": 0.9, "Bulk_Density": 1.40, "Particle_Density": 2.15, "Open_Porosity": 0.39, "Water_Absorption": 0.31, "Crushing_Strength": 34.8, "pH": 7.3, "EC": 0.49, "Water_Retention": 0.68, "Plant_Growth_Index": 0.76},
            {"SAC": 0.68, "Lime": 0.32, "Temperature_C": 1080.0, "Holding_Time_min": 14.0, "Heating_Rate_C_per_min": 11.0, "Pellet_Size_mm": 8.0, "Shell_Thickness_mm": 1.1, "Bulk_Density": 1.31, "Particle_Density": 2.05, "Open_Porosity": 0.44, "Water_Absorption": 0.26, "Crushing_Strength": 37.4, "pH": 7.0, "EC": 0.42, "Water_Retention": 0.73, "Plant_Growth_Index": 0.85},
            {"SAC": 0.63, "Lime": 0.37, "Temperature_C": 1120.0, "Holding_Time_min": 16.0, "Heating_Rate_C_per_min": 9.0, "Pellet_Size_mm": 8.0, "Shell_Thickness_mm": 1.0, "Bulk_Density": 1.37, "Particle_Density": 2.12, "Open_Porosity": 0.41, "Water_Absorption": 0.29, "Crushing_Strength": 35.6, "pH": 7.2, "EC": 0.47, "Water_Retention": 0.70, "Plant_Growth_Index": 0.79},
            {"SAC": 0.72, "Lime": 0.28, "Temperature_C": 1030.0, "Holding_Time_min": 11.0, "Heating_Rate_C_per_min": 13.0, "Pellet_Size_mm": 7.0, "Shell_Thickness_mm": 1.3, "Bulk_Density": 1.25, "Particle_Density": 1.98, "Open_Porosity": 0.47, "Water_Absorption": 0.22, "Crushing_Strength": 39.5, "pH": 6.8, "EC": 0.38, "Water_Retention": 0.76, "Plant_Growth_Index": 0.90},
            {"SAC": 0.58, "Lime": 0.42, "Temperature_C": 1170.0, "Holding_Time_min": 20.0, "Heating_Rate_C_per_min": 7.0, "Pellet_Size_mm": 9.0, "Shell_Thickness_mm": 0.8, "Bulk_Density": 1.43, "Particle_Density": 2.18, "Open_Porosity": 0.37, "Water_Absorption": 0.33, "Crushing_Strength": 33.2, "pH": 7.4, "EC": 0.51, "Water_Retention": 0.66, "Plant_Growth_Index": 0.73},
            {"SAC": 0.66, "Lime": 0.34, "Temperature_C": 1100.0, "Holding_Time_min": 15.0, "Heating_Rate_C_per_min": 10.0, "Pellet_Size_mm": 8.0, "Shell_Thickness_mm": 1.0, "Bulk_Density": 1.33, "Particle_Density": 2.08, "Open_Porosity": 0.43, "Water_Absorption": 0.27, "Crushing_Strength": 36.9, "pH": 7.05, "EC": 0.44, "Water_Retention": 0.72, "Plant_Growth_Index": 0.83},
        ])
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(8):
        raw = rng.uniform(0.2, 1.0, size=len(materials))
        frac = raw / raw.sum()
        row = dict(zip(materials, np.round(frac, 4)))
        row.update({
            "Temperature_C": round(float(rng.uniform(1000, 1200)), 1),
            "Holding_Time_min": round(float(rng.uniform(10, 25)), 1),
            "Heating_Rate_C_per_min": round(float(rng.uniform(5, 15)), 1),
            "Pellet_Size_mm": round(float(rng.uniform(6, 10)), 1),
            "Shell_Thickness_mm": round(float(rng.uniform(0.8, 1.5)), 2),
            "Bulk_Density": round(float(rng.uniform(1.0, 1.5)), 3),
            "Particle_Density": round(float(rng.uniform(1.8, 2.3)), 3),
            "Open_Porosity": round(float(rng.uniform(0.30, 0.50)), 3),
            "Water_Absorption": round(float(rng.uniform(0.18, 0.35)), 3),
            "Crushing_Strength": round(float(rng.uniform(28, 42)), 2),
            "pH": round(float(rng.uniform(6.5, 7.6)), 2),
            "EC": round(float(rng.uniform(0.30, 0.60)), 3),
            "Water_Retention": round(float(rng.uniform(0.60, 0.80)), 3),
            "Plant_Growth_Index": round(float(rng.uniform(0.65, 0.92)), 3),
        })
        rows.append(row)
    return pd.DataFrame(rows)


if not sample_data_path.exists():
    if training_extended_data_path.exists():
        sample_df = pd.read_csv(training_extended_data_path)
    elif training_data_path.exists():
        sample_df = pd.read_csv(training_data_path)
    else:
        sample_df = _default_bootstrap_sample(st.session_state["materials"])
    sample_df.to_csv(sample_data_path, index=False)

persisted_path, persisted_source = load_persisted_data_selection(PROJECT_DIR)
if "selected_data_path" not in st.session_state:
    if persisted_path is not None and persisted_path.exists():
        st.session_state["selected_data_path"] = str(persisted_path)
        st.session_state["selected_data_source"] = persisted_source or "persisted"
    else:
        selected_data_path, selected_source = select_data_source(PROJECT_DIR)
        st.session_state["selected_data_path"] = str(selected_data_path)
        st.session_state["selected_data_source"] = selected_source

st.title("Hydroponic Growing Medium Design & Optimization")
st.caption("Configurable multi-material development platform for hydroponic growing media")

menu = st.sidebar.radio("Workflow", ["DOE", "Data Entry", "Modeling", "Optimization", "Reports"])

if menu == "DOE":
    st.subheader("Materials")
    st.caption("Add, remove, or rename the mixture materials. Every material here becomes a column that must sum to 1.0 across every recipe. Changes apply the next time you generate a DOE.")
    materials_df = pd.DataFrame({"Material": st.session_state["materials"]})
    edited_materials = st.data_editor(
        materials_df, num_rows="dynamic", key="materials_editor", width='stretch',
        column_config={"Material": st.column_config.TextColumn("Material", required=True)},
    )
    if st.button("Save materials"):
        new_list = [str(m).strip() for m in edited_materials["Material"].tolist() if str(m).strip()]
        try:
            saved = save_materials(PROJECT_DIR, new_list)
            st.session_state["materials"] = saved
            st.success(f"Materials saved: {', '.join(saved)}")
        except ValueError as exc:
            st.error(str(exc))

    materials = st.session_state["materials"]
    st.divider()
    st.subheader("Mixture Design")
    method = st.selectbox("DOE Method", ["manual", "simplex_lattice", "simplex_centroid", "d_optimal"])

    manual_rows = None
    levels, n_points = 4, 12
    if method == "manual":
        st.caption(f"Enter the exact ratio of each material for as many rows (experiments) as you want. Each row is normalized to sum to 1.0 automatically -- you do not need to hit exactly 1.0 yourself. Materials: {', '.join(materials)}")
        default_val = round(1.0 / len(materials), 3)
        grid_df = pd.DataFrame([{m: default_val for m in materials} for _ in range(5)])
        edited_grid = st.data_editor(
            grid_df, num_rows="dynamic", key="manual_ratio_editor", width='stretch',
            column_config={m: st.column_config.NumberColumn(m, min_value=0.0, step=0.01) for m in materials},
        )
        manual_rows = edited_grid.to_dict(orient="records")
    elif method == "simplex_lattice":
        levels = st.slider("Lattice level", 3, 8, 4)
    elif method == "d_optimal":
        n_points = st.slider("Number of D-optimal points", 6, 20, 12)

    if st.button("Generate DOE"):
        try:
            doe = build_doe(method=method, levels=levels, n_points=n_points,
                             manual_rows=manual_rows, components=materials)
            doe_path = DIRS["data"] / "doe_design.csv"
            doe.to_csv(doe_path, index=False)
            st.success(f"DOE generated and saved to {doe_path}")
            st.dataframe(doe)
        except ValueError as exc:
            st.error(str(exc))

elif menu == "Data Entry":
    st.subheader("Laboratory Data")
    st.caption(f"Current materials: {', '.join(st.session_state['materials'])} -- change these from the DOE tab if needed before uploading.")
    uploaded = st.file_uploader("Upload experimental CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        save_path = DIRS["data"] / "experimental_data.csv"
        df.to_csv(save_path, index=False)
        save_persisted_data_selection(PROJECT_DIR, save_path, "uploaded")
        st.session_state["selected_data_path"] = str(save_path)
        st.session_state["selected_data_source"] = "uploaded"
        detected = infer_mixture_components(df)
        if detected and set(detected) != set(st.session_state["materials"]):
            st.session_state["materials"] = detected
            save_materials(PROJECT_DIR, detected)
            st.info(f"Materials list updated from your upload: {', '.join(detected)}")
        st.success(f"Saved to {save_path}")
        st.dataframe(df.head())
    else:
        st.info("Upload a CSV containing mixture, process, and response columns.")

elif menu == "Modeling":
    st.subheader("Model Training")
    selected_data_path = Path(st.session_state.get("selected_data_path", str(DIRS["data"] / "experimental_data.csv")))
    selected_source = st.session_state.get("selected_data_source", "auto")
    st.caption(f"Using data source: {selected_source} -> {selected_data_path.name}")
    st.info(f"Training will use: {selected_data_path}")

    if selected_data_path.exists():
        df = pd.read_csv(selected_data_path)
        st.metric("Rows loaded", len(df))
        st.metric("Columns loaded", len(df.columns))

        if st.button("Train Models", type="primary"):
            cached = load_training_artifacts(PROJECT_DIR, selected_data_path, df)
            if cached is not None:
                st.session_state["model_results"] = cached
                st.session_state["training_data_path"] = str(selected_data_path)
                st.session_state["training_summary"] = [
                    {
                        "target": target,
                        "model": bundle["best_model_name"],
                        "r2": bundle["best_metrics"].get("r2"),
                        "rmse": bundle["best_metrics"].get("rmse"),
                    }
                    for target, bundle in cached.items()
                ]
                st.info("Loaded cached training results")
            else:
                try:
                    with st.spinner("Training models..."):
                        results = train_all_targets(df)
                except ValueError as exc:
                    st.error(f"Could not train models: {exc}")
                    results = None
                if results is not None:
                    st.session_state["model_results"] = results
                    st.session_state["training_data_path"] = str(selected_data_path)
                    st.session_state["training_summary"] = [
                        {
                            "target": target,
                            "model": bundle["best_model_name"],
                            "r2": bundle["best_metrics"].get("r2"),
                            "rmse": bundle["best_metrics"].get("rmse"),
                        }
                        for target, bundle in results.items()
                    ]
                    save_training_artifacts(PROJECT_DIR, results, selected_data_path, df)
            if "model_results" in st.session_state:
                save_persisted_data_selection(PROJECT_DIR, selected_data_path, selected_source)
                st.success(f"Training completed using {selected_data_path.name}")
                st.dataframe(pd.DataFrame(st.session_state["training_summary"]))

        if st.button("Run Optimization + Generate Report"):
            if "model_results" not in st.session_state:
                st.warning("Train models first")
            else:
                try:
                    with st.spinner("Running optimization and generating report..."):
                        data_path = Path(st.session_state.get("training_data_path", str(DIRS["data"] / "experimental_data.csv")))
                        ranked = run_optimization(st.session_state["model_results"], pd.read_csv(data_path))
                        ranked.to_csv(DIRS["results"] / "ranked_formulations.csv", index=False)
                        st.session_state["optimization_results"] = ranked
                        best_row = ranked.iloc[0]
                        materials_now = infer_mixture_components(ranked)
                        st.session_state["report_summary"] = {
                            "best_formulation": {m: float(best_row[m]) for m in materials_now},
                            "score": float(best_row["score"]),
                            "row_count": int(len(ranked)),
                        }
                    st.success("Optimization and report generation completed")
                    st.dataframe(ranked.head(10))
                except Exception as exc:
                    st.error(f"Optimization failed: {exc}")

        if "training_summary" in st.session_state:
            st.caption("Latest training summary")
            st.dataframe(pd.DataFrame(st.session_state["training_summary"]))
    else:
        st.info("Upload experimental data first")

elif menu == "Optimization":
    st.subheader("Multi-objective Optimization")
    if "model_results" in st.session_state:
        model_results = st.session_state["model_results"]
        if st.button("Run Optimization"):
            try:
                data_path = Path(st.session_state.get("training_data_path", str(DIRS["data"] / "experimental_data.csv")))
                ranked = run_optimization(model_results, pd.read_csv(data_path))
                ranked.to_csv(DIRS["results"] / "ranked_formulations.csv", index=False)
                st.session_state["optimization_results"] = ranked
                st.success("Optimization completed")
                st.dataframe(ranked)
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")

        if "optimization_results" in st.session_state:
            st.caption("Latest optimization output")
            st.dataframe(st.session_state["optimization_results"])
    else:
        st.info("Train models first")

elif menu == "Reports":
    st.subheader("Reports and Figures")
    data_path = Path(st.session_state.get("training_data_path", str(DIRS["data"] / "experimental_data.csv")))
    if data_path.exists():
        df = pd.read_csv(data_path)
        materials = infer_mixture_components(df) or st.session_state["materials"]

        present_targets = [t for t in TARGET_COLUMNS if t in df.columns]
        st.caption("Pick the hydroponic properties you care about most -- charts below are built around this selection.")
        selected_props = st.multiselect(
            "Properties that matter to you", options=present_targets,
            default=present_targets, key="reports_selected_props",
        )

        st.write("General figures")
        plot_mixture_triangle(df, str(DIRS["figures"] / "mixture_triangle.png"), materials=materials)
        st.image(str(DIRS["figures"] / "mixture_triangle.png"))

        if selected_props:
            surface_target = st.selectbox("Response-surface property", selected_props, key="surface_target")
            plot_response_surface(df, surface_target, str(DIRS["figures"] / "response_surface.png"))
            st.image(str(DIRS["figures"] / "response_surface.png"))

        plot_correlation_heatmap(df, str(DIRS["figures"] / "correlation_heatmap.png"))
        st.image(str(DIRS["figures"] / "correlation_heatmap.png"))

        if "model_results" in st.session_state:
            importances = {
                name: bundle["best_metrics"].get("r2", 0.0) or 0.0
                for name, bundle in st.session_state["model_results"].items()
            }
            plot_feature_importance(importances, str(DIRS["figures"] / "feature_importance.png"))
        else:
            plot_feature_importance({}, str(DIRS["figures"] / "feature_importance.png"))
        st.image(str(DIRS["figures"] / "feature_importance.png"))

        ranked = st.session_state.get("optimization_results")
        progress = list(ranked.attrs.get("optimization_progress", [])) if ranked is not None else []
        plot_optimization_progress(progress, str(DIRS["figures"] / "optimization_progress.png"))
        st.image(str(DIRS["figures"] / "optimization_progress.png"))

        if ranked is not None and selected_props:
            st.write("Property-focused results (from the latest optimization run)")
            for prop in selected_props:
                fig_path = DIRS["figures"] / f"ranking_{prop}.png"
                plot_property_ranking(ranked, prop, str(fig_path))
                st.image(str(fig_path))

            if len(selected_props) >= 3:
                radar_path = DIRS["figures"] / "radar_comparison.png"
                plot_radar_comparison(ranked, selected_props, str(radar_path))
                st.image(str(radar_path))
            else:
                st.info("Pick at least 3 properties above to also see a radar comparison of the top formulations.")

            best = ranked.iloc[0]
            best_desc = ", ".join(f"{m}={best[m]:.3f}" for m in materials if m in best)
            st.success(f"Best ranked formulation: {best_desc} (score={best['score']:.3f})")
        elif selected_props:
            st.info("Run an optimization (Modeling or Optimization tab) to unlock property-focused charts and the radar comparison.")
    else:
        st.info("Upload data to create reports")
