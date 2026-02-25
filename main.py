# src/main.py
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pyomo.environ as pyo
import streamlit as st
from pyomo.opt import SolverFactory

from src.loaders.ucblock_thermal import load_ucblock_thermal
from src.loaders.ucblock_hydro import load_ucblock_hydro
from src.model.pyomo_uc import build_model


def extract_time_series(model):
    """Pulls useful series from the solved Pyomo model for reporting/plots."""
    times = list(model.T)

    demand = [pyo.value(model.demand[t]) for t in times]
    thermal = {l: [pyo.value(model.p[l, t]) for t in times] for l in model.L}
    commitment = {l: [pyo.value(model.y[l, t]) for t in times] for l in model.L}
    # Only keep "real" reservoirs: exclude run-of-river nodes (Vmax=0) and the
    # unbounded discharge sink (Vmax >= 1e10). Showing them would crush the y-axis.
    volumes = {
        r: [pyo.value(model.V[r, t]) for t in times]
        for r in model.R
        if 0 < pyo.value(model.Vmax[r]) < 1e10
    }

    # Hydro contribution (can be zero with the dummy data in main)
    hydro_gen = [sum(pyo.value(model.Pplus[k, t]) for k in model.K) for t in times]
    pump_use = [
        sum(pyo.value(model.rho_pump[k] * model.fminus[k, t]) for k in model.K)
        for t in times
    ]
    hydro_net = [g - u for g, u in zip(hydro_gen, pump_use)]

    total_prod = [
        sum(thermal[l][i] for l in model.L) + hydro_net[i] for i in range(len(times))
    ]

    return {
        "times": times,
        "demand": demand,
        "thermal": thermal,
        "commitment": commitment,
        "volumes": volumes,
        "hydro_gen": hydro_gen,
        "pump_use": pump_use,
        "hydro_net": hydro_net,
        "total_prod": total_prod,
    }


def build_figures(series):
    """Create Plotly figures for the Streamlit dashboard."""
    times = series["times"]
    demand = series["demand"]
    hydro_gen = series["hydro_gen"]
    pump_use = series["pump_use"]
    total = series["total_prod"]
    thermal = series["thermal"]
    commitment = series["commitment"]
    volumes = series["volumes"]

    # Dispatch stacked areas
    dispatch_fig = go.Figure()
    for l, values in thermal.items():
        dispatch_fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=f"Thermique {l}",
                stackgroup="prod",
            )
        )
    dispatch_fig.add_trace(
        go.Scatter(
            x=times,
            y=hydro_gen,
            mode="lines",
            name="Hydro (prod)",
            stackgroup="prod",
        )
    )
    dispatch_fig.add_trace(
        go.Scatter(
            x=times,
            y=[-u for u in pump_use],
            mode="lines",
            name="Pompage (conso)",
            stackgroup="cons",
            line=dict(color="grey"),
        )
    )
    dispatch_fig.add_trace(
        go.Scatter(
            x=times,
            y=demand,
            mode="lines",
            name="Demande",
            line=dict(dash="dash", color="black"),
        )
    )
    dispatch_fig.add_trace(
        go.Scatter(
            x=times,
            y=total,
            mode="lines",
            name="Production totale",
            line=dict(color="firebrick", width=3),
        )
    )
    dispatch_fig.update_layout(
        title="Dispatch par pas de temps",
        xaxis_title="Temps",
        yaxis_title="Puissance",
        legend_orientation="h",
        legend_y=-0.2,
        hovermode="x unified",
    )

    # Commitment steps
    commit_fig = go.Figure()
    for l, values in commitment.items():
        commit_fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                line_shape="hv",
                name=f"Unité {l}",
            )
        )
    commit_fig.update_layout(
        title="Plan de fonctionnement des unités",
        xaxis_title="Temps",
        yaxis_title="Etat (0/1)",
        yaxis=dict(range=[-0.1, 1.1]),
        hovermode="x unified",
        legend_orientation="h",
        legend_y=-0.2,
    )

    # Reservoir volumes
    volumes_fig = go.Figure()
    for r, values in volumes.items():
        volumes_fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines+markers",
                name=f"Réservoir {r}",
            )
        )
    volumes_fig.update_layout(
        title="Evolution des volumes des réservoirs",
        xaxis_title="Temps",
        yaxis_title="Volume",
        hovermode="x unified",
        legend_orientation="h",
        legend_y=-0.2,
    )

    # Animated "water level" bars per réservoir
    reservoirs = list(volumes.keys())
    base_frame = go.Bar(
        x=[f"R{r}" for r in reservoirs],
        y=[volumes[r][0] for r in reservoirs],
        marker=dict(color="royalblue", opacity=0.75),
    )

    frames = []
    for idx, t in enumerate(times):
        frames.append(
            go.Frame(
                name=str(t),
                data=[
                    go.Bar(
                        x=[f"R{r}" for r in reservoirs],
                        y=[volumes[r][idx] for r in reservoirs],
                        marker=dict(color="royalblue", opacity=0.8),
                    )
                ],
                layout=go.Layout(title_text=f"Niveaux des réservoirs – t={t}"),
            )
        )

    anim_layout = go.Layout(
        title="Animation des volumes (play)",
        xaxis=dict(title="Réservoir"),
        yaxis=dict(title="Volume", range=[0, (max(max(v) for v in volumes.values()) * 1.1) if volumes else 1]),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": str(t),
                        "method": "animate",
                    }
                    for t in times
                ],
                "transition": {"duration": 0},
                "x": 0.05,
                "len": 0.9,
            }
        ],
    )
    anim_fig = go.Figure(data=[base_frame], frames=frames, layout=anim_layout)

    return {
        "dispatch": dispatch_fig,
        "commitment": commit_fig,
        "volumes": volumes_fig,
        "animation": anim_fig,
    }


def to_dataframes(series):
    """Convert series to tabular formats for Streamlit display."""
    times = series["times"]

    dispatch_df = pd.DataFrame(
        {
            "Demande": series["demand"],
            "Hydro (prod)": series["hydro_gen"],
            "Pompage (conso)": series["pump_use"],
            "Hydro net": series["hydro_net"],
            "Production totale": series["total_prod"],
        },
        index=times,
    )

    for l, values in series["thermal"].items():
        dispatch_df[f"Thermique {l}"] = values

    commitment_df = pd.DataFrame(series["commitment"], index=times)
    volumes_df = pd.DataFrame(series["volumes"], index=times)

    return dispatch_df, commitment_df, volumes_df


def run_optimization(
    thermal_path: str | None = None,
    hydro_path: str | None = None,
):
    """Build, solve and collect outputs for the Streamlit dashboard."""
    project_root = Path(__file__).resolve().parents[0]

    if thermal_path is None:
        thermal_path = str(project_root / "data" / "10_0_1_w.nc4")

    if hydro_path is None:
        hydro_path = str(
            project_root
            / "data"
            / "smspp-hydro-units-main-Given Data"
            / "Given Data"
            / "20090907_pHydro_1A_none.nc4"
        )

    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Thermal data ──────────────────────────────────────────────────────────
    thermal = load_ucblock_thermal(thermal_path)

    # ── Hydro data ────────────────────────────────────────────────────────────
    # T is taken from thermal (demand drives the horizon).
    # Hydro dicts are defined for t=1..hydro.T; if hydro.T >= thermal.T the
    # extra entries are simply never accessed by build_model().
    hydro = load_ucblock_hydro(hydro_path, smax=1)

    m = build_model(
        T=thermal.T,
        dt=thermal.dt,
        L_list=thermal.L_list,
        K_list=hydro.K_list,
        Smax=hydro.Smax,
        # Thermal
        tau_plus_dict=thermal.tau_plus_dict,
        tau_minus_dict=thermal.tau_minus_dict,
        demand_dict=thermal.demand_dict,
        c_dict=thermal.c_dict,
        startup_dict=thermal.startup_dict,
        Pmin_dict=thermal.Pmin_dict,
        Pmax_dict=thermal.Pmax_dict,
        g_dict=thermal.g_dict,
        # Hydro – turbine
        Fplus_min_dict=hydro.Fplus_min_dict,
        Fplus_max_dict=hydro.Fplus_max_dict,
        gplus_dict=hydro.gplus_dict,
        Pplus_min_dict=hydro.Pplus_min_dict,
        Pplus_max_dict=hydro.Pplus_max_dict,
        # Hydro – pompage (désactivé)
        Fminus_min_dict=hydro.Fminus_min_dict,
        Fminus_max_dict=hydro.Fminus_max_dict,
        gminus_dict=hydro.gminus_dict,
        rho_pump_dict=hydro.rho_pump_dict,
        # Hydro – courbe puissance (linéaire par morceaux)
        f_breakpoints_dict=hydro.f_bp_dict,
        P_breakpoints_dict=hydro.P_bp_dict,
        # Hydro – réservoirs
        Vmin_dict=hydro.Vmin_dict,
        Vmax_dict=hydro.Vmax_dict,
        inflow_dict=hydro.inflow_dict,
        V0_dict=hydro.V0_dict,
        # Hydro – connectivité des arcs (topologie générale)
        arc_start_dict=hydro.arc_start_dict,
        arc_end_dict=hydro.arc_end_dict,
    )

    lp_path = out_dir / "uc_model.lp"
    m.write(str(lp_path), io_options={"symbolic_solver_labels": True})

    solver = SolverFactory("glpk")
    # tmlim: wall-clock time limit in seconds (GLPK option).
    # pHydro files have T=96 and ~149 thermal units → very large MILP;
    # without a limit GLPK can run indefinitely.
    results = solver.solve(m, tee=False, options={"tmlim": 300})

    termination = str(results.solver.termination_condition)
    # GLPK may return "maxTimeLimit" or "other" when the time limit is hit;
    # still try to extract the incumbent objective if one was found.
    try:
        objective = pyo.value(m.obj)
    except Exception:
        objective = float("nan")

    series = extract_time_series(m)

    return {
        "series": series,
        "objective": objective,
        "solver_status": str(results.solver.status),
        "termination": termination,
        "lp_path": lp_path,
        "topology_warning": hydro.is_non_simple_topology,
        "T_thermal": thermal.T,
        "T_hydro": hydro.T,
        "n_units": len(thermal.L_list),
        "n_arcs": len(hydro.K_list),
    }


def main():
    st.set_page_config(page_title="Dashboard optimisation UC", layout="wide")
    st.title("Optimisation UC hydro-thermique")

    project_root = Path(__file__).resolve().parents[0]
    thermal_dir = project_root / "data"
    hydro_dir   = project_root / "data" / "smspp-hydro-units-main-Given Data" / "Given Data"

    thermal_files = sorted(thermal_dir.glob("*.nc4")) if thermal_dir.exists() else []
    hydro_files   = sorted(hydro_dir.glob("*.nc4"))   if hydro_dir.exists()   else []

    with st.sidebar:
        st.header("Données thermiques")
        if thermal_files:
            chosen_thermal = st.selectbox(
                "Fichier thermique",
                options=[str(f) for f in thermal_files],
                format_func=lambda p: Path(p).name,
            )
        else:
            chosen_thermal = None
            st.warning("Aucun fichier thermique trouvé dans data/")

        st.header("Données hydrauliques")
        if hydro_files:
            chosen_hydro = st.selectbox(
                "Fichier hydro (pHydro)",
                options=[str(f) for f in hydro_files],
                format_func=lambda p: Path(p).name,
            )
        else:
            chosen_hydro = None
            st.warning("Aucun fichier hydro trouvé dans data/")

    cache_key = (chosen_thermal, chosen_hydro)
    if "run_results" not in st.session_state or st.session_state.get("last_cache_key") != cache_key:
        with st.spinner("Résolution du modèle en cours..."):
            st.session_state.run_results = run_optimization(chosen_thermal, chosen_hydro)
            st.session_state.last_cache_key = cache_key

    if st.button("Relancer l'optimisation"):
        with st.spinner("Nouvelle résolution..."):
            st.session_state.run_results = run_optimization(chosen_thermal, chosen_hydro)
            st.session_state.last_cache_key = cache_key

    results = st.session_state.run_results
    series = results["series"]

    # ── Avertissements ────────────────────────────────────────────────────────
    T_th, T_hy = results["T_thermal"], results["T_hydro"]
    if T_th != T_hy:
        st.info(
            f"Horizons différents : thermique T={T_th}, hydro T={T_hy}. "
            f"Le modèle utilise T={T_th} (thermique). "
            "Les données hydro pour t > T sont ignorées."
        )
    if results.get("topology_warning"):
        st.warning(
            "**Topologie non-simple détectée.**  \n"
            "Le fichier hydro contient des arcs parallèles ou des bifurcations. "
            "Le bilan hydraulique est correct (formulation DAG générale), "
            "mais vérifiez que la topologie correspond bien à votre instance."
        )

    col1, col2, col3, col4 = st.columns(4)
    obj_val = results["objective"]
    col1.metric("Coût total", f"{obj_val:.2f}" if obj_val == obj_val else "N/A (time limit)")
    col2.metric("Unités thermiques", results["n_units"])
    col3.metric("Arcs hydro", results["n_arcs"])
    col4.write(f"**Solveur :** {results['solver_status']}  \n**Term. :** {results['termination']}")

    st.caption(f"Fichier LP généré : `{results['lp_path']}`")

    figs = build_figures(series)
    st.plotly_chart(figs["dispatch"], width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(figs["commitment"], width='stretch')
    with col_b:
        st.plotly_chart(figs["volumes"], width='stretch')

    st.subheader("Animation volumes réservoirs")
    st.plotly_chart(figs["animation"], width='stretch')

    dispatch_df, commitment_df, volumes_df = to_dataframes(series)

    st.subheader("Dispatch & hydrologie")
    st.dataframe(dispatch_df.style.format("{:.2f}"))

    col_commit, col_vol = st.columns(2)
    with col_commit:
        st.subheader("Commitment (0/1)")
        st.dataframe(commitment_df)
    with col_vol:
        st.subheader("Volumes des réservoirs")
        st.dataframe(volumes_df.style.format("{:.2f}"))


if __name__ == "__main__":
    main()
