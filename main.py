# src/main.py
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pyomo.environ as pyo
import streamlit as st
from pyomo.opt import SolverFactory

from src.loaders.ucblock_thermal import load_ucblock_thermal
from src.model.pyomo_uc import build_model


def extract_time_series(model):
    """Pulls useful series from the solved Pyomo model for reporting/plots."""
    times = list(model.T)

    demand = [pyo.value(model.demand[t]) for t in times]
    thermal = {l: [pyo.value(model.p[l, t]) for t in times] for l in model.L}
    commitment = {l: [pyo.value(model.y[l, t]) for t in times] for l in model.L}
    volumes = {r: [pyo.value(model.V[r, t]) for t in times] for r in model.R}

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
        yaxis=dict(title="Volume", range=[0, max(max(v) for v in volumes.values()) * 1.1]),
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


def run_optimization():
    """Build, solve and collect outputs for the Streamlit dashboard."""
    project_root = Path(__file__).resolve().parents[0]

    data_path = project_root / "data" / "10_0_1_w.nc4"
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_ucblock_thermal(str(data_path))

    # Données hydro plausibles (4 liens -> 5 réservoirs en cascade)
    K_list = [1, 2, 3, 4]
    Smax = 1

    # Turbines k=1..4
    Fplus_min_dict = {(k, t): 0.0 for k in K_list for t in range(1, data.T + 1)}
    Fplus_max_dict = {(k, t): 30.0 for k in K_list for t in range(1, data.T + 1)}  # débit max
    gplus_dict = {k: 10.0 for k in K_list}  # rampe débit
    Pplus_min_dict = {k: 0.0 for k in K_list}
    Pplus_max_dict = {k: 25.0 for k in K_list}  # puissance max associée

    # Pompage k=1..4 (aval -> amont)
    Fminus_min_dict = {(k, t): 0.0 for k in K_list for t in range(1, data.T + 1)}
    Fminus_max_dict = {(k, t): 15.0 for k in K_list for t in range(1, data.T + 1)}
    gminus_dict = {k: 10.0 for k in K_list}
    rho_pump_dict = {k: 1.1 for k in K_list}  # coût énergétique du pompage

    # Courbe puissance turbine (linéaire avec Smax=1 => 2 points) pour chaque lien
    f_breakpoints_dict = {(k, 1): 0.0 for k in K_list}
    f_breakpoints_dict.update({(k, 2): 30.0 for k in K_list})
    P_breakpoints_dict = {(k, 1): 0.0 for k in K_list}
    P_breakpoints_dict.update({(k, 2): 25.0 for k in K_list})

    # Réservoirs r=1..5
    reservoirs = [1, 2, 3, 4, 5]
    Vmin_dict = {r: 50.0 for r in reservoirs}
    Vmax_dict = {r: 500.0 for r in reservoirs}
    inflow_dict = {(r, t): 2.0 for r in reservoirs for t in range(1, data.T + 1)}  # apports naturels
    V0_dict = {1: 350.0, 2: 320.0, 3: 280.0, 4: 260.0, 5: 240.0}  # volumes initiaux différenciés

    m = build_model(
        T=data.T,
        dt=data.dt,
        L_list=data.L_list,
        K_list=K_list,
        Smax=Smax,
        tau_plus_dict=data.tau_plus_dict,
        tau_minus_dict=data.tau_minus_dict,
        demand_dict=data.demand_dict,
        c_dict=data.c_dict,
        startup_dict=data.startup_dict,
        Pmin_dict=data.Pmin_dict,
        Pmax_dict=data.Pmax_dict,
        g_dict=data.g_dict,
        Fplus_min_dict=Fplus_min_dict,
        Fplus_max_dict=Fplus_max_dict,
        gplus_dict=gplus_dict,
        Pplus_min_dict=Pplus_min_dict,
        Pplus_max_dict=Pplus_max_dict,
        Fminus_min_dict=Fminus_min_dict,
        Fminus_max_dict=Fminus_max_dict,
        gminus_dict=gminus_dict,
        rho_pump_dict=rho_pump_dict,
        f_breakpoints_dict=f_breakpoints_dict,
        P_breakpoints_dict=P_breakpoints_dict,
        Vmin_dict=Vmin_dict,
        Vmax_dict=Vmax_dict,
        inflow_dict=inflow_dict,
        V0_dict=V0_dict,
    )

    lp_path = out_dir / "uc_model.lp"
    m.write(str(lp_path), io_options={"symbolic_solver_labels": True})

    solver = SolverFactory("glpk")
    results = solver.solve(m, tee=False)

    series = extract_time_series(m)
    objective = pyo.value(m.obj)

    return {
        "series": series,
        "objective": objective,
        "solver_status": str(results.solver.status),
        "termination": str(results.solver.termination_condition),
        "lp_path": lp_path,
    }


def main():
    st.set_page_config(page_title="Dashboard optimisation UC", layout="wide")
    st.title("Optimisation UC hydro-thermique")

    if "run_results" not in st.session_state:
        with st.spinner("Résolution du modèle en cours..."):
            st.session_state.run_results = run_optimization()

    if st.button("Relancer l'optimisation"):
        with st.spinner("Nouvelle résolution..."):
            st.session_state.run_results = run_optimization()

    results = st.session_state.run_results
    series = results["series"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Coût total", f"{results['objective']:.2f}")
    col2.write(f"Statut solveur : {results['solver_status']}")
    col3.write(f"Termination : {results['termination']}")

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
