# src/main.py
from pathlib import Path

import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from src.loaders.ucblock_thermal import load_ucblock_thermal
from src.model.pyomo_uc import build_model


def extract_time_series(model):
    """Pulls useful series from the solved Pyomo model for reporting/plots."""
    times = list(model.T)

    demand = [pyo.value(model.demand[t]) for t in times]
    thermal = {l: [pyo.value(model.p[l, t]) for t in times] for l in model.L}
    commitment = {l: [pyo.value(model.y[l, t]) for t in times] for l in model.L}

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
        "hydro_net": hydro_net,
        "total_prod": total_prod,
    }


def plot_results(series, out_dir: Path):
    """Generate matplotlib figures for dispatch and commitment schedules."""
    times = series["times"]
    demand = series["demand"]
    hydro_net = series["hydro_net"]
    total = series["total_prod"]
    thermal = series["thermal"]
    commitment = series["commitment"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # --- Dispatch plot ---
    ax0 = axes[0]
    thermal_labels = list(thermal.keys())
    thermal_stack = [thermal[l] for l in thermal_labels]
    if thermal_stack:
        ax0.stackplot(times, thermal_stack, labels=thermal_labels, alpha=0.8)

    if any(abs(v) > 1e-6 for v in hydro_net):
        ax0.plot(times, hydro_net, color="teal", label="Hydro net")

    ax0.plot(times, demand, "--", color="black", label="Demande")
    ax0.plot(times, total, color="firebrick", linewidth=2, label="Production totale")
    ax0.set_ylabel("Puissance")
    ax0.legend(loc="upper right")
    ax0.set_title("Dispatch par pas de temps")

    # --- Commitment (on/off) plot ---
    ax1 = axes[1]
    for l, y_series in commitment.items():
        ax1.step(times, y_series, where="post", label=l, linewidth=1.6)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("Temps")
    ax1.set_ylabel("Etat (0/1)")
    ax1.legend(loc="upper right", ncol=max(1, len(commitment) // 2))
    ax1.set_title("Plan de fonctionnement des unités")

    fig.tight_layout()
    out_path = out_dir / "dispatch.png"
    #fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    #plt.close(fig)
    #print(f"Figure enregistrée dans {out_path}")


def main():
    # Racine projet = dossier qui contient "src/"
    project_root = Path(__file__).resolve().parents[0]
    
    # ==========
    # Paths
    # ==========
    data_path = project_root / "data" / "10_0_1_w.nc4"
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==========
    # Load data
    # ==========
    data = load_ucblock_thermal(str(data_path))

    # ==========
    # TEMPORAIRE : données hydro factices pour tester
    # (le fichier UCBlock est thermique-only)
    # ==========
    K_list = [1]
    Smax = 1

    Fplus_min_dict = {(1, t): 0.0 for t in range(1, data.T + 1)}
    Fplus_max_dict = {(1, t): 0.0 for t in range(1, data.T + 1)}
    gplus_dict = {1: 0.0}
    Pplus_min_dict = {1: 0.0}
    Pplus_max_dict = {1: 0.0}

    Fminus_min_dict = {(1, t): 0.0 for t in range(1, data.T + 1)}
    Fminus_max_dict = {(1, t): 0.0 for t in range(1, data.T + 1)}
    gminus_dict = {1: 0.0}
    rho_pump_dict = {1: 0.0}

    # Smax=1 => points i=1..2
    f_breakpoints_dict = {(1, 1): 0.0, (1, 2): 0.0}
    P_breakpoints_dict = {(1, 1): 0.0, (1, 2): 0.0}

    # K_list=[1] => R = max(K_list)+1 = 2 reservoirs
    Vmin_dict = {1: 0.0, 2: 0.0}
    Vmax_dict = {1: 100.0, 2: 100.0}
    inflow_dict = {(r, t): 0.0 for r in [1, 2] for t in range(1, data.T + 1)}

    # ==========
    # Build model
    # ==========
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
    )

    # ==========
    # Export LP
    # ==========
    lp_path = out_dir / "uc_model.lp"
    m.write(str(lp_path), io_options={"symbolic_solver_labels": True})
    print(f"Wrote {lp_path}")

    # =========
    # Solve model
    # =========
    solver = SolverFactory("glpk")
    results = solver.solve(m, tee=True)
    print(results.solver.status, results.solver.termination_condition)
    print(f"Coût total = {pyo.value(m.obj):.2f}")

    # =========
    # Export & plots
    # =========
    series = extract_time_series(m)
    plot_results(series, out_dir)


if __name__ == "__main__":
    main()
