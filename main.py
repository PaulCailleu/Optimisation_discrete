# src/main.py
from pathlib import Path

from src.loaders.ucblock_thermal import load_ucblock_thermal
from src.model.pyomo_uc import build_model
from pyomo.opt import SolverFactory

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


if __name__ == "__main__":
    main()
