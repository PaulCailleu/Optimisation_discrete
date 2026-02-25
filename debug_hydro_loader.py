#!/usr/bin/env python3
"""
debug_hydro_loader.py
---------------------
End-to-end demonstration of the hydro data loader:

  1. Load thermal data from a pHydro NC4 file (ThermalUnitBlocks only).
  2. Load hydro data from the same file (HydroUnitBlock).
  3. Print a mapping report (model parameters vs. loaded values).
  4. Build the Pyomo model with the real data.
  5. Export the model to outputs/uc_hydro.lp.
  6. (Optional) solve with GLPK and report cost.

Usage:
    python debug_hydro_loader.py [path/to/pHydro_file.nc4] [--smax N] [--no-solve]

Defaults:
    path  = data/smspp-hydro-units-main-Given Data/Given Data/20090907_pHydro_1_none.nc4
    smax  = 1  (linear power curve – fully compatible with current Pyomo model)
    solve = True
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ── project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.ucblock_thermal import load_ucblock_thermal
from src.loaders.ucblock_hydro   import load_ucblock_hydro, HydroUCData
from src.model.pyomo_uc          import build_model

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("debug_hydro")


# ── helpers ───────────────────────────────────────────────────────────────────

def _stats(d: dict, label: str) -> None:
    """Print min/max/mean for a numeric dict."""
    if not d:
        print(f"  {label}: (empty)")
        return
    vals = list(d.values())
    print(f"  {label}: n={len(vals)}, min={min(vals):.4g}, max={max(vals):.4g}, "
          f"mean={sum(vals)/len(vals):.4g}")


def print_mapping_report(hydro: HydroUCData) -> None:
    print("\n" + "=" * 60)
    print("MAPPING REPORT — HydroUCData → build_model()")
    print("=" * 60)
    print(f"  T          = {hydro.T}   (time steps)")
    print(f"  dt         = {hydro.dt}")
    print(f"  K_list     = {hydro.K_list}  ({len(hydro.K_list)} arcs)")
    print(f"  R          = {hydro.R}   (reservoirs, incl. discharge at r={hydro.R})")
    print(f"  Smax       = {hydro.Smax}  (piecewise segments per arc)")

    print("\n── Turbine flow bounds ──")
    _stats(hydro.Fplus_min_dict, "Fplus_min  (k,t)")
    _stats(hydro.Fplus_max_dict, "Fplus_max  (k,t)")

    print("\n── Turbine ramp & power ──")
    print(f"  gplus:     {hydro.gplus_dict}")
    print(f"  Pplus_min: {hydro.Pplus_min_dict}")
    print(f"  Pplus_max: {hydro.Pplus_max_dict}")

    print("\n── Piecewise breakpoints ──")
    arcs_shown = sorted({k for k, _ in hydro.f_bp_dict})
    for k in arcs_shown:
        pts = [(i, hydro.f_bp_dict[(k, i)], hydro.P_bp_dict[(k, i)])
               for i in range(1, hydro.Smax + 2)]
        pts_str = "  ".join(f"F={f:.3g}/P={p:.3g}" for _, f, p in pts)
        print(f"  arc k={k}: {pts_str}")

    print("\n── Reservoir volumes ──")
    for r in sorted(hydro.Vmin_dict):
        print(f"  r={r}: V0={hydro.V0_dict[r]:.4g}, "
              f"Vmin={hydro.Vmin_dict[r]:.4g}, Vmax={hydro.Vmax_dict[r]:.4g}")

    print("\n── Inflows ──")
    for r in sorted({r for r, _ in hydro.inflow_dict}):
        vals = [hydro.inflow_dict[(r, t)] for t in range(1, hydro.T + 1)]
        print(f"  r={r}: min={min(vals):.3g}, max={max(vals):.3g}, "
              f"mean={sum(vals)/len(vals):.3g}")

    print("\n── Pumping (no pumping expected) ──")
    max_fminus = max(hydro.Fminus_max_dict.values(), default=0.0)
    print(f"  Fminus_max (all should be 0): max = {max_fminus:.4g}")
    print("=" * 60 + "\n")


def run_unit_tests(hydro: HydroUCData) -> None:
    """Very lightweight sanity checks."""
    print("── Unit tests ──────────────────────────────────────────────")
    errors = []

    # 1. K_list must be 1-indexed consecutive
    assert hydro.K_list == list(range(1, len(hydro.K_list) + 1)), \
        "K_list is not consecutive 1-indexed"

    # 2. R must equal len(K_list) + 1
    assert hydro.R == len(hydro.K_list) + 1, "R != n_arcs + 1"

    # 3. f_bp must be monotone per arc
    for k in hydro.K_list:
        fs = [hydro.f_bp_dict[(k, i)] for i in range(1, hydro.Smax + 2)]
        assert fs == sorted(fs), f"f_bp arc k={k} not monotone: {fs}"

    # 4. Fplus_min <= Fplus_max
    for key in hydro.Fplus_min_dict:
        lo = hydro.Fplus_min_dict[key]
        hi = hydro.Fplus_max_dict[key]
        if lo > hi + 1e-6:
            errors.append(f"Fplus_min > Fplus_max at {key}: {lo} > {hi}")

    # 5. Vmin <= V0 <= Vmax (excluding discharge reservoir)
    for r in range(1, hydro.R):  # exclude discharge reservoir
        v0   = hydro.V0_dict.get(r, 0.0)
        vmin = hydro.Vmin_dict.get(r, 0.0)
        vmax = hydro.Vmax_dict.get(r, 1e12)
        if not (vmin - 1e-6 <= v0 <= vmax + 1e-6):
            errors.append(
                f"V0[{r}]={v0:.4g} not in [Vmin={vmin:.4g}, Vmax={vmax:.4g}]"
            )

    # 6. Piecewise breakpoints: all covered
    expected_keys = {(k, i) for k in hydro.K_list for i in range(1, hydro.Smax + 2)}
    missing = expected_keys - set(hydro.f_bp_dict.keys())
    if missing:
        errors.append(f"Missing f_bp keys: {sorted(missing)[:5]}…")

    # 7. inflow_dict covers all (r, t)
    expected_inflow = {(r, t) for r in range(1, hydro.R + 1) for t in range(1, hydro.T + 1)}
    missing_inflow = expected_inflow - set(hydro.inflow_dict.keys())
    if missing_inflow:
        errors.append(f"Missing inflow keys: {sorted(missing_inflow)[:5]}…")

    if errors:
        for e in errors:
            print(f"  FAIL  {e}")
        print(f"  {len(errors)} test(s) FAILED")
    else:
        print("  All tests PASSED")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug hydro loader")
    parser.add_argument(
        "nc4_path",
        nargs="?",
        default=str(
            PROJECT_ROOT
            / "data"
            / "smspp-hydro-units-main-Given Data"
            / "Given Data"
            / "20090907_pHydro_1A_none.nc4"
        ),
        help="Path to a pHydro NC4 file",
    )
    parser.add_argument(
        "--smax", type=int, default=1,
        help="Piecewise segments (1=linear, -1=from data, default=1)",
    )
    parser.add_argument(
        "--no-solve", action="store_true",
        help="Skip GLPK solve (only build and export LP)",
    )
    args = parser.parse_args()

    nc4_path = Path(args.nc4_path)
    if not nc4_path.exists():
        log.error("File not found: %s", nc4_path)
        sys.exit(1)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load thermal ──────────────────────────────────────────────────
    log.info("Loading thermal data from '%s'", nc4_path.name)
    thermal = load_ucblock_thermal(str(nc4_path))
    print(f"\nThermal data loaded:")
    print(f"  T          = {thermal.T}")
    print(f"  units      = {len(thermal.L_list)}  ({thermal.L_list[:3]}…)")
    print(f"  demand     : min={min(thermal.demand_dict.values()):.4g}, "
          f"max={max(thermal.demand_dict.values()):.4g}")

    # ── 2. Load hydro ────────────────────────────────────────────────────
    log.info("Loading hydro data from '%s'", nc4_path.name)
    hydro = load_ucblock_hydro(str(nc4_path), smax=args.smax)

    if hydro.T != thermal.T:
        log.warning(
            "T mismatch: thermal T=%d, hydro T=%d. "
            "Both come from the same file so this should not happen.",
            thermal.T, hydro.T,
        )

    # ── 3. Mapping report ────────────────────────────────────────────────
    print_mapping_report(hydro)

    # ── 4. Unit tests ────────────────────────────────────────────────────
    run_unit_tests(hydro)

    # ── 5. Build Pyomo model ─────────────────────────────────────────────
    log.info("Building Pyomo model (T=%d, %d thermal, %d hydro arcs)…",
             thermal.T, len(thermal.L_list), len(hydro.K_list))

    model = build_model(
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
        # Hydro – piecewise
        f_breakpoints_dict=hydro.f_bp_dict,
        P_breakpoints_dict=hydro.P_bp_dict,
        # Hydro – pump (disabled)
        Fminus_min_dict=hydro.Fminus_min_dict,
        Fminus_max_dict=hydro.Fminus_max_dict,
        gminus_dict=hydro.gminus_dict,
        rho_pump_dict=hydro.rho_pump_dict,
        # Hydro – reservoirs
        Vmin_dict=hydro.Vmin_dict,
        Vmax_dict=hydro.Vmax_dict,
        inflow_dict=hydro.inflow_dict,
        V0_dict=hydro.V0_dict,
        # Hydro – arc connectivity (general DAG topology)
        arc_start_dict=hydro.arc_start_dict,
        arc_end_dict=hydro.arc_end_dict,
    )
    log.info("Model built successfully.")

    # ── 6. Export LP ─────────────────────────────────────────────────────
    lp_path = out_dir / "uc_hydro.lp"
    model.write(str(lp_path), io_options={"symbolic_solver_labels": True})
    log.info("LP exported → %s  (%d KB)", lp_path, lp_path.stat().st_size // 1024)

    # ── 7. Solve (optional) ──────────────────────────────────────────────
    if not args.no_solve:
        log.info("Solving with GLPK…")
        solver = SolverFactory("glpk")
        if not solver.available():
            log.warning("GLPK not available – skipping solve.")
        else:
            results = solver.solve(model, tee=False)
            status      = str(results.solver.status)
            termination = str(results.solver.termination_condition)
            log.info("Solver status: %s  |  termination: %s", status, termination)

            if termination == "optimal":
                obj = pyo.value(model.obj)
                print(f"\n  Optimal cost = {obj:.4f}")

                # Summary of hydro production
                times = list(model.T)
                hydro_gen = [
                    sum(pyo.value(model.Pplus[k, t]) for k in model.K)
                    for t in times
                ]
                vols = {
                    r: [pyo.value(model.V[r, t]) for t in times]
                    for r in model.R
                }
                print(f"  Hydro net generation: "
                      f"min={min(hydro_gen):.3g}, max={max(hydro_gen):.3g}, "
                      f"mean={sum(hydro_gen)/len(hydro_gen):.3g}")
                print("  Reservoir final volumes:")
                for r, vs in vols.items():
                    print(f"    r={r}: V(T)={vs[-1]:.4g}")
            else:
                log.warning("No optimal solution found (status=%s, term=%s)",
                            status, termination)
    else:
        log.info("--no-solve flag set; skipping GLPK.")

    print("\nDone.")


if __name__ == "__main__":
    main()
