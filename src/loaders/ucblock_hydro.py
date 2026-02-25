"""
Hydro data loader for SMS++ UCBlock files containing a HydroUnitBlock.

Mapping summary
---------------
HydroUnitBlock (NC4, 0-indexed)        → Pyomo build_model() (1-indexed)
----------------------------------------  ----------------------------------------
NumberArcs  arcs  k_data = 0..n-1        K_list = [1..n_arcs]
EndArc.max() = n_arcs → n_arcs+1 nodes  R = n_arcs + 1  (last node = discharge)
NumberReservoirs = n_arcs                r_pyomo = r_data + 1  (r=1..n_arcs)
                                         r_discharge = n_arcs + 1
Inflows[r_data, t_data]                 inflow_dict[(r_data+1, t_data+1)]
InitialVolumetric[r_data]               V0_dict[r_data+1]
MinVolumetric[r_data, :]               Vmin_dict[r_data+1]  (constant over time)
MaxVolumetric[r_data, :]               Vmax_dict[r_data+1]  (constant over time)
MinFlow[t_data, k_data]                Fplus_min_dict[(k_data+1, t_data+1)]
MaxFlow[t_data, k_data]                Fplus_max_dict[(k_data+1, t_data+1)]
DeltaRampUp[0, k_data]                 gplus_dict[k_data+1]
MinPower[0, k_data]                    Pplus_min_dict[k_data+1]
MaxPower[0, k_data]                    Pplus_max_dict[k_data+1]
LinearTerm / ConstantTerm / NumberPieces → f_bp_dict / P_bp_dict (reconstructed)
UphillFlow = [0..0]  → no pumping       Fminus_*=0, gminus/rho_pump=1 (dummy)

Known mismatches / simplifications
-----------------------------------
1. PIECEWISE CURVE: The NC4 stores piecewise slopes/intercepts (LinearTerm,
   ConstantTerm, NumberPieces). The Pyomo model expects explicit breakpoints
   (f_bp, P_bp) for a shared Smax across all arcs.
   → For smax=1 (default): use (MinFlow, MinPower) and (MaxFlow, MaxPower).
   → For smax=-1 (from data): use max(NumberPieces); reconstruct breakpoints by
     equal-interval sampling over [MinFlow, MaxFlow] and evaluating P(F) via
     the natural intersection points of consecutive segments.

2. VOLUME BOUNDS: Data provides time-varying (n_res, T) arrays; model expects
   scalars per reservoir. We take the global min/max over time.

3. PUMPING: UphillFlow = [0..0] means no pumping in these instances. All pump
   parameters are set to zero (Fminus_max=0).

4. DISCHARGE RESERVOIR (r = n_arcs+1): Not in the NC4 data. Initialised with
   V0=0, Vmin=0, Vmax=1e12 (effectively unbounded sink).

5. T ALIGNMENT: The hydro file's TimeHorizon must match the thermal file's T
   when both are used together. The pHydro files have T=96; the thermal-only
   10_0_1_w.nc4 files have T=24. Use the same source NC4 for both loaders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from netCDF4 import Dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class HydroUCData:
    """
    All hydro parameters ready for build_model(), indexed as Pyomo expects.
    Time:       t ∈ {1, …, T}   (1-indexed)
    Links:      k ∈ K_list       (1-indexed integers)
    Reservoirs: r ∈ {1, …, R}   (1-indexed; r=R is the discharge end)
    """
    T: int
    dt: float
    K_list: List[int]       # [1, 2, …, n_arcs]
    R: int                  # n_arcs + 1

    Smax: int               # number of piecewise segments (same for all arcs)

    # Turbine flow bounds (k, t) → float
    Fplus_min_dict: Dict[Tuple[int, int], float]
    Fplus_max_dict: Dict[Tuple[int, int], float]

    # Turbine ramp rates   k → float
    gplus_dict: Dict[int, float]

    # Turbine power bounds k → float
    Pplus_min_dict: Dict[int, float]
    Pplus_max_dict: Dict[int, float]

    # Piecewise breakpoints  (k, i) where i = 1 … Smax+1
    f_bp_dict: Dict[Tuple[int, int], float]
    P_bp_dict: Dict[Tuple[int, int], float]

    # Pump parameters  (disabled when no pumping: Fminus_max = 0)
    Fminus_min_dict: Dict[Tuple[int, int], float]
    Fminus_max_dict: Dict[Tuple[int, int], float]
    gminus_dict: Dict[int, float]
    rho_pump_dict: Dict[int, float]

    # Reservoir parameters
    Vmin_dict:    Dict[int, float]               # r → min volume
    Vmax_dict:    Dict[int, float]               # r → max volume
    inflow_dict:  Dict[Tuple[int, int], float]   # (r, t) → natural inflow
    V0_dict:      Dict[int, float]               # r → initial volume

    # Arc connectivity (1-indexed k → 1-indexed reservoir)
    # These encode the actual DAG topology read from StartArc/EndArc in the NC4.
    arc_start_dict: Dict[int, int]   # k → source reservoir (upstream)
    arc_end_dict:   Dict[int, int]   # k → destination reservoir (downstream)

    # True when the file has a non-simple cascade (parallel arcs / bifurcations).
    is_non_simple_topology: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arc_breakpoints(
    n_pieces: int,
    a: np.ndarray,
    b: np.ndarray,
    f_lo: float,
    f_hi: float,
    smax: int,
) -> Tuple[List[float], List[float]]:
    """
    Compute smax+1 flow breakpoints (equally spaced in [f_lo, f_hi]) and
    their corresponding power values P(F) = a_i*F + b_i via the piecewise
    linear turbine curve.

    Strategy
    --------
    * Build a segment map: list of (f_start, piece_index) pairs.
      Only genuine piece transitions (different a or b) create boundaries;
      identical adjacent pieces (same slope AND same intercept) are merged.
    * For each sampled flow F, look up the active piece and evaluate P.
    """
    # Build segment map: list of (f_boundary, piece_index)
    seg_map: List[Tuple[float, int]] = [(f_lo, 0)]
    for i in range(n_pieces - 1):
        da = abs(float(a[i]) - float(a[i + 1]))
        db = abs(float(b[i]) - float(b[i + 1]))
        if da > 1e-15:
            # Compute intersection point
            f_j = (float(b[i + 1]) - float(b[i])) / (float(a[i]) - float(a[i + 1]))
            if f_lo < f_j < f_hi:
                seg_map.append((float(f_j), i + 1))
            # If intersection is outside [f_lo, f_hi], the piece transition
            # happens at the boundary — use the dominant piece in [f_lo, f_hi].
            elif f_j <= f_lo:
                seg_map[0] = (f_lo, i + 1)   # transition before range: use later piece
        elif db > 1e-10:
            # Same slope, different intercept: continuous only at intersection
            # (should be outside range for physical piecewise); use equal split
            f_j = f_lo + (i + 1) * (f_hi - f_lo) / n_pieces
            if f_lo < f_j < f_hi:
                seg_map.append((float(f_j), i + 1))
        # else: identical piece → no new boundary

    # Sort by flow boundary (should already be sorted for well-formed data)
    seg_map.sort(key=lambda x: x[0])

    def _eval(f: float) -> float:
        """Return P(F) using the segment map."""
        piece = seg_map[-1][1]   # default: last piece
        for j in range(len(seg_map) - 1, -1, -1):
            if f >= seg_map[j][0] - 1e-10:
                piece = seg_map[j][1]
                break
        return float(a[piece]) * f + float(b[piece])

    # Degenerate arc (f_lo == f_hi)
    if f_hi <= f_lo:
        P0 = _eval(f_lo)
        return [f_lo] * (smax + 1), [P0] * (smax + 1)

    # Sample smax+1 equally-spaced points
    f_breaks = [f_lo + i * (f_hi - f_lo) / smax for i in range(smax + 1)]
    P_breaks = [max(0.0, _eval(f)) for f in f_breaks]

    return f_breaks, P_breaks


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_ucblock_hydro(
    path: str,
    block: str = "Block_0",
    smax: int = 1,
    dt: float = 1.0,
) -> HydroUCData:
    """
    Load a HydroUnitBlock from an SMS++ UCBlock NC4 file.

    Parameters
    ----------
    path  : path to the .nc4 file (typically a pHydro_*.nc4 instance)
    block : root group name (default "Block_0")
    smax  : number of piecewise segments for the turbine power curve.
            1  → linear approximation from (MinFlow, MinPower) to (MaxFlow, MaxPower).
            -1 → derive from data: max(NumberPieces) over all arcs.
    dt    : time step size used in the Pyomo model (default 1.0)

    Returns
    -------
    HydroUCData with all parameters indexed as build_model() expects.
    """
    log.info("Loading hydro data from '%s'", path)
    nc = Dataset(path, "r")
    root = nc.groups[block]

    # ── time horizon ──────────────────────────────────────────────────────
    if "TimeHorizon" not in root.dimensions:
        raise KeyError(
            f"{block} has no 'TimeHorizon' dimension. "
            f"Found: {list(root.dimensions)}"
        )
    T = int(root.dimensions["TimeHorizon"].size)
    log.info("  TimeHorizon = %d", T)

    # ── find HydroUnitBlock ───────────────────────────────────────────────
    unit_groups = sorted(
        [g for g in root.groups if g.startswith("UnitBlock_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    hydro_group = None
    for gname in reversed(unit_groups):
        ug = root.groups[gname]
        if getattr(ug, "type", None) == "HydroUnitBlock":
            hydro_group = ug
            log.info("  Found HydroUnitBlock: %s", gname)
            break

    if hydro_group is None:
        raise ValueError(
            f"No HydroUnitBlock found in '{path}/{block}'. "
            f"Last 3 unit groups: {unit_groups[-3:]}. "
            "Make sure the file is a pHydro instance."
        )

    hg = hydro_group

    # ── dimensions ────────────────────────────────────────────────────────
    n_reservoirs  = int(hg.dimensions["NumberReservoirs"].size)
    n_arcs        = int(hg.dimensions["NumberArcs"].size)
    n_intervals   = int(hg.dimensions["NumberIntervals"].size)
    total_pieces  = int(hg.dimensions["TotalNumberPieces"].size)

    log.info(
        "  Dimensions: %d reservoirs, %d arcs, %d intervals, %d total pieces",
        n_reservoirs, n_arcs, n_intervals, total_pieces,
    )

    if n_intervals != T:
        raise ValueError(
            f"HydroUnitBlock NumberIntervals={n_intervals} "
            f"!= Block TimeHorizon={T}. "
            "Cannot align hydro and demand time series."
        )

    # ── read raw arrays ───────────────────────────────────────────────────
    def _arr(name: str) -> np.ndarray:
        assert name in hg.variables, (
            f"Missing variable '{name}' in HydroUnitBlock. "
            f"Available: {list(hg.variables)}"
        )
        return np.array(hg.variables[name][:])

    start_arc     = _arr("StartArc").astype(int)    # (n_arcs,)
    end_arc       = _arr("EndArc").astype(int)       # (n_arcs,)
    inflows       = _arr("Inflows")                  # (n_reservoirs, T)
    init_vol      = _arr("InitialVolumetric")        # (n_reservoirs,)
    min_vol       = _arr("MinVolumetric")            # (n_reservoirs, T)
    max_vol       = _arr("MaxVolumetric")            # (n_reservoirs, T)
    min_flow      = _arr("MinFlow")                  # (T, n_arcs)
    max_flow      = _arr("MaxFlow")                  # (T, n_arcs)
    min_power     = _arr("MinPower")                 # (T, n_arcs)
    max_power     = _arr("MaxPower")                 # (T, n_arcs)
    ramp_up       = _arr("DeltaRampUp")              # (T, n_arcs)
    n_pieces_arr  = _arr("NumberPieces").astype(int) # (n_arcs,)
    linear_terms  = _arr("LinearTerm")               # (TotalPieces,)
    const_terms   = _arr("ConstantTerm")             # (TotalPieces,)

    nc.close()

    # ── validate cascade structure ─────────────────────────────────────────
    expected_start = list(range(n_arcs))
    expected_end   = list(range(1, n_arcs + 1))
    _is_simple_chain = (
        start_arc.tolist() == expected_start
        and end_arc.tolist() == expected_end
    )
    if not _is_simple_chain:
        log.warning(
            "Non-simple cascade topology detected:\n"
            "  StartArc = %s\n"
            "  EndArc   = %s\n"
            "  The Pyomo model assumes a linear chain "
            "(StartArc[k]=k, EndArc[k]=k+1). "
            "Parallel links or bifurcations are mapped by arc index but "
            "the water balance constraints will be incorrect. "
            "Consider using pHydro_1A_none.nc4 (simple chain topology).",
            start_arc.tolist(), end_arc.tolist(),
        )

    pieces_sum = int(np.sum(n_pieces_arr))
    assert pieces_sum == total_pieces, (
        f"sum(NumberPieces)={pieces_sum} != TotalNumberPieces={total_pieces}"
    )

    # ── Pyomo indices ──────────────────────────────────────────────────────
    K_list  = list(range(1, n_arcs + 1))   # 1-indexed arcs
    # R_total must equal n_arcs + 1, because pyomo_uc.py always computes
    #   R = max(K_list) + 1 = n_arcs + 1
    # and defines m.R = RangeSet(1, R), so Vmin/Vmax must be defined for all
    # r in 1..n_arcs+1.  Using max(EndArc)+1 can be < n_arcs+1 for non-simple
    # topologies (parallel arcs), which would leave some Vmin[r] undefined.
    R_total = n_arcs + 1

    # Arc connectivity: 0-indexed NC4 node ids → 1-indexed Pyomo reservoir ids
    # (NC4 node n  ↔  Pyomo reservoir n+1)
    arc_start_dict: Dict[int, int] = {k: int(start_arc[k - 1]) + 1 for k in K_list}
    arc_end_dict:   Dict[int, int] = {k: int(end_arc[k - 1])   + 1 for k in K_list}

    # ── resolve smax ──────────────────────────────────────────────────────
    if smax < 0:
        smax = int(np.max(n_pieces_arr))
        log.info("  Smax derived from data = %d (max NumberPieces)", smax)
    else:
        log.info("  Smax = %d (provided)", smax)

    # ── piecewise breakpoints ─────────────────────────────────────────────
    f_bp_dict: Dict[Tuple[int, int], float] = {}
    P_bp_dict: Dict[Tuple[int, int], float] = {}
    piece_offset = 0

    for ka in range(n_arcs):
        k   = ka + 1                              # 1-indexed
        n_p = int(n_pieces_arr[ka])
        a   = linear_terms[piece_offset: piece_offset + n_p]
        b   = const_terms [piece_offset: piece_offset + n_p]
        piece_offset += n_p

        # Flow range for this arc: use t=0 values (typically constant over time)
        f_lo = float(min_flow[0, ka])
        f_hi = float(max_flow[0, ka])

        f_breaks, P_breaks = _arc_breakpoints(n_p, a, b, f_lo, f_hi, smax)

        for i, (fv, Pv) in enumerate(zip(f_breaks, P_breaks)):
            f_bp_dict[(k, i + 1)] = float(fv)
            P_bp_dict[(k, i + 1)] = max(0.0, float(Pv))

    log.info(
        "  Breakpoints computed for %d arcs (Smax=%d, %d points each)",
        n_arcs, smax, smax + 1,
    )

    # ── turbine flow bounds ────────────────────────────────────────────────
    Fplus_min_dict: Dict[Tuple[int, int], float] = {}
    Fplus_max_dict: Dict[Tuple[int, int], float] = {}
    for ka in range(n_arcs):
        k = ka + 1
        for t_data in range(T):
            t = t_data + 1
            Fplus_min_dict[(k, t)] = float(min_flow[t_data, ka])
            Fplus_max_dict[(k, t)] = float(max_flow[t_data, ka])

    # ── ramp rates  (take t=0; data is constant in all tested files) ───────
    gplus_dict: Dict[int, float] = {}
    for ka in range(n_arcs):
        gplus_dict[ka + 1] = float(ramp_up[0, ka])

    # ── power bounds (t=0 value per arc) ──────────────────────────────────
    Pplus_min_dict: Dict[int, float] = {ka + 1: float(min_power[0, ka]) for ka in range(n_arcs)}
    Pplus_max_dict: Dict[int, float] = {ka + 1: float(max_power[0, ka]) for ka in range(n_arcs)}

    # ── pump parameters (no pumping in these instances) ───────────────────
    Fminus_min_dict: Dict[Tuple[int, int], float] = {
        (k, t): 0.0 for k in K_list for t in range(1, T + 1)
    }
    Fminus_max_dict: Dict[Tuple[int, int], float] = {
        (k, t): 0.0 for k in K_list for t in range(1, T + 1)
    }
    # Use 1.0 (not 0.0) for gminus / rho_pump to avoid division-by-zero in
    # pump constraints that multiply by these params, even when flow = 0.
    gminus_dict:    Dict[int, float] = {k: 1.0 for k in K_list}
    rho_pump_dict:  Dict[int, float] = {k: 1.0 for k in K_list}

    # ── reservoir parameters ───────────────────────────────────────────────
    Vmin_dict:   Dict[int, float]               = {}
    Vmax_dict:   Dict[int, float]               = {}
    V0_dict:     Dict[int, float]               = {}
    inflow_dict: Dict[Tuple[int, int], float]   = {}

    for r_data in range(n_reservoirs):
        r = r_data + 1
        # Volume bounds: take global min/max over time (constant in practice)
        Vmin_dict[r] = float(np.min(min_vol[r_data, :]))
        Vmax_dict[r] = float(np.max(max_vol[r_data, :]))
        V0_dict[r]   = float(init_vol[r_data])
        # Warn if Vmax == 0 (run-of-river node: might cause infeasibility)
        if Vmax_dict[r] == 0.0:
            log.warning(
                "  Reservoir r=%d has Vmax=0 (run-of-river node; V0 must also be 0).",
                r,
            )
        for t_data in range(T):
            inflow_dict[(r, t_data + 1)] = float(inflows[r_data, t_data])

    # Discharge reservoir(s): node indices NOT covered by NC4 reservoir data.
    # NumberReservoirs covers nodes 0..n_reservoirs-1 → Pyomo r=1..n_reservoirs.
    # Any Pyomo r > n_reservoirs up to R_total gets dummy (sink) values.
    for r in range(n_reservoirs + 1, R_total + 1):
        Vmin_dict[r]  = 0.0
        Vmax_dict[r]  = 1e12   # unbounded sink
        V0_dict[r]    = 0.0
        for t in range(1, T + 1):
            inflow_dict[(r, t)] = 0.0
    r_disch = R_total  # kept for logging below

    # ── summary log ───────────────────────────────────────────────────────
    log.info("  K_list = %s", K_list)
    log.info("  R = %d  (reservoirs 1..%d, discharge at %d)", R_total, n_arcs, r_disch)
    log.info(
        "  V0  : %s",
        {r: f"{v:.4g}" for r, v in V0_dict.items() if r != r_disch},
    )
    log.info(
        "  Vmax: %s",
        {r: f"{v:.4g}" for r, v in Vmax_dict.items() if r != r_disch},
    )
    log.info(
        "  MaxFlow (t=1) : %s",
        {k + 1: f"{max_flow[0, k]:.4g}" for k in range(n_arcs)},
    )
    log.info(
        "  MaxPower(t=1) : %s",
        {k + 1: f"{max_power[0, k]:.4g}" for k in range(n_arcs)},
    )
    log.info(
        "  Inflow ranges : %s",
        {
            r + 1: f"[{inflows[r].min():.3g}, {inflows[r].max():.3g}]"
            for r in range(n_reservoirs)
        },
    )

    return HydroUCData(
        T=T,
        dt=dt,
        K_list=K_list,
        R=R_total,
        Smax=smax,
        Fplus_min_dict=Fplus_min_dict,
        Fplus_max_dict=Fplus_max_dict,
        gplus_dict=gplus_dict,
        Pplus_min_dict=Pplus_min_dict,
        Pplus_max_dict=Pplus_max_dict,
        f_bp_dict=f_bp_dict,
        P_bp_dict=P_bp_dict,
        Fminus_min_dict=Fminus_min_dict,
        Fminus_max_dict=Fminus_max_dict,
        gminus_dict=gminus_dict,
        rho_pump_dict=rho_pump_dict,
        Vmin_dict=Vmin_dict,
        Vmax_dict=Vmax_dict,
        inflow_dict=inflow_dict,
        V0_dict=V0_dict,
        arc_start_dict=arc_start_dict,
        arc_end_dict=arc_end_dict,
        is_non_simple_topology=not _is_simple_chain,
    )
