from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from netCDF4 import Dataset


@dataclass
class ThermalUCData:
    T: int
    dt: float              # si pas fourni dans le fichier, on met 1.0 par défaut
    L_list: List[str]

    demand_dict: Dict[int, float]
    c_dict: Dict[Tuple[str, int], float]
    startup_dict: Dict[str, float]
    Pmin_dict: Dict[Tuple[str, int], float]
    Pmax_dict: Dict[Tuple[str, int], float]
    g_dict: Dict[Tuple[str, int], float]
    tau_plus_dict: Dict[str, int]
    tau_minus_dict: Dict[str, int]


def _read_scalar_or_1d(var) -> float:
    a = np.array(var[:])
    if a.size == 1:
        return float(a.reshape(-1)[0])
    # si jamais c'est un vecteur, on prend le premier
    return float(a.reshape(-1)[0])


def load_ucblock_thermal(path: str, block: str = "Block_0") -> ThermalUCData:
    """
    Reads a UCBlock thermal-only instance structured as:
      /Block_0/ (dims: TimeHorizon, NumberUnits, NumberIntervals)
      /Block_0/ActivePowerDemand (typically indexed by time)
      /Block_0/UnitBlock_i/ with scalar or time-indexed vars

    Returns dicts indexed exactly like your Pyomo model expects:
      t in 1..T, unit labels in L_list
    """
    nc = Dataset(path, "r")
    root = nc.groups[block]

    # --- dimensions ---
    # Most important: TimeHorizon
    if "TimeHorizon" not in root.dimensions:
        raise KeyError(f"{block} has no 'TimeHorizon' dimension. Found: {list(root.dimensions.keys())}")
    T = len(root.dimensions["TimeHorizon"])

    # dt is not shown in your group listing; default to 1.0 unless you know otherwise.
    dt = 1.0

    # --- demand ---
    if "ActivePowerDemand" not in root.variables:
        raise KeyError(f"{block} has no 'ActivePowerDemand'. Found: {list(root.variables.keys())}")

    dem = np.array(root.variables["ActivePowerDemand"][:]).reshape(-1)
    if dem.size not in (T,):
        # sometimes stored on NumberIntervals; keep robust:
        dem = dem[:T]
    demand_dict = {t + 1: float(dem[t]) for t in range(T)}

    # --- units: UnitBlock_0 ... UnitBlock_{U-1} ---
    unit_groups = [g for g in root.groups.keys() if g.startswith("UnitBlock_")]
    unit_groups_sorted = sorted(unit_groups, key=lambda s: int(s.split("_")[-1]))
    if not unit_groups_sorted:
        raise ValueError(f"No UnitBlock_i groups found under {block}/")

    L_list = [f"U{int(name.split('_')[-1])}" for name in unit_groups_sorted]

    # Prepare dicts
    c_dict: Dict[Tuple[str, int], float] = {}
    startup_dict: Dict[str, float] = {}
    Pmin_dict: Dict[Tuple[str, int], float] = {}
    Pmax_dict: Dict[Tuple[str, int], float] = {}
    g_dict: Dict[Tuple[str, int], float] = {}
    tau_plus_dict: Dict[str, int] = {}
    tau_minus_dict: Dict[str, int] = {}

    # Helper to read a variable that can be scalar or time series
    def read_unit_time_or_scalar(ug, varname: str) -> np.ndarray:
        if varname not in ug.variables:
            raise KeyError(f"Missing {varname} in group {ug.path}. Found: {list(ug.variables.keys())}")
        arr = np.array(ug.variables[varname][:]).reshape(-1)
        if arr.size == 1:
            return np.repeat(arr[0], T)
        # if longer than T, truncate; if shorter, error
        if arr.size < T:
            raise ValueError(f"{ug.path}/{varname} has length {arr.size} < T={T}")
        return arr[:T]

    for u_label, ug_name in zip(L_list, unit_groups_sorted):
        ug = root.groups[ug_name]

        # Power limits
        pmin = read_unit_time_or_scalar(ug, "MinPower")
        pmax = read_unit_time_or_scalar(ug, "MaxPower")

        # Ramp: you have DeltaRampUp and DeltaRampDown.
        # Your Pyomo model uses one g[l,t] for both up/down constraints.
        # Conservative choice: g = min(ramp_up, ramp_down) (or take ramp_up if symmetric).
        ramp_up = read_unit_time_or_scalar(ug, "DeltaRampUp")
        ramp_dn = read_unit_time_or_scalar(ug, "DeltaRampDown")
        ramp = np.minimum(ramp_up, ramp_dn)

        # Costs: UCBlock provides quadratic + linear + constant terms.
        # Your Pyomo objective uses c[l,t] * p[l,t] (linear variable cost).
        # So we map LinearTerm -> c. (QuadTerm ignored unless you change model.)
        lin = read_unit_time_or_scalar(ug, "LinearTerm")
        su = _read_scalar_or_1d(ug.variables["StartUpCost"]) if "StartUpCost" in ug.variables else 0.0

        # Min up/down times
        mu = int(_read_scalar_or_1d(ug.variables["MinUpTime"])) if "MinUpTime" in ug.variables else 1
        md = int(_read_scalar_or_1d(ug.variables["MinDownTime"])) if "MinDownTime" in ug.variables else 1

        startup_dict[u_label] = float(su)
        tau_plus_dict[u_label] = mu
        tau_minus_dict[u_label] = md

        for t in range(T):
            tt = t + 1
            Pmin_dict[(u_label, tt)] = float(pmin[t])
            Pmax_dict[(u_label, tt)] = float(pmax[t])
            g_dict[(u_label, tt)] = float(ramp[t])
            c_dict[(u_label, tt)] = float(lin[t])

    nc.close()

    return ThermalUCData(
        T=T, dt=dt, L_list=L_list,
        demand_dict=demand_dict,
        c_dict=c_dict, startup_dict=startup_dict,
        Pmin_dict=Pmin_dict, Pmax_dict=Pmax_dict, g_dict=g_dict,
        tau_plus_dict=tau_plus_dict, tau_minus_dict=tau_minus_dict,
    )
