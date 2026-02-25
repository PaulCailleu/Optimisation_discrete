import pyomo.environ as pyo

def build_model(
    T, dt, L_list, K_list, Smax,
    tau_plus_dict, tau_minus_dict,
    demand_dict, c_dict, startup_dict,
    Pmin_dict, Pmax_dict, g_dict,
    Fplus_min_dict, Fplus_max_dict, gplus_dict, Pplus_max_dict, Pplus_min_dict,
    Fminus_min_dict, Fminus_max_dict, gminus_dict, rho_pump_dict,
    f_breakpoints_dict, P_breakpoints_dict,
    Vmin_dict, Vmax_dict, inflow_dict, V0_dict,
    arc_start_dict, arc_end_dict,
):

    m = pyo.ConcreteModel()

    # =========================================================
    # Sets
    # =========================================================
    # T: time steps
    m.T = pyo.RangeSet(1, T)

    # L: thermal units (given)
    m.L = pyo.Set(initialize=L_list)

    # Cascade chain: R reservoirs, K links between reservoirs
    # We assume here that K_list corresponds to the LINKS k=1..R-1
    # and that reservoirs are r=1..R with R = max(K_list)+1.
    m.K = pyo.Set(initialize=K_list, ordered=True)          # links (turbine/pump) between k -> k+1
    R = max(K_list) + 1
    m.R = pyo.RangeSet(1, R)                                # reservoirs

    # Piecewise segments for turbine power curve
    m.S = pyo.RangeSet(1, Smax)                             # segments s=1..Smax, points i=1..Smax+1

    # =========================================================
    # Params
    # =========================================================
    m.dt = pyo.Param(initialize=dt)

    # Thermal min up/down times (per unit)
    m.tau_plus = pyo.Param(m.L, initialize=tau_plus_dict, within=pyo.PositiveIntegers)
    m.tau_minus = pyo.Param(m.L, initialize=tau_minus_dict, within=pyo.PositiveIntegers)

    # Demand and thermal costs
    m.demand = pyo.Param(m.T, initialize=demand_dict)
    m.c = pyo.Param(m.L, m.T, initialize=c_dict)
    m.startup = pyo.Param(m.L, initialize=startup_dict)

    # Thermal limits + ramping
    m.Pmin = pyo.Param(m.L, m.T, initialize=Pmin_dict)
    m.Pmax = pyo.Param(m.L, m.T, initialize=Pmax_dict)
    m.g = pyo.Param(m.L, m.T, initialize=g_dict)

    # Turbine (on each link k)
    # Note: use lambda initializers for time-indexed hydro params so that Pyomo
    # only looks up valid (k, t) in m.K × m.T, even when the supplied dicts
    # contain more time steps than T (e.g. hydro file has T=96, thermal T=24).
    m.Fplus_min = pyo.Param(m.K, m.T, initialize=lambda m, k, t: Fplus_min_dict[k, t])
    m.Fplus_max = pyo.Param(m.K, m.T, initialize=lambda m, k, t: Fplus_max_dict[k, t])
    m.gplus = pyo.Param(m.K, initialize=gplus_dict)
    m.Pplus_max = pyo.Param(m.K, initialize=Pplus_max_dict)
    m.Pplus_min = pyo.Param(m.K, initialize=Pplus_min_dict)

    # Pump (on each link k, pumping from k+1 -> k)
    m.Fminus_min = pyo.Param(m.K, m.T, initialize=lambda m, k, t: Fminus_min_dict[k, t])
    m.Fminus_max = pyo.Param(m.K, m.T, initialize=lambda m, k, t: Fminus_max_dict[k, t])
    m.gminus = pyo.Param(m.K, initialize=gminus_dict)
    m.rho_pump = pyo.Param(m.K, initialize=rho_pump_dict)

    # Piecewise breakpoints for turbine power curve (per link k)
    # Points i=1..Smax+1
    m.f_bp = pyo.Param(m.K, pyo.RangeSet(1, Smax + 1), initialize=f_breakpoints_dict)
    m.P_bp = pyo.Param(m.K, pyo.RangeSet(1, Smax + 1), initialize=P_breakpoints_dict)

    # Reservoir parameters (per reservoir r)
    m.Vmin = pyo.Param(m.R, initialize=Vmin_dict)
    m.Vmax = pyo.Param(m.R, initialize=Vmax_dict)
    m.inflow = pyo.Param(m.R, m.T, initialize=lambda m, r, t: inflow_dict[r, t])
    m.V0 = pyo.Param(m.R, initialize=V0_dict)

    # =========================================================
    # Variables
    # =========================================================
    # Thermal
    m.p = pyo.Var(m.L, m.T, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.L, m.T, within=pyo.Binary)
    m.u = pyo.Var(m.L, m.T, within=pyo.Binary)  # startup
    m.d = pyo.Var(m.L, m.T, within=pyo.Binary)  # shutdown

    # Hydro - links
    m.fplus = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)   # turbine flow k -> k+1
    m.Pplus = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)   # turbine power
    m.fminus = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)  # pump flow k+1 -> k
    m.z = pyo.Var(m.K, m.T, within=pyo.Binary)                 # pump on/off

    # Piecewise helpers (for turbine power curve)
    m.zseg = pyo.Var(m.K, m.S, m.T, within=pyo.Binary)
    m.theta = pyo.Var(m.K, m.S, m.T, bounds=(0, 1))

    # Reservoir volumes
    m.V = pyo.Var(
        m.R, m.T,
        bounds=lambda m, r, t: (pyo.value(m.Vmin[r]), pyo.value(m.Vmax[r]))
    )

    # Initial volume fix
    def init_volume_rule(m, r):
        return m.V[r, m.T.first()] == m.V0[r]
    m.init_volume = pyo.Constraint(m.R, rule=init_volume_rule)

    # =========================================================
    # Objective
    # =========================================================
    def obj_rule(m):
        thermal_cost = sum(m.c[l, t] * m.p[l, t] * m.dt for l in m.L for t in m.T)
        startup_cost = sum(m.startup[l] * m.u[l, t] for l in m.L for t in m.T)
        return thermal_cost + startup_cost

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # =========================================================
    # Constraints - Thermal
    # =========================================================
    def power_min_rule(m, l, t):
        return m.p[l, t] >= m.Pmin[l, t] * m.y[l, t]
    m.power_min = pyo.Constraint(m.L, m.T, rule=power_min_rule)

    def power_max_rule(m, l, t):
        return m.p[l, t] <= m.Pmax[l, t] * m.y[l, t]
    m.power_max = pyo.Constraint(m.L, m.T, rule=power_max_rule)

    def ramp_up_rule(m, l, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.p[l, t] - m.p[l, t - 1] <= m.g[l, t] * m.dt
    m.ramp_up = pyo.Constraint(m.L, m.T, rule=ramp_up_rule)

    def ramp_down_rule(m, l, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.p[l, t - 1] - m.p[l, t] <= m.g[l, t] * m.dt
    m.ramp_down = pyo.Constraint(m.L, m.T, rule=ramp_down_rule)

    def transition_rule(m, l, t):
        if t == m.T.first():
            return pyo.Constraint.Skip  # handle initial state separately if needed
        return m.u[l, t] - m.d[l, t] == m.y[l, t] - m.y[l, t - 1]
    m.transition = pyo.Constraint(m.L, m.T, rule=transition_rule)

    def no_simultaneous_start_stop_rule(m, l, t):
        return m.u[l, t] + m.d[l, t] <= 1
    m.no_simul = pyo.Constraint(m.L, m.T, rule=no_simultaneous_start_stop_rule)

    def min_up_time_rule(m, l, t):
        UT = int(pyo.value(m.tau_plus[l]))
        if t + UT - 1 > m.T.last():
            return pyo.Constraint.Skip
        return sum(m.y[l, tt] for tt in range(t, t + UT)) >= UT * m.u[l, t]
    m.min_up = pyo.Constraint(m.L, m.T, rule=min_up_time_rule)

    def min_down_time_rule(m, l, t):
        DT = int(pyo.value(m.tau_minus[l]))
        if t + DT - 1 > m.T.last():
            return pyo.Constraint.Skip
        return sum(1 - m.y[l, tt] for tt in range(t, t + DT)) >= DT * m.d[l, t]
    m.min_down = pyo.Constraint(m.L, m.T, rule=min_down_time_rule)

    # =========================================================
    # Constraints - Hydro (flows + ramping + pump on/off)
    # =========================================================
    def turb_flow_min_rule(m, k, t):
        return m.fplus[k, t] >= m.Fplus_min[k, t]
    m.turb_flow_min = pyo.Constraint(m.K, m.T, rule=turb_flow_min_rule)

    def turb_flow_max_rule(m, k, t):
        return m.fplus[k, t] <= m.Fplus_max[k, t]
    m.turb_flow_max = pyo.Constraint(m.K, m.T, rule=turb_flow_max_rule)

    def turb_ramp_up_rule(m, k, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.fplus[k, t] - m.fplus[k, t - 1] <= m.gplus[k] * m.dt
    m.turb_ramp_up = pyo.Constraint(m.K, m.T, rule=turb_ramp_up_rule)

    def turb_ramp_down_rule(m, k, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.fplus[k, t - 1] - m.fplus[k, t] <= m.gplus[k] * m.dt
    m.turb_ramp_down = pyo.Constraint(m.K, m.T, rule=turb_ramp_down_rule)

    def pump_flow_min_rule(m, k, t):
        return m.fminus[k, t] >= m.Fminus_min[k, t] * m.z[k, t]
    m.pump_flow_min = pyo.Constraint(m.K, m.T, rule=pump_flow_min_rule)

    def pump_flow_max_rule(m, k, t):
        return m.fminus[k, t] <= m.Fminus_max[k, t] * m.z[k, t]
    m.pump_flow_max = pyo.Constraint(m.K, m.T, rule=pump_flow_max_rule)

    def pump_ramp_up_rule(m, k, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.fminus[k, t] - m.fminus[k, t - 1] <= m.gminus[k] * m.dt
    m.pump_ramp_up = pyo.Constraint(m.K, m.T, rule=pump_ramp_up_rule)

    def pump_ramp_down_rule(m, k, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return m.fminus[k, t - 1] - m.fminus[k, t] <= m.gminus[k] * m.dt
    m.pump_ramp_down = pyo.Constraint(m.K, m.T, rule=pump_ramp_down_rule)

    def turb_power_max_rule(m, k, t):
        return m.Pplus[k, t] <= m.Pplus_max[k]
    m.turb_power_max = pyo.Constraint(m.K, m.T, rule=turb_power_max_rule)

    def turb_power_min_rule(m, k, t):
        return m.Pplus[k, t] >= m.Pplus_min[k]
    m.turb_power_min = pyo.Constraint(m.K, m.T, rule=turb_power_min_rule)

    # =========================================================
    # Piecewise linear turbine power curve (segments)
    # =========================================================
    def seg_select_rule(m, k, t):
        return sum(m.zseg[k, s, t] for s in m.S) == 1
    m.seg_select = pyo.Constraint(m.K, m.T, rule=seg_select_rule)

    def theta_cap_rule(m, k, s, t):
        return m.theta[k, s, t] <= m.zseg[k, s, t]
    m.theta_cap = pyo.Constraint(m.K, m.S, m.T, rule=theta_cap_rule)

    def flow_piece_rule(m, k, t):
        return m.fplus[k, t] == sum(
            m.f_bp[k, s] * m.zseg[k, s, t]
            + (m.f_bp[k, s + 1] - m.f_bp[k, s]) * m.theta[k, s, t]
            for s in m.S
        )
    m.flow_piece = pyo.Constraint(m.K, m.T, rule=flow_piece_rule)

    def power_piece_rule(m, k, t):
        return m.Pplus[k, t] == sum(
            m.P_bp[k, s] * m.zseg[k, s, t]
            + (m.P_bp[k, s + 1] - m.P_bp[k, s]) * m.theta[k, s, t]
            for s in m.S
        )
    m.power_piece = pyo.Constraint(m.K, m.T, rule=power_piece_rule)

    # =========================================================
    # Water balance — general DAG topology
    # arc_start_dict[k] = upstream reservoir of arc k (1-indexed)
    # arc_end_dict[k]   = downstream reservoir of arc k (1-indexed)
    #
    # For reservoir r at time t:
    #   V[r,t+1] = V[r,t] + inflow[r,t]*dt
    #     + sum(fplus[k,t]  for k where arc_end[k]==r)   *dt  (turbine arrives at r)
    #     - sum(fplus[k,t]  for k where arc_start[k]==r) *dt  (turbine leaves r)
    #     - sum(fminus[k,t] for k where arc_end[k]==r)   *dt  (pump takes water from r)
    #     + sum(fminus[k,t] for k where arc_start[k]==r) *dt  (pump returns water to r)
    #
    # sum([]) = 0 → handles source/sink nodes without extra guards.
    # =========================================================

    # Precompute adjacency lists (Python dicts, not Pyomo Params — faster in rules)
    arcs_ending_at:   dict = {r: [] for r in range(1, R + 1)}
    arcs_starting_at: dict = {r: [] for r in range(1, R + 1)}
    for k in K_list:
        arcs_ending_at[arc_end_dict[k]].append(k)
        arcs_starting_at[arc_start_dict[k]].append(k)

    def vol_balance_rule(m, r, t):
        if t == m.T.last():
            return pyo.Constraint.Skip
        return (
            m.V[r, t + 1]
            == m.V[r, t]
            + m.inflow[r, t] * m.dt
            + sum(m.fplus[k, t]  for k in arcs_ending_at[r])   * m.dt
            - sum(m.fplus[k, t]  for k in arcs_starting_at[r]) * m.dt
            - sum(m.fminus[k, t] for k in arcs_ending_at[r])   * m.dt
            + sum(m.fminus[k, t] for k in arcs_starting_at[r]) * m.dt
        )

    m.vol_balance = pyo.Constraint(m.R, m.T, rule=vol_balance_rule)

    # =========================================================
    # Power balance
    # Pump consumption = rho_pump[k] * fminus[k,t]
    # =========================================================
    def demand_balance_rule(m, t):
        return sum(m.p[l, t] for l in m.L) \
            + sum(m.Pplus[k, t] for k in m.K) \
            - sum(m.rho_pump[k] * m.fminus[k, t] for k in m.K) \
            == m.demand[t]

    m.demand_balance = pyo.Constraint(m.T, rule=demand_balance_rule)
    return m
