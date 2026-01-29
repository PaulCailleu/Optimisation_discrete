import pyomo.environ as pyo

m = pyo.ConcreteModel()

# -----------------
# Sets
# -----------------
m.T = pyo.RangeSet(1, T)          # temps 1..T
m.L = pyo.Set(initialize=L_list)  # thermiques
m.K = pyo.Set(initialize=K_list)  # hydro arcs (turbines/pompes)
m.S = pyo.RangeSet(1, Smax)       # segments piecewise s=1..Smax (Smax = nb_points-1)

# -----------------
# Params
# -----------------
m.dt = pyo.Param(initialize=dt)                             # time step
m.tau_plus = pyo.Param(initialize=tau_plus)                 # min uptime for thermal plant
m.tau_minus = pyo.Param(initialize=tau_minus)               # min downtime for thermal plant

m.demand = pyo.Param(m.T, initialize=demand_dict)          # d[t]           demand
m.c = pyo.Param(m.L, m.T, initialize=c_dict)               # c[l,t]         cost for thermal plant
m.startup = pyo.Param(m.L, initialize=startup_dict)        # startup[l]     startup cost

m.Pmin = pyo.Param(m.L, m.T, initialize=Pmin_dict)          # Pmin[l,t]     min power for thermal plant
m.Pmax = pyo.Param(m.L, m.T, initialize=Pmax_dict)          # Pmax[l,t]     max power for thermal plant
m.g = pyo.Param(m.L, m.T, initialize=g_dict)                # g[l,t]        ramping for thermal plant

m.Fplus_min = pyo.Param(m.K, m.T, initialize=Fplus_min_dict)    # Fplus_min[k,t]    min flux for turbine
m.Fplus_max = pyo.Param(m.K, m.T, initialize=Fplus_max_dict)    # Fplus_max[k,t]    max flux for turbine
m.gplus = pyo.Param(m.K, initialize=gplus_dict)                 # gplus[k]          ramping for turbine
m.Pplus_max = pyo.Param(m.K, initialize=Pplus_max_dict)         # Pplus_max[k]      max power for turbine
m.Pplus_min = pyo.Param(m.K, initialize=Pplus_min_dict)         # Pplus_min[k]      min power for turbine

m.Fminus_min = pyo.Param(m.K, m.T, initialize=Fminus_min_dict)    # Fminus_min[k,t]    min flux for pump
m.Fminus_max = pyo.Param(m.K, m.T, initialize=Fminus_max_dict)    # Fminus_max[k,t]    max flux for pump
m.gminus = pyo.Param(m.K, initialize=gminus_dict)                 # gminus[k]          ramping for pump
m.rho_pump = pyo.Param(m.K, initialize=rho_pump_dict)             # rho_pump[k]        rho for pump power

# Piecewise data: breakpoints per (k,s) for s=1..Smax+1
m.f_bp = pyo.Param(m.K, pyo.RangeSet(1, Smax+1), initialize=f_breakpoints_dict)  # f_k_i
m.P_bp = pyo.Param(m.K, pyo.RangeSet(1, Smax+1), initialize=P_breakpoints_dict)  # P_k_i

# Reservoir (single reservoir version)
m.Vmin = pyo.Param(m.K, initialize=Vmin)                        # Vmin[k]   min volume for reservoir
m.Vmax = pyo.Param(m.K, initialize=Vmax)                        # Vmax[k]   max volume for reservoir
m.inflow = pyo.Param(m.K, m.T, initialize=inflow_dict)          # a[k,t]

# -----------------
# Variables
# -----------------
# Thermal
m.p = pyo.Var(m.L, m.T, within=pyo.NonNegativeReals)        # power for thermal plant
m.y = pyo.Var(m.L, m.T, within=pyo.Binary)                  # on/off for thermal plant
m.u = pyo.Var(m.L, m.T, within=pyo.Binary)                  # start for thermal plant
m.d = pyo.Var(m.L, m.T, within=pyo.Binary)                  # stop for thermal plant

# Hydro
m.fplus = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)    # flux for turbine
m.Pplus = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)    # power for turbine

m.fminus = pyo.Var(m.K, m.T, within=pyo.NonNegativeReals)   # flux for pump

m.z = pyo.Var(m.K, m.T, within=pyo.Binary)                  # on/off for pump

# Piecewise helpers
m.zseg = pyo.Var(m.K, m.S, m.T, within=pyo.Binary)
m.theta = pyo.Var(m.K, m.S, m.T, bounds=(0, 1))

# Volume
m.V = pyo.Var(m.K, m.T, bounds=lambda m,k,t: (pyo.value(m.Vmin[k]), pyo.value(m.Vmax[k])))  # Volume for reservoir

# -----------------
# Objective
# -----------------
def obj_rule(m):
    return sum(m.c[l,t] * m.p[l,t] * m.dt for l in m.L for t in m.T) + sum(m.startup[l] * m.u[l,t] for l in m.L for t in m.T)
m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# -----------------
# Constraints - Thermal
# -----------------
def power_min_rule(m, l, t):
    return m.p[l,t] >= m.Pmin[l,t] * m.y[l,t]
m.power_min = pyo.Constraint(m.L, m.T, rule=power_min_rule)

def power_max_rule(m, l, t):
    return m.p[l,t] <= m.Pmax[l,t] * m.y[l,t]
m.power_max = pyo.Constraint(m.L, m.T, rule=power_max_rule)

def ramp_up_rule(m, l, t):
    if t == 1:
        return pyo.Constraint.Skip
    return m.p[l,t] - m.p[l,t-1] <= m.g[l,t] * m.dt
m.ramp_up = pyo.Constraint(m.L, m.T, rule=ramp_up_rule)

def ramp_down_rule(m, l, t):
    if t == 1:
        return pyo.Constraint.Skip
    return m.p[l,t-1] - m.p[l,t] <= m.g[l,t] * m.dt
m.ramp_down = pyo.Constraint(m.L, m.T, rule=ramp_down_rule)

def startup_def_rule1(m, l, t):
    if t == 1:
        return pyo.Constraint.Skip
    return m.u[l,t] - m.d[l,t] == m.y[l,t] - m.y[l,t-1]
m.startup_def1 = pyo.Constraint(m.L, m.T, rule=startup_def_rule1)

def startup_def_rule2(m, l, t):
    if t == 1:
        return pyo.Constraint.Skip
    return m.u[l,t] + m.d[l,t] <= 1
m.startup_def2 = pyo.Constraint(m.L, m.T, rule=startup_def_rule2)

def min_up_time_rule(m, l, t):
    UT = m.tau_plus
    if t + UT - 1 > m.T.last():
        return pyo.Constraint.Skip
    return sum(m.y[l,tt] for tt in range(t, t + UT)) >= UT * m.u[l,t]
m.min_up = pyo.Constraint(m.L, m.T, rule=min_up_time_rule)

def min_down_time_rule(m, l, t):
    DT = m.tau_minus
    if t + DT - 1 > m.T.last():
        return pyo.Constraint.Skip
    return sum(1 - m.y[l,tt] for tt in range(t, t + DT)) >= DT * m.d[l,t]
m.min_down = pyo.Constraint(m.L, m.T, rule=min_down_time_rule)

# -----------------
# Constraints - Hydro
# -----------------
def turb_flow_min_rule(m, k, t):
    return m.fplus[k,t] >= m.Fplus_min[k,t]
m.turb_flow_min = pyo.Constraint(m.K, m.T, rule=turb_flow_min_rule)

def turb_flow_max_rule(m, k, t):
    return m.fplus[k,t] <= m.Fplus_max[k,t]
m.turb_flow_max = pyo.Constraint(m.K, m.T, rule=turb_flow_max_rule)

def turb_ramp_up_rule(m, k, t):
    if t == 1:
        return pyo.Constraint.Skip
    return m.fplus[k,t] - m.fplus[k,t-1] <= m.gplus[k] * m.dt
m.turb_ramp_up = pyo.Constraint(m.K, m.T, rule=turb_ramp_up_rule)

def turb_ramp_down_rule(m, k, t):
    if t == 1:
        return pyo.Constraint.Skip
    return m.fplus[k,t-1] - m.fplus[k,t] <= m.gplus[k] * m.dt
m.turb_ramp_down = pyo.Constraint(m.K, m.T, rule=turb_ramp_down_rule)


def pump_flow_min_rule(m, k, t):
    return m.fminus[k,t] >= m.Fminus_min[k,t] * m.z[k,t]
m.pump_flow_min = pyo.Constraint(m.K, m.T, rule=pump_flow_min_rule)

def pump_flow_max_rule(m, k, t):
    return m.fminus[k,t] <= m.Fminus_max[k,t] * m.z[k,t]
m.pump_flow_max = pyo.Constraint(m.K, m.T, rule=pump_flow_max_rule)

def pump_ramp_up_rule(m, k, t):
    if t == 1:
        return pyo.Constraint.Skip
    return m.fminus[k,t] - m.fminus[k,t-1] <= m.gminus[k] * m.dt
m.pump_ramp_up = pyo.Constraint(m.K, m.T, rule=pump_ramp_up_rule)

def pump_ramp_down_rule(m, k, t):
    if t == 1:
        return pyo.Constraint.Skip
    return m.fminus[k,t-1] - m.fminus[k,t] <= m.gminus[k] * m.dt
m.pump_ramp_down = pyo.Constraint(m.K, m.T, rule=pump_ramp_down_rule)

def turb_power_max_rule(m, k, t):
    return m.Pplus[k,t] <= m.Pplus_max[k]
m.turb_power_max = pyo.Constraint(m.K, m.T, rule=turb_power_max_rule)

def turb_power_min_rule(m, k, t):
    return m.Pplus[k,t] >= m.Pplus_min[k]
m.turb_power_min = pyo.Constraint(m.K, m.T, rule=turb_power_min_rule)

# -----------------
# Piecewise linear: segment formulation (zseg + theta)
# -----------------
def seg_select_rule(m, k, t):
    return sum(m.zseg[k,s,t] for s in m.S) == 1
m.seg_select = pyo.Constraint(m.K, m.T, rule=seg_select_rule)

def theta_cap_rule(m, k, s, t):
    return m.theta[k,s,t] <= m.zseg[k,s,t]
m.theta_cap = pyo.Constraint(m.K, m.S, m.T, rule=theta_cap_rule)

def flow_piece_rule(m, k, t):
    return m.fplus[k,t] == sum(m.f_bp[k,s] * m.zseg[k,s,t]
                              + (m.f_bp[k,s+1] - m.f_bp[k,s]) * m.theta[k,s,t]
                              for s in m.S)
m.flow_piece = pyo.Constraint(m.K, m.T, rule=flow_piece_rule)

def power_piece_rule(m, k, t):
    return m.Pplus[k,t] == sum(m.P_bp[k,s] * m.zseg[k,s,t]
                              + (m.P_bp[k,s+1] - m.P_bp[k,s]) * m.theta[k,s,t]
                              for s in m.S)
m.power_piece = pyo.Constraint(m.K, m.T, rule=power_piece_rule)

# -----------------
# Water balance
# -----------------
def vol_balance_rule(m, k, t):
    if t == T:
        return pyo.Constraint.Skip
    return m.V[k,t+1] == m.V[k,t] + (m.inflow[k,t] + m.fminus[k,t] - m.fplus[k,t] + m.fplus[k-1,t] - m.fminus[k-1,t]) * m.dt 
m.vol_balance = pyo.Constraint(m.T, rule=vol_balance_rule)

# -----------------
# Power balance
# -----------------
def demand_balance_rule(m, t):
    return sum(m.p[l,t] for l in m.L) + sum(m.Pplus[k,t] for k in m.K) - sum(m.fminus[k,t] * m.rho_pump[k] for k in m.K) == m.demand[t]
m.demand_balance = pyo.Constraint(m.T, rule=demand_balance_rule)

# -----------------
# Export
# -----------------
m.write("uc_model.lp", io_options={"symbolic_solver_labels": True})
# or: m.write("uc_model.mps")
print("Wrote uc_model.lp")
