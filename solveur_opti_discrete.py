#!/usr/bin/env python3
"""Exemple minimal Pyomo :
Minimiser 3x + 4y sous contraintes simples.
"""

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, SolverFactory, value

# Modèle
model = ConcreteModel()

# Variables (>= 0)
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)

# Fonction objectif : min 3x + 4y
model.cost = Objective(expr=3 * model.x + 4 * model.y, sense=1)

# Contraintes
model.cons1 = Constraint(expr=2 * model.x + model.y >= 8)
model.cons2 = Constraint(expr=model.x + 2 * model.y >= 8)

# Résolution
solver = SolverFactory("glpk")  # installer glpk ou remplacer par cbc/gurobi...
result = solver.solve(model, tee=False)

# Affichage
print(result.solver.status, result.solver.termination_condition)
print(f"x = {value(model.x):.2f}")
print(f"y = {value(model.y):.2f}")
print(f"Coût = {value(model.cost):.2f}")
