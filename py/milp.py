#!/usr/bin/python3
# Copyright (C) 2019 Gabriel Gouvine - All Rights Reserved

"""
@author: Gabriel Gouvine
"""
import argparse
import collections
import math
import pdb
import random
import sys
import time as time_mod

import numpy as np
import docplex.mp


from docplex.mp.model import Model
from cplex.callbacks import LazyConstraintCallback, UserCutCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin

import common
import constraint_gen
import quantile

RESOURCES_STR = 'Resources'
SEASONS_STR = 'Seasons'
INTERVENTIONS_STR = 'Interventions'
EXCLUSIONS_STR = 'Exclusions'
T_STR = 'T'
SCENARIO_NUMBER = 'Scenarios_number'
RESOURCE_CHARGE_STR = 'workload'
TMAX_STR = 'tmax'
DELTA_STR = 'Delta'
MAX_STR = 'max'
MIN_STR = 'min'
RISK_STR = 'risk'
START_STR = 'start'
QUANTILE_STR = "Quantile"
ALPHA_STR = "Alpha"
MODEL_STR = "Model"

RiskTuple = collections.namedtuple("RiskTuple", ["time", "risk", "min", "max", "mean"])

class Subproblem:
    def __init__(self, problem):
        self.pb = problem

class Exclusions(Subproblem):
    def __init__(self, problem):
        super().__init__(problem)
        self.durations = []
        for intervention_name in self.pb.intervention_names:
            intervention = self.pb.instance[INTERVENTIONS_STR][intervention_name]
            deltas = [int(d) for d in intervention[DELTA_STR]]
            self.durations.append(deltas)
        empty_set = frozenset()
        empty_conflicts = [empty_set for i in range(self.pb.nb_interventions)]
        self.conflicts = [empty_conflicts for i in range(self.pb.nb_timesteps)]
        for season_name, season_times in self.pb.instance[SEASONS_STR].items():
            # Gather exclusions per season
            exclusions = self.pb.instance[EXCLUSIONS_STR]
            conflicts = [set() for i in range(self.pb.nb_interventions)]
            for exclusion in exclusions.values():
                [i1_name, i2_name, excl_season] = exclusion
                if excl_season != season_name:
                    continue
                i1 = self.pb.name_to_intervention[i1_name]
                i2 = self.pb.name_to_intervention[i2_name]
                conflicts[i1].add(i2)
                conflicts[i2].add(i1)
            conflicts = [frozenset(s) for s in conflicts]
            # Setup exclusions for each timestep of this season
            for time_str in season_times:
                time = int(time_str) - 1
                self.conflicts[time] = conflicts
            # TODO: check that each timestep is in only one season

class Resources(Subproblem):
    def __init__(self, problem):
        super().__init__(problem)
        self.lower_bounds = np.zeros( (self.pb.nb_timesteps, self.pb.nb_resources) )
        self.upper_bounds = np.zeros( (self.pb.nb_timesteps, self.pb.nb_resources) )
        resources = self.pb.instance[RESOURCES_STR]
        for i, resource_name in enumerate(self.pb.resource_names):
            resource = resources[resource_name]
            self.lower_bounds[:,i] = np.array(resource[MIN_STR])
            self.upper_bounds[:,i] = np.array(resource[MAX_STR])
        self.intervention_demands = [[[] for j in range(self.pb.nb_timesteps)] for i in range(self.pb.nb_interventions)]
        self.resource_usage = [[[] for j in range(self.pb.nb_timesteps)] for i in range(self.pb.nb_resources)]
        for i, intervention_name in enumerate(self.pb.intervention_names):
            intervention = self.pb.instance[INTERVENTIONS_STR][intervention_name]
            intervention_workload = intervention[RESOURCE_CHARGE_STR]
            for j, resource_name in enumerate(self.pb.resource_names):
                if resource_name not in intervention_workload:
                    continue
                for resource_time_str, timestep_usages in intervention_workload[resource_name].items():
                    resource_time = int(resource_time_str)-1
                    for intervention_time_str, usage in timestep_usages.items():
                        # TODO: only if it is within the duration of the intervention
                        intervention_time = int(intervention_time_str)-1
                        self.intervention_demands[i][intervention_time].append( (resource_time, j, usage) )
                        self.resource_usage[j][resource_time].append( (intervention_time, i, usage) )

class MeanRisk(Subproblem):
    def __init__(self, problem):
        super().__init__(problem)
        alpha = self.pb.instance[ALPHA_STR]
        scenario_numbers = self.pb.instance[SCENARIO_NUMBER]
        self.mean_risk_contribution = np.zeros( (self.pb.nb_interventions, self.pb.nb_timesteps) )
        for i, intervention_name in enumerate(self.pb.intervention_names):
            intervention = self.pb.instance[INTERVENTIONS_STR][intervention_name]
            intervention_risk = intervention[RISK_STR]
            max_start_time = self.pb.max_start_times[i]
            for start_time in range(max_start_time):
                mean_contribution = 0.0
                delta = int(intervention[DELTA_STR][start_time])
                for time in range(start_time, start_time + delta):
                    factor = alpha / (scenario_numbers[time] * self.pb.nb_timesteps)
                    for additional_risk in intervention_risk[str(time + 1)][str(start_time+1)]:
                        mean_contribution += factor * additional_risk
                self.mean_risk_contribution[i, start_time] = mean_contribution

class QuantileRisk(Subproblem):
    def __init__(self, problem):
        super().__init__(problem)
        scenario_numbers = self.pb.instance[SCENARIO_NUMBER]
        quantile = self.pb.instance[QUANTILE_STR]
        alpha = self.pb.instance[ALPHA_STR]
        self.nb_scenarios = np.array(scenario_numbers, dtype=np.int32)
        self.quantile_scenario = np.array([int(np.ceil(n * quantile))-1 for n in self.nb_scenarios])
        self.risk_contribution = [[[] for j in range(self.pb.nb_timesteps)] for i in range(self.pb.nb_interventions)]
        self.risk_origin = [[[] for i in range(self.pb.nb_interventions)] for t in range(self.pb.nb_timesteps)]
        self.risk_from_times = [{} for t in range(self.pb.nb_timesteps)]
        self.min_risk_from_times = [{} for t in range(self.pb.nb_timesteps)]
        self.max_risk_from_times = [{} for t in range(self.pb.nb_timesteps)]
        for i, intervention_name in enumerate(self.pb.intervention_names):
            intervention = self.pb.instance[INTERVENTIONS_STR][intervention_name]
            intervention_risk = intervention[RISK_STR]
            for risk_time_str, timestep_risks in intervention_risk.items():
                risk_time = int(risk_time_str)-1
                for intervention_time_str, risks in timestep_risks.items():
                    # TODO: only if it is within the duration of the intervention
                    intervention_time = int(intervention_time_str)-1
                    risk_array = alpha / self.pb.nb_timesteps * np.array(risks)
                    self.risk_contribution[i][intervention_time].append( (risk_time, risk_array) )
                    tp = RiskTuple(time=intervention_time, risk=risk_array,
                                   min=risk_array.min(), max=risk_array.max(),
                                   mean=np.mean(risk_array))
                    self.risk_origin[risk_time][i].append(tp)
                    elt = (i, intervention_time)
                    self.risk_from_times[risk_time][elt] = risk_array
                    self.min_risk_from_times[risk_time][elt] = risk_array.min()
                    self.max_risk_from_times[risk_time][elt] = risk_array.max()

class Problem:
    def __init__(self, instance: dict, args):
        self.args = args
        self.instance = instance
        self.intervention_names = list(instance[INTERVENTIONS_STR].keys())
        self.resource_names = list(instance[RESOURCES_STR].keys())
        self.name_to_intervention = {name: i for i, name in enumerate(self.intervention_names)}
        self.name_to_resource = {name: i for i, name in enumerate(self.resource_names)}
        self.nb_timesteps = instance[T_STR]
        self.nb_interventions = len(self.intervention_names)
        self.nb_resources = len(self.resource_names)
        self.max_start_times = np.zeros(self.nb_interventions, dtype=np.int32)
        for i, intervention_name in enumerate(self.intervention_names):
            intervention = instance[INTERVENTIONS_STR][intervention_name]
            max_start_time = int(intervention[TMAX_STR])
            self.max_start_times[i] = max_start_time

        self.log_file = None
        if args.log_file is not None:
            self.log_file = open(args.log_file, "w")

        self.exclusions = Exclusions(self)
        self.resources = Resources(self)
        self.mean_risk = MeanRisk(self)
        self.quantile_risk = QuantileRisk(self)
        self.best_value = float("inf")

    def compute_quantile_value(self, risk, time):
        pos = self.quantile_risk.quantile_scenario[time]
        return np.partition(risk, pos)[pos]

    def compute_quantile_subset(self, risk, time):
        pos = self.quantile_risk.quantile_scenario[time]
        return np.sort(np.argpartition(risk, pos)[pos:])

    def compute_scenario_bounds(self):
        self.scenario_bounds = []
        for t in range(self.nb_timesteps):
            self.scenario_bounds.append([])
            for s in range(self.quantile_risk.nb_scenarios[t]):
                sum_contrib = 0.0
                for i in range(self.nb_interventions):
                    max_contrib = 0.0
                    for tp in self.quantile_risk.risk_origin[t][i]:
                        max_contrib = max(max_contrib, tp.risk[s])
                    sum_contrib += max_contrib
                self.scenario_bounds[-1].append(sum_contrib)

    def compute_scenario_bounds_milp(self):
        self.scenario_bounds = []
        for t in range(self.nb_timesteps):
            self.scenario_bounds.append([])
            for s in range(self.quantile_risk.nb_scenarios[t]):
                m = Model(name=f"Scenario_bound_{t}_{s}")
                # Create decisions to model t
                intervention_decisions = [[m.binary_var(name=f"i_{i}_{t}") for t in range(s)] for i, s in enumerate(self.max_start_times)]
                for ivars in intervention_decisions:
                    m.add_constraint(m.sum(ivars) <= 1)
                # Look for maximum scenario impact
                decisions = []
                for i in range(self.nb_interventions):
                    for tp in self.quantile_risk.risk_origin[t][i]:
                        decisions.append(tp.risk[s] * intervention_decisions[i][tp.time])
                m.max_risk = m.sum(decisions)
                m.maximize(m.max_risk)
                # Under resource constraints
                for resource_time in range(self.nb_timesteps):
                    for resource in range(self.nb_resources):
                        decisions = []
                        for intervention_time, intervention, usage in self.resources.resource_usage[resource][resource_time]:
                            decisions.append(usage * intervention_decisions[intervention][intervention_time])
                        m.add_constraint(m.sum(decisions) <= self.resources.upper_bounds[resource_time, resource])
                        m.add_constraint(m.sum(decisions) >= self.resources.lower_bounds[resource_time, resource])
                m.solve()
                self.scenario_bounds[-1].append(m.max_risk.solution_value)

    def create_model(self):
        m = Model(name="RTE_planification_Lazy")
        self.model = m

        # Basic decisions
        if self.args.verbosity >= 2:
            print("Creating decision variables")
        self.intervention_decisions = [[m.binary_var(name=f"i_{i}_{t}") for t in range(s)] for i, s in enumerate(self.max_start_times)]
        for ivars in self.intervention_decisions:
            m.add_constraint(m.sum(ivars) == 1)

        # Exclusion constraints
        if self.args.verbosity >= 2:
            print("Creating exclusion constraints")
        self.create_exclusions()

        # Resource constraints
        if self.args.verbosity >= 2:
            print("Creating resource constraints")
        for resource_time in range(self.nb_timesteps):
            for resource in range(self.nb_resources):
                decisions = []
                for intervention_time, intervention, usage in self.resources.resource_usage[resource][resource_time]:
                    decisions.append(usage * self.intervention_decisions[intervention][intervention_time])
                m.add_constraint(m.sum(decisions) <= self.resources.upper_bounds[resource_time, resource])
                m.add_constraint(m.sum(decisions) >= self.resources.lower_bounds[resource_time, resource])

        # Mean risk objective
        if self.args.verbosity >= 2:
            print("Creating mean risk objective")
        mean_risk_expr = []
        for i in range(self.nb_interventions):
            for j in range(self.max_start_times[i]):
                mean_risk_expr.append(self.intervention_decisions[i][j] * self.mean_risk.mean_risk_contribution[i, j])
        m.mean_risk_objective = m.sum(mean_risk_expr)
        m.add_kpi(m.mean_risk_objective, "mean_risk_objective")

        # Excess risk objective
        if self.args.verbosity >= 2:
            print("Creating excess risk objective")
        self.mean_risk_dec = [m.continuous_var(name=f"mr_{i}") for i in range(self.nb_timesteps)]
        self.quantile_risk_dec = [m.continuous_var(name=f"qr_{i}") for i in range(self.nb_timesteps)]
        self.indicator_dec = []
        excess_risk_expr = []
        self.scenario_risk = []
        for i in range(self.nb_timesteps):
            if self.args.skip_quantile_risk:
                continue
            if self.quantile_risk.nb_scenarios[i] <= 1:
                continue

            # Do not count quantile risk when it's lower than the mean risk
            m.add_constraint(self.quantile_risk_dec[i] - self.mean_risk_dec[i] >= 0.0)

            # Express mean risk
            mean_risk_expr = [-self.mean_risk_dec[i]]
            for intervention in range(self.nb_interventions):
                for tp in self.quantile_risk.risk_origin[i][intervention]:
                    mean_risk_expr.append(self.intervention_decisions[intervention][tp.time] * tp.mean)
            m.add_constraint(m.sum(mean_risk_expr) == 0)

            if args.full and self.quantile_risk.nb_scenarios[i] > 1:
                # Complete case, without lazy constraints: use indicator variables
                indicators = []
                self.scenario_risk.append([])
                for s in range(self.quantile_risk.nb_scenarios[i]):
                    self.scenario_risk[i].append(m.continuous_var(name=f"sr_{i}_{s}", lb=0, ub=self.scenario_bounds[i][s]))
                    expr = [self.scenario_risk[i][s]]
                    for intervention in range(self.nb_interventions):
                        for tp in self.quantile_risk.risk_origin[i][intervention]:
                            expr.append(-tp.risk[s] * self.intervention_decisions[intervention][tp.time])
                    m.add_constraint(m.sum(expr) == 0)
                    indicator = m.binary_var(name=f"ind_{i}_{s}")
                    c = m.add_indicator(indicator, self.quantile_risk_dec[i] - self.scenario_risk[i][s] >= 0.0)
                    indicators.append(indicator)
                m.add_constraint(m.sum(indicators) >= self.quantile_risk.quantile_scenario[i] + 1)
                self.indicator_dec.append(indicators)
            excess_risk_expr.append(self.quantile_risk_dec[i])
            excess_risk_expr.append(-self.mean_risk_dec[i])

        m.excess_risk_objective = m.sum(excess_risk_expr)
        m.add_kpi(m.excess_risk_objective, "excess_risk_objective")
        m.minimize(m.mean_risk_objective + m.excess_risk_objective)

        # Additional useful constraints for better bounds
        if args.root_constraints:
            self.add_root_constraints()
        elif not args.no_root_constraints:
            self.add_simple_root_constraints()

        if self.args.verbosity >= 2:
            print("Finished model creation")

        # Setup time limit and other strategies
        params = m.parameters
        if args.two_solves and args.skip_quantile_risk and any(nb > 1 for nb in self.quantile_risk.nb_scenarios):
            # No need for a big precision
            params.mip.tolerances.mipgap = 0.1
        else:
            params.mip.tolerances.mipgap = 1.0e-6
        #params.mip.limits.cutsfactor = 100.0
        params.mip.display = 3
        #params.emphasis.mip = 2  # Optimality
        params.mip.strategy.file = 2  # Reduce memory usage by saving to disk
        params.workmem = 2048
        params.mip.limits.treememory = 60000  # Bound disk usage
        params.randomseed = args.seed

        if args.subset_cuts:
            cut_cb = m.register_callback(QuantileCutSubsetCallback)
            cut_cb.register_pb(self)
        if args.full and args.polyhedral_cuts:
            cut_cb = m.register_callback(QuantileCutPolyhedralCallback)
            cut_cb.register_pb(self)

        # Lazy constraints
        if not args.full and len(excess_risk_expr) >= 1:
            lazyct_cb = m.register_callback(QuantileLazyCallback)
            lazyct_cb.register_pb(self)

        # Case where a solution is already given
        if args.reoptimize:
            self.read_back()

    def create_exclusions(self):
        m = self.model
        presence = [ [[] for t in range(self.nb_timesteps)] for i in range(self.nb_interventions)]
        for i in range(self.nb_interventions):
            for start_time in range(self.max_start_times[i]):
                dec = self.intervention_decisions[i][start_time]
                for t in range(start_time, start_time + self.exclusions.durations[i][start_time]):
                    presence[i][t].append(dec)
        constraints = []
        for t in range(self.nb_timesteps):
            for i1 in range(self.nb_interventions):
                for i2 in self.exclusions.conflicts[t][i1]:
                    expr = presence[i1][t] + presence[i2][t]
                    constraints.append(m.sum(expr) <= 1)
        m.add_constraints(constraints)

    def add_simple_root_constraints(self):
        if self.args.verbosity >= 2:
            print("Adding simple root constraints")
        for i in range(self.nb_timesteps):
            if self.quantile_risk.nb_scenarios[i] <= 1:
                continue
            if self.args.skip_quantile_risk:
                continue
            expr = [self.quantile_risk_dec[i]]
            for intervention in range(self.nb_interventions):
                # TODO: compute a non-trivial subset
                for tp in self.quantile_risk.risk_origin[i][intervention]:
                    expr.append(- tp.min * self.intervention_decisions[intervention][tp.time])
                self.model.add_constraint(self.model.sum(expr) >= -self.args.tolerance)

    def add_root_constraints(self):
        if self.args.verbosity >= 2:
            print("Adding advanced root constraints")
        for i in range(self.nb_timesteps):
            if self.quantile_risk.nb_scenarios[i] <= 1:
                continue
            if self.args.skip_quantile_risk:
                continue
            nb = 0
            max_contrib_seen = [0.0 for j in range(self.nb_interventions)]
            for intervention in range(self.nb_interventions):
                for tp in self.quantile_risk.risk_origin[i][intervention]:
                    if tp.max - tp.min <= 1.0e-6:
                        # Already handled by the simple root constraints
                        continue
                    subset = self.compute_quantile_subset(tp.risk, i)
                    contrib = tp.risk[subset].min()
                    abs_margin = 1.0e-6
                    rel_margin = 1.0e-6
                    if contrib <= max_contrib_seen[intervention] * (1.0 + rel_margin) + abs_margin:
                        # Not that good compared to cases we have seen with other subsets already
                        continue
                    expr = [self.quantile_risk_dec[i]]
                    for intervention2 in range(self.nb_interventions):
                        for tp2 in self.quantile_risk.risk_origin[i][intervention2]:
                            contrib = tp2.risk[subset].min()
                            max_contrib_seen[intervention2] = max(max_contrib_seen[intervention2], contrib)
                            expr.append(- contrib * self.intervention_decisions[intervention2][tp2.time])
                    self.model.add_constraint(self.model.sum(expr) >= -self.args.tolerance)
                    nb += 1
            if self.args.verbosity >= 3:
                print(f"\tAdded {nb} root constraints for timestep {i}")

    def read_back(self):
        sol = docplex.mp.solution.SolveSolution(self.model)
        start_times = []
        for i, intervention_name in enumerate(self.intervention_names):
            start_time = self.instance[INTERVENTIONS_STR][intervention_name][START_STR] - 1
            sol.add_var_value(self.intervention_decisions[i][start_time], 1.0)
            for t in range(self.max_start_times[i]):
                if t != start_time:
                    sol.add_var_value(self.intervention_decisions[i][t], 0.0)
            start_times.append(start_time)

        if self.args.full:
            risk = [np.zeros(s) for s in self.quantile_risk.nb_scenarios]
            for i, start_time in enumerate(start_times):
                for risk_time, contrib in self.quantile_risk.risk_contribution[i][start_time]:
                    risk[risk_time] += contrib
            for i in range(self.nb_timesteps):
                sorted_contrib = np.argsort(risk[i])
                pos = self.quantile_risk.quantile_scenario[i]
                indicators_one = sorted_contrib[:pos+1]
                indicators_zero = sorted_contrib[pos+1:]
                for ind in indicators_one:
                    sol.add_var_value(self.indicator_dec[i][ind], 1.0)
                for ind in indicators_zero:
                    sol.add_var_value(self.indicator_dec[i][ind], 0.0)

        self.model.add_mip_start(sol)

    def write_back(self):
        mean_risk = self.model.mean_risk_objective.solution_value
        excess_risk = self.model.excess_risk_objective.solution_value
        if self.args.verbosity >= 1:
            print(f"MILP: final solution with risk {mean_risk + excess_risk:.5f} ({mean_risk:.2f} + {excess_risk:.2f})")
        if self.instance is not None:
            for i, intervention_name in enumerate(self.intervention_names):
                start_time = [t for t, d in enumerate(self.intervention_decisions[i]) if d.solution_value > 0.5]
                assert len(start_time) == 1
                start_time = start_time[0]
                self.instance[INTERVENTIONS_STR][intervention_name][START_STR] = start_time + 1
        if mean_risk + excess_risk < self.best_value:
            with open(self.args.solution_file, 'w') as f:
                for i, intervention_name in enumerate(self.intervention_names):
                    start_time = [t for t, d in enumerate(self.intervention_decisions[i]) if d.solution_value > 0.5]
                    assert len(start_time) == 1
                    start_time = start_time[0]
                    print(f'{intervention_name} {start_time+1}', file=f)

    def get_quantile_risk(self, time, intervention_times):
        risk = np.zeros(self.quantile_risk.nb_scenarios[time])
        for it in intervention_times:
            risk += self.quantile_risk.risk_from_times[time][it]
        return self.compute_quantile_value(risk, time)

    def get_quantile_risk_subset(self, time, intervention_times):
        risk = np.zeros(self.quantile_risk.nb_scenarios[time])
        for it in intervention_times:
            risk += self.quantile_risk.risk_from_times[time][it]
        return self.compute_quantile_subset(risk, time)

    def get_subset_risk(self, time, intervention_times, subset):
        risk = np.zeros(self.quantile_risk.nb_scenarios[time])
        for it in intervention_times:
            risk += self.quantile_risk.risk_from_times[time][it]
        return risk[subset].min()

    def check_subset(self, time, subset):
        assert len(subset) == self.quantile_risk.nb_scenarios[time] - self.quantile_risk.quantile_scenario[time]

    def get_lazy_constraint(self, time, intervention_times, extend=False):
        """
        Get a simple lazy constraint on these intervention times.
        """
        quantile_risk = self.get_quantile_risk(time, intervention_times)
        decisions = [it for it in intervention_times]
        coefs = []
        for it in intervention_times:
            contrib = self.quantile_risk.max_risk_from_times[time][it]
            contrib = min(contrib, quantile_risk)
            coefs.append(-contrib)
        rhs = quantile_risk + sum(coefs) - self.args.tolerance
        if extend:
            # Add other interventions and times as required
            interventions_used = set(intervention_times)
            for it, risk_min in self.quantile_risk.min_risk_from_times[time].items():
                if it not in interventions_used:
                    decisions.append(it)
                    coefs.append(-risk_min)
        return rhs, coefs, decisions

    def get_subset_lazy_constraint(self, time, intervention_times, extend=False):
        """
        Get a subset lazy constraint for these intervention times.
        """
        subset = self.get_quantile_risk_subset(time, intervention_times)
        self.check_subset(time, subset)
        quantile_risk = self.get_quantile_risk(time, intervention_times)
        decisions = [it for it in intervention_times]
        coefs = []
        for it in intervention_times:
            contrib = self.quantile_risk.risk_from_times[time][it][subset].max()
            if not extend:
                # Only safe in this case
                contrib = min(contrib, quantile_risk)
            coefs.append(-contrib)
        rhs = quantile_risk + sum(coefs) - self.args.tolerance
        if extend:
            # Add other interventions and times as required
            interventions_used = set(intervention_times)
            for it, risk in self.quantile_risk.risk_from_times[time].items():
                if it not in interventions_used:
                    decisions.append(it)
                    coefs.append(-risk[subset].min())
        return rhs, coefs, decisions

    def get_subset_constraint(self, time, subset, intervention_times=None):
        """
        Get a subset constraint for these intervention times
        """
        if intervention_times is None:
            intervention_times = dict()
        self.check_subset(time, subset)

        risk = np.zeros(self.quantile_risk.nb_scenarios[time])
        for it, beta in intervention_times.items():
            risk += beta * self.quantile_risk.risk_from_times[time][it]
        base_risk = risk[subset].min()

        decisions = []
        coefs = []
        rhs = base_risk - self.args.tolerance
        for it, beta in intervention_times.items():
            data = self.quantile_risk.risk_from_times[time][it][subset]
            contrib = beta * data.max() + (1-beta) * data.min()
            decisions.append(it)
            coefs.append(-contrib)
            rhs -= beta * data.max()

        # Add other interventions and times as required
        interventions_used = set(intervention_times.keys())
        for it, risk in self.quantile_risk.risk_from_times[time].items():
            if it not in interventions_used:
                decisions.append(it)
                coefs.append(-risk[subset].min())
        return rhs, coefs, decisions


class QuantileLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.pb = None
        self.nb_calls = 0
        self.nb_constraints = 0
        self.intervention_history = None

    def register_pb(self, pb):
        self.pb = pb
        self.intervention_history = [set() for i in range(pb.nb_interventions)]

    def add_constraint(self, time, rhs, coefs, decisions):
        pb = self.pb
        var_decisions = [pb.intervention_decisions[it[0]][it[1]].index for it in decisions]
        coefs = list(coefs)
        var_decisions.append(pb.quantile_risk_dec[time].index)
        coefs.append(1.0)
        self.add([var_decisions, coefs], "G", rhs)
        self.nb_constraints += 1

    def write_solution(self, start_times):
        pb = self.pb
        with open(pb.args.solution_file, 'w') as f:
            for intervention_name, start_time in zip(pb.intervention_names, start_times):
                print(f'{intervention_name} {start_time+1}', file=f)

    def __call__(self):
        """Add lazy constraints as needed to compute the actual quantile values"""
        pb = self.pb
        if pb.log_file is not None:
            call_start_time = time_mod.perf_counter()
        start_times = []
        for i in range(pb.nb_interventions):
            values = self.get_values([d.index for d in pb.intervention_decisions[i]])
            start_time = [t for t, v in enumerate(values) if v > 0.5]
            assert len(start_time) == 1
            start_times.append(start_time[0])
            self.intervention_history[i].add(start_time[0])
        risk = [np.zeros(s) for s in pb.quantile_risk.nb_scenarios]
        contributors = [[] for s in pb.quantile_risk.nb_scenarios]
        tot_mean_risk = 0.0
        tot_excess_risk = 0.0
        for i, start_time in enumerate(start_times):
            # Compute the quantile risks
            for risk_time, contrib in pb.quantile_risk.risk_contribution[i][start_time]:
                risk[risk_time] += contrib
                contributors[risk_time].append(i)
            tot_mean_risk += pb.mean_risk.mean_risk_contribution[i, start_time]
        for time in range(pb.nb_timesteps):
            quantile_risk = pb.compute_quantile_value(risk[time], time)
            mean_risk = np.mean(risk[time])
            model_quantile_risk  = self.get_values(pb.quantile_risk_dec[time].index)
            model_mean_risk  = self.get_values(pb.mean_risk_dec[time].index)
            tot_excess_risk += max(quantile_risk - mean_risk, 0.0)
            intervention_times = tuple((inter, start_times[inter]) for inter in contributors[time])
            if model_quantile_risk < quantile_risk - 1.0e-7:
                if pb.log_file is not None:
                    print(f"Adding lazy constraint at time {time} with gap {quantile_risk-model_quantile_risk:.4f} "
                          f"(value {quantile_risk:.4f}, target {model_quantile_risk:.4f}) "
                          f"for {intervention_times}", file=pb.log_file)
                
                if pb.args.subset_constraints:
                    rhs, coefs, decisions = pb.get_subset_lazy_constraint(time, intervention_times, extend=False)
                    self.add_constraint(time, rhs, coefs, decisions)
                    rhs, coefs, decisions = pb.get_subset_lazy_constraint(time, intervention_times, extend=True)
                    self.add_constraint(time, rhs, coefs, decisions)
                else:
                    rhs, coefs, decisions = pb.get_lazy_constraint(time, intervention_times, extend=True)
                    self.add_constraint(time, rhs, coefs, decisions)
                self.nb_calls += 1
            elif model_quantile_risk > quantile_risk + 1.0e-2:
                print(f"WARNING: the quantile risk found at time {time} is pessimistic ({model_quantile_risk:.2f} vs {quantile_risk:.2f})")
        if tot_mean_risk + tot_excess_risk < pb.best_value:
            pb.best_value = tot_mean_risk + tot_excess_risk
            if self.pb.args.verbosity >= 2:
                print(f"MILP: new solution with risk {pb.best_value:.5f} ({tot_mean_risk:.2f} + {tot_excess_risk:.2f})")
            self.write_solution(start_times)

        if pb.log_file is not None:
            call_end_time = time_mod.perf_counter()
            print(f"Evaluated solution #{self.nb_calls} with objective {tot_mean_risk + tot_excess_risk:.4f} "
                  f"(mean risk {tot_mean_risk:.2f}, excess risk {tot_excess_risk:.2f}), "
                  f"in {call_end_time-call_start_time:.2f}s", file=pb.log_file)
            pb.log_file.flush()


class QuantileCutSubsetCallback(ConstraintCallbackMixin, UserCutCallback):
    def __init__(self, env):
        UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.pb = None
        self.nb_calls = 0
        self.nb_constraints = 0

    def register_pb(self, pb):
        self.pb = pb
        self.nb_fail = [0 for i in range(pb.nb_timesteps)]

    def add_constraint(self, time, rhs, coefs, decisions):
        pb = self.pb
        var_decisions = [pb.intervention_decisions[it[0]][it[1]].index for it in decisions]
        coefs = list(coefs)
        var_decisions.append(pb.quantile_risk_dec[time].index)
        coefs.append(1.0)
        self.add([var_decisions, coefs], "G", rhs)
        self.nb_constraints += 1

    def __call__(self):
        #if self.get_node_ID() != 0:
        #    # Only at root node, this stuff is heavy enough as it is
        #    return
        pb = self.pb
        self.nb_calls += 1
        #if self.nb_calls >= 50:
        #    return
        if pb.log_file is not None:
            call_start_time = time_mod.perf_counter()
        intervention_values = [
            self.get_values([d.index for d in pb.intervention_decisions[i]])
            for i in range(pb.nb_interventions)]
        values = dict()
        for i in range(pb.nb_interventions):
            for t, v in enumerate(intervention_values[i]):
                if v > 1.0e-4:
                    if i not in values:
                        values[i] = dict()
                    values[i][t] = v
        quantile_values = self.get_values([d.index for d in pb.quantile_risk_dec])
        for time in range(pb.nb_timesteps):
            if self.nb_fail[time] >= 2:
                continue
            _, subset_val = constraint_gen.SubsetCutCoefModeler.run(pb, time, values, quantile_value=quantile_values[time])
            subset, betas, generic_val = constraint_gen.GenericSubsetCutModeler.run(pb, time, values, quantile_value=quantile_values[time])
            #print(f"Subset value is {subset_val:.2f}, generic {generic_val:.2f}, original {quantile_values[time]:.2f}")
            if subset is None:
                self.nb_fail[time] += 1
                continue
            rhs, coefs, decisions = pb.get_subset_constraint(time, subset, betas)
            self.add_constraint(time, rhs, coefs, decisions)

        if pb.log_file is not None:
            call_end_time = time_mod.perf_counter()
            print(f"Evaluated cut #{self.nb_calls} "
                  f"in {call_end_time-call_start_time:.2f}s", file=pb.log_file)
            pb.log_file.flush()


class QuantileCutPolyhedralCallback(ConstraintCallbackMixin, UserCutCallback):
    def __init__(self, env):
        UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.pb = None
        self.nb_calls = 0
        self.nb_constraints = 0

    def register_pb(self, pb):
        self.pb = pb

    def add_constraint(self, time, constant, coefs):
        pb = self.pb
        var_decisions = [d.index for d in pb.scenario_risk[time]]
        var_decisions.append(pb.quantile_risk_dec[time].index)
        coefs = list(coefs)
        coefs.append(-1.0)
        self.add([var_decisions, coefs], "L", -constant)
        self.nb_constraints += 1

    def __call__(self):
        if self.get_node_ID() != 0:
            # Only at root node, this stuff is heavy enough as it is
            return
        pb = self.pb
        self.nb_calls += 1
        #if self.nb_calls >= 50:
        #    return
        if pb.log_file is not None:
            call_start_time = time_mod.perf_counter()
        quantile_values = self.get_values([d.index for d in pb.quantile_risk_dec])
        for time in range(pb.nb_timesteps):
            if pb.quantile_risk.nb_scenarios[time] <= 1:
                continue
            scenario_values = np.array(self.get_values([d.index for d in pb.scenario_risk[time]]))
            current_quantile = quantile_values[time]
            lbs = np.zeros(pb.quantile_risk.nb_scenarios[time])
            ubs = np.array(pb.scenario_bounds[time])
            k = pb.quantile_risk.nb_scenarios[time] - pb.quantile_risk.quantile_scenario[time]
            scenario_values = np.maximum(scenario_values, lbs)
            scenario_values = np.minimum(scenario_values, ubs)
            assert 1 <= k <= pb.quantile_risk.nb_scenarios[time]
            if pb.log_file is not None:
                print(f"Look for the cut at time {time}, quantile {k}, ub {ubs.mean():.2f}, val {current_quantile:.2f}", file=pb.log_file)
                print(k, file=pb.log_file)
                print(scenario_values, file=pb.log_file)
                print(lbs, file=pb.log_file)
                print(ubs, file=pb.log_file)
                print(current_quantile, file=pb.log_file)
                pb.log_file.flush()

            constraint = quantile.most_violated_constraint(k, scenario_values, lbs, ubs, threshold=current_quantile+1.0e-6)
            if constraint is None:
                continue
            coefs, constant = constraint
            val = constant
            for c, s in zip(coefs, scenario_values):
                val += c * s
            actual_quantile = quantile.quantile(k, scenario_values)

            if pb.log_file is not None:
                print(f"Found a valid cut at time {time} ({val:.2f} vs {current_quantile:.2f}, actual {actual_quantile:.2f})", file=pb.log_file)
                pb.log_file.flush()
            assert actual_quantile >= val - 1.0e-6
            self.add_constraint(time, constant, coefs)
 
        if pb.log_file is not None:
            call_end_time = time_mod.perf_counter()
            print(f"Evaluated cut #{self.nb_calls} "
                  f"in {call_end_time-call_start_time:.2f}s", file=pb.log_file)
            pb.log_file.flush()



def run(args):
    if args.name:
        print("J3")
        sys.exit(0)
    starting_time = time_mod.perf_counter()
    instance = common.read_json(args.instance_file)
    if args.reoptimize:
        common.read_solution_from_txt(instance, args.solution_file)

    pb = Problem(instance, args)
    if args.verbosity >= 1:
        nb_scenarios = pb.quantile_risk.nb_scenarios.max()
        print(f"MILP: problem with "
              f"{pb.nb_interventions} interventions, "
              f"{pb.nb_resources} resources, "
              f"{pb.nb_timesteps} timesteps, "
              f"{nb_scenarios} scenarios")
    pb.compute_scenario_bounds()

    if args.two_solves:
        args.skip_quantile_risk = True
        pb.create_model()
        solve_starting_time = time_mod.perf_counter()
        if args.time is not None:
            #safe_limit = 0.99 * args.time - (solve_starting_time - starting_time)
            safe_limit = args.time
            if safe_limit <= 0.0:
                print("Not enough time remaining to solve the MILP model")
                sys.exit(1)
            pb.model.parameters.timelimit = safe_limit

        # Solve
        pb.model.solve(log_output=args.verbosity >= 2)
        pb.write_back()
        args.skip_quantile_risk = False
        args.reoptimize = True
        pb = Problem(instance, args)
    pb.create_model()
    solve_starting_time = time_mod.perf_counter()
    if args.time is not None:
        #safe_limit = 0.99 * args.time - (solve_starting_time - starting_time)
        safe_limit = args.time
        if safe_limit <= 0.0:
            print("Not enough time remaining to solve the MILP model")
            sys.exit(1)
        pb.model.parameters.timelimit = safe_limit

    # Release the problem data (quite huge actually)
    pb.instance = None
    instance = None

    # Solve
    pb.model.solve(log_output=args.verbosity >= 2)
    pb.write_back()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", "-p", help="Instance file name (.json)", dest="instance_file")
    parser.add_argument("--output", "-o", help="Output file name (.txt)", dest="solution_file")
    parser.add_argument("--seed", "-s", help="Random seed", type=int, default=0)
    parser.add_argument("--time-limit", "-t", help="Time limit", type=int, dest="time")
    parser.add_argument("--verbosity", "-v", help="Verbosity level", type=int, default=1)
    parser.add_argument("-name", help="Print the team's name (J3)", action='store_true')

    parser.add_argument("--log-file", help="Log file for the cuts and lazy constraints", dest="log_file")
    parser.add_argument("--warm-start", help="Improve an existing solution", action='store_true', dest="reoptimize")
    parser.add_argument("--full", help="Use a complete model without lazy constraints", action='store_true')
    parser.add_argument('--skip-quantile-risk', action='store_true', help="Completely skip quantile risk modeling")
    parser.add_argument('--two-solves', action='store_true', help="Solve twice, once without evaluating the quantiles")
    parser.add_argument("--tolerance", help="Tolerance on the constraints", type=float, default=1.0e-6)

    g1 = parser.add_mutually_exclusive_group()
    g1.add_argument('--root-constraints', action='store_true', dest="root_constraints", help="Enable additional root constraints")
    g1.add_argument('--no-root-constraints', action='store_true', dest="no_root_constraints", help="Disable all root constraints")

    g2 = parser.add_mutually_exclusive_group()
    g2.add_argument('--subset-constraints', action='store_true', dest="subset_constraints", help="Enable subset lazy constraints")
    g2.add_argument('--no-subset-constraints', action='store_false', dest="subset_constraints", help="Enable subset lazy constraints")

    parser.add_argument('--polyhedral-cuts', action='store_true', help="Enable polyhedral cuts at root node")
    parser.add_argument('--subset-cuts', action='store_true', help="Enable subset cuts during solve")

    parser.set_defaults(subset_constraints=True)

    args = parser.parse_args()
    if not args.name and not args.instance_file:
        parser.error("An instance file is required")
    if not args.name and not args.solution_file:
        parser.error("An output file is required")
    random.seed(args.seed)
    run(args)
