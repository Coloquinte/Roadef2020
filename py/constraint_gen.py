# Copyright (C) 2019 Gabriel Gouvine - All Rights Reserved

"""
@author: Gabriel Gouvine
"""

import numpy as np
import docplex.mp

from docplex.mp.model import Model
from cplex.callbacks import LazyConstraintCallback, UserCutCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from docplex.util.status import JobSolveStatus


class HeuristicSubsetSolver:
    def __init__(self, mat, k):
        self.mat = mat
        self.n = mat.shape[0]
        self.k = k
        self.subset = None

    def compute_value(self, subset):
        return self.mat[subset].min(axis=0).sum()

    def restart(self):
        self.subset = np.zeros(self.n, dtype=bool)
        indices = np.random.choice(range(self.n), self.k, replace=False)
        self.subset[indices] = True
        self.value = self.compute_value(self.subset)

    def try_move(self):
        from_ind = np.random.choice(np.where(self.subset)[0])
        to_ind = np.random.choice(np.where(~self.subset)[0])
        new_subset = np.copy(self.subset)
        new_subset[from_ind] = False
        new_subset[to_ind] = True
        new_value = self.compute_value(new_subset)
        if new_value >= self.value:
            self.subset = new_subset
            self.value = new_value

    def extend_subset(self, subset):
        assert subset.sum() < self.k
        vals = self.mat[subset].min(axis=0, keepdims=True)
        incumbents = np.minimum(self.mat[~subset], vals).sum(axis=1)
        to_ind = np.where(~subset)[0][np.argmax(incumbents)]
        subset[to_ind] = True

    def best_move(self):
        from_ind = np.random.choice(np.where(self.subset)[0])
        new_subset = np.copy(self.subset)
        new_subset[from_ind] = False
        self.extend_subset(new_subset)
        self.subset = new_subset
        self.value = self.compute_value(new_subset)

    def greedy(self, nb):
        nb = min(nb, self.k)
        from_ind = np.random.choice(np.where(self.subset)[0], nb, replace=False)
        new_subset = np.copy(self.subset)
        new_subset[from_ind] = False
        for i in range(nb):
            self.extend_subset(new_subset)
        new_value = self.compute_value(new_subset)
        if new_value >= self.value:
            self.subset = new_subset
            self.value = new_value

    def solve(self):
        candidates = []
        for i in range(10):
            self.restart()
            for i in range(1000):
                choice = np.random.rand()
                if choice <= 0.8:
                    self.best_move()
                elif choice <= 0.95:
                    self.greedy(2)
                else:
                    self.greedy(3)
            candidates.append(self.subset)
        best_ind = np.argmax([self.compute_value(s) for s in candidates])
        self.subset = candidates[best_ind]


class SubsetCutCoefModeler:
    @staticmethod
    def run(pb, time, values, quantile_value):
        modeler = SubsetCutCoefModeler(pb, time, values, quantile_value)
        if len(modeler.values) == 0:
            return None
        modeler.create_model()
        modeler.m.solve()
        return modeler.get_result()

    def __init__(self, problem, time, values, quantile_value):
        self.pb = problem
        self.time = time
        self.intervention_risks = None
        self.values = values
        self.quantile_value = quantile_value

        # Model and decisions
        self.m = Model(name="user_cut_coef_model")
        self.subset_decs = None
        self.intervention_vals = None

    def create_model(self):
        self.subset_decs = [self.m.binary_var(name=f"in_subset_{i}") for i in range(self.pb.quantile_risk.nb_scenarios[self.time])]
        k = self.pb.quantile_risk.nb_scenarios[self.time] - self.pb.quantile_risk.quantile_scenario[self.time]
        scenario_index = self.pb.quantile_risk.quantile_scenario[self.time]
        self.m.add_constraint(self.m.sum(self.subset_decs) == k)
        self.intervention_vals = list()
        overall_risk = np.zeros(self.pb.quantile_risk.nb_scenarios[self.time])
        for intervention in self.values.keys():
            for tp in self.pb.quantile_risk.risk_origin[self.time][intervention]:
                if tp.time not in self.values[intervention]:
                    continue
                frac_value = self.values[intervention][tp.time]
                if frac_value <= 1.0e-4:
                    continue
                overall_risk += frac_value * tp.risk
                dec = self.m.continuous_var(name=f"intervention_bound_{intervention}_{tp.time}")
                self.intervention_vals.append(dec)
                for i, risk in enumerate(tp.risk):
                    self.m.add_indicator(self.subset_decs[i], dec <= frac_value * risk)
                # Trivial constraint to strengthen it a bit
                simple_bound = np.partition(tp.risk, scenario_index)[scenario_index]
                self.m.add_constraint(dec <= frac_value * simple_bound)
        self.m.total_risk = self.m.sum(self.intervention_vals)
        self.m.maximize(self.m.total_risk)

        # Compute a not-too-bad starting point
        sol = docplex.mp.solution.SolveSolution(self.m)
        order = np.argpartition(overall_risk, scenario_index)
        for i in order[:scenario_index]:
            sol.add_var_value(self.subset_decs[i], 0.0)
        for i in order[scenario_index:]:
            sol.add_var_value(self.subset_decs[i], 1.0)
        subset = order[scenario_index:]
        start_values = []
        for intervention in self.values.keys():
            for tp in self.pb.quantile_risk.risk_origin[self.time][intervention]:
                if tp.time not in self.values[intervention]:
                    continue
                frac_value = self.values[intervention][tp.time]
                if frac_value <= 1.0e-4:
                    continue
                start_values.append(frac_value * tp.risk[subset].min())
        for dec, value in zip(self.intervention_vals, start_values):
            sol.add_var_value(dec, value)
        self.m.add_mip_start(sol)

    def solve_heuristic(self):
        n = self.pb.quantile_risk.nb_scenarios[self.time]
        k = n - self.pb.quantile_risk.quantile_scenario[self.time]
        risk_mat = []
        for intervention in self.values.keys():
            for tp in self.pb.quantile_risk.risk_origin[self.time][intervention]:
                if tp.time not in self.values[intervention]:
                    continue
                frac_value = self.values[intervention][tp.time]
                if frac_value <= 1.0e-4:
                    continue
                risk_mat.append(frac_value * tp.risk)
        if len(risk_mat) == 0:
            return None
        risk_mat = np.stack(risk_mat, axis=1)
        solver = HeuristicSubsetSolver(risk_mat, k)
        solver.solve()
        if solver.value >= self.quantile_value + 1.0e-4:
            return np.where(solver.subset)[0]
        else:
            return None

    def check(self, subset, cut_value):
        actual_value = 0.0
        for intervention in self.values.keys():
            for tp in self.pb.quantile_risk.risk_origin[self.time][intervention]:
                if tp.time not in self.values[intervention]:
                    continue
                frac_value = self.values[intervention][tp.time]
                if frac_value <= 1.0e-4:
                    continue
                actual_value += frac_value * tp.risk[subset].min()
        if np.abs(actual_value - cut_value) >= 1.0e-4:
            print("WARNING: difference ({cut_value:.2f} vs {actual_value:.2f})")

    def get_result(self):
        values = [dec.solution_value for dec in self.subset_decs]
        values = [x > 0.5 for x in values]
        cut_value = self.m.total_risk.solution_value
        subset = np.array([i for i, x in enumerate(values) if x])
        #print(f"Subset cut at time {self.time} with {cut_value:.2f} risk vs {self.quantile_value:.2f}")
        if cut_value > self.quantile_value + 1.0e-4:
            #print(f"Stronger cut at time {self.time} with {cut_value:.2f} risk vs {self.quantile_value:.2f}")
            return subset, cut_value
        else:
            #print(f"Weaker cut at time {self.time} with {cut_value:.2f} risk vs {self.quantile_value:.2f}")
            return None, cut_value


class GenericSubsetCutModeler:
    @staticmethod
    def run(pb, time, values, quantile_value):
        modeler = GenericSubsetCutModeler(pb, time, values, quantile_value)
        if len(modeler.values) == 0:
            return None
        modeler.create_model()
        modeler.m.parameters.timelimit = pb.args.subset_cuts_time
        s = modeler.m.solve()
        if s is None or s.solve_status not in [JobSolveStatus.OPTIMAL_SOLUTION, JobSolveStatus.FEASIBLE_SOLUTION]:
            return None, None, modeler.quantile_value
        return modeler.get_result()

    def __init__(self, problem, time, values, quantile_value):
        self.pb = problem
        self.time = time
        self.intervention_risks = None
        self.values = values
        self.quantile_value = quantile_value

        # Model and decisions
        self.m = Model(name="user_cut_coef_model")
        self.subset_decs = None
        self.intervention_vals = None

    def create_model(self):
        self.subset_decs = [self.m.binary_var(name=f"in_subset_{i}") for i in range(self.pb.quantile_risk.nb_scenarios[self.time])]
        n = self.pb.quantile_risk.nb_scenarios[self.time]
        scenario_index = self.pb.quantile_risk.quantile_scenario[self.time]
        k = n - scenario_index
        self.m.add_constraint(self.m.sum(self.subset_decs) == k)
        self.intervention_vals = []
        self.remaining_risk = [[] for i in range(n)]
        self.betas = dict()
        overall_bound = []
        for intervention in self.values.keys():
            for tp in self.pb.quantile_risk.risk_origin[self.time][intervention]:
                if tp.time not in self.values[intervention]:
                    continue
                frac_value = self.values[intervention][tp.time]
                min_dec = self.m.continuous_var(name=f"min_bound_{intervention}_{tp.time}")
                max_dec = self.m.continuous_var(name=f"max_bound_{intervention}_{tp.time}")
                beta = self.m.continuous_var(name=f"beta_{intervention}_{tp.time}", lb=0.0, ub=1.0)
                self.intervention_vals.append(min_dec)
                self.intervention_vals.append(max_dec)
                self.betas[(intervention, tp.time)] = beta
                # min over the subset
                for i, risk in enumerate(tp.risk):
                    self.m.add_indicator(self.subset_decs[i], min_dec <= frac_value * risk * (1-beta))
                for i, risk in enumerate(tp.risk):
                    self.m.add_indicator(self.subset_decs[i], max_dec <= (frac_value-1) * risk * beta)
                for i, risk in enumerate(tp.risk):
                    self.remaining_risk[i].append(risk * beta)
                # Simple bounds to speed up the solution process
                # Minimum of positive elements
                min_bound = np.partition(tp.risk, scenario_index)[scenario_index]
                self.m.add_constraint(min_dec <= frac_value * min_bound * (1-beta))
                # Minimum of negative elements due to (frac_value-1) becomes a max
                max_bound = np.partition(tp.risk, k-1)[k-1]
                self.m.add_constraint(max_dec <= (frac_value-1) * max_bound * beta)
                # Bound on the remaining risk
                overall_bound.append(tp.risk.max() * beta)
        self.remaining_risk_dec = self.m.continuous_var(name=f"remaining_risk")
        for i, risks in enumerate(self.remaining_risk):
            self.m.add_indicator(self.subset_decs[i], self.remaining_risk_dec <= self.m.sum(risks))
        # Pessimistic bound
        self.m.add_constraint(self.remaining_risk_dec <= self.m.sum(overall_bound))
        self.m.total_risk = self.m.sum(self.intervention_vals) + self.remaining_risk_dec
        self.m.add_constraint(self.m.total_risk >= self.quantile_value + 5.0e-5)
        self.m.maximize(self.m.total_risk)

    def get_result(self):
        values = [dec.solution_value > 0.5 for dec in self.subset_decs]
        betas = {a: dec.solution_value for a, dec in self.betas.items()}
        cut_value = self.m.total_risk.solution_value
        subset = np.array([i for i, x in enumerate(values) if x])
        #print(f"General cut at time {self.time} with {cut_value:.2f} risk vs {self.quantile_value:.2f}")
        if cut_value > self.quantile_value + 1.0e-4:
            #print(f"Stronger cut at time {self.time} with {cut_value:.2f} risk vs {self.quantile_value:.2f}")
            return subset, betas, cut_value
        else:
            #print(f"Weaker cut at time {self.time} with {cut_value:.2f} risk vs {self.quantile_value:.2f}")
            return None, None, cut_value



class UserCutCoefModeler:
    @staticmethod
    def run(pb, time, values, cutoff=None, time_limit=None):
        modeler = UserCutCoefModeler(pb, time, values)
        if cutoff is not None:
            modeler.cutoff = cutoff
        if time_limit is not None:
            modeler.time_limit = time_limit
        modeler.create_intervention_risks()
        if len(modeler.intervention_risks) == 0:
            # No variable at all
            return None, None, None
        modeler.create_decisions()
        modeler.create_objective()
        modeler.add_simple_optimistic_constraints()
        lazyct_cb = modeler.m.register_callback(LazyConstraintCoefCallback)
        lazyct_cb.modeler = modeler
        if modeler.time_limit is not None:
            modeler.m.parameters.timelimit = modeler.time_limit
        #print(f"Starting cut model for time {time}")
        s = modeler.m.solve()
        if s is None or s.solve_status != JobSolveStatus.OPTIMAL_SOLUTION:
            #print("No cut solution found")
            return None, None, None
        assert s is not None and s.solve_status == JobSolveStatus.OPTIMAL_SOLUTION, "Cut model was not solved"
        #print("Done")
        return modeler.get_result()

    def __init__(self, problem, time, values):
        self.pb = problem
        self.time = time
        self.intervention_risks = None
        self.values = values

        self.cutoff = float("inf")
        self.time_limit = None

        # Model and decisions
        self.m = Model(name="user_cut_coef_model")
        self.a_decs = None
        self.b_dec = None

    def create_intervention_risks(self):
        # Gather risks for every intervention we want
        self.intervention_risks = dict()
        for intervention in range(self.pb.nb_interventions):
            if intervention not in self.values:
                continue
            for tp in self.pb.quantile_risk.risk_origin[self.time][intervention]:
                if tp.time not in self.values[intervention]:
                    continue
                if self.values[intervention][tp.time] <= 1.0e-4:
                    continue
                if intervention not in self.intervention_risks:
                    self.intervention_risks[intervention] = dict()
                self.intervention_risks[intervention][tp.time] = tp.risk

    def create_decisions(self):
        # Create the decision variables, including a dummy integer to enable lazy constraint support
        self.m.binary_var(name="dummy")
        self.a_decs = dict()
        for intervention, risks in self.intervention_risks.items():
            self.a_decs[intervention] = dict( (t, self.m.continuous_var(name=f"a_{intervention}_{t}", lb=r.min(), ub=r.max())) for t, r in risks.items())
        self.b_dec = self.m.continuous_var(name="b", lb=-self.m.infinity, ub=0)
        # TODO: b should be bigger than -sum(a)

    def compute_risk(self, assignment):
        # Compute the actual risk associated with an assignment
        nb = self.pb.quantile_risk.nb_scenarios[self.time]
        pos = self.pb.quantile_risk.quantile_scenario[self.time]
        risk = np.zeros(nb)
        for intervention, t in assignment.items():
            risk += self.intervention_risks[intervention][t]
        return np.partition(risk, pos)[pos]

    def compute_expected_risk(self, b_coef, a_coefs, assignment):
        expected_risk = b_coef
        for i, t in assignment.items():
            expected_risk += a_coefs[i][t]
        return expected_risk

    def add_optimistic_constraint(self, assignment):
        risk = self.compute_risk(assignment)
        if risk >= self.cutoff:
            return
        expr = [self.b_dec]
        for intervention, t in assignment.items():
            expr.append(self.a_decs[intervention][t])
        self.m.add_constraint(self.m.sum(expr) <= risk)

    def add_simple_optimistic_constraints(self):
        # Add a few simple constraints to remove the basic cases
        self.add_optimistic_constraint({})
        for intervention, risks in self.intervention_risks.items():
            for t in risks.keys():
                self.add_optimistic_constraint({intervention: t})
        interventions = list(self.intervention_risks.keys())
        for i1 in interventions:
            for i2 in interventions:
                if i1 >= i2:
                    continue
                for t1 in self.intervention_risks[i1].keys():
                    for t2 in self.intervention_risks[i2].keys():
                        self.add_optimistic_constraint({i1: t1, i2: t2})

    def create_objective(self):
        # Maximize the value given at the target point
        expr = [self.b_dec]
        for intervention, decs in self.a_decs.items():
            for time, dec in decs.items():
                expr.append(self.values[intervention][time] * dec)
        self.m.quantile_risk_objective = self.m.sum(expr)
        # Slight penalization of the b coef to help generalization
        self.m.maximize(self.m.quantile_risk_objective + 0.0001 * self.b_dec)
        #self.m.maximize(self.m.quantile_risk_objective)

    def get_result(self):
        # Retrieve the coefficients computed by the model
        b_coef = self.b_dec.solution_value
        a_coefs = dict()
        for intervention, decs in self.a_decs.items():
            a_coefs[intervention] = dict()
            for t, dec in decs.items():
                a_coefs[intervention][t] = dec.solution_value
        for intervention, risks in self.intervention_risks.items():
            if intervention not in a_coefs:
                a_coefs[intervention] = dict()
            for t, r in risks.items():
                if t not in a_coefs[intervention]:
                    a_coefs[intervention][t] = r.min()
        obj = self.m.quantile_risk_objective.solution_value
        # Check that the solution is correct i.e. no violated assignment
        assignment = self.find_violated_assignment(b_coef, a_coefs)
        if assignment is None:
            return None, None, None
        constraint_value = self.compute_expected_risk(b_coef, a_coefs, assignment)
        actual_value = self.compute_risk(assignment)
        assert constraint_value <= actual_value + 1.0e-3, f"Found a violated assignment: {constraint_value:.4f} to {actual_value:.4f} ({len(assignment)} interventions, {assignment})"
        return b_coef, a_coefs, obj

    def find_violated_assignment(self, b_coef, a_coefs):
        """
        Find an intervention assignment that is not handled correctly by those coefficients
        Returns the assignment, the value that is expected and the value obtained with those coefficients
        """
        m = Model(name="violated_assignment_modeler")
    
        m.constraint_value = m.continuous_var(name="constraint_value", lb=-m.infinity)
        m.actual_value = m.continuous_var(name="actual_value", lb=-m.infinity)
    
        # Decisions on interventions
        intervention_decisions = dict()
        for intervention, risks in self.intervention_risks.items():
            intervention_decisions[intervention] = dict()
            for t, r in risks.items():
                intervention_decisions[intervention][t] = m.binary_var(name=f"i_{intervention}_{t}")
    
        # Each intervention present at most once
        for intervention, decs in intervention_decisions.items():
            m.add_constraint(m.sum(decs.values()) <= 1)
    
        # Constraint value computation
        expr = [b_coef]
        for intervention, decs in intervention_decisions.items():
            assert intervention in a_coefs
            for t, dec in decs.items():
                assert t in a_coefs[intervention]
                expr.append(a_coefs[intervention][t] * dec)
        m.add_constraint(m.sum(expr) == m.constraint_value) 
    
        # Actual value computation (quantile with indicators)
        indicators = [m.binary_var(name=f"ind_{s}") for s in range(self.pb.quantile_risk.nb_scenarios[self.time])]
        m.add_constraint(m.sum(indicators) == self.pb.quantile_risk.quantile_scenario[self.time] + 1)
        for s, indicator in enumerate(indicators):
            expr = [m.actual_value]
            for intervention, risks in self.intervention_risks.items():
                for t, r in risks.items():
                    expr.append(-r[s] * intervention_decisions[intervention][t])
            m.add_indicator(indicator, m.sum(expr) >= 0.0)

        # Cutoff to only get cases with low enough risk
        if self.cutoff < float("inf"):
            m.add_constraint(m.constraint_value <= self.cutoff)

        # Eliminate infeasible cases
        for resource in range(self.pb.nb_resources):
            decisions = []
            for intervention_time, intervention, usage in self.pb.resources.resource_usage[resource][self.time]:
                if intervention in self.intervention_risks and intervention_time in self.intervention_risks[intervention]:
                    decisions.append(usage * intervention_decisions[intervention][intervention_time])
            m.add_constraint(m.sum(decisions) <= self.pb.resources.upper_bounds[self.time, resource])
            m.add_constraint(m.sum(decisions) >= self.pb.resources.lower_bounds[self.time, resource])

        # Quick cutoff to get more cuts quickly instead of finding the overall maximum
        m.add_constraint(m.constraint_value - m.actual_value <= 0.1)

        # Maximize violation
        m.maximize(m.constraint_value - m.actual_value)
    
        if self.time_limit is not None:
            m.parameters.timelimit = self.time_limit / 4

        s = m.solve()
        if s is None or s.solve_status != JobSolveStatus.OPTIMAL_SOLUTION:
            #s = m.solve(log_output=True)
            #print("No assignment solution found")
            return None
        assert s is not None and s.solve_status == JobSolveStatus.OPTIMAL_SOLUTION, f"Expected optimal job status, got {s.solve_status if s is not None else 'empty solution'}"
    
        # Extract intervention placement
        violated_assignment = dict()
        for intervention, decs in intervention_decisions.items():
            times = [t for t, v in decs.items() if v.solution_value >= 0.5]
            assert len(times) <= 1
            if len(times) > 0:
                violated_assignment[intervention] = times[0]
        return violated_assignment


class LazyConstraintCoefCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.modeler = None

    def try_add_constraint(self, b_coef, a_coefs, assignment):
        m = self.modeler
        constraint_value = m.compute_expected_risk(b_coef, a_coefs, assignment)
        actual_value = m.compute_risk(assignment)
        if constraint_value - actual_value < 5.0e-4:
            # No violation
            return False
        if constraint_value >= m.cutoff:
            # Beyond the cutoff
            return False
        #print(f"\tAdding new lazy constraint: {constraint_value:.4f} to {actual_value:.4f} ({len(assignment)} interventions, {assignment})")
        var_decisions = [m.b_dec.index]
        for intervention, t in assignment.items():
            var_decisions.append(m.a_decs[intervention][t].index)
        coefs = [1.0] * len(var_decisions)
        self.add([var_decisions, coefs], "L", actual_value)
        return True

    def get_coefs(self):
        m = self.modeler
        a_coefs = dict()
        for intervention, decs in m.a_decs.items():
            a_coefs[intervention] = dict()
            for t, dec in decs.items():
                a_coefs[intervention][t] = self.get_values([dec.index])[0]
        b_coef = self.get_values([m.b_dec.index])[0]
        return b_coef, a_coefs

    def __call__(self):
        m = self.modeler
        #print(f"\tLooking for a violated assignment ({m.time})")
        b_coef, a_coefs = self.get_coefs()
        assignment = self.modeler.find_violated_assignment(b_coef, a_coefs)
        if assignment is None:
            self.abort()
            return
        constraint_added = self.try_add_constraint(b_coef, a_coefs, assignment)
        if constraint_added:
            # Agressively try a few more assignments similar to this one
            # They may not satisfy the resource constraints, but that is not an issue
            for i, r in m.intervention_risks.items():
                for t in r.keys():
                    attempt = dict(assignment)
                    attempt[i] = t
                    self.try_add_constraint(b_coef, a_coefs, attempt)
                if i in assignment:
                    attempt = dict(assignment)
                    del attempt[i]
                    self.try_add_constraint(b_coef, a_coefs, attempt)
