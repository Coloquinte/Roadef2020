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


class LazyConstraintCoefModeler:
    @staticmethod
    def run(pb, time, assignment=None, cutoff=None, used_times=None, time_limit=None):
        modeler = LazyConstraintCoefModeler(pb, time)
        if assignment is not None:
            modeler.assignment = assignment
        if cutoff is not None:
            modeler.cutoff = cutoff
        if used_times is not None:
            modeler.used_times = used_times
        modeler.create_intervention_risks()
        modeler.create_decisions()
        modeler.create_objective()
        modeler.add_target_constraint()
        modeler.add_simple_optimistic_constraints()
        lazyct_cb = modeler.m.register_callback(LazyConstraintCoefCallback)
        lazyct_cb.modeler = modeler
        if time_limit is not None:
            params = modeler.m.parameters
            params.timelimit = time_limit
        #print(f"Starting solve for time {time}, with {len(modeler.assignment)} interventions assigned")
        s = modeler.m.solve()
        #print("Done")
        if s is None:
            return None, None
        return modeler.get_result()

    def __init__(self, problem, time):
        self.pb = problem
        self.time = time
        self.intervention_risks = None

        self.assignment = dict()
        self.cutoff = float("inf")
        self.used_times = None

        # Model and decisions
        self.m = Model(name="lazy_constraint_coef_model")
        self.a_decs = None
        self.b_dec = None

    def create_intervention_risks(self):
        # Gather risks for every intervention we want
        self.intervention_risks = dict()
        for intervention in range(self.pb.nb_interventions):
            for tp in self.pb.quantile_risk.risk_origin[self.time][intervention]:
                if self.used_times is not None and tp.time not in self.used_times[intervention]:
                    continue
                if intervention not in self.intervention_risks:
                    self.intervention_risks[intervention] = dict()
                self.intervention_risks[intervention][tp.time] = tp.risk
        self.assignment = dict([(i, t) for i, t in self.assignment.items() if i in self.intervention_risks and t in self.intervention_risks[i]])
     
    def create_decisions(self):
        # Create the decision variables, including a dummy integer to enable lazy constraint support
        self.m.binary_var(name="dummy")
        self.a_decs = dict()
        for intervention, risks in self.intervention_risks.items():
            self.a_decs[intervention] = dict( (t, self.m.continuous_var(name=f"a_{intervention}_{t}", lb=r.min(), ub=r.max())) for t, r in risks.items())
        self.b_dec = self.m.continuous_var(name="b", lb=-self.m.infinity, ub=0)

    def compute_risk(self, assignment):
        # Compute the actual risk associated with an assignment
        nb = self.pb.quantile_risk.nb_scenarios[self.time]
        pos = self.pb.quantile_risk.quantile_scenario[self.time]
        risk = np.zeros(nb)
        for intervention, t in assignment.items():
            risk += self.intervention_risks[intervention][t]
        return np.partition(risk, pos)[pos]

    def add_target_constraint(self):
        # Force the resulting lazy constraint to be tight for this assignment
        expr = [self.b_dec]
        for intervention, t in self.assignment.items():
            expr.append(self.a_decs[intervention][t])
        self.m.add_constraint(self.m.sum(expr) == self.compute_risk(self.assignment))

    def add_optimistic_constraint(self, assignment):
        expr = [self.b_dec]
        for intervention, t in assignment.items():
            expr.append(self.a_decs[intervention][t])
        self.m.add_constraint(self.m.sum(expr) <= self.compute_risk(assignment))

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
        # Optimize first for the constant coefficient, then maximize the impact of the other coefficients
        # TODO: coefficients present in the target assignment should be minimized
        # TODO: using the dual values would be beneficial
        pos_coefs = []
        neg_coefs = []
        for i, decs in self.a_decs.items():
            for t, dec in decs.items():
                if i in self.assignment and t == self.assignment[i]:
                    neg_coefs.append(dec)
                else:
                    pos_coefs.append(dec)
        #all_coefs = [dec for i, decs in self.a_decs.items() for t, dec in decs.items()]
        self.m.maximize(self.m.sum(pos_coefs) - 1000 * self.m.sum(neg_coefs))

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
        return b_coef, a_coefs

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
    
        s = m.solve()
        if s is None:
            s = m.solve(log_output=True)
        assert s is not None and s.solve_status == JobSolveStatus.OPTIMAL_SOLUTION, f"Expected optimal job status, got {s.solve_status if s is not None else 'empty solution'}"
    
        # Extract intervention placement
        violated_assignment = dict()
        for intervention, decs in intervention_decisions.items():
            times = [t for t, v in decs.items() if v.solution_value >= 0.5]
            assert len(times) <= 1
            if len(times) > 0:
                violated_assignment[intervention] = times[0]
        constraint_value = m.constraint_value.solution_value
        actual_value = self.compute_risk(violated_assignment)
    
        return violated_assignment, constraint_value, actual_value

class UserCutCoefModeler:
    @staticmethod
    def run(pb, time, values, cutoff=None, time_limit=None):
        modeler = UserCutCoefModeler(pb, time, values)
        if cutoff is not None:
            modeler.cutoff = cutoff
        modeler.create_intervention_risks()
        if len(modeler.intervention_risks) == 0:
            # No variable at all
            return None, None, None
        modeler.create_decisions()
        modeler.create_objective()
        modeler.add_simple_optimistic_constraints()
        lazyct_cb = modeler.m.register_callback(LazyConstraintCoefCallback)
        lazyct_cb.modeler = modeler
        if time_limit is not None:
            params = modeler.m.parameters
            params.timelimit = time_limit
        #print(f"Starting cut model for time {time}")
        s = modeler.m.solve()
        if s is None or s.solve_status != JobSolveStatus.OPTIMAL_SOLUTION:
            #print("No cut solution found")
            return None, None, None
        assert s is not None and s.solve_status == JobSolveStatus.OPTIMAL_SOLUTION, "Cut model was not solved"
        #print("Done")
        b_coef, a_coefs, value = modeler.get_result()
        assignment, constraint_value, actual_value = modeler.find_violated_assignment(b_coef, a_coefs)
        assert constraint_value <= actual_value + 1.0e-6, f"Found a violated assignment: {constraint_value:.4f} to {actual_value:.4f} ({len(assignment)} interventions, {assignment})"
        return b_coef, a_coefs, value

    def __init__(self, problem, time, values):
        self.pb = problem
        self.time = time
        self.intervention_risks = None
        self.values = values

        self.cutoff = float("inf")
        self.used_times = None

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
                if self.values[intervention][tp.time] <= 1.0e-6:
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

    def add_optimistic_constraint(self, assignment):
        expr = [self.b_dec]
        for intervention, t in assignment.items():
            expr.append(self.a_decs[intervention][t])
        self.m.add_constraint(self.m.sum(expr) <= self.compute_risk(assignment))

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
    
        s = m.solve()
        if s is None or s.solve_status != JobSolveStatus.OPTIMAL_SOLUTION:
            #s = m.solve(log_output=True)
            #print("No assignment solution found")
            return None, None, None
        assert s is not None and s.solve_status == JobSolveStatus.OPTIMAL_SOLUTION, f"Expected optimal job status, got {s.solve_status if s is not None else 'empty solution'}"
    
        # Extract intervention placement
        violated_assignment = dict()
        for intervention, decs in intervention_decisions.items():
            times = [t for t, v in decs.items() if v.solution_value >= 0.5]
            assert len(times) <= 1
            if len(times) > 0:
                violated_assignment[intervention] = times[0]
        constraint_value = m.constraint_value.solution_value
        actual_value = self.compute_risk(violated_assignment)
    
        return violated_assignment, constraint_value, actual_value


class LazyConstraintCoefCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.modeler = None

    def __call__(self):
        #print(f"\tLooking for a violated assignment ({self.modeler.time})")
        m = self.modeler
        a_coefs = dict()
        for intervention, decs in m.a_decs.items():
            a_coefs[intervention] = dict()
            for t, dec in decs.items():
                a_coefs[intervention][t] = self.get_values([dec.index])[0]
        b_coef = self.get_values([m.b_dec.index])[0]
        assignment, constraint_value, actual_value = m.find_violated_assignment(b_coef, a_coefs)
        if assignment is None:
            #print(f"\tSearch for violated assignment failed ({self.modeler.time})")
            self.abort()
            return
        if constraint_value - actual_value >= 1.0e-7:
            #print(f"\tAdding new lazy constraint: {constraint_value:.4f} to {actual_value:.4f} ({len(assignment)} interventions, {assignment})")
            var_decisions = [m.b_dec.index]
            for intervention, t in assignment.items():
                var_decisions.append(m.a_decs[intervention][t].index)
            coefs = [1.0] * len(var_decisions)
            self.add([var_decisions, coefs], "L", actual_value)



