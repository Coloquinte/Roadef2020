#!/usr/bin/python3

"""
@author: rte-challenge-roadef-2020-team
"""
import argparse
import sys
import numpy as np

import common

####################
### Utils ##########
####################

## Global variables
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

################################
## Results processing ##########
################################

def compute_resources(Instance: dict):
    """Compute effective workload (i.e. resources consumption values) for every resource and every time step"""

    # Retrieve usefull infos
    Interventions = Instance[INTERVENTIONS_STR]
    T_max = Instance[T_STR]
    Resources = Instance[RESOURCES_STR]
    # Init resource usage dictionnary for each resource and time
    resources_usage = {}
    for resource_name in Resources.keys():
        resources_usage[resource_name] = np.zeros(T_max)
    # Compute value for each resource and time step
    for intervention_name, intervention in Interventions.items():
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = start_time - 1 #index of list starts at 0
        intervention_worload = intervention[RESOURCE_CHARGE_STR]
        intervention_delta = int(intervention[DELTA_STR][start_time_idx])
        for resource_name, intervention_resource_worload in intervention_worload.items():
            for time in range(start_time_idx, start_time_idx + intervention_delta):
                # null values are not available
                if str(time+1) in intervention_resource_worload and str(start_time) in intervention_resource_worload[str(time+1)]:
                    resources_usage[resource_name][time] += intervention_resource_worload[str(time+1)][str(start_time)]

    return resources_usage


def compute_presence(Instance: dict):
    Interventions = Instance[INTERVENTIONS_STR]
    T_max = Instance[T_STR]
    presence = np.zeros(T_max, dtype=np.int32)
    starting = np.zeros(T_max, dtype=np.int32)
    for intervention_name, intervention in Interventions.items():
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = start_time - 1 #index of list starts at 0
        starting[start_time_idx] += 1
        intervention_delta = int(intervention[DELTA_STR][start_time_idx])
        for time in range(start_time_idx, start_time_idx + intervention_delta):
            presence[time] += 1
    return starting, presence


## Retrieve effective risk distribution given starting times solution
def compute_risk_distribution(Interventions: dict, T_max: int, scenario_numbers):
    """Compute risk distributions for all time steps, given the interventions starting times"""

    # Init risk table
    risk = [scenario_numbers[t] * [0] for t in range(T_max)]
    # Compute for each intervention independently
    for intervention in Interventions.values():
        # Retrieve Intervention's usefull infos
        intervention_risk = intervention[RISK_STR]
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = int(start_time) - 1 # index for list getter
        delta = int(intervention[DELTA_STR][start_time_idx])
        for time in range(start_time_idx, start_time_idx + delta):
            for i, additional_risk in enumerate(intervention_risk[str(time + 1)][str(start_time)]):
                risk[time][i] += additional_risk

    return risk

## Compute mean for each period
def compute_mean_risk(risk, T_max: int, scenario_numbers):
    """Compute mean risk values over each time period"""

    # Init mean risk
    mean_risk = np.zeros(T_max)
    # compute mean
    for t in range(T_max):
        mean_risk[t] = sum(risk[t]) / scenario_numbers[t]

    return mean_risk

## Compute quantile for each period
def compute_quantile(risk, T_max: int, scenario_numbers, quantile):
    """Compute Quantile values over each time period"""

    # Init quantile
    q = np.zeros(T_max)
    for t in range(T_max):
        risk[t].sort()
        q[t] = risk[t][int(np.ceil(scenario_numbers[t] * quantile))-1]

    return q

def compute_max(risk, T_max: int, scenario_numbers):
    """Compute max values over each time period"""

    # Init quantile
    q = np.zeros(T_max)
    for t in range(T_max):
        q[t] = max(risk[t])

    return q


## Compute both objectives: mean risk and quantile
def compute_objective(Instance: dict):
    """Compute objectives (mean and expected excess)"""

    # Retrieve usefull infos
    T_max = Instance[T_STR]
    scenario_numbers = Instance[SCENARIO_NUMBER]
    Interventions = Instance[INTERVENTIONS_STR]
    quantile = Instance[QUANTILE_STR]
    # Retrieve risk final distribution
    risk = compute_risk_distribution(Interventions, T_max, scenario_numbers)
    # Compute mean risk
    mean_risk = compute_mean_risk(risk, T_max, scenario_numbers)
    # Compute quantile
    q = compute_quantile(risk, T_max, scenario_numbers, quantile)
    # Compute max
    m = compute_max(risk, T_max, scenario_numbers)

    return mean_risk, q, m



##################################
## Constraints checkers ##########
##################################

## Launch each Constraint checks
def check_all_constraints(Instance: dict):
    """Run all constraint checks"""

    # Schedule constraints
    check_schedule(Instance)
    # Resources constraints
    check_resources(Instance)
    # Exclusions constraints
    check_exclusions(Instance)

## Schedule constraints: §4.1 in model description
def check_schedule(Instance: dict):
    """Check schedule constraints"""

    # Continuous interventions: §4.1.1
    #   This constraint is implicitly checked by the resource computation:
    #   computation is done under continuity hypothesis, and resource bounds will ensure the feasibility
    # Checks is done on each Intervention
    Interventions = Instance[INTERVENTIONS_STR]
    error_cnt = 0
    for intervention_name, intervention in Interventions.items():
        # Interventions are planned once: §4.1.2
        #   assert a starting time has been assigned to the intervention
        if not START_STR in intervention:
            print('ERROR: Schedule constraint 4.1.2: Intervention ' + intervention_name + ' has not been scheduled.')
            error_cnt += 1
            continue
        # Starting time validity: no explicit constraint
        start_time = intervention[START_STR]
        horizon_end = Instance[T_STR]
        if not (1 <= start_time <= horizon_end):
            print('ERROR: Schedule constraint 4.1 time validity: Intervention ' + intervention_name + ' starting time ' + str(start_time)
            + ' is not a valid starting date. Expected value between 1 and ' + str(horizon_end) + '.')
            # Remove start time to avoid later access errors
            del intervention[START_STR]
            error_cnt += 1
            continue
        # No work left: §4.1.3
        #   assert intervention is not ongoing after time limit or end of horizon
        time_limit = int(intervention[TMAX_STR])
        if time_limit < start_time:
            print('ERROR: Schedule constraint 4.1.3: Intervention ' + intervention_name + ' realization exceeds time limit.'
            + ' It starts at ' + str(start_time) + ' while time limit is ' + str(time_limit) + '.')
            # Remove start time to avoid later access errors
            del intervention[START_STR]
            error_cnt += 1
            continue
    assert error_cnt == 0, f"{error_cnt} scheduling errors"

## Resources constraints: §4.2 in model description
def check_resources(Instance: dict, skip_check=False):
    """Check resources constraints"""

    T_max = Instance[T_STR]
    Resources = Instance[RESOURCES_STR]
    # Bounds are checked with a tolerance value
    tolerance = 1e-5
    # Compute resource usage
    resource_usage = compute_resources(Instance) # dict on resources and time
    resource_error = 0.0
    # Compare bounds to usage
    for resource_name, resource in Resources.items():
        for time in range(T_max):
            # retrieve bounds values
            upper_bound = resource[MAX_STR][time]
            lower_bound = resource[MIN_STR][time]
            # Consumed value
            workload = resource_usage[resource_name][time]
            # Check max
            if workload > upper_bound + tolerance:
                if not skip_check:
                    print('ERROR: Resources constraint 4.2 upper bound: Worload on Resource ' + resource_name + ' at time ' + str(time+1) + ' exceeds upper bound.'
                    + ' Value ' + str(workload) + ' is greater than bound ' + str(upper_bound) + ' plus tolerance ' + str(tolerance) + '.')
                resource_error += workload - upper_bound - tolerance
            # Check min
            if workload < lower_bound - tolerance:
                if not skip_check:
                    print('ERROR: Resources constraint 4.2 lower bound: Worload on Resource ' + resource_name + ' at time ' + str(time+1) + ' does not match lower bound.'
                    + ' Value ' + str(workload) + ' is lower than bound ' + str(lower_bound) + ' minus tolerance ' + str(tolerance) + '.')
                resource_error += lower_bound - tolerance - workload
    assert skip_check or resource_error == 0.0, f"{resource_error} discrepancy in resource usage"
    return resource_error

## Exclusions constraints: §4.3 in model description
def check_exclusions(Instance: dict, skip_check=False):
    """Check exclusions constraints"""

    # Retrieve Interventions and Exclusions
    Interventions = Instance[INTERVENTIONS_STR]
    Exclusions = Instance[EXCLUSIONS_STR]
    # Assert every exclusion holds
    error_cnt = 0
    for exclusion in Exclusions.values():
        # Retrieve exclusion infos
        [intervention_1_name, intervention_2_name, season] = exclusion
        # Retrieve concerned interventions...
        intervention_1 = Interventions[intervention_1_name]
        intervention_2 = Interventions[intervention_2_name]
        # start time should be defined (already checked in scheduled constraint checker)
        if (not START_STR in intervention_1) or (not START_STR in intervention_2):
            continue
        # ... their respective starting times...
        intervention_1_start_time = intervention_1[START_STR]
        intervention_2_start_time = intervention_2[START_STR]
        # ... and their respective deltas (duration)
        intervention_1_delta = int(intervention_1[DELTA_STR][intervention_1_start_time - 1]) # get index in list
        intervention_2_delta = int(intervention_2[DELTA_STR][intervention_2_start_time - 1]) # get index in list
        # Check overlaps for each time step of the season
        for time_str in Instance[SEASONS_STR][season]:
            time = int(time_str)
            if (intervention_1_start_time <= time < intervention_1_start_time + intervention_1_delta) and (intervention_2_start_time <= time < intervention_2_start_time + intervention_2_delta):
                if not skip_check:
                    print('ERROR: Exclusions constraint 4.3: Interventions ' + intervention_1_name + ' and ' + intervention_2_name
                        + ' are both ongoing at time ' + str(time) + '.')
                error_cnt += 1
    assert skip_check or error_cnt == 0, f"{error_cnt} exclusion errors"
    return error_cnt


#######################
## Displayer ##########
#######################

## Basic printing
def display_basic(Instance: dict, mean_risk, quantile, max_risk):
    """Print main infos"""
    alpha = Instance[ALPHA_STR]
    q = Instance[QUANTILE_STR]
    T_max = Instance[T_STR]
    Interventions = Instance[INTERVENTIONS_STR]

    durations = []
    for intervention_name, intervention in Interventions.items():
        deltas = [int(d) for d in intervention[DELTA_STR]]
        durations.append(np.mean(deltas))
    print('Instance infos:')
    print('\tInterventions: ', len(Instance[INTERVENTIONS_STR]))
    print(f'\tScenarios (avg):  {np.mean(Instance[SCENARIO_NUMBER]):.1f}')
    print(f'\tDurations (avg):  {np.mean(durations):.1f}')
    print('\tTimesteps: ', T_max)
    print('\tAlpha: ', alpha)
    print('\tQuantile: ', q)

    obj_1 = np.mean(mean_risk)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    proxy = np.mean(np.max(np.vstack((max_risk - mean_risk, tmp)), axis=0))
    contrib_1 = alpha * obj_1
    contrib_2 = (1-alpha) * obj_2
    obj_tot = contrib_1 + contrib_2
    print('Solution evaluation:')
    #print('\tMean risk over time: ', mean_risk)
    #print('\tQuantile (Q' + str(q) + ') over time: ', quantile)
    #print('\tMax risk over time: ', max_risk)
    print(f'\tObjective 1 (mean risk): {obj_1:.1f} ({contrib_1:.2f})')
    print(f'\tObjective 2 (quantile excess): {obj_2:.1f} ({contrib_2:.2f})')
    print(f'\tProxy objective (max excess): {proxy:.1f} ({(1-alpha)*proxy:.2f})')
    print(f'\tTotal objective: {obj_tot:.5f}')
    print()

def export_csv(Instance: dict, mean_risk, quantile, max_risk, args):
    if args.csv is None:
        return
    with open(args.csv, "w") as f:
        T_max = Instance[T_STR]
        starting, presence = compute_presence(Instance)
        print("Time\tStarting\tPresent\tMean\tQuantile\tMax", file=f)
        for t in range(T_max):
            print(f"{t}\t"
                  f"{starting[t]}\t{presence[t]}\t"
                  f"{mean_risk[t]:.2f}\t"
                  f"{quantile[t]:.2f}\t"
                  f"{max_risk[t]:.2f}\t", file=f)

######################
## Launcher ##########
######################

def check_and_display_instance(instance, args):
    # Check all constraints
    check_all_constraints(instance)
    # Compute indicators
    mean_risk, quantile, max_risk = compute_objective(instance)
    # Display Solution
    display_basic(instance, mean_risk, quantile, max_risk)
    export_csv(instance, mean_risk, quantile, max_risk, args)


def check_and_display(args):
    """Control checker actions"""

    # Read Instance
    instance = common.read_json(args.instance_file)
    # Read Solution
    common.read_solution_from_txt(instance, args.solution_file)
    check_and_display_instance(instance, args)


def get_compound_objective(instance):
    check_schedule(instance)
    exclusion_err = check_exclusions(instance, True)
    resource_err = check_resources(instance, True)
    mean_risk, quantile, max_risk = compute_objective(instance)
    alpha = instance[ALPHA_STR]
    obj_1 = alpha * np.mean(mean_risk)
    tmp = np.zeros(len(quantile))
    obj_2 = (1-alpha) * np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    return (exclusion_err, resource_err, obj_1 + obj_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_file", help="Instance file")
    parser.add_argument("solution_file", help="Solution file to write")
    parser.add_argument("--csv", help="Write a CSV file with more information", type=str)
    args = parser.parse_args()
    check_and_display(args)
