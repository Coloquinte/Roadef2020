# Copyright (C) 2019 Gabriel Gouvine - All Rights Reserved

"""
@author: Gabriel Gouvine
"""

import json
import gzip

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


def read_json(filename: str):
    """Read a json file and return data as a dict object"""
    if filename.endswith(".gz"):
        with gzip.open(filename, 'r') as f:
            Instance = json.load(f)
    else:
        with open(filename, 'r') as f:
            Instance = json.load(f)
    return Instance

def read_solution_from_txt(Instance: dict, solution_filename: str):
    """Read a txt formated Solution file, and store the solution informations in Instance"""

    # Load interventions
    Interventions = Instance[INTERVENTIONS_STR]
    # Read file line by line, and store starting time value (no checks yet, except format and duplicate)
    solution_file = open(solution_filename, 'r')
    for line in solution_file:
        # Split line to retrive infos: Intervention name and decided starting date
        tmp = line.split(' ')
        intervention_name = tmp[0]
        start_time_str = tmp[1].split('\n')[0]
        # Assert Intervention exists
        if not intervention_name in Interventions:
            print('ERROR: Unexpected Intervention ' + intervention_name + ' in solution file ' + solution_filename + '.')
            continue
        # Assert starting date is an integer
        start_time: int
        try:
            start_time = int(start_time_str)
        except ValueError:
            print('ERROR: Unexpected starting time ' + start_time_str + ' for Intervention ' + intervention_name + '. Expect integer value.')
            continue
        # Assert no duplicate
        if START_STR in Interventions[intervention_name]:
            print('ERROR: Duplicate entry for Intervention ' + intervention_name + '. Only first read value is being considered.')
            continue
        # Store starting time
        Interventions[intervention_name][START_STR] = start_time
    solution_file.close()

def write_solution_to_txt(Instance: dict, solution_filename: str):
    Interventions = Instance[INTERVENTIONS_STR]
    with open(solution_filename, 'w') as f:
        for intervention_name, intervention in Interventions.items():
            print(f'{intervention_name} {intervention[START_STR]}', file=f)

