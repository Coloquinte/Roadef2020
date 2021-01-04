#!/usr/bin/python3
# Copyright (C) 2019 Gabriel Gouvine - All Rights Reserved

"""
@author: Gabriel Gouvine
"""

import argparse
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", "-p", help="Instance file name (.json)", dest="instance_file")
    parser.add_argument("--output", "-o", help="Output file name (.txt)", dest="solution_file")
    parser.add_argument("--seed", "-s", help="Random seed", type=int, default=0)
    parser.add_argument("--time-limit", "-t", help="Time limit", type=int, dest="time", default=3600)
    parser.add_argument("-name", help="Print the team's name (J3)", action='store_true')

    parser.set_defaults(root_constraints=True, subset_constraints=True, root_cuts=False)

    args = parser.parse_args()
    if not args.name and not args.instance_file:
        parser.error("An instance file is required")
    if not args.name and not args.solution_file:
        parser.error("An output file is required")
    if args.name:
        print("J3")
        sys.exit(0)
    milp_args = ["python3", "py/milp.py",
                 "-p", args.instance_file,
                 "-o", args.solution_file + ".milp",
                 "-s", str(args.seed)]
    first_beam_args = ["bin/beam_search.bin",
                 "-p", args.instance_file,
                 "-o", args.solution_file,
                 "-s", str(args.seed)]
    second_beam_args = ["bin/beam_search.bin",
                 "-p", args.instance_file,
                 "-o", args.solution_file,
                 "-s", str(args.seed),
                 "-warm-start"]
    milp_args += ["-t", str(args.time // 2)]
    first_beam_args += ["-t", str(args.time // 2)]
    second_beam_args += ["-t", str(args.time // 2)]
    milp_proc = subprocess.Popen(milp_args)
    beam_proc = subprocess.Popen(first_beam_args)
    milp_proc.wait()
    beam_proc.wait()
    beam_proc = subprocess.Popen(second_beam_args)
    beam_proc.wait()
