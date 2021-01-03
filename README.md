
# A hybrid exact/heuristic approach for the ROADEF 2020 challenge

The subject of the ROADEF 2020 challenge features a complex objective function. It is both computationally expensive, with up to 600 different scenarios, and highly non-convex, which makes it a challenge for exact approaches using MILP solvers.
At the same time, the simplest case, with only one scenario, is a relatively simple packing problem, which is a good match for these solvers.

We use a hybrid approach to leverage the power of MILP solvers on the simplest problems without sacrificing solution quality in the most complex case.
Our MILP model uses a custom lazy constraint generation routine: it is complete, proves optimality on simple instances and is able to provide solutions for large instances.
Simultaneously, our heuristic method uses beam search with a simple bounding procedure, and allows us to intensify the search for good solutions.

# Requirements

* Numpy
* CPLEX (used from the Python API)

# Executing the program

Launching
```bash
python3 roadef.py -p INSTANCEFILE -o SOLUTIONFILE
```
will execute the auxiliary scripts and export the solution.
