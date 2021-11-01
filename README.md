
# A hybrid exact/heuristic approach for the ROADEF 2020 challenge

The subject of the ROADEF 2020 challenge features a complex objective function. It is both computationally expensive, with up to 600 different scenarios, and highly non-convex, which makes it a challenge for exact approaches using MILP solvers.
At the same time, the simplest case, with only one scenario, is a relatively simple packing problem, which is a good match for these solvers.

We use a hybrid approach to leverage the power of MILP solvers on the simplest problems without sacrificing solution quality in the most complex case.
Our MILP model uses a custom lazy constraint generation routine: it is complete, proves optimality on simple instances and is able to provide solutions for large instances.
Simultaneously, our heuristic method uses beam search with a simple bounding procedure, and allows us to intensify the search for good solutions.

## MILP model

We implemented a typical MILP model with indicator variables to model the quantiles exactly. This is, of course, extremely difficult to solve.
In order to obtain faster solution times, we chose not to model the risk per scenario directly, and the function's landscape is built on-demand using lazy constraints when an incumbent solution is found.
We then introduce new cutting plane families that allow us to strengthen the relaxation at the root node.

To reproduce the conditions of the paper, use the options:
* ```py/milp.py -p INSTANCE.json -o SOL.txt -v3 --no-root-constraints --full -v3``` for the full model (Full)
* ```py/milp.py -p INSTANCE.json -o SOL.txt -v3 --root-constraints --full -v3``` for the full model with additional "subset constraints" (Full+S)
* ```py/milp.py -p INSTANCE.json -o SOL.txt -v3 --polyhedral-cuts --no-root-constraints --full -v3``` for the full model with cutting planes for the quantile function at the root node (Full+C)
* ```py/milp.py -p INSTANCE.json -o SOL.txt -v3 --no-root-constraints -v3``` for the constraint generation method (CGen)
* ```py/milp.py -p INSTANCE.json -o SOL.txt -v3 --root-constraints -v3``` for the constraint generation method with additional "subset constraints" (CGen+S)
* ```py/milp.py -p INSTANCE.json -o SOL.txt -v3 --subset-cuts --no-root-constraints -v3```  for the constraint generation method with the best "subset constraints" at the root node (very slow) (Full+C)

## Beam search

The beam search we use is comparatively simple. It makes use of various backtracking methods, restart choices and node choice heuristics. Additionally, we only compute an estimate of the objective function at some stages in order to save time.


# Requirements

* Python3
* Numpy
* CPLEX (used from the Python API)

# Executing the program

Launching
```bash
python3 roadef2020_J3.py -p INSTANCEFILE -o SOLUTIONFILE
```
will execute the auxiliary scripts and export the solution.
