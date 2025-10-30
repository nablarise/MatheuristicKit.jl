# MatheuristicKit.jl

[![Build Status](https://github.com/nablarise/MatheuristicKit.jl/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/nablarise/MatheuristicKit.jl/actions/workflows/test.yml) [![codecov](https://codecov.io/gh/nablarise/MatheuristicKit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nablarise/NablaMatheuristicKit.jl)

A Julia toolkit for building 
matheuristics.

## Column generation
### Quick Start

This example solves a Generalized Assignment Problem using column generation.

```julia
using MatheuristicKit, ReformulationKit, JuMP, GLPK

# Problem data: 3 machines, 5 jobs
capacities = [15, 15, 16]
consumption = [10 4 5 9 1; 5 6 1 4 3; 10 7 2 2 3]
cost = [3 7 1 1 4; 4 6 7 8 4; 4 5 7 4 2]

# Create full model
model = Model()
@variable(model, assignment[1:3, 1:5], Bin)
@constraint(model, knapsack[m=1:3], sum(consumption[m,j] * assignment[m,j] for j in 1:5) <= capacities[m])
@constraint(model, coverage[j=1:5], sum(assignment[m,j] for m in 1:3) >= 1)
@objective(model, Min, sum(cost[m,j] * assignment[m,j] for m in 1:3, j in 1:5))

# Decompose with declarative syntax
reformulation = @dantzig_wolfe model begin
   assignment[m, _] => subproblem(m) # Variables assignment[m,j] go to subproblem m
   coverage[_] => master()          # Coverage constraints go to master
   knapsack[m] => subproblem(m)      # Knapsack constraints go to subproblem m
end

# Set optimizers for master and subproblems
JuMP.set_optimizer(ReformulationKit.master(reformulation), GLPK.Optimizer)
for (sp_id, sp_model) in ReformulationKit.subproblems(reformulation)
    JuMP.set_optimizer(sp_model, GLPK.Optimizer)
end

# Run column generation
result = MatheuristicKit.ColGen.run_column_generation(reformulation)
println("Optimal value: $(result.master_lp_obj)")
```
You'll see in the terminal

```
Iter 1 | Cols: 3 | DB: -59955.0 | LP: 50000.0 | IP: N/A
Iter 2 | Cols: 3 | DB: -19985.0 | LP: 27.0 | IP: N/A
Iter 3 | Cols: 3 | DB: 2.0 | LP: 21.0 | IP: N/A
Iter 4 | Cols: 3 | DB: 2.0 | LP: 15.0 | IP: N/A
Iter 5 | Cols: 3 | DB: 2.0 | LP: 15.0 | IP: N/A
Iter 6 | Cols: 2 | DB: 11.0 | LP: 15.0 | IP: N/A
Iter 7 | Cols: 1 | DB: 13.0 | LP: 15.0 | IP: N/A
Iter 8 | Cols: 1 | DB: 13.0 | LP: 15.0 | IP: N/A
Iter 9 | Cols: 2 | DB: 13.0 | LP: 13.0 | IP: N/A
MatheuristicKit.ColGen.ColGenOutput(13.0, 13.000000000000004)
```


## Work in progress

This package also contains : 
- [`MathOptState`](https://github.com/nablarise/MatheuristicKit.jl/tree/main/src/MathOptState#readme) an interface to manage changes in MathOptInterface models, particularly in tree search algorithms.
- [`TreeSearch`](https://github.com/nablarise/MatheuristicKit.jl/tree/main/src/TreeSearch#readme) a collection of tree search algorithms

