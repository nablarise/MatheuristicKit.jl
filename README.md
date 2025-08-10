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
model = Model(GLPK.Optimizer)
@variable(model, assignment[1:3, 1:5], Bin)
@constraint(model, knapsack[m=1:3], sum(consumption[m,j] * assignment[m,j] for j in 1:5) <= capacities[m])
@constraint(model, coverage[j=1:5], sum(assignment[m,j] for m in 1:3) >= 1)
@objective(model, Min, sum(cost[m,j] * assignment[m,j] for m in 1:3, j in 1:5))

# Decomposition annotation function
dw_annotation(::Val{:assignment}, machine, job) = ReformulationKit.dantzig_wolfe_subproblem(machine)
dw_annotation(::Val{:coverage}, job) = ReformulationKit.dantzig_wolfe_master()
dw_annotation(::Val{:knapsack}, machine) = ReformulationKit.dantzig_wolfe_subproblem(machine)

# Apply Dantzig-Wolfe decomposition
reformulation = ReformulationKit.dantzig_wolfe_decomposition(model, dw_annotation)

# Set optimizers for master and subproblems
JuMP.set_optimizer(ReformulationKit.master(reformulation), GLPK.Optimizer)
for (sp_id, sp_model) in ReformulationKit.subproblems(reformulation)
    JuMP.set_optimizer(sp_model, GLPK.Optimizer)
end

# Run column generation
result = MatheuristicKit.ColGen.run_column_generation(reformulation)
println("Optimal value: $(result.master_lp_obj)")
```
