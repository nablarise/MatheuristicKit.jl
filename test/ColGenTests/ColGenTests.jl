module ColGenTests

using JuMP, MathOptInterface, GLPK
# =====================================================================
# Example: Problem-specific Model and Solver Invocation
# =====================================================================
using GLPK
using Test
using MatheuristicKit, ReformulationKit

const MK = MatheuristicKit
const RK = ReformulationKit
const MOI = MathOptInterface

include("helpers.jl")
include("dw_colgen.jl")
include("dw_colgen_iteration.jl")
include("optimizer_validation.jl")

dw_annotation(::Val{:assignment}, machine, job) = RK.dantzig_wolfe_subproblem(machine);
dw_annotation(::Val{:coverage}, job) = RK.dantzig_wolfe_master();
dw_annotation(::Val{:knapsack}, machine) = RK.dantzig_wolfe_subproblem(machine);

function run()
    # Run helper tests
    test_unit_helpers()
    
    # Run Dantzig-Wolfe column generation tests
    test_dw_colgen()
    
    test_unit_solution()
    
    # Run optimizer validation tests
    test_unit_optimizer_validation()
    
    # Run column generation example
    machines = 1:3;
    jobs = 1:15;
    costs = [12.4 22.8 9.2 20.5 13.3 12.7 24.5 19.4 11.2 17.7 24.4 7.1 21.4 14.6 10.2; 19.4 24.5 24.7 23.3 16.4 20.3 15.3 9.2 8.2 11.6 22.3 8.3 21.8 14.4 23.5; 18.3 14.4 22.4 10.2 24.5 24.2 21.1 12.6 17.4 12.2 18.4 10.4 8.8 9.2 7.4; 13.4 15.9 17.1 16.4 8.7 17.2 17.6 12.4 17.2 22.3 19.6 14.9 18.5 19.3 24.5];
    weights = [63 70 57 82 51 74 98 64 86 80 69 79 60 76 78; 50 57 61 83 81 79 63 99 82 59 83 91 59 99 91; 91 81 66 63 59 81 87 90 65 55 57 68 92 91 86; 62 79 73 60 75 66 68 99 69 60 56 100 67 68 54];
    capacities = [1020 1460 1530];

    model = Model(GLPK.Optimizer)
    @variable(model, assignment[machine in machines, job in jobs], Bin);
    @constraint(model, coverage[job in jobs], sum(assignment[machine, job] for machine in machines) >= 1);
    @constraint(model, knapsack[machine in machines], sum(weights[machine, job] * assignment[machine, job] for job in jobs) <= capacities[machine]);
    @objective(model, Min, sum(costs[machine, job] * assignment[machine, job] for machine in machines, job in jobs));

    reformulation = RK.dantzig_wolfe_decomposition(model, dw_annotation)
    JuMP.set_optimizer(RK.master(reformulation), GLPK.Optimizer)


    MK.ColGen.run_column_generation(reformulation)
end

end