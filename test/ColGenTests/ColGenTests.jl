module ColGenTests

using JuMP, MathOptInterface, HiGHS
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
include("test_utils.jl")
include("dw_colgen.jl")
include("dw_colgen_iteration.jl")
include("optimizer_validation.jl")
include("master_primal_solution_printing.jl")
include("master_dual_solution_printing.jl")
include("gap_e2e_tests.jl")

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
    
    # Run MasterPrimalSolution printing tests
    test_unit_master_primal_solution_printing()
    
    # Run MasterDualSolution printing tests
    test_unit_master_dual_solution_printing()

    # Run GAP E2E tests with different constraint types
    test_gap_e2e_all()
end

end