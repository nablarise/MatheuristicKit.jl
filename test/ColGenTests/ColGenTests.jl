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
include("master_optimization_tests.jl")
include("reduced_costs_tests.jl")
include("pricing_optimization_tests.jl")
include("dual_bounds_tests.jl")
include("column_insertion_tests.jl")
include("ip_management_tests.jl")
include("dw_colgen_iteration.jl")
include("optimizer_validation.jl")
include("master_primal_solution_printing.jl")
include("master_dual_solution_printing.jl")
include("gap_e2e_tests.jl")

dw_annotation(::Val{:assignment}, machine, job) = RK.dantzig_wolfe_subproblem(machine);
dw_annotation(::Val{:coverage}, job) = RK.dantzig_wolfe_master();
dw_annotation(::Val{:knapsack}, machine) = RK.dantzig_wolfe_subproblem(machine);

# Annotation function for penalty test with unassigned variables
dw_annotation_with_penalty(::Val{:assignment}, machine, job) = RK.dantzig_wolfe_subproblem(machine);
dw_annotation_with_penalty(::Val{:coverage}, job) = RK.dantzig_wolfe_master();
dw_annotation_with_penalty(::Val{:knapsack}, machine) = RK.dantzig_wolfe_subproblem(machine);
dw_annotation_with_penalty(::Val{:unassigned}, job) = RK.dantzig_wolfe_master();

function run()
    # Run helper tests
    # test_unit_helpers()
    
    # # Run Dantzig-Wolfe column generation tests
    # test_dw_colgen()
    
    # # Run modular column generation tests
    # test_unit_master_optimization()
    # test_unit_reduced_costs()
    # test_unit_pricing_optimization()
    # test_unit_dual_bounds()
    # test_unit_column_insertion()
    # test_unit_ip_management()
    
    # # Run legacy test suite (for backward compatibility)
    # test_unit_solution()
    
    # # Run optimizer validation tests
    # test_unit_optimizer_validation()
    
    # # Run MasterPrimalSolution printing tests
    # test_unit_master_primal_solution_printing()
    
    # # Run MasterDualSolution printing tests
    # test_unit_master_dual_solution_printing()

    # Run GAP E2E tests with different constraint types
    test_gap_e2e_all()
end

end