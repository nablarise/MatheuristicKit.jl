# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_pricing_strategy_basic()
    # Test DefaultPricingStrategy functionality
    master_model = Model(GLPK.Optimizer)
    @objective(master_model, Min, 0)
    
    # Create subproblems with required extensions
    subproblem1 = Model()
    subproblem1.ext[:dw_coupling_constr_mapping] = RK.CouplingConstraintMapping()
    subproblem1.ext[:dw_sp_var_original_cost] = RK.OriginalCostMapping()
    
    subproblem2 = Model()
    subproblem2.ext[:dw_coupling_constr_mapping] = RK.CouplingConstraintMapping()
    subproblem2.ext[:dw_sp_var_original_cost] = RK.OriginalCostMapping()
    
    # Create minimal reformulation with subproblems
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict(1 => subproblem1, 2 => subproblem2),  # Two subproblems with extensions
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Test getting pricing strategy
    strategy = MK.ColGen.get_pricing_strategy(context, MK.ColGen.MixedPhase1and2())
    @test strategy isa MK.ColGen.DefaultPricingStrategy
    
    # Test strategy iteration
    first_result = MK.ColGen.pricing_strategy_iterate(strategy)
    @test first_result !== nothing
    (sp_id1, pricing_sp1), state1 = first_result
    @test sp_id1 in [1, 2]
    
    second_result = MK.ColGen.pricing_strategy_iterate(strategy, state1)
    @test second_result !== nothing
    (sp_id2, pricing_sp2), state2 = second_result
    @test sp_id2 in [1, 2]
    @test sp_id2 != sp_id1  # Should be the other subproblem
    
    # Third iteration should return nothing (exhausted)
    third_result = MK.ColGen.pricing_strategy_iterate(strategy, state2)
    @test third_result === nothing
end

function test_pricing_solution_types()
    # Test PricingSolution and PricingPrimalMoiSolution creation and accessors
    
    # Create a PricingPrimalMoiSolution
    variable_values = Dict(MOI.VariableIndex(1) => 2.5, MOI.VariableIndex(2) => 3.0)
    unified_solution = MK.ColGen.PrimalMoiSolution(-1.5, variable_values)
    pricing_sol = MK.ColGen.PricingPrimalMoiSolution(1, unified_solution, true)
    
    @test pricing_sol.subproblem_id == 1
    @test pricing_sol.solution.obj_value == -1.5
    @test pricing_sol.solution.variable_values == variable_values
    @test pricing_sol.is_improving == true
    
    # Create a PricingSolution
    solution = MK.ColGen.PricingSolution(false, false, -1.5, -0.8, [pricing_sol])
    
    @test MK.ColGen.is_infeasible(solution) == false
    @test MK.ColGen.is_unbounded(solution) == false
    @test MK.ColGen.get_primal_bound(solution) == -1.5
    @test MK.ColGen.get_dual_bound(solution) == -0.8
    @test length(MK.ColGen.get_primal_sols(solution)) == 1
    @test MK.ColGen.get_primal_sols(solution)[1] === pricing_sol
end

function test_column_set_management()
    # Test set_of_columns and push_in_set! functionality
    master_model = Model(GLPK.Optimizer)
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    column_set = MK.ColGen.set_of_columns(context)
    
    @test column_set isa MK.ColGen.PricingPrimalMoiSolutionToInsert
    @test length(column_set.collection) == 0
    
    # Create improving and non-improving columns
    variable_values = Dict(MOI.VariableIndex(1) => 1.0)
    improving_sol = MK.ColGen.PricingPrimalMoiSolution(
        1, 
        MK.ColGen.PrimalMoiSolution(-2.0, variable_values), 
        true
    )
    non_improving_sol = MK.ColGen.PricingPrimalMoiSolution(
        2, 
        MK.ColGen.PrimalMoiSolution(1.0, variable_values), 
        false
    )
    
    # Test adding improving column
    result1 = MK.ColGen.push_in_set!(column_set, improving_sol)
    @test result1 == true
    @test length(column_set.collection) == 1
    
    # Test adding non-improving column (should be filtered out)
    result2 = MK.ColGen.push_in_set!(column_set, non_improving_sol)
    @test result2 == false
    @test length(column_set.collection) == 1  # Still only 1
    
    @test column_set.collection[1] === improving_sol
end

function test_initial_bounds()
    # Test initial dual and primal bounds computation
    master_model = Model(GLPK.Optimizer)
    @objective(master_model, Min, 0)
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Test minimization bounds
    db = MK.ColGen.compute_sp_init_db(context, nothing)
    pb = MK.ColGen.compute_sp_init_pb(context, nothing)
    
    @test db == -Inf
    @test pb == Inf
end

function test_unit_pricing_optimization()
    @testset "[pricing_optimization] strategy and solution management" begin
        test_pricing_strategy_basic()
        test_pricing_solution_types()
        test_column_set_management()
        test_initial_bounds()
    end
end