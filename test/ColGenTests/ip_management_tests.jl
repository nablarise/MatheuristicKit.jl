# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_ip_feasibility_check_basic()
    # Test basic IP feasibility checking functionality
    master_model = Model(GLPK.Optimizer)
    @objective(master_model, Min, 0)
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Create a mock master primal solution
    variable_values = Dict(MOI.VariableIndex(1) => 1.5)
    primal_sol = MK.ColGen.MasterPrimalSolution(
        MK.ColGen.PrimalMoiSolution(10.0, variable_values)
    )
    
    # Test IP feasibility check
    projected_sol, is_feasible = MK.ColGen.check_primal_ip_feasibility!(
        primal_sol, 
        context, 
        MK.ColGen.MixedPhase1and2()
    )
    
    @test projected_sol isa MK.ColGen.ProjectedIpPrimalSol
    @test is_feasible == false  # Current implementation always returns false
end

function test_update_incumbent_solution()
    # Test updating incumbent primal solution
    master_model = Model(GLPK.Optimizer)
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Create projected IP solution
    projected_sol = MK.ColGen.ProjectedIpPrimalSol()
    
    # Test incumbent update - should not error
    result = MK.ColGen.update_inc_primal_sol!(context, nothing, projected_sol)
    @test result === nothing  # Function returns nothing
end

function test_better_primal_solution_comparison()
    # Test is_better_primal_sol utility function
    variable_values = Dict(MOI.VariableIndex(1) => 2.0)
    primal_sol = MK.ColGen.MasterPrimalSolution(
        MK.ColGen.PrimalMoiSolution(5.0, variable_values)
    )
    
    # Test comparing solution with nothing (should always be better)
    @test MK.ColGen.is_better_primal_sol(primal_sol, nothing) == true
end

function test_ip_management_types()
    # Test IP management type construction
    projected_sol = MK.ColGen.ProjectedIpPrimalSol()
    @test projected_sol isa MK.ColGen.ProjectedIpPrimalSol
end

function test_master_constraint_dual_update()
    # Test update_master_constrs_dual_vals! functionality
    master_model = Model(GLPK.Optimizer)
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Create mock dual solution
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex}, Dict{Int64, Float64}}()
    dual_sol = MK.ColGen.MasterDualSolution(MK.ColGen.DualMoiSolution(0.0, constraint_duals))
    
    # Test dual values update - should not error and return nothing
    result = MK.ColGen.update_master_constrs_dual_vals!(context, dual_sol)
    @test result === nothing
end

function test_unit_ip_management()
    @testset "[ip_management] integer programming utilities" begin
        test_ip_feasibility_check_basic()
        test_update_incumbent_solution()
        test_better_primal_solution_comparison()
        test_ip_management_types()
        test_master_constraint_dual_update()
    end
end