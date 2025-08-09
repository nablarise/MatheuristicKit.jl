# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_convexity_contrib_basic()
    # Test _convexity_contrib with simplified mock setup
    # Note: This test validates that _convexity_contrib function can be called
    # In practice, accurate testing requires full MOI constraint setup with actual optimization
    master_model = Model(GLPK.Optimizer)
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),  # Empty convexity constraints for simplicity
        Dict{Any,Any}()   
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Create mock dual solution
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex}, Dict{Int64, Float64}}()
    dual_sol = MK.ColGen.MasterDualSolution(MK.ColGen.DualMoiSolution(0.0, constraint_duals))
    
    # Test that function executes without error (returns 0.0 for empty constraints)
    result = MK.ColGen._convexity_contrib(context, dual_sol)
    @test result ≈ 0.0 rtol=1e-10
end

function test_subprob_contrib_basic()
    # Test _subprob_contrib with known reduced costs and multiplicity bounds
    master_model = Model(GLPK.Optimizer)
    
    # Create variables for subproblems with different multiplicity bounds
    @variable(master_model, λ1 >= 1.5)  # Lower bound = 1.5
    @variable(master_model, λ2 >= 2.0)  # Lower bound = 2.0
    ub1 = @constraint(master_model, λ1 <= 3.0)  # Upper bound = 3.0
    ub2 = @constraint(master_model, λ2 <= 4.0)  # Upper bound = 4.0
    lb1 = @constraint(master_model, λ1 >= 1.5)
    lb2 = @constraint(master_model, λ2 >= 2.0)
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict(1 => lb1, 2 => lb2),
        Dict(1 => ub1, 2 => ub2)
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Test data: reduced costs for subproblems
    sps_db = Dict{Int64,Float64}(
        1 => -2.0,  # Negative (improving): use upper multiplicity 3.0 → contribution = -2.0 * 3.0 = -6.0
        2 => 1.5    # Positive (non-improving): use lower multiplicity 2.0 → contribution = 1.5 * 2.0 = 3.0
    )
    
    # Expected total contribution = -6.0 + 3.0 = -3.0
    result = MK.ColGen._subprob_contrib(context, sps_db)
    @test result ≈ -3.0 rtol=1e-10
end

function test_compute_dual_bound_basic()
    # Test compute_dual_bound integration with simplified setup
    master_model = Model(GLPK.Optimizer)
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Create dual solution with known objective value
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex}, Dict{Int64, Float64}}()
    dual_sol = MK.ColGen.MasterDualSolution(MK.ColGen.DualMoiSolution(10.0, constraint_duals))
    
    # Test data: basic subproblem contributions
    sps_db = Dict{Int64,Float64}(1 => -1.5)  # Reduced cost = -1.5
    
    # Dual bound = (obj_value - convexity_contrib) + subprob_contrib
    # = (10.0 - 0.0) + 0.0 = 10.0 (both contributions are 0 with empty constraints)
    result = MK.ColGen.compute_dual_bound(context, MK.ColGen.MixedPhase1and2(), sps_db, dual_sol)
    @test result ≈ 10.0 rtol=1e-10
end

function test_subproblem_convexity_contrib_basic()
    # Test _subproblem_convexity_contrib with simplified setup
    master_model = Model(GLPK.Optimizer)
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Create dual solution
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex}, Dict{Int64, Float64}}()
    dual_sol = MK.ColGen.MasterDualSolution(MK.ColGen.DualMoiSolution(0.0, constraint_duals))
    
    # Test contribution for any subproblem (should return 0.0 with empty constraints)
    result1 = MK.ColGen._subproblem_convexity_contrib(context, 1, dual_sol)
    @test result1 ≈ 0.0 rtol=1e-10
    
    result2 = MK.ColGen._subproblem_convexity_contrib(context, 2, dual_sol)
    @test result2 ≈ 0.0 rtol=1e-10
    
    # Test for non-existent subproblem: should return 0.0
    result3 = MK.ColGen._subproblem_convexity_contrib(context, 99, dual_sol)
    @test result3 ≈ 0.0 rtol=1e-10
end

function test_unit_dual_bounds()
    @testset "[dual_bounds] convexity and subproblem contributions" begin
        test_convexity_contrib_basic()
        test_subprob_contrib_basic()
        test_compute_dual_bound_basic()
        test_subproblem_convexity_contrib_basic()
    end
end