# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_optimize_master_lp_primal_integration()
    # Simple integration test: create minimal LP and test that optimization works
    master_model = Model(GLPK.Optimizer)

    @variable(master_model, x >= 1.0)
    @constraint(master_model, x <= 5.0)
    @objective(master_model, Min, x)

    # Create minimal reformulation 
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )

    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    MK.ColGen.setup_reformulation!(context, MK.ColGen.MixedPhase1and2())

    # Test that optimization returns proper MasterSolution with dual solution
    master_solution = MK.ColGen.optimize_master_lp_problem!(MK.ColGen.get_master(context), context)

    @test master_solution isa MK.ColGen.MasterSolution

    primal_solution = MK.ColGen.get_primal_sol(master_solution)
    @test primal_solution isa MK.ColGen.MasterPrimalSolution
    @test primal_solution.obj_value == 1.0
    @test primal_solution.variable_values[JuMP.index(x)] == 1.0
end

function test_optimize_master_lp_dual_integration()
    # Simple integration test: create minimal LP and test that optimization works
    master_model = Model(GLPK.Optimizer)

    @variable(master_model, x >= 0)
    @variable(master_model, y >= 1)
    @constraint(master_model, cstr1, x <= 5.0)
    @constraint(master_model, cstr2, x + y == 5)
    @objective(master_model, Min, x + 3y)

    #

    # Create minimal reformulation 
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )

    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    MK.ColGen.setup_reformulation!(context, MK.ColGen.MixedPhase1and2())

    # Test that optimization returns proper MasterSolution with dual solution
    master_solution = MK.ColGen.optimize_master_lp_problem!(MK.ColGen.get_master(context), context)

    @test master_solution isa MK.ColGen.MasterSolution

    dual_solution = MK.ColGen.get_dual_sol(master_solution)
    @test dual_solution isa MK.ColGen.MasterDualSolution
    @test dual_solution.obj_value == 7.0

    @test dual_solution.constraint_duals[MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}][JuMP.index(JuMP.LowerBoundRef(x)).value] == 0
    @test dual_solution.constraint_duals[MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}][JuMP.index(JuMP.LowerBoundRef(y)).value] == 2
    @test dual_solution.constraint_duals[MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}][JuMP.index(cstr1).value] == 0
    @test dual_solution.constraint_duals[MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}][JuMP.index(cstr2).value] == 1
end

function test_unit_solution()
    @testset "[solution] integration test" begin
        test_optimize_master_lp_primal_integration()
        test_optimize_master_lp_dual_integration()
    end
end