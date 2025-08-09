# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_run_column_generation_no_optimizer_error()
    # Create a JuMP master problem WITHOUT optimizer
    master = Model()  # No optimizer attached
    
    # Variables and constraints
    @variable(master, x1 >= 0)
    @variable(master, x2 >= 0)
    @constraint(master, x1 + x2 == 5.0)
    @objective(master, Min, x1 + 2*x2)
    
    # Create reformulation
    subproblems = Dict{Any, Model}()
    convexity_constraints_lb = Dict{Any, Any}()
    convexity_constraints_ub = Dict{Any, Any}()
    
    reformulation = RK.DantzigWolfeReformulation(
        master, 
        subproblems, 
        convexity_constraints_lb, 
        convexity_constraints_ub
    )
    
    # Test that run_column_generation throws appropriate error
    @test_throws ErrorException MK.ColGen.run_column_generation(reformulation)
    
    # Verify error message contains expected content
    try
        MK.ColGen.run_column_generation(reformulation)
        @test false  # Should not reach here
    catch e
        @test e isa ErrorException
        @test contains(string(e), "No optimizer attached to the master problem")
        @test contains(string(e), "JuMP.set_optimizer")
        @test contains(string(e), "HiGHS.Optimizer")
    end
end

function test_dantzig_wolfe_constructor_no_optimizer_assert()
    # Create a JuMP master problem WITHOUT optimizer
    master = Model()  # No optimizer attached
    
    # Variables and constraints
    @variable(master, x1 >= 0)
    @variable(master, x2 >= 0)
    @constraint(master, x1 + x2 == 5.0)
    @objective(master, Min, x1 + 2*x2)
    
    # Create reformulation
    subproblems = Dict{Any, Model}()
    convexity_constraints_lb = Dict{Any, Any}()
    convexity_constraints_ub = Dict{Any, Any}()
    
    reformulation = RK.DantzigWolfeReformulation(
        master, 
        subproblems, 
        convexity_constraints_lb, 
        convexity_constraints_ub
    )
    
    # Test that constructor throws assertion error when called directly
    @test_throws AssertionError MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Verify assertion message
    try
        MK.ColGen.DantzigWolfeColGenImpl(reformulation)
        @test false  # Should not reach here
    catch e
        @test e isa AssertionError
        @test contains(string(e), "Master must have optimizer attached")
    end
end

function test_successful_optimizer_attachment()
    # Create a JuMP master problem WITH optimizer
    master = Model(GLPK.Optimizer)  # Optimizer attached
    
    # Variables and constraints
    @variable(master, x1 >= 0)
    @variable(master, x2 >= 0)
    @constraint(master, x1 + x2 == 5.0)
    @objective(master, Min, x1 + 2*x2)
    
    # Create reformulation
    subproblems = Dict{Any, Model}()
    convexity_constraints_lb = Dict{Any, Any}()
    convexity_constraints_ub = Dict{Any, Any}()
    
    reformulation = RK.DantzigWolfeReformulation(
        master, 
        subproblems, 
        convexity_constraints_lb, 
        convexity_constraints_ub
    )
    
    # Test that constructor succeeds when optimizer is attached
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    @test context isa MK.ColGen.DantzigWolfeColGenImpl
    @test MK.ColGen.get_reform(context) === reformulation
    
    # Verify optimizer is actually attached to master
    master_backend = JuMP.backend(RK.master(reformulation))
    @test master_backend.optimizer !== nothing
end

function test_optimizer_attachment_after_model_creation()
    # Create a JuMP master problem without optimizer initially
    master = Model()
    
    # Variables and constraints
    @variable(master, x1 >= 0)
    @variable(master, x2 >= 0)
    @constraint(master, x1 + x2 == 5.0)
    @objective(master, Min, x1 + 2*x2)
    
    # Attach optimizer after model creation
    JuMP.set_optimizer(master, GLPK.Optimizer)
    
    # Create reformulation
    subproblems = Dict{Any, Model}()
    convexity_constraints_lb = Dict{Any, Any}()
    convexity_constraints_ub = Dict{Any, Any}()
    
    reformulation = RK.DantzigWolfeReformulation(
        master, 
        subproblems, 
        convexity_constraints_lb, 
        convexity_constraints_ub
    )
    
    # Test that constructor succeeds when optimizer is attached later
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    @test context isa MK.ColGen.DantzigWolfeColGenImpl
    
    # Test that run_column_generation also succeeds
    # Note: This might fail due to incomplete implementation, but optimizer validation should pass
    try
        MK.ColGen.run_column_generation(reformulation)
        # If it succeeds, great! If it fails for other reasons, that's also fine for this test
    catch e
        # Make sure the error is NOT about missing optimizer
        @test !contains(string(e), "No optimizer attached")
    end
end

function test_optimize_master_lp_problem_variable_extraction()
    # Test that optimize_master_lp_problem! correctly extracts variable values
    @testset "Variable Value Extraction" begin
        # Create a simple master problem with optimizer
        master = Model(GLPK.Optimizer)
        @variable(master, x1 >= 0)
        @variable(master, x2 >= 0)
        @constraint(master, x1 + x2 == 5.0)
        @objective(master, Min, x1 + 2*x2)
        
        # Create reformulation
        subproblems = Dict{Any, Model}()
        convexity_constraints_lb = Dict{Any, Any}()
        convexity_constraints_ub = Dict{Any, Any}()
        
        reformulation = RK.DantzigWolfeReformulation(
            master, 
            subproblems, 
            convexity_constraints_lb, 
            convexity_constraints_ub
        )
        
        # Create DantzigWolfeColGenImpl context
        context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
        master_wrapper = MK.ColGen.get_master(context)
        
        # Call optimize_master_lp_problem!
        result = MK.ColGen.optimize_master_lp_problem!(master_wrapper, context)
        
        # Extract primal solution using get_primal_sol
        primal_sol = MK.ColGen.get_primal_sol(result)
        
        # Test that result contains MasterPrimalSolution with variable values
        @test isa(primal_sol, MK.ColGen.MasterPrimalSolution)
        @test isa(primal_sol.sol.variable_values, Dict{MOI.VariableIndex, Float64})
        
        # Test objective value
        @test primal_sol.sol.obj_value ≈ 5.0 atol=1e-6
        
        # If solution is feasible, should have variable values
        if result.moi_primal_status == MOI.FEASIBLE_POINT
            @test length(primal_sol.sol.variable_values) >= 2  # At least x1 and x2
            
            # Get variable indices from JuMP model for comparison
            x1_idx = JuMP.index(x1)
            x2_idx = JuMP.index(x2)
            
            # Test expected optimal solution: x1 = 5, x2 = 0
            @test haskey(primal_sol.sol.variable_values, x1_idx)
            @test haskey(primal_sol.sol.variable_values, x2_idx)
            @test primal_sol.sol.variable_values[x1_idx] ≈ 5.0 atol=1e-6
            @test primal_sol.sol.variable_values[x2_idx] ≈ 0.0 atol=1e-6
            
            # Check that all variable values are finite
            for (var_idx, value) in primal_sol.sol.variable_values
                @test isfinite(value)
                @test isa(var_idx, MOI.VariableIndex)
            end
        end
    end
end

function test_unit_optimizer_validation()
    @testset "[optimizer_validation] run_column_generation error handling" begin
        test_run_column_generation_no_optimizer_error()
    end
    
    @testset "[optimizer_validation] constructor assert behavior" begin
        test_dantzig_wolfe_constructor_no_optimizer_assert()
    end
    
    @testset "[optimizer_validation] successful optimizer attachment" begin
        test_successful_optimizer_attachment()
        test_optimizer_attachment_after_model_creation()
    end
    
    @testset "[optimizer_validation] variable value storage and retrieval" begin
        test_optimize_master_lp_problem_variable_extraction()
    end
end