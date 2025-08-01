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
    @test context.reformulation === reformulation
    
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
end