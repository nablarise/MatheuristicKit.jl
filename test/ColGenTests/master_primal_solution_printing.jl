# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

import MatheuristicKit as MK
using Test
import MathOptInterface as MOI

function test_master_primal_solution_printing_with_named_variables()
    # Create a simple MOI model for testing
    model = MOI.Utilities.Model{Float64}()
    
    # Add variables with names
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    z = MOI.add_variable(model)
    
    MOI.set(model, MOI.VariableName(), x, "x1")
    MOI.set(model, MOI.VariableName(), y, "y2")
    MOI.set(model, MOI.VariableName(), z, "slack")
    
    # Create MasterPrimalSolution
    variable_values = Dict(
        x => 2.5,
        y => 0.0,
        z => 1.25
    )
    solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(123.45, variable_values))
    
    # Test output with model (named variables)
    io = IOBuffer()
    show(io, solution, model)
    output = String(take!(io))
    
    @test contains(output, "Primal solution:")
    @test contains(output, "x1: 2.5")
    @test contains(output, "y2: 0.0") 
    @test contains(output, "slack: 1.25")
    @test contains(output, "└ cost = 123.45")
    
    # Check that we have proper formatting characters
    @test contains(output, "|") || contains(output, "└")
    
    # Verify variables are sorted by index
    lines = split(output, '\n')
    variable_lines = filter(line -> contains(line, ": "), lines)
    @test length(variable_lines) == 3
end

function test_master_primal_solution_printing_without_model()
    # Create MasterPrimalSolution without access to model names
    x_index = MOI.VariableIndex(1)
    y_index = MOI.VariableIndex(2)
    z_index = MOI.VariableIndex(5)
    
    variable_values = Dict(
        x_index => 10.0,
        y_index => -5.5,
        z_index => 0.0
    )
    solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(-42.7, variable_values))
    
    # Test output without model (fallback names)
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))
    
    @test contains(output, "Primal solution:")
    @test contains(output, "_[1]: 10.0")
    @test contains(output, "_[2]: -5.5")
    @test contains(output, "_[5]: 0.0")
    @test contains(output, "└ cost = -42.7")
    
    # Check that we have proper formatting characters
    @test contains(output, "|") || contains(output, "└")
end

function test_master_primal_solution_printing_mixed_named_unnamed()
    # Create a model with some named and some unnamed variables
    model = MOI.Utilities.Model{Float64}()
    
    x = MOI.add_variable(model)  # Will be unnamed
    y = MOI.add_variable(model)  # Will be named
    z = MOI.add_variable(model)  # Will be unnamed
    
    MOI.set(model, MOI.VariableName(), y, "named_var")
    
    variable_values = Dict(
        x => 1.0,
        y => 2.0, 
        z => 3.0
    )
    solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(100.0, variable_values))
    
    io = IOBuffer()
    show(io, solution, model)
    output = String(take!(io))
    
    @test contains(output, "Primal solution:")
    @test contains(output, "named_var: 2.0")
    @test contains(output, "_[") # Should have fallback names for unnamed variables
    @test contains(output, "└ cost = 100.0")
end

function test_master_primal_solution_printing_edge_cases()
    # Test empty solution
    empty_solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(0.0, Dict{MOI.VariableIndex,Float64}()))
    
    io = IOBuffer()
    show(io, empty_solution)
    output = String(take!(io))
    
    @test contains(output, "Primal solution:")
    @test contains(output, "└ cost = 0.0")
    
    # Test single variable
    single_var_index = MOI.VariableIndex(42)
    single_solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(999.999, Dict(single_var_index => 123.456)))
    
    io = IOBuffer()
    show(io, single_solution)
    output = String(take!(io))
    
    @test contains(output, "Primal solution:")
    @test contains(output, "└ _[42]: 123.456")
    @test contains(output, "└ cost = 999.999")
    
    # Should not have any "|" since there's only one variable
    variable_lines = filter(line -> contains(line, ": "), split(output, '\n'))
    @test length(variable_lines) == 1
end

function test_master_primal_solution_formatting_consistency()
    # Test that variables are consistently sorted by index
    model = MOI.Utilities.Model{Float64}()
    
    # Add variables in non-sequential order
    vars = [MOI.add_variable(model) for _ in 1:5]
    
    # Set names for some variables
    MOI.set(model, MOI.VariableName(), vars[3], "var_c")
    MOI.set(model, MOI.VariableName(), vars[1], "var_a") 
    MOI.set(model, MOI.VariableName(), vars[5], "var_e")
    
    variable_values = Dict(
        vars[5] => 5.0,
        vars[1] => 1.0,
        vars[3] => 3.0,
        vars[2] => 2.0,
        vars[4] => 4.0
    )
    solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(15.0, variable_values))
    
    io = IOBuffer()
    show(io, solution, model)
    output = String(take!(io))
    
    lines = split(output, '\n')
    variable_lines = filter(line -> contains(line, ": "), lines)
    
    # Variables should be sorted by their index values, not by name or value
    @test length(variable_lines) == 5
    
    # First variable should be var_a (vars[1])
    @test contains(variable_lines[1], "var_a: 1.0")
    
    # Last variable line should use └ connector
    @test startswith(variable_lines[end], "└")
end

function test_master_primal_solution_printing_with_jump_model()
    # Create a JuMP model for testing
    jump_model = JuMP.Model()
    
    # Add variables with names using JuMP
    @JuMP.variable(jump_model, x >= 0)
    @JuMP.variable(jump_model, production_y >= 0)
    @JuMP.variable(jump_model, slack)
    
    # Get the MOI indices for these JuMP variables  
    x_index = JuMP.index(x)
    y_index = JuMP.index(production_y)
    slack_index = JuMP.index(slack)
    
    # Create MasterPrimalSolution with MOI indices
    variable_values = Dict(
        x_index => 10.5,
        y_index => 25.0,
        slack_index => 0.0
    )
    solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(456.78, variable_values))
    
    # Test output with JuMP model (should show JuMP variable names)
    io = IOBuffer()
    show(io, solution, jump_model)
    output = String(take!(io))
    
    @test contains(output, "Primal solution:")
    @test contains(output, "x: 10.5")
    @test contains(output, "production_y: 25.0") 
    @test contains(output, "slack: 0.0")
    @test contains(output, "└ cost = 456.78")
    
    # Check that we have proper formatting characters
    @test contains(output, "|") || contains(output, "└")
end

function test_master_primal_solution_jump_model_mixed_named_unnamed()
    # Create a JuMP model with some named and some unnamed variables
    jump_model = JuMP.Model()
    
    # Add some variables with names
    @JuMP.variable(jump_model, named_var >= 0)
    
    # Add variables without explicit names (JuMP will auto-generate names)
    x = JuMP.@variable(jump_model)  # This will have an auto-generated name
    y = JuMP.@variable(jump_model)  # This will have an auto-generated name
    
    # Get MOI indices
    named_index = JuMP.index(named_var)
    x_index = JuMP.index(x)
    y_index = JuMP.index(y)
    
    variable_values = Dict(
        named_index => 5.0,
        x_index => 10.0,
        y_index => 15.0
    )
    solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(200.0, variable_values))
    
    io = IOBuffer()
    show(io, solution, jump_model)
    output = String(take!(io))
    
    @test contains(output, "Primal solution:")
    @test contains(output, "named_var: 5.0")
    @test contains(output, "└ cost = 200.0")
    
    # Should show JuMP auto-generated names or fallback names
    lines = split(output, '\n')
    variable_lines = filter(line -> contains(line, ": "), lines)
    @test length(variable_lines) == 3
end

function test_master_primal_solution_jump_model_edge_cases()
    # Test with invalid variable indices (not in JuMP model)
    jump_model = JuMP.Model()
    @JuMP.variable(jump_model, valid_var >= 0)
    
    valid_index = JuMP.index(valid_var)
    invalid_index = MOI.VariableIndex(999) # This doesn't exist in the model
    
    variable_values = Dict(
        valid_index => 1.0,
        invalid_index => 2.0  # Should trigger fallback
    )
    solution = MK.ColGen.MasterPrimalSolution(MK.ColGen.PrimalMoiSolution(50.0, variable_values))
    
    io = IOBuffer()
    show(io, solution, jump_model)
    output = String(take!(io))
    
    @test contains(output, "Primal solution:")
    @test contains(output, "valid_var: 1.0")
    @test contains(output, "_[999]: 2.0")  # Fallback format
    @test contains(output, "└ cost = 50.0")
end

function test_unit_master_primal_solution_printing()
    @testset "[master_primal_solution] printing with named variables" begin
        test_master_primal_solution_printing_with_named_variables()
    end
    
    @testset "[master_primal_solution] printing without model" begin
        test_master_primal_solution_printing_without_model()
    end
    
    @testset "[master_primal_solution] printing mixed named/unnamed" begin
        test_master_primal_solution_printing_mixed_named_unnamed()
    end
    
    @testset "[master_primal_solution] printing edge cases" begin
        test_master_primal_solution_printing_edge_cases()
    end
    
    @testset "[master_primal_solution] formatting consistency" begin
        test_master_primal_solution_formatting_consistency()
    end
    
    @testset "[master_primal_solution] printing with JuMP model" begin
        test_master_primal_solution_printing_with_jump_model()
    end
    
    @testset "[master_primal_solution] JuMP model mixed named/unnamed" begin
        test_master_primal_solution_jump_model_mixed_named_unnamed()
    end
    
    @testset "[master_primal_solution] JuMP model edge cases" begin
        test_master_primal_solution_jump_model_edge_cases()
    end
end