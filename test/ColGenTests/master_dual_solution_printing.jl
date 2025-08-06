# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

import MatheuristicKit as MK
using Test
import MathOptInterface as MOI

function test_master_dual_solution_printing_with_named_constraints()
    # Create a simple MOI model for testing
    model = MOI.Utilities.Model{Float64}()
    
    # Add variables
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    
    # Add constraints with names
    eq_constraint = MOI.add_constraint(model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)], 0.0), MOI.EqualTo(5.0))
    leq_constraint = MOI.add_constraint(model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(2.0, x)], 0.0), MOI.LessThan(10.0))
    
    MOI.set(model, MOI.ConstraintName(), eq_constraint, "balance_constraint")
    MOI.set(model, MOI.ConstraintName(), leq_constraint, "capacity_constraint")
    
    # Create constraint_duals structure manually (simulating _populate_constraint_duals)
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    
    eq_type = typeof(eq_constraint)
    leq_type = typeof(leq_constraint)
    
    constraint_duals[eq_type] = Dict{Int64,Float64}(eq_constraint.value => 2.5)
    constraint_duals[leq_type] = Dict{Int64,Float64}(leq_constraint.value => 1.0)
    
    solution = MK.ColGen.MasterDualSolution(123.45, constraint_duals)
    
    # Test output with model (named constraints)
    io = IOBuffer()
    show(io, solution, model)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    @test contains(output, "balance_constraint: 2.5")
    @test contains(output, "capacity_constraint: 1.0")
    @test contains(output, "└ cost = 123.45")
    
    # Check that we have proper formatting characters
    @test contains(output, "|") || contains(output, "└")
    
    # Verify we have the right number of constraints
    lines = split(output, '\n')
    constraint_lines = filter(line -> contains(line, ": "), lines)
    @test length(constraint_lines) == 2
end

function test_master_dual_solution_printing_without_model()
    # Create MasterDualSolution without access to model names
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    
    # Add some mock constraint types and values
    eq_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}
    leq_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}
    
    constraint_duals[eq_type] = Dict{Int64,Float64}(1 => 3.5, 3 => -2.0)
    constraint_duals[leq_type] = Dict{Int64,Float64}(2 => 0.5)
    
    solution = MK.ColGen.MasterDualSolution(-42.7, constraint_duals)
    
    # Test output without model (fallback names)
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    @test contains(output, "constr[$(eq_type)][1]: 3.5")
    @test contains(output, "constr[$(eq_type)][3]: -2.0")
    @test contains(output, "constr[$(leq_type)][2]: 0.5")
    @test contains(output, "└ cost = -42.7")
    
    # Check that we have proper formatting characters
    @test contains(output, "|") || contains(output, "└")
    
    # Verify we have the right number of constraints
    lines = split(output, '\n')
    constraint_lines = filter(line -> contains(line, ": "), lines)
    @test length(constraint_lines) == 3
end

function test_master_dual_solution_printing_mixed_named_unnamed()
    # Create a model with some named and some unnamed constraints
    model = MOI.Utilities.Model{Float64}()
    
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    
    # Add constraints
    named_constraint = MOI.add_constraint(model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0), MOI.EqualTo(1.0))
    unnamed_constraint = MOI.add_constraint(model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, y)], 0.0), MOI.LessThan(2.0))
    
    # Name only one constraint
    MOI.set(model, MOI.ConstraintName(), named_constraint, "named_constraint")
    
    # Create constraint_duals
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    named_type = typeof(named_constraint)
    unnamed_type = typeof(unnamed_constraint)
    
    constraint_duals[named_type] = Dict{Int64,Float64}(named_constraint.value => 1.5)
    constraint_duals[unnamed_type] = Dict{Int64,Float64}(unnamed_constraint.value => 2.5)
    
    solution = MK.ColGen.MasterDualSolution(100.0, constraint_duals)
    
    io = IOBuffer()
    show(io, solution, model)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    @test contains(output, "named_constraint: 1.5")
    @test contains(output, "constr[") # Should have fallback names for unnamed constraints
    @test contains(output, "└ cost = 100.0")
end

function test_master_dual_solution_printing_edge_cases()
    # Test empty solution
    empty_constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    empty_solution = MK.ColGen.MasterDualSolution(0.0, empty_constraint_duals)
    
    io = IOBuffer()
    show(io, empty_solution)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    @test contains(output, "└ cost = 0.0")
    
    # Test single constraint
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    eq_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}
    constraint_duals[eq_type] = Dict{Int64,Float64}(42 => 123.456)
    
    single_solution = MK.ColGen.MasterDualSolution(999.999, constraint_duals)
    
    io = IOBuffer()
    show(io, single_solution)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    @test contains(output, "└ constr[$(eq_type)][42]: 123.456")
    @test contains(output, "└ cost = 999.999")
    
    # Should not have any "|" since there's only one constraint
    constraint_lines = filter(line -> contains(line, ": "), split(output, '\n'))
    @test length(constraint_lines) == 1
end

function test_master_dual_solution_formatting_consistency()
    # Test that constraints are consistently sorted
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    
    # Add constraints in non-sequential order with different types
    eq_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}
    leq_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}
    geq_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}
    
    constraint_duals[leq_type] = Dict{Int64,Float64}(3 => 3.0, 1 => 1.0)
    constraint_duals[eq_type] = Dict{Int64,Float64}(2 => 2.0)
    constraint_duals[geq_type] = Dict{Int64,Float64}(4 => 4.0)
    
    solution = MK.ColGen.MasterDualSolution(15.0, constraint_duals)
    
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))
    
    lines = split(output, '\n')
    constraint_lines = filter(line -> contains(line, ": "), lines)
    
    # Should have 4 constraint lines
    @test length(constraint_lines) == 4
    
    # Last constraint line should use └ connector
    @test startswith(constraint_lines[end], "└")
    
    # First few should use | connector
    for i in 1:(length(constraint_lines)-1)
        @test startswith(constraint_lines[i], "|")
    end
end

function test_master_dual_solution_printing_with_jump_model()
    # Create a JuMP model for testing
    jump_model = JuMP.Model()
    
    # Add variables
    @JuMP.variable(jump_model, x >= 0)
    @JuMP.variable(jump_model, y >= 0)
    
    # Add constraints with names
    @JuMP.constraint(jump_model, balance_constraint, x + y == 5)
    @JuMP.constraint(jump_model, capacity_constraint, 2*x <= 10)
    
    # Get MOI constraint indices
    balance_index = JuMP.index(balance_constraint)
    capacity_index = JuMP.index(capacity_constraint)
    
    # Create constraint_duals structure
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    balance_type = typeof(balance_index)
    capacity_type = typeof(capacity_index)
    
    constraint_duals[balance_type] = Dict{Int64,Float64}(balance_index.value => 2.5)
    constraint_duals[capacity_type] = Dict{Int64,Float64}(capacity_index.value => 1.0)
    
    solution = MK.ColGen.MasterDualSolution(456.78, constraint_duals)
    
    # Test output with JuMP model (should show JuMP constraint names)
    io = IOBuffer()
    show(io, solution, jump_model)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    @test contains(output, "balance_constraint: 2.5")
    @test contains(output, "capacity_constraint: 1.0")
    @test contains(output, "└ cost = 456.78")
    
    # Check that we have proper formatting characters
    @test contains(output, "|") || contains(output, "└")
end

function test_master_dual_solution_jump_model_mixed_named_unnamed()
    # Create a JuMP model with some named and some unnamed constraints
    jump_model = JuMP.Model()
    
    @JuMP.variable(jump_model, x >= 0)
    @JuMP.variable(jump_model, y >= 0)
    
    # Add constraint with explicit name
    @JuMP.constraint(jump_model, named_constraint, x >= 1)
    
    # Add constraint without explicit name (JuMP will auto-generate)
    unnamed_constraint = JuMP.@constraint(jump_model, y <= 2)
    
    # Get MOI indices
    named_index = JuMP.index(named_constraint)
    unnamed_index = JuMP.index(unnamed_constraint)
    
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    named_type = typeof(named_index)
    unnamed_type = typeof(unnamed_index)
    
    constraint_duals[named_type] = Dict{Int64,Float64}(named_index.value => 5.0)
    constraint_duals[unnamed_type] = Dict{Int64,Float64}(unnamed_index.value => 10.0)
    
    solution = MK.ColGen.MasterDualSolution(200.0, constraint_duals)
    
    io = IOBuffer()
    show(io, solution, jump_model)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    @test contains(output, "named_constraint: 5.0")
    @test contains(output, "└ cost = 200.0")
    
    # Should show JuMP auto-generated names or fallback names
    lines = split(output, '\n')
    constraint_lines = filter(line -> contains(line, ": "), lines)
    @test length(constraint_lines) == 2
end

function test_master_dual_solution_jump_model_edge_cases()
    # Test with invalid constraint indices (not in JuMP model)
    jump_model = JuMP.Model()
    @JuMP.variable(jump_model, x >= 0)
    @JuMP.constraint(jump_model, valid_constraint, x >= 0)
    
    valid_index = JuMP.index(valid_constraint)
    
    # Create constraint_duals with valid and invalid indices
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    valid_type = typeof(valid_index)
    
    constraint_duals[valid_type] = Dict{Int64,Float64}(
        valid_index.value => 1.0,
        999 => 2.0  # Invalid index - should trigger fallback
    )
    
    solution = MK.ColGen.MasterDualSolution(50.0, constraint_duals)
    
    io = IOBuffer()
    show(io, solution, jump_model)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    @test contains(output, "valid_constraint: 1.0")
    @test contains(output, "constr[$(valid_type)][999]: 2.0")  # Fallback format
    @test contains(output, "└ cost = 50.0")
end

function test_master_dual_solution_variable_bounds_display()
    # Test variable bounds display enhancement
    moi_model = MOI.Utilities.Model{Float64}()
    
    # Add variables
    x = MOI.add_variable(moi_model)
    y = MOI.add_variable(moi_model)
    z = MOI.add_variable(moi_model)
    
    # Set variable names
    MOI.set(moi_model, MOI.VariableName(), x, "production_x")
    MOI.set(moi_model, MOI.VariableName(), y, "production_y")
    # z remains unnamed
    
    # Add different types of variable bounds
    lb_x = MOI.add_constraint(moi_model, x, MOI.GreaterThan(0.0))
    ub_y = MOI.add_constraint(moi_model, y, MOI.LessThan(100.0))
    eq_z = MOI.add_constraint(moi_model, z, MOI.EqualTo(50.0))
    
    # Add a regular constraint for comparison
    regular_constraint = MOI.add_constraint(moi_model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)], 0.0), MOI.LessThan(200.0))
    MOI.set(moi_model, MOI.ConstraintName(), regular_constraint, "total_capacity")
    
    # Create constraint_duals structure
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    constraint_duals[typeof(lb_x)] = Dict{Int64,Float64}(lb_x.value => 2.5)
    constraint_duals[typeof(ub_y)] = Dict{Int64,Float64}(ub_y.value => -1.0)
    constraint_duals[typeof(eq_z)] = Dict{Int64,Float64}(eq_z.value => 3.5)
    constraint_duals[typeof(regular_constraint)] = Dict{Int64,Float64}(regular_constraint.value => 1.25)
    
    solution = MK.ColGen.MasterDualSolution(150.0, constraint_duals)
    
    io = IOBuffer()
    show(io, solution, moi_model)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    # Variable bounds should show in readable format
    @test contains(output, "production_x >= 0.0: 2.5")
    @test contains(output, "production_y <= 100.0: -1.0")
    @test contains(output, "var[$(z.value)] == 50.0: 3.5")  # Unnamed variable fallback
    # Regular constraint should show name
    @test contains(output, "total_capacity: 1.25")
    @test contains(output, "└ cost = 150.0")
end

function test_master_dual_solution_variable_bounds_jump_model()
    # Test variable bounds display with JuMP model
    jump_model = JuMP.Model()
    
    @JuMP.variable(jump_model, x >= 5)  # Lower bound
    @JuMP.variable(jump_model, y <= 20) # Upper bound
    @JuMP.constraint(jump_model, capacity, x + y <= 50)  # Regular constraint
    
    # Get constraint indices
    moi_backend = JuMP.backend(jump_model)
    all_constraint_types = MOI.get(moi_backend, MOI.ListOfConstraintTypesPresent())
    
    # Create constraint_duals with sample values
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    
    for (F, S) in all_constraint_types
        constraint_indices = MOI.get(moi_backend, MOI.ListOfConstraintIndices{F,S}())
        if !isempty(constraint_indices)
            constraint_type = typeof(first(constraint_indices))
            constraint_duals[constraint_type] = Dict{Int64,Float64}()
            
            # Add sample dual values
            for (i, ci) in enumerate(constraint_indices)
                constraint_duals[constraint_type][ci.value] = i * 1.5
            end
        end
    end
    
    solution = MK.ColGen.MasterDualSolution(75.0, constraint_duals)
    
    io = IOBuffer()
    show(io, solution, jump_model)
    output = String(take!(io))
    
    @test contains(output, "Dual solution:")
    # Should show variable bounds in readable format with JuMP variable names
    @test contains(output, "x >= 5.0:")
    @test contains(output, "y <= 20.0:")
    # Regular constraint should show JuMP constraint name
    @test contains(output, "capacity:")
    @test contains(output, "└ cost = 75.0")
end

function test_master_dual_solution_recompute_cost()
    # Test basic cost recomputation with various constraint types
    moi_model = MOI.Utilities.Model{Float64}()
    
    # Add variables
    x = MOI.add_variable(moi_model)
    y = MOI.add_variable(moi_model)
    
    # Add different types of constraints with known RHS values
    eq_constraint = MOI.add_constraint(moi_model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)], 0.0), MOI.EqualTo(10.0))
    leq_constraint = MOI.add_constraint(moi_model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(2.0, x)], 0.0), MOI.LessThan(20.0))
    geq_constraint = MOI.add_constraint(moi_model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, y)], 0.0), MOI.GreaterThan(5.0))
    
    # Create constraint_duals with known dual values
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    constraint_duals[typeof(eq_constraint)] = Dict{Int64,Float64}(eq_constraint.value => 2.0)    # dual = 2.0, RHS = 10.0 -> contribution = 20.0
    constraint_duals[typeof(leq_constraint)] = Dict{Int64,Float64}(leq_constraint.value => 1.5)  # dual = 1.5, RHS = 20.0 -> contribution = 30.0
    constraint_duals[typeof(geq_constraint)] = Dict{Int64,Float64}(geq_constraint.value => 3.0)  # dual = 3.0, RHS = 5.0 -> contribution = 15.0
    
    # Expected total cost = 20.0 + 30.0 + 15.0 = 65.0
    solution = MK.ColGen.MasterDualSolution(999.999, constraint_duals)  # Use different obj_value to verify independent computation
    
    recomputed_cost = MK.ColGen.recompute_cost(solution, moi_model)
    
    @test recomputed_cost ≈ 65.0 atol=1e-6
end

function test_master_dual_solution_recompute_cost_empty()
    # Test with empty dual solution
    moi_model = MOI.Utilities.Model{Float64}()
    
    empty_constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    empty_solution = MK.ColGen.MasterDualSolution(123.45, empty_constraint_duals)
    
    recomputed_cost = MK.ColGen.recompute_cost(empty_solution, moi_model)
    
    @test recomputed_cost ≈ 0.0 atol=1e-6
end

function test_master_dual_solution_recompute_cost_with_invalid_constraints()
    # Test handling of invalid/non-existent constraints
    moi_model = MOI.Utilities.Model{Float64}()
    
    # Add one valid constraint
    x = MOI.add_variable(moi_model)
    valid_constraint = MOI.add_constraint(moi_model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0), MOI.EqualTo(5.0))
    
    # Create constraint_duals with valid and invalid constraint indices
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    valid_type = typeof(valid_constraint)
    
    constraint_duals[valid_type] = Dict{Int64,Float64}(
        valid_constraint.value => 2.0,  # Valid: dual = 2.0, RHS = 5.0 -> contribution = 10.0
        999 => 10.0  # Invalid index - should be skipped
    )
    
    solution = MK.ColGen.MasterDualSolution(0.0, constraint_duals)
    
    recomputed_cost = MK.ColGen.recompute_cost(solution, moi_model)
    
    # Should only include the valid constraint: 2.0 * 5.0 = 10.0
    @test recomputed_cost ≈ 10.0 atol=1e-6
end

function test_master_dual_solution_recompute_cost_variable_bounds()
    # Test with variable bounds constraints
    moi_model = MOI.Utilities.Model{Float64}()
    
    # Add variables with bounds
    x = MOI.add_variable(moi_model)
    y = MOI.add_variable(moi_model)
    
    # Add variable bounds
    lb_x = MOI.add_constraint(moi_model, x, MOI.GreaterThan(0.0))
    ub_y = MOI.add_constraint(moi_model, y, MOI.LessThan(100.0))
    eq_z = MOI.add_variable(moi_model)
    eq_bound = MOI.add_constraint(moi_model, eq_z, MOI.EqualTo(50.0))
    
    # Create constraint_duals for variable bounds
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    constraint_duals[typeof(lb_x)] = Dict{Int64,Float64}(lb_x.value => 1.0)      # dual = 1.0, RHS = 0.0 -> contribution = 0.0
    constraint_duals[typeof(ub_y)] = Dict{Int64,Float64}(ub_y.value => 2.0)      # dual = 2.0, RHS = 100.0 -> contribution = 200.0
    constraint_duals[typeof(eq_bound)] = Dict{Int64,Float64}(eq_bound.value => 0.5)  # dual = 0.5, RHS = 50.0 -> contribution = 25.0
    
    # Expected total cost = 0.0 + 200.0 + 25.0 = 225.0
    solution = MK.ColGen.MasterDualSolution(0.0, constraint_duals)
    
    recomputed_cost = MK.ColGen.recompute_cost(solution, moi_model)
    
    @test recomputed_cost ≈ 225.0 atol=1e-6
end

function test_master_dual_solution_recompute_cost_zero_duals()
    # Test with zero dual values
    moi_model = MOI.Utilities.Model{Float64}()
    
    x = MOI.add_variable(moi_model)
    constraint = MOI.add_constraint(moi_model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0), MOI.LessThan(15.0))
    
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    constraint_duals[typeof(constraint)] = Dict{Int64,Float64}(constraint.value => 0.0)
    
    solution = MK.ColGen.MasterDualSolution(42.0, constraint_duals)
    
    recomputed_cost = MK.ColGen.recompute_cost(solution, moi_model)
    
    @test recomputed_cost ≈ 0.0 atol=1e-6
end

function test_unit_master_dual_solution_printing()
    @testset "[master_dual_solution] printing with named constraints" begin
        test_master_dual_solution_printing_with_named_constraints()
    end
    
    @testset "[master_dual_solution] printing without model" begin
        test_master_dual_solution_printing_without_model()
    end
    
    @testset "[master_dual_solution] printing mixed named/unnamed" begin
        test_master_dual_solution_printing_mixed_named_unnamed()
    end
    
    @testset "[master_dual_solution] printing edge cases" begin
        test_master_dual_solution_printing_edge_cases()
    end
    
    @testset "[master_dual_solution] formatting consistency" begin
        test_master_dual_solution_formatting_consistency()
    end
    
    @testset "[master_dual_solution] printing with JuMP model" begin
        test_master_dual_solution_printing_with_jump_model()
    end
    
    @testset "[master_dual_solution] JuMP model mixed named/unnamed" begin
        test_master_dual_solution_jump_model_mixed_named_unnamed()
    end
    
    @testset "[master_dual_solution] JuMP model edge cases" begin
        test_master_dual_solution_jump_model_edge_cases()
    end
    
    @testset "[master_dual_solution] variable bounds display" begin
        test_master_dual_solution_variable_bounds_display()
    end
    
    @testset "[master_dual_solution] variable bounds with JuMP model" begin
        test_master_dual_solution_variable_bounds_jump_model()
    end
    
    @testset "[master_dual_solution] recompute cost basic" begin
        test_master_dual_solution_recompute_cost()
    end
    
    @testset "[master_dual_solution] recompute cost empty" begin
        test_master_dual_solution_recompute_cost_empty()
    end
    
    @testset "[master_dual_solution] recompute cost with invalid constraints" begin
        test_master_dual_solution_recompute_cost_with_invalid_constraints()
    end
    
    @testset "[master_dual_solution] recompute cost variable bounds" begin
        test_master_dual_solution_recompute_cost_variable_bounds()
    end
    
    @testset "[master_dual_solution] recompute cost zero duals" begin
        test_master_dual_solution_recompute_cost_zero_duals()
    end
end