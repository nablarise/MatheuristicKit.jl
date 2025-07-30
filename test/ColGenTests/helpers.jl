# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_add_variable_continuous()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add a continuous variable with no bounds
    var = MK.ColGen.add_variable!(model)
    
    @test MOI.is_valid(model, var)
    lower_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}())
    upper_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.LessThan{Float64}}())
    @test length(lower_constraints) == 0
    @test length(upper_constraints) == 0
end

function test_add_variable_with_bounds1()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add variable with lower bound
    var1 = MK.ColGen.add_variable!(model; lower_bound=0.0)
    @test var1 isa MOI.VariableIndex
    
    # Check lower bound constraint exists and has correct value
    lower_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}())
    @test length(lower_constraints) == 1
    lower_set = MOI.get(model, MOI.ConstraintSet(), lower_constraints[1])
    @test lower_set.lower == 0.0
end

function test_add_variable_with_bounds2()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add variable with upper bound
    var2 = MK.ColGen.add_variable!(model; upper_bound=10.0)
    @test var2 isa MOI.VariableIndex
    
    # Check upper bound constraint exists and has correct value
    upper_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.LessThan{Float64}}())
    @test length(upper_constraints) == 1
    upper_set = MOI.get(model, MOI.ConstraintSet(), upper_constraints[1])
    @test upper_set.upper == 10.0
end

function test_add_variable_with_bounds3()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add variable with both bounds
    var3 = MK.ColGen.add_variable!(model; lower_bound=1.0, upper_bound=5.0)
    @test var3 isa MOI.VariableIndex
    
    # Check both bounds exist with correct values
    lower_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}())
    upper_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.LessThan{Float64}}())
    @test length(lower_constraints) == 1  # var3
    @test length(upper_constraints) == 1  # var3
    
    # Find and verify the specific constraints for var3
    @test MOI.get(model, MOI.ConstraintFunction(), lower_constraints[1]) == var3
    @test MOI.get(model, MOI.ConstraintFunction(), upper_constraints[1]) == var3
    @test MOI.get(model, MOI.ConstraintSet(), lower_constraints[1]).lower == 1.0
    @test MOI.get(model, MOI.ConstraintSet(), upper_constraints[1]).upper == 5.0
end

function test_add_variable_binary()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add binary variable
    var_bin = MK.ColGen.add_variable!(model; variable_type=MOI.ZeroOne())
    @test var_bin isa MOI.VariableIndex
    
    # Check binary constraint exists
    binary_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.ZeroOne}())
    @test length(binary_constraints) == 1
    @test MOI.get(model, MOI.ConstraintFunction(), binary_constraints[1]) == var_bin
end

function test_add_variable_integer()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add integer variable with bounds
    var_int = MK.ColGen.add_variable!(model; variable_type=MOI.Integer(), lower_bound=0.0, upper_bound=100.0)
    @test var_int isa MOI.VariableIndex
    
    # Check integer constraint exists
    integer_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())
    @test length(integer_constraints) == 1
    @test MOI.get(model, MOI.ConstraintFunction(), integer_constraints[1]) == var_int
    
    # Check bounds for integer variable
    lower_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}())
    upper_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.LessThan{Float64}}())
    @test length(lower_constraints) == 1
    @test length(upper_constraints) == 1
    
    # Verify bound values for integer variable
    @test MOI.get(model, MOI.ConstraintFunction(), lower_constraints[1]) == var_int
    @test MOI.get(model, MOI.ConstraintFunction(), upper_constraints[1]) == var_int
    @test MOI.get(model, MOI.ConstraintSet(), lower_constraints[1]).lower == 0.0
    @test MOI.get(model, MOI.ConstraintSet(), upper_constraints[1]).upper == 100.0
end

function test_add_variable_with_constraint_coeffs()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add an initial variable and constraint: 1.0*var1 <= 10.0
    var1 = MOI.add_variable(model)
    func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, var1)], 0.0)
    set = MOI.LessThan(10.0)
    constraint = MOI.add_constraint(model, func, set)
    
    # Add new variable with coefficient 2.0 in existing constraint
    constraint_coeffs = Dict(constraint => 2.0)
    var2 = MK.ColGen.add_variable!(model; constraint_coeffs=constraint_coeffs)
    
    @test var2 isa MOI.VariableIndex
    
    # Check that constraint was updated: should now be 1.0*var1 + 2.0*var2 <= 10.0
    updated_func = MOI.get(model, MOI.ConstraintFunction(), constraint)
    @test length(updated_func.terms) == 2
    @test updated_func.constant == 0.0
    
    # Verify coefficients are correct
    terms_dict = Dict(term.variable => term.coefficient for term in updated_func.terms)
    @test terms_dict[var1] == 1.0
    @test terms_dict[var2] == 2.0
    
    # Verify constraint set (RHS) is unchanged
    constraint_set = MOI.get(model, MOI.ConstraintSet(), constraint)
    @test constraint_set.upper == 10.0
end

function test_add_variable_with_objective_coeff()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Set initial objective: 1.0*var1 + 0.0
    var1 = MOI.add_variable(model)
    obj_func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, var1)], 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    
    # Add variable with objective coefficient 3.0
    var2 = MK.ColGen.add_variable!(model; objective_coeff=3.0)
    
    @test var2 isa MOI.VariableIndex
    
    # Check objective was updated: should now be 1.0*var1 + 3.0*var2 + 0.0
    updated_obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    @test length(updated_obj.terms) == 2
    @test updated_obj.constant == 0.0
    
    # Verify objective coefficients are correct
    obj_terms_dict = Dict(term.variable => term.coefficient for term in updated_obj.terms)
    @test obj_terms_dict[var1] == 1.0
    @test obj_terms_dict[var2] == 3.0
end

function test_add_constraint_equality()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add variables
    var1 = MOI.add_variable(model)
    var2 = MOI.add_variable(model)
    
    # Add equality constraint: 1.0*x1 + 2.0*x2 = 5.0
    coeffs = Dict(var1 => 1.0, var2 => 2.0)
    constraint = MK.ColGen.add_constraint!(model, coeffs, MOI.EqualTo(5.0))
    
    @test constraint isa MOI.ConstraintIndex
    @test MOI.is_valid(model, constraint)
    
    # Verify constraint function coefficients
    constraint_func = MOI.get(model, MOI.ConstraintFunction(), constraint)
    @test length(constraint_func.terms) == 2
    @test constraint_func.constant == 0.0
    
    terms_dict = Dict(term.variable => term.coefficient for term in constraint_func.terms)
    @test terms_dict[var1] == 1.0
    @test terms_dict[var2] == 2.0
    
    # Verify constraint set (RHS)
    constraint_set = MOI.get(model, MOI.ConstraintSet(), constraint)
    @test constraint_set isa MOI.EqualTo{Float64}
    @test constraint_set.value == 5.0
end

function test_add_constraint_inequality()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add variables
    var1 = MOI.add_variable(model)
    var2 = MOI.add_variable(model)
    
    # Add less-than-or-equal constraint: 1.0*x1 + 1.0*x2 <= 10.0
    coeffs_leq = Dict(var1 => 1.0, var2 => 1.0)
    constraint_leq = MK.ColGen.add_constraint!(model, coeffs_leq, MOI.LessThan(10.0))
    @test constraint_leq isa MOI.ConstraintIndex
    
    # Verify LEQ constraint
    leq_func = MOI.get(model, MOI.ConstraintFunction(), constraint_leq)
    leq_set = MOI.get(model, MOI.ConstraintSet(), constraint_leq)
    @test length(leq_func.terms) == 2
    @test leq_func.constant == 0.0
    @test leq_set isa MOI.LessThan{Float64}
    @test leq_set.upper == 10.0
    
    leq_terms_dict = Dict(term.variable => term.coefficient for term in leq_func.terms)
    @test leq_terms_dict[var1] == 1.0
    @test leq_terms_dict[var2] == 1.0
    
    # Add greater-than-or-equal constraint: 1.0*x1 + (-1.0)*x2 >= 0.0
    coeffs_geq = Dict(var1 => 1.0, var2 => -1.0)
    constraint_geq = MK.ColGen.add_constraint!(model, coeffs_geq, MOI.GreaterThan(0.0))
    @test constraint_geq isa MOI.ConstraintIndex
    
    # Verify GEQ constraint
    geq_func = MOI.get(model, MOI.ConstraintFunction(), constraint_geq)
    geq_set = MOI.get(model, MOI.ConstraintSet(), constraint_geq)
    @test length(geq_func.terms) == 2
    @test geq_func.constant == 0.0
    @test geq_set isa MOI.GreaterThan{Float64}
    @test geq_set.lower == 0.0
    
    geq_terms_dict = Dict(term.variable => term.coefficient for term in geq_func.terms)
    @test geq_terms_dict[var1] == 1.0
    @test geq_terms_dict[var2] == -1.0
end

function test_add_constraint_with_coeffs()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Add variables
    var1 = MOI.add_variable(model)
    var2 = MOI.add_variable(model)
    var3 = MOI.add_variable(model)
    
    # Add constraint with mixed coefficients: 2.0*x1 + (-3.0)*x2 + 1.0*x3 <= 15.0
    coeffs = Dict(var1 => 2.0, var2 => -3.0, var3 => 1.0)
    constraint = MK.ColGen.add_constraint!(model, coeffs, MOI.LessThan(15.0))
    
    @test constraint isa MOI.ConstraintIndex
    
    # Check constraint function coefficients
    func = MOI.get(model, MOI.ConstraintFunction(), constraint)
    @test length(func.terms) == 3
    @test func.constant == 0.0
    
    # Verify all coefficients are correct
    terms_dict = Dict(term.variable => term.coefficient for term in func.terms)
    @test terms_dict[var1] == 2.0
    @test terms_dict[var2] == -3.0
    @test terms_dict[var3] == 1.0
    
    # Verify constraint set (RHS)
    constraint_set = MOI.get(model, MOI.ConstraintSet(), constraint)
    @test constraint_set isa MOI.LessThan{Float64}
    @test constraint_set.upper == 15.0
end

function test_unit_helpers()
    @testset "[helpers] add_variable! continuous" begin
        test_add_variable_continuous()
    end
    
    @testset "[helpers] add_variable! with bounds" begin
        test_add_variable_with_bounds1()
        test_add_variable_with_bounds2()
        test_add_variable_with_bounds3()
    end
    
    @testset "[helpers] add_variable! binary" begin
        test_add_variable_binary()
    end
    
    @testset "[helpers] add_variable! integer" begin
        test_add_variable_integer()
    end
    
    @testset "[helpers] add_variable! with constraint coefficients" begin
        test_add_variable_with_constraint_coeffs()
    end
    
    @testset "[helpers] add_variable! with objective coefficient" begin
        test_add_variable_with_objective_coeff()
    end
    
    @testset "[helpers] add_constraint! equality" begin
        test_add_constraint_equality()
    end
    
    @testset "[helpers] add_constraint! inequality" begin
        test_add_constraint_inequality()
    end
    
    @testset "[helpers] add_constraint! with coefficients" begin
        test_add_constraint_with_coeffs()
    end
end