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

function test_artificial_variables_setup()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Create a simple problem with equality constraints
    # Variables: x1, x2
    x1 = MOI.add_variable(model)
    x2 = MOI.add_variable(model)
    
    # Add equality constraint: x1 + x2 = 5.0
    eq_func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1), MOI.ScalarAffineTerm(1.0, x2)], 0.0)
    eq_constraint = MOI.add_constraint(model, eq_func, MOI.EqualTo(5.0))
    
    # Set objective: minimize x1 + 2*x2
    obj_func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1), MOI.ScalarAffineTerm(2.0, x2)], 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    
    # Count variables and constraints before artificial variables
    vars_before = MOI.get(model, MOI.ListOfVariableIndices())
    eq_constraints_before = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}())
    
    @test length(vars_before) == 2  # x1, x2
    @test length(eq_constraints_before) == 1  # x1 + x2 = 5
    
    # Use artificial variable cost of 1000.0 for testing
    
    # Simulate the artificial variables setup by calling our helper functions directly
    # Add positive artificial variable (s⁺)
    s_pos = MK.ColGen.add_variable!(model; 
        lower_bound=0.0, 
        constraint_coeffs=Dict(eq_constraint => 1.0),
        objective_coeff=1000.0
    )
    
    # Add negative artificial variable (s⁻)  
    s_neg = MK.ColGen.add_variable!(model;
        lower_bound=0.0,
        constraint_coeffs=Dict(eq_constraint => -1.0),
        objective_coeff=1000.0
    )
    
    # Verify artificial variables were added
    vars_after = MOI.get(model, MOI.ListOfVariableIndices())
    @test length(vars_after) == 4  # x1, x2, s_pos, s_neg
    
    # Verify the constraint now includes artificial variables: x1 + x2 + s_pos - s_neg = 5
    updated_constraint_func = MOI.get(model, MOI.ConstraintFunction(), eq_constraint)
    @test length(updated_constraint_func.terms) == 4
    
    # Verify coefficients
    terms_dict = Dict(term.variable => term.coefficient for term in updated_constraint_func.terms)
    @test terms_dict[x1] == 1.0
    @test terms_dict[x2] == 1.0
    @test terms_dict[s_pos] == 1.0
    @test terms_dict[s_neg] == -1.0
    
    # Verify constraint RHS is unchanged
    constraint_set = MOI.get(model, MOI.ConstraintSet(), eq_constraint)
    @test constraint_set.value == 5.0
    
    # Verify objective includes artificial variables with correct costs
    updated_obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    @test length(updated_obj.terms) == 4  # x1, x2, s_pos, s_neg
    
    obj_terms_dict = Dict(term.variable => term.coefficient for term in updated_obj.terms)
    @test obj_terms_dict[x1] == 1.0
    @test obj_terms_dict[x2] == 2.0
    @test obj_terms_dict[s_pos] == 1000.0
    @test obj_terms_dict[s_neg] == 1000.0
    
    # Verify bounds on artificial variables
    lower_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}())
    @test length(lower_constraints) == 2  # s_pos >= 0, s_neg >= 0
    
    # Find bounds for our artificial variables
    s_pos_bound_found = false
    s_neg_bound_found = false
    for bound_constraint in lower_constraints
        bound_var = MOI.get(model, MOI.ConstraintFunction(), bound_constraint)
        bound_set = MOI.get(model, MOI.ConstraintSet(), bound_constraint)
        if bound_var == s_pos
            @test bound_set.lower == 0.0
            s_pos_bound_found = true
        elseif bound_var == s_neg
            @test bound_set.lower == 0.0
            s_neg_bound_found = true
        end
    end
    @test s_pos_bound_found
    @test s_neg_bound_found
end

function test_artificial_variables_inequality_constraints()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()
    )
    
    # Create variables: x1, x2
    x1 = MOI.add_variable(model)
    x2 = MOI.add_variable(model)
    
    # Add less-than-or-equal constraint: x1 + x2 ≤ 10.0
    leq_func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1), MOI.ScalarAffineTerm(1.0, x2)], 0.0)
    leq_constraint = MOI.add_constraint(model, leq_func, MOI.LessThan(10.0))
    
    # Add greater-than-or-equal constraint: x1 - x2 ≥ 2.0
    geq_func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1), MOI.ScalarAffineTerm(-1.0, x2)], 0.0)
    geq_constraint = MOI.add_constraint(model, geq_func, MOI.GreaterThan(2.0))
    
    # Set objective: minimize x1 + x2
    obj_func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1), MOI.ScalarAffineTerm(1.0, x2)], 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    
    # Count constraints before artificial variables
    vars_before = MOI.get(model, MOI.ListOfVariableIndices())
    leq_constraints_before = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}())
    geq_constraints_before = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}())
    
    @test length(vars_before) == 2  # x1, x2
    @test length(leq_constraints_before) == 1  # x1 + x2 ≤ 10
    @test length(geq_constraints_before) == 1  # x1 - x2 ≥ 2
    
    # Add artificial variable for LEQ constraint: x1 + x2 ≤ 10 becomes x1 + x2 + s_leq = 10
    s_leq = MK.ColGen.add_variable!(model; 
        lower_bound=0.0, 
        constraint_coeffs=Dict(leq_constraint => 1.0),
        objective_coeff=500.0
    )
    
    # Add artificial variable for GEQ constraint: x1 - x2 ≥ 2 becomes x1 - x2 - s_geq = 2
    s_geq = MK.ColGen.add_variable!(model;
        lower_bound=0.0,
        constraint_coeffs=Dict(geq_constraint => -1.0),
        objective_coeff=500.0
    )
    
    # Verify artificial variables were added
    vars_after = MOI.get(model, MOI.ListOfVariableIndices())
    @test length(vars_after) == 4  # x1, x2, s_leq, s_geq
    
    # Verify LEQ constraint: x1 + x2 + s_leq ≤ 10
    updated_leq_func = MOI.get(model, MOI.ConstraintFunction(), leq_constraint)
    @test length(updated_leq_func.terms) == 3
    
    leq_terms_dict = Dict(term.variable => term.coefficient for term in updated_leq_func.terms)
    @test leq_terms_dict[x1] == 1.0
    @test leq_terms_dict[x2] == 1.0
    @test leq_terms_dict[s_leq] == 1.0
    
    leq_set = MOI.get(model, MOI.ConstraintSet(), leq_constraint)
    @test leq_set.upper == 10.0
    
    # Verify GEQ constraint: x1 - x2 - s_geq ≥ 2
    updated_geq_func = MOI.get(model, MOI.ConstraintFunction(), geq_constraint)
    @test length(updated_geq_func.terms) == 3
    
    geq_terms_dict = Dict(term.variable => term.coefficient for term in updated_geq_func.terms)
    @test geq_terms_dict[x1] == 1.0
    @test geq_terms_dict[x2] == -1.0
    @test geq_terms_dict[s_geq] == -1.0
    
    geq_set = MOI.get(model, MOI.ConstraintSet(), geq_constraint)
    @test geq_set.lower == 2.0
    
    # Verify objective includes artificial variables
    updated_obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    @test length(updated_obj.terms) == 4  # x1, x2, s_leq, s_geq
    
    obj_terms_dict = Dict(term.variable => term.coefficient for term in updated_obj.terms)
    @test obj_terms_dict[x1] == 1.0
    @test obj_terms_dict[x2] == 1.0
    @test obj_terms_dict[s_leq] == 500.0
    @test obj_terms_dict[s_geq] == 500.0
    
    # Verify bounds on artificial variables
    lower_constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}())
    @test length(lower_constraints) == 2  # s_leq ≥ 0, s_geq ≥ 0
    
    # Find bounds for our artificial variables
    s_leq_bound_found = false
    s_geq_bound_found = false
    for bound_constraint in lower_constraints
        bound_var = MOI.get(model, MOI.ConstraintFunction(), bound_constraint)
        bound_set = MOI.get(model, MOI.ConstraintSet(), bound_constraint)
        if bound_var == s_leq
            @test bound_set.lower == 0.0
            s_leq_bound_found = true
        elseif bound_var == s_geq
            @test bound_set.lower == 0.0
            s_geq_bound_found = true
        end
    end
    @test s_leq_bound_found
    @test s_geq_bound_found
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
    
    @testset "[helpers] artificial variables setup - equality" begin
        test_artificial_variables_setup()
    end
    
    @testset "[helpers] artificial variables setup - inequality" begin
        test_artificial_variables_inequality_constraints()
    end
end