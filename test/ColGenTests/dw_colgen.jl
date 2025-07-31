# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

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

function test_dw_colgen()
    @testset "[dw_colgen] artificial variables setup - equality" begin
        test_artificial_variables_setup()
    end
    
    @testset "[dw_colgen] artificial variables setup - inequality" begin
        test_artificial_variables_inequality_constraints()
    end
end