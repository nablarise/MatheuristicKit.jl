# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
add_variable!(model; lower_bound, upper_bound, variable_type, constraint_coeffs, objective_coeff)

Add a new variable to a MOI model with specified bounds and constraint coefficients.

Keyword Arguments:
- lower_bound: Lower bound for the variable (default: nothing)
- upper_bound: Upper bound for the variable (default: nothing)
- variable_type: MOI constraint for variable type (e.g., MOI.Integer(), MOI.ZeroOne()) (default: nothing)
- constraint_coeffs: Dict mapping constraint references to coefficients (default: empty)
- objective_coeff: Objective coefficient for the new variable (default: 0.0)

Returns:
- MOI.VariableIndex: Reference to the created variable
"""
function add_variable!(
    model;
    lower_bound = nothing,
    upper_bound = nothing,
    variable_type = nothing,
    constraint_coeffs::Dict{<:MOI.ConstraintIndex, Float64} = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, <:MOI.AbstractSet}, Float64}(),
    objective_coeff::Float64 = 0.0
)
    # Add the variable to the model
    var = MOI.add_variable(model)
    
    # Set variable bounds if specified
    if lower_bound !== nothing
        MOI.add_constraint(model, var, MOI.GreaterThan(lower_bound))
    end
    if upper_bound !== nothing
        MOI.add_constraint(model, var, MOI.LessThan(upper_bound))
    end
    
    # Apply variable type constraint if specified
    if variable_type !== nothing
        MOI.add_constraint(model, var, variable_type)
    end
    # Continuous variables don't need additional constraints
    
    # Update existing constraints with new variable coefficients
    for (constraint_ref, coeff) in constraint_coeffs
        if coeff != 0.0
            # Get the current constraint function
            current_func = MOI.get(model, MOI.ConstraintFunction(), constraint_ref)
            
            # Add the new variable term
            new_term = MOI.ScalarAffineTerm(coeff, var)
            new_func = MOI.ScalarAffineFunction([current_func.terms..., new_term], current_func.constant)
            
            # Update the constraint
            MOI.set(model, MOI.ConstraintFunction(), constraint_ref, new_func)
        end
    end
    
    # Update objective function if coefficient is non-zero
    if objective_coeff != 0.0
        current_obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        new_term = MOI.ScalarAffineTerm(objective_coeff, var)
        new_obj = MOI.ScalarAffineFunction([current_obj.terms..., new_term], current_obj.constant)
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), new_obj)
    end
    
    return var
end

"""
add_constraint!(model, coeffs, constraint_set)

Add a new linear constraint to a MOI model.

Arguments:
- model: MOI model to modify
- coeffs: Dict mapping variable references to coefficients
- constraint_set: MOI constraint set instance (e.g., MOI.EqualTo(5.0), MOI.LessThan(10.0))

Returns:
- MOI.ConstraintIndex: Reference to the created constraint
"""
function add_constraint!(
    model,
    coeffs::Dict{MOI.VariableIndex, Float64},
    constraint_set::MOI.AbstractSet
)
    # Create the constraint function
    terms = [MOI.ScalarAffineTerm(coeff, var) for (var, coeff) in coeffs if coeff != 0.0]
    func = MOI.ScalarAffineFunction(terms, 0.0)
    
    # Add the constraint to the model using the constraint set directly
    constraint_ref = MOI.add_constraint(model, func, constraint_set)
    
    return constraint_ref
end