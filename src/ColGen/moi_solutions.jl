# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    PrimalMoiSolution

Unified primal solution type for both master and pricing problems.

Fields:
- obj_value::Float64: Objective function value of the solution
- variable_values::Dict{MOI.VariableIndex,Float64}: Variable index to value mapping
"""
struct PrimalMoiSolution
    obj_value::Float64
    variable_values::Dict{MOI.VariableIndex,Float64}
end

"""
    DualMoiSolution

Unified dual solution type for both master and pricing problems.

Fields:
- obj_value::Float64: Dual objective function value
- constraint_duals::Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}: Constraint dual values organized by constraint type
"""
struct DualMoiSolution
    obj_value::Float64
    constraint_duals::Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}
end

function Base.show(io::IO, sol::PrimalMoiSolution, model)
    println(io, "Primal solution:")
    
    # Sort variables by index for consistent output
    sorted_vars = sort(collect(sol.variable_values), by = x -> x[1].value)
    
    for (i, (var_index, value)) in enumerate(sorted_vars)
        # Get variable name if it exists
        var_name = MOI.get(model, MOI.VariableName(), var_index)
        if isempty(var_name)
            var_name = "_[$(var_index.value)]"
        end
        
        # Use appropriate connector: | for middle items, └ for last item
        connector = i == length(sorted_vars) ? "└" : "|"
        println(io, "$connector $var_name: $value")
    end
    
    print(io, "└ cost = $(sol.obj_value)")
end

function Base.show(io::IO, sol::PrimalMoiSolution, jump_model::JuMP.Model)
    println(io, "Primal solution:")
    
    # Sort variables by index for consistent output
    sorted_vars = sort(collect(sol.variable_values), by = x -> x[1].value)
    
    for (i, (var_index, value)) in enumerate(sorted_vars)
        # Convert MOI.VariableIndex to JuMP.VariableRef to access JuMP variable names
        var_name = try
            var_ref = JuMP.VariableRef(jump_model, var_index)
            jump_name = JuMP.name(var_ref)
            if isempty(jump_name)
                "_[$(var_index.value)]"
            else
                jump_name
            end
        catch
            # Fallback if variable doesn't exist in JuMP model
            "_[$(var_index.value)]"
        end
        
        # Use appropriate connector: | for middle items, └ for last item
        connector = i == length(sorted_vars) ? "└" : "|"
        println(io, "$connector $var_name: $value")
    end
    
    print(io, "└ cost = $(sol.obj_value)")
end

function Base.show(io::IO, sol::PrimalMoiSolution)
    println(io, "Primal solution:")
    
    # Sort variables by index for consistent output
    sorted_vars = sort(collect(sol.variable_values), by = x -> x[1].value)
    
    for (i, (var_index, value)) in enumerate(sorted_vars)
        var_name = "_[$(var_index.value)]"
        
        # Use appropriate connector: | for middle items, └ for last item
        connector = i == length(sorted_vars) ? "└" : "|"
        println(io, "$connector $var_name: $value")
    end
    
    print(io, "└ cost = $(sol.obj_value)")
end

function Base.show(io::IO, sol::DualMoiSolution, model)
    println(io, "Dual solution:")
    
    # Collect all constraints with their types and sort them
    all_constraints = []
    for (constraint_type, constraint_dict) in sol.constraint_duals
        for (index_value, dual_value) in constraint_dict
            # Reconstruct the MOI.ConstraintIndex from type and value
            constraint_index = constraint_type(index_value)
            push!(all_constraints, (constraint_type, constraint_index, dual_value))
        end
    end
    
    # Sort by constraint type name, then by index value for consistency
    sort!(all_constraints, by = x -> (string(x[1]), x[2].value))
    
    for (i, (constraint_type, constraint_index, dual_value)) in enumerate(all_constraints)
        # Get constraint name if it exists, with special handling for variable bounds
        constraint_name = try
            # Check if this is a variable bound constraint (function is MOI.VariableIndex)
            constraint_func = MOI.get(model, MOI.ConstraintFunction(), constraint_index)
            if constraint_func isa MOI.VariableIndex
                # This is a variable bound constraint
                var_index = constraint_func
                var_name = MOI.get(model, MOI.VariableName(), var_index)
                if isempty(var_name)
                    var_name = "var[$(var_index.value)]"
                end
                
                # Get the constraint set to determine bound type and value
                constraint_set = MOI.get(model, MOI.ConstraintSet(), constraint_index)
                if constraint_set isa MOI.GreaterThan
                    "$(var_name) >= $(constraint_set.lower)"
                elseif constraint_set isa MOI.LessThan
                    "$(var_name) <= $(constraint_set.upper)"
                elseif constraint_set isa MOI.EqualTo
                    "$(var_name) == $(constraint_set.value)"
                else
                    # Other bound types (like Interval, etc.)
                    "$(var_name) in $(constraint_set)"
                end
            else
                # Regular constraint - try to get its name
                name = MOI.get(model, MOI.ConstraintName(), constraint_index)
                if isempty(name)
                    "constr[$(constraint_type)][$(constraint_index.value)]"
                else
                    name
                end
            end
        catch
            # Fallback if constraint doesn't exist in model
            "constr[$(constraint_type)][$(constraint_index.value)]"
        end
        
        # Use appropriate connector: | for middle items, └ for last item
        connector = i == length(all_constraints) ? "└" : "|"
        println(io, "$connector $constraint_name: $dual_value")
    end
    
    print(io, "└ cost = $(sol.obj_value)")
end

function Base.show(io::IO, sol::DualMoiSolution, jump_model::JuMP.Model)
    println(io, "Dual solution:")
    
    # Collect all constraints with their types and sort them
    all_constraints = []
    for (constraint_type, constraint_dict) in sol.constraint_duals
        for (index_value, dual_value) in constraint_dict
            # Reconstruct the MOI.ConstraintIndex from type and value
            constraint_index = constraint_type(index_value)
            push!(all_constraints, (constraint_type, constraint_index, dual_value))
        end
    end
    
    # Sort by constraint type name, then by index value for consistency
    sort!(all_constraints, by = x -> (string(x[1]), x[2].value))
    
    for (i, (constraint_type, constraint_index, dual_value)) in enumerate(all_constraints)
        # Get constraint name from JuMP model if it exists, with special handling for variable bounds
        constraint_name = try
            # Get MOI backend to check constraint function type
            moi_backend = JuMP.backend(jump_model)
            constraint_func = MOI.get(moi_backend, MOI.ConstraintFunction(), constraint_index)
            
            if constraint_func isa MOI.VariableIndex
                # This is a variable bound constraint
                var_index = constraint_func
                # Convert to JuMP variable reference to get name
                var_ref = JuMP.VariableRef(jump_model, var_index)
                var_name = JuMP.name(var_ref)
                if isempty(var_name)
                    var_name = "var[$(var_index.value)]"
                end
                
                # Get the constraint set to determine bound type and value
                constraint_set = MOI.get(moi_backend, MOI.ConstraintSet(), constraint_index)
                if constraint_set isa MOI.GreaterThan
                    "$(var_name) >= $(constraint_set.lower)"
                elseif constraint_set isa MOI.LessThan
                    "$(var_name) <= $(constraint_set.upper)"
                elseif constraint_set isa MOI.EqualTo
                    "$(var_name) == $(constraint_set.value)"
                else
                    # Other bound types (like Interval, etc.)
                    "$(var_name) in $(constraint_set)"
                end
            else
                # Regular constraint - try to get JuMP constraint name
                constraint_ref = JuMP.constraint_ref_with_index(jump_model, constraint_index)
                jump_name = JuMP.name(constraint_ref)
                if isempty(jump_name)
                    "constr[$(constraint_type)][$(constraint_index.value)]"
                else
                    jump_name
                end
            end
        catch
            # Fallback if constraint doesn't exist in JuMP model
            "constr[$(constraint_type)][$(constraint_index.value)]"
        end
        
        # Use appropriate connector: | for middle items, └ for last item
        connector = i == length(all_constraints) ? "└" : "|"
        println(io, "$connector $constraint_name: $dual_value")
    end
    
    print(io, "└ cost = $(sol.obj_value)")
end

function Base.show(io::IO, sol::DualMoiSolution)
    println(io, "Dual solution:")
    
    # Collect all constraints with their types and sort them
    all_constraints = []
    for (constraint_type, constraint_dict) in sol.constraint_duals
        for (index_value, dual_value) in constraint_dict
            push!(all_constraints, (constraint_type, index_value, dual_value))
        end
    end
    
    # Sort by constraint type name, then by index value for consistency
    sort!(all_constraints, by = x -> (string(x[1]), x[2]))
    
    for (i, (constraint_type, index_value, dual_value)) in enumerate(all_constraints)
        constraint_name = "constr[$(constraint_type)][$(index_value)]"
        
        # Use appropriate connector: | for middle items, └ for last item
        connector = i == length(all_constraints) ? "└" : "|"
        println(io, "$connector $constraint_name: $dual_value")
    end
    
    print(io, "└ cost = $(sol.obj_value)")
end

"""
    recompute_cost(dual_sol::DualMoiSolution, model)::Float64

Recompute the dual objective cost by multiplying dual values with RHS values and adding the objective constant.
The formula is: ∑(dual_value × rhs_value) + objective_constant
"""
function recompute_cost(dual_sol::DualMoiSolution, model)::Float64
    total_cost = 0.0
    
    # Iterate through all constraint types and their dual values
    for (constraint_type, constraint_dict) in dual_sol.constraint_duals
        for (index_value, dual_value) in constraint_dict
            # Reconstruct the MOI.ConstraintIndex from type and value
            constraint_index = constraint_type(index_value)
            
            try
                # Get the constraint set to extract RHS value
                constraint_set = MOI.get(model, MOI.ConstraintSet(), constraint_index)
                
                # Extract RHS based on constraint set type
                rhs_value = if constraint_set isa MOI.LessThan
                    constraint_set.upper
                elseif constraint_set isa MOI.GreaterThan
                    constraint_set.lower
                elseif constraint_set isa MOI.EqualTo
                    constraint_set.value
                else
                    # For other constraint types (like Interval), we might need more sophisticated handling
                    # For now, skip these constraints
                    continue
                end
                
                # Accumulate: dual_value * rhs_value
                total_cost += dual_value * rhs_value
                
            catch e
                # If constraint doesn't exist in model or other error, skip it
                # This handles cases where constraint indices might be stale
                continue
            end
        end
    end
    
    # TODO.: ugly
    # Add objective constant term
    try
        objective_function = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        objective_constant = objective_function.constant
        total_cost += objective_constant
    catch
        # If objective function is not ScalarAffineFunction or doesn't exist, 
        # try other common objective types or default to 0
        try
            objective_function = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
            objective_constant = objective_function.constant
            total_cost += objective_constant
        catch
            # Default to 0 if we can't extract constant (e.g., single variable objective)
        end
    end
    
    return total_cost
end