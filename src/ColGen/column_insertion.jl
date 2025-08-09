# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function _compute_original_column_cost(column::PricingPrimalMoiSolution, original_cost_mapping::RK.OriginalCostMapping)
    # Compute the original cost of the column using costs from the compact formulation
    # This is ∑(c_i * x_i) where c_i are original variable costs and x_i are solution values
    original_cost = 0.0
    for (var_index, var_value) in column.solution.variable_values
        if haskey(original_cost_mapping, var_index)
            original_cost += original_cost_mapping[var_index] * var_value
        end
    end
    return original_cost
end

function _compute_master_constraint_membership(
    column::PricingPrimalMoiSolution, 
    coupling_mapping::RK.CouplingConstraintMapping,
    master::Master
)
    constraint_coeffs = Dict{MOI.ConstraintIndex, Float64}()
    sp_id = column.subproblem_id
    
    # Compute coupling constraint memberships (A * x for each constraint)
    for (var_index, var_value) in column.solution.variable_values
        coefficients = RK.get_variable_coefficients(coupling_mapping, var_index)
        for (constraint_type, constraint_value, coeff) in coefficients
            constraint_ref = constraint_type(constraint_value)
            constraint_coeffs[constraint_ref] = get(constraint_coeffs, constraint_ref, 0.0) + coeff * var_value
        end
    end
    
    # Add convexity constraint membership (coefficient = 1.0)
    if haskey(master.convexity_constraints_ub, sp_id)
        conv_constraint_ref = master.convexity_constraints_ub[sp_id]
        constraint_coeffs[conv_constraint_ref] = 1.0
    end
    if haskey(master.convexity_constraints_lb, sp_id)
        conv_constraint_ref = master.convexity_constraints_lb[sp_id]
        constraint_coeffs[conv_constraint_ref] = 1.0
    end
    
    return constraint_coeffs
end

function insert_columns!(context::DantzigWolfeColGenImpl, ::MixedPhase1and2, columns_to_insert::PricingPrimalMoiSolutionToInsert)
    master = get_master(context)
    master_moi = moi_master(master)
    pricing_subprobs = get_pricing_subprobs(context)
    
    cols_inserted = 0
    
    for column in columns_to_insert.collection
        # Get subproblem information
        sp_id = column.subproblem_id
        pricing_sp = pricing_subprobs[sp_id]
        
        # Compute original column cost (from compact formulation variable costs)
        original_cost = _compute_original_column_cost(column, pricing_sp.original_cost_mapping)
        
        # Compute master constraint membership (how much this solution contributes to each constraint)
        constraint_memberships = _compute_master_constraint_membership(
            column, 
            pricing_sp.coupling_constr_mapping,
            master
        )
        
        # Add the column variable to master
        # - Lower bound 0.0: convex combination coefficients are non-negative
        # - Constraint coeffs: membership values computed above
        # - Objective coeff: original cost from compact formulation
        column_var = add_variable!(
            master_moi;
            lower_bound = 0.0,
            constraint_coeffs = constraint_memberships,
            objective_coeff = original_cost
        )
        
        cols_inserted += 1
    end
    
    return cols_inserted
end