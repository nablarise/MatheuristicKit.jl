# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function _compute_original_column_cost(
    column::PricingPrimalMoiSolution,
    callbacks,#::AbstractColumnGenerationCallbacks,
)
    # Delegate to callback to compute the original cost from the compact formulation
    return RK.compute_column_cost(callbacks, column.solution)
end

function _compute_master_constraint_membership(
    column::PricingPrimalMoiSolution,
    callbacks,#::AbstractColumnGenerationCallbacks,
    subproblem_id,
    master::Master
)
    # Get coupling constraint coefficients from callback
    # Note: Callback returns only coupling constraints, not convexity
    constraint_coeffs = RK.compute_column_coefficients(
        callbacks,
        column.solution,
    )

    # Add convexity constraint memberships (MatheuristicKit's responsibility)
    # Coefficient = 1.0 for both lb and ub convexity constraints
    if haskey(master.convexity_constraints_ub, subproblem_id)
        conv_constraint_ref = master.convexity_constraints_ub[subproblem_id]
        constraint_coeffs[conv_constraint_ref] = 1.0
    end
    if haskey(master.convexity_constraints_lb, subproblem_id)
        conv_constraint_ref = master.convexity_constraints_lb[subproblem_id]
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
        
        # Compute original column cost using callbacks
        original_cost = _compute_original_column_cost(
            column,
            pricing_sp.callbacks
        )

        # Compute master constraint membership using callbacks
        constraint_memberships = _compute_master_constraint_membership(
            column,
            pricing_sp.callbacks,
            sp_id,
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