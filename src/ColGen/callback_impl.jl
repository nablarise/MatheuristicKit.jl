# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

variable_values(solution::PrimalMoiSolution) = solution.variable_values
variable_costs(callbacks::RK.MappingBasedCallbacks) = callbacks.original_cost_mapping

function get_constraint_dual_value(solution::DualMoiSolution, constraint_type, constraint_idx)
    haskey(solution.constraint_duals, constraint_type) || return 0.0
    constraint_dict = master_dual_solution.constraint_duals[constraint_type]
    return get(constraint_dict, constraint_idx, 0.0)
end

function RK.compute_column_cost(
    callbacks,#::RK.MappingBasedCallbacks,
    solution::PrimalMoiSolution
)
    cost = 0.0
    for (var_idx, var_value) in variable_values(solution)
        original_cost = RK.get_original_cost(callbacks, var_idx)
        cost += original_cost * var_value
    end
    return cost
end

function RK.compute_column_coefficients(
    callbacks,#::RK.MappingBasedCallbacks,
    solution::PrimalMoiSolution,
)
    constraint_coeffs = Dict{MOI.ConstraintIndex, Float64}()

    # Iterate over non-zero solution variable values (sparse)
    for (var_idx, var_value) in variable_values(solution)
        # Get all constraints this variable appears in
        coefficients = RK.get_variable_coefficients_in_coupling_constraints(callbacks, var_idx)

        # Accumulate contributions to each constraint
        for (constraint_type, constraint_value, coeff) in coefficients
            constraint_ref = constraint_type(constraint_value)
            constraint_coeffs[constraint_ref] = get(constraint_coeffs, constraint_ref, 0.0) + coeff * var_value
        end
    end

    return constraint_coeffs
end

function RK.compute_reduced_costs(
    callbacks,#::RK.MappingBasedCallbacks,
    master_dual_solution::DualMoiSolution
)
    sp_reduced_costs = Dict{MOI.VariableIndex, Float64}()

    # Iterate over all variables in the subproblem
    for (var_idx, original_cost) in variable_costs(callbacks)
        dual_contribution = 0.0

        # Get constraint coefficients for this variable
        coefficients = RK.get_variable_coefficients_in_coupling_constraints(callbacks, var_idx)

        # Subtract dual contributions from coupling constraints
        for (constraint_type, constraint_idx, coeff) in coefficients
            dual_value = get_constraint_dual_value(master_dual_solution, constraint_type, constraint_idx)
            dual_contribution += coeff * dual_value
        end

        sp_reduced_costs[var_idx] = original_cost - dual_contribution
    end

    return sp_reduced_costs
end
