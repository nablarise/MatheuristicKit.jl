# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function RK.compute_column_cost(
    callbacks::RK.MappingBasedCallbacks,
    solution::PrimalMoiSolution
)
    cost = 0.0
    for (var_idx, var_value) in solution.variable_values
        if haskey(callbacks.original_cost_mapping, var_idx)
            cost += callbacks.original_cost_mapping[var_idx] * var_value
        end
    end
    return cost
end

function RK.compute_column_coefficients(
    callbacks::RK.MappingBasedCallbacks,
    solution::PrimalMoiSolution,
)
    constraint_coeffs = Dict{MOI.ConstraintIndex, Float64}()

    # Iterate over non-zero solution variable values (sparse)
    for (var_idx, var_value) in solution.variable_values
        # Get all constraints this variable appears in
        coefficients = RK.get_variable_coefficients(callbacks.coupling_constr_mapping, var_idx)

        # Accumulate contributions to each constraint
        for (constraint_type, constraint_value, coeff) in coefficients
            constraint_ref = constraint_type(constraint_value)
            constraint_coeffs[constraint_ref] = get(constraint_coeffs, constraint_ref, 0.0) + coeff * var_value
        end
    end

    return constraint_coeffs
end

function RK.compute_reduced_costs(
    callbacks::RK.MappingBasedCallbacks,
    master_dual_solution::DualMoiSolution
)
    sp_reduced_costs = Dict{MOI.VariableIndex, Float64}()

    # Iterate over all variables in the subproblem
    for (var_idx, original_cost) in callbacks.original_cost_mapping
        dual_contribution = 0.0

        # Get constraint coefficients for this variable
        coefficients = RK.get_variable_coefficients(callbacks.coupling_constr_mapping, var_idx)

        # Subtract dual contributions from coupling constraints
        for (constraint_type, constraint_value, coeff) in coefficients
            if haskey(master_dual_solution.constraint_duals, constraint_type)
                constraint_dict = master_dual_solution.constraint_duals[constraint_type]
                if haskey(constraint_dict, constraint_value)
                    dual_value = constraint_dict[constraint_value]
                    dual_contribution += coeff * dual_value
                end
            end
        end

        sp_reduced_costs[var_idx] = original_cost - dual_contribution
    end

    return sp_reduced_costs
end
