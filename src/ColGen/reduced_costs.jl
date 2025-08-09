# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct ReducedCosts
    values::Dict{Any,Dict{MOI.VariableIndex,Float64}}
end

_constr_sign(::Type{MOI.ConstraintIndex{F, S}}) where {F,S <: MOI.GreaterThan{Float64}} = 1.0
_constr_sign(::Type{MOI.ConstraintIndex{F, S}}) where {F,S <: MOI.LessThan{Float64}} = 1.0
_constr_sign(::Type{MOI.ConstraintIndex{F, S}}) where {F,S <: MOI.EqualTo{Float64}} = 1.0

function compute_reduced_costs!(context::DantzigWolfeColGenImpl, phase::MixedPhase1and2, mast_dual_sol::MasterDualSolution)
    reduced_costs_dict = Dict{Any,Dict{MOI.VariableIndex,Float64}}()

    for (sp_id, pricing_sp) in get_pricing_subprobs(context)
        sp_reduced_costs = Dict{MOI.VariableIndex,Float64}()

        # Direct access to mappings from PricingSubproblem
        coupling_mapping = pricing_sp.coupling_constr_mapping

        # Compute reduced costs: original_cost - dual_contribution
        for (var_index, original_cost) in pricing_sp.original_cost_mapping
            dual_contribution = 0.0

            # Get constraint coefficients for this variable using new RK structure
            coefficients = RK.get_variable_coefficients(coupling_mapping, var_index)

            for (constraint_type, constraint_value, coeff) in coefficients
                # Direct lookup in type-stable dual solution structure
                if haskey(mast_dual_sol.sol.constraint_duals, constraint_type)
                    constraint_dict = mast_dual_sol.sol.constraint_duals[constraint_type]
                    constr_sign = _constr_sign(constraint_type)
                    if haskey(constraint_dict, constraint_value)
                        dual_value = constraint_dict[constraint_value]
                        dual_contribution += constr_sign * coeff * dual_value
                    end
                end
            end
            sp_reduced_costs[var_index] = original_cost - dual_contribution
        end

        reduced_costs_dict[sp_id] = sp_reduced_costs
    end

    return ReducedCosts(reduced_costs_dict)
end

function update_reduced_costs!(context::DantzigWolfeColGenImpl, ::MixedPhase1and2, red_costs::ReducedCosts)
    # Update objective coefficients in each subproblem with reduced costs
    for (sp_id, pricing_sp) in get_pricing_subprobs(context)
        if haskey(red_costs.values, sp_id)
            sp_reduced_costs = red_costs.values[sp_id]

            # Update objective coefficients directly in the MOI model
            for (var_index, reduced_cost) in sp_reduced_costs
                # Use MOI to modify the objective coefficient
                MOI.modify(pricing_sp.moi_model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                    MOI.ScalarCoefficientChange(var_index, reduced_cost))
            end
        end
    end
end

# Initial subproblem dual & primal bounds

compute_sp_init_db(impl::DantzigWolfeColGenImpl, _) = is_minimization(impl) ? -Inf : Inf
compute_sp_init_pb(impl::DantzigWolfeColGenImpl, _) = is_minimization(impl) ? Inf : -Inf