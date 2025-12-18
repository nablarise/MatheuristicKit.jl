# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct ReducedCosts
    values::Dict{Any,Dict{MOI.VariableIndex,Float64}}
end

function compute_reduced_costs!(context::DantzigWolfeColGenImpl, phase::MixedPhase1and2, mast_dual_sol::MasterDualSolution)
    reduced_costs_dict = Dict{Any,Dict{MOI.VariableIndex,Float64}}()

    for (sp_id, pricing_sp) in get_pricing_subprobs(context)
        # Use callback to compute all reduced costs for this subproblem
        sp_reduced_costs = RK.compute_reduced_costs(
            pricing_sp.callbacks,
            mast_dual_sol.sol
        )

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