# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Pricing strategy

struct DefaultPricingStrategy{PricingSubproblemIterator}
    pricing_sps::PricingSubproblemIterator
end
get_pricing_strategy(impl::DantzigWolfeColGenImpl, ::MixedPhase1and2) = DefaultPricingStrategy(get_pricing_subprobs(impl))
pricing_strategy_iterate(strategy::DefaultPricingStrategy) = iterate(strategy.pricing_sps)
pricing_strategy_iterate(strategy::DefaultPricingStrategy, state) = iterate(strategy.pricing_sps, state)

# Pricing solution

struct PricingSolution{PricingPrimalSolution}
    is_infeasible::Bool
    is_unbounded::Bool
    primal_bound::Float64
    dual_bound::Float64
    primal_sols::Vector{PricingPrimalSolution}
end

is_infeasible(sol::PricingSolution) = sol.is_infeasible
is_unbounded(sol::PricingSolution) = sol.is_unbounded
get_primal_sols(sol::PricingSolution) = sol.primal_sols
get_primal_bound(sol::PricingSolution) = sol.primal_bound
get_dual_bound(sol::PricingSolution) = sol.dual_bound

struct PricingPrimalMoiSolution
    subproblem_id::Any  # Subproblem that generated this solution
    solution::PrimalMoiSolution  # Wraps unified solution type
    is_improving::Bool  # Whether this solution has an improving reduced cost
end

# Set of columns

struct PricingPrimalMoiSolutionToInsert
    collection::Vector{PricingPrimalMoiSolution}
end
set_of_columns(::DantzigWolfeColGenImpl) = PricingPrimalMoiSolutionToInsert(PricingPrimalMoiSolution[])

function push_in_set!(set::PricingPrimalMoiSolutionToInsert, sol::PricingPrimalMoiSolution)
    # Only add columns with improving reduced costs
    if sol.is_improving
        push!(set.collection, sol)
        return true
    else
        return false  # Column filtered out due to non-improving reduced cost
    end
end

# Pricing

struct SubproblemMoiOptimizer end
# TODO: implement pricing callback.
get_pricing_subprob_optimizer(::ExactStage, ::PricingSubproblem) = SubproblemMoiOptimizer()

function optimize_pricing_problem!(context::DantzigWolfeColGenImpl, sp_id::Any, pricing_sp::PricingSubproblem, ::SubproblemMoiOptimizer, mast_dual_sol::MasterDualSolution, stab_changes_mast_dual_sol)    
    MOI.optimize!(moi_pricing_sp(pricing_sp))

    # Get objective value from subproblem (includes coupling constraint reduced costs)
    subproblem_obj_value = MOI.get(moi_pricing_sp(pricing_sp), MOI.ObjectiveValue())
    
    # Compute convexity constraint contribution to get true reduced cost
    master = get_master(context)
    
    lb_dual = 0.0
    ub_dual = 0.0
    
    # Lower bound dual
    if haskey(master.convexity_constraints_lb, sp_id)
        constraint_index = master.convexity_constraints_lb[sp_id]
        constraint_type = typeof(constraint_index)
        constraint_value = constraint_index.value
        
        if haskey(mast_dual_sol.sol.constraint_duals, constraint_type)
            constraint_dict = mast_dual_sol.sol.constraint_duals[constraint_type]
            if haskey(constraint_dict, constraint_value)
                lb_dual = constraint_dict[constraint_value]
            end
        end
    end
    
    # Upper bound dual  
    if haskey(master.convexity_constraints_ub, sp_id)
        constraint_index = master.convexity_constraints_ub[sp_id]
        constraint_type = typeof(constraint_index)
        constraint_value = constraint_index.value
        
        if haskey(mast_dual_sol.sol.constraint_duals, constraint_type)
            constraint_dict = mast_dual_sol.sol.constraint_duals[constraint_type]
            if haskey(constraint_dict, constraint_value)
                ub_dual = constraint_dict[constraint_value]
            end
        end
    end
    
    convexity_contrib = lb_dual + ub_dual
    
    # True reduced cost = subproblem objective - convexity contribution
    reduced_cost = subproblem_obj_value - convexity_contrib
    
    # Determine if this solution has an improving reduced cost
    # For minimization: negative reduced cost is improving
    # For maximization: positive reduced cost is improving
    is_improving = if is_minimization(context)
        reduced_cost < -1e-6
    else
        reduced_cost > 1e-6
    end

    # Get variable primal values
    variable_values = _populate_variable_values(moi_pricing_sp(pricing_sp))
    unified_solution = PrimalMoiSolution(reduced_cost, variable_values)
    primal_sol = PricingPrimalMoiSolution(sp_id, unified_solution, is_improving)

    moi_termination_status = MOI.get(moi_pricing_sp(pricing_sp), MOI.TerminationStatus())

    is_infeasible = moi_termination_status == MOI.INFEASIBLE
    is_unbounded = moi_termination_status == MOI.DUAL_INFEASIBLE || moi_termination_status == MOI.INFEASIBLE_OR_UNBOUNDED

    return PricingSolution(
        is_infeasible,
        is_unbounded,
        reduced_cost,
        subproblem_obj_value,
        [primal_sol]
    )
end