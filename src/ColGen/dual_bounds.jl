# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function _subproblem_convexity_contrib(impl::DantzigWolfeColGenImpl, sp_id::Any, mast_dual_sol::MasterDualSolution)
    master = get_master(impl)
    convexity_contribution = 0.0
    
    # Process convexity upper bound constraint (≤) for this subproblem
    if haskey(master.convexity_constraints_ub, sp_id)
        constraint_index = master.convexity_constraints_ub[sp_id]
        constraint_type = typeof(constraint_index)
        constraint_value = constraint_index.value
        
        if haskey(mast_dual_sol.sol.constraint_duals, constraint_type)
            constraint_dict = mast_dual_sol.sol.constraint_duals[constraint_type]
            if haskey(constraint_dict, constraint_value)
                dual_value = constraint_dict[constraint_value]
                convexity_contribution += dual_value
            end
        end
    end
    
    # Process convexity lower bound constraint (≥) for this subproblem
    if haskey(master.convexity_constraints_lb, sp_id)
        constraint_index = master.convexity_constraints_lb[sp_id]
        constraint_type = typeof(constraint_index)
        constraint_value = constraint_index.value
        
        if haskey(mast_dual_sol.sol.constraint_duals, constraint_type)
            constraint_dict = mast_dual_sol.sol.constraint_duals[constraint_type]
            if haskey(constraint_dict, constraint_value)
                dual_value = constraint_dict[constraint_value]
                convexity_contribution += dual_value
            end
        end
    end
    
    return convexity_contribution
end

function _convexity_contrib(impl::DantzigWolfeColGenImpl, sep_mast_dual_sol::MasterDualSolution)
    master = get_master(impl)
    convexity_contribution = 0.0
    
    # Process convexity upper bound constraints (≤)
    for (sp_id, constraint_index) in master.convexity_constraints_ub
        constraint_type = typeof(constraint_index)
        constraint_value = constraint_index.value
        constraint_set = MOI.get(master.moi_master, MOI.ConstraintSet(), constraint_index)
        rhs = constraint_set.upper
        
        if haskey(sep_mast_dual_sol.sol.constraint_duals, constraint_type)
            constraint_dict = sep_mast_dual_sol.sol.constraint_duals[constraint_type]
            if haskey(constraint_dict, constraint_value)
                dual_value = constraint_dict[constraint_value]
                convexity_contribution += rhs * dual_value
            end
        end
    end
    
    # Process convexity lower bound constraints (≥)
    for (sp_id, constraint_index) in master.convexity_constraints_lb
        constraint_type = typeof(constraint_index)
        constraint_value = constraint_index.value
        constraint_set = MOI.get(master.moi_master, MOI.ConstraintSet(), constraint_index)
        rhs = constraint_set.lower
        
        if haskey(sep_mast_dual_sol.sol.constraint_duals, constraint_type)
            constraint_dict = sep_mast_dual_sol.sol.constraint_duals[constraint_type]
            if haskey(constraint_dict, constraint_value)
                dual_value = constraint_dict[constraint_value]
                convexity_contribution += rhs * dual_value
            end
        end
    end
    
    return convexity_contribution
end

function _subprob_contrib(impl::DantzigWolfeColGenImpl, sps_db::Dict{Int64,Float64})
    # Compute contribution from subproblem variables using multiplicity bounds
    # Contribution = dual_bound * multiplicity, where multiplicity depends on reduced cost sign
    master = get_master(impl)
    subprob_contribution = 0.0
    sense = is_minimization(impl) ? 1 : -1
    
    for (sp_id, dual_bound) in sps_db
        multiplicity = 0.0
        
        # Determine multiplicity based on dual_bound sign
        if sense * dual_bound < 0  
            if haskey(master.convexity_constraints_ub, sp_id)
                constraint_index = master.convexity_constraints_ub[sp_id]
                constraint_set = MOI.get(master.moi_master, MOI.ConstraintSet(), constraint_index)
                multiplicity = constraint_set.upper
            end
        else
            if haskey(master.convexity_constraints_lb, sp_id)
                constraint_index = master.convexity_constraints_lb[sp_id]
                constraint_set = MOI.get(master.moi_master, MOI.ConstraintSet(), constraint_index)
                multiplicity = constraint_set.lower
            end
        end
        
        subprob_contribution += dual_bound * multiplicity
    end
    
    return subprob_contribution
end

function compute_dual_bound(impl::DantzigWolfeColGenImpl, ::MixedPhase1and2, sps_db::Dict{Int64,Float64}, mast_dual_sol::MasterDualSolution)
    master = get_master(impl)
    recomputed_cost = recompute_cost(mast_dual_sol.sol, master.moi_master)
    @assert abs(recomputed_cost - mast_dual_sol.sol.obj_value) < 1e-6 "Dual solution cost mismatch: recomputed=$recomputed_cost, stored=$(mast_dual_sol.sol.obj_value)"

    sp_contrib = _subprob_contrib(impl, sps_db)
    
    return mast_dual_sol.sol.obj_value - _convexity_contrib(impl, mast_dual_sol) + _subprob_contrib(impl, sps_db)
end