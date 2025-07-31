struct MasterSolution 
    moi_termination_status::MOI.TerminationStatus
    moi_primal_status::MOI.PrimalStatus
    moi_dual_status::MOI.DualStatus
end
is_infeasible(::MasterSolution) = false
is_unbounded(::MasterSolution) = false
get_obj_val(::MasterSolution) = 0.0

struct MasterPrimalSolution end
get_primal_sol(::MasterSolution) = MasterPrimalSolution()
is_better_primal_sol(::MasterPrimalSolution, ::Nothing) = true

function optimize_master_lp_problem!(master, ::DantzigWolfeColGenImpl)
    MOI.optimize!(moi_master(master))
    return MasterSolution(
        MOI.get(moi_master(master), MOI.TerminationStatus()),
        MOI.get(moi_master(master), MOI.PrimalStatus()),
        MOI.get(moi_master(master), MOI.DualStatus())
    )
end

struct ProjectedIpPrimalSol end

function check_primal_ip_feasibility!(::MasterPrimalSolution, ::DantzigWolfeColGenImpl, ::MixedPhase1and2)
    return ProjectedIpPrimalSol(), false
end

function update_inc_primal_sol!(::DantzigWolfeColGenImpl, ::Nothing, ::ProjectedIpPrimalSol)
    
end

struct MasterDualSolution end

get_dual_sol(::MasterSolution) = MasterDualSolution()




function update_master_constrs_dual_vals!(::DantzigWolfeColGenImpl, ::MasterDualSolution)
    # We do not support non-robust cuts.
end





struct ReducedCosts end

function compute_reduced_costs!(context::DantzigWolfeColGenImpl, phase::MixedPhase1and2, mast_dual_sol::MasterDualSolution)
    return ReducedCosts()
end

function optimize_pricing_problem!(::DantzigWolfeColGenImpl, ::JuMP.Model, ::SubproblemOptimizer, ::MasterDualSolution, stab_changes_mast_dual_sol)
    @assert !stab_changes_mast_dual_sol
    return PricingSolution()
end

function compute_dual_bound(impl::DantzigWolfeColGenImpl, ::MixedPhase1and2, sps_db::Dict{Int64, Nothing}, generated_columns::SetOfColumns, sep_mast_dual_sol::MasterDualSolution)
    return 0.0
end



struct PricingSolution end

is_infeasible(::PricingSolution) = false
is_unbounded(::PricingSolution) = false


struct PricingPrimalSolution end
get_primal_sols(::PricingSolution) = [PricingPrimalSolution(), PricingPrimalSolution()]
push_in_set!(::DantzigWolfeColGenImpl, ::SetOfColumns, ::PricingPrimalSolution) = true

get_primal_bound(::PricingSolution) = nothing
get_dual_bound(::PricingSolution) = nothing





function insert_columns!(::DantzigWolfeColGenImpl, ::MixedPhase1and2, ::SetOfColumns)
    return 0
end


function update_reduced_costs!(::DantzigWolfeColGenImpl, ::MixedPhase1and2, ::ReducedCosts)
    # compute reduced costs.
    # update reducted costs in subproblems.
end