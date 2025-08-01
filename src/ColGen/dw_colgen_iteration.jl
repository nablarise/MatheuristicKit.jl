struct MasterPrimalSolution 
    obj_value::Float64
    variable_values::Dict{MOI.VariableIndex, Float64}
end

struct MasterDualSolution 
    obj_value::Float64
    constraint_duals::Dict{Type{<:MOI.ConstraintIndex}, Dict{Int64, Float64}}
end

struct MasterSolution
    moi_termination_status::MOI.TerminationStatusCode
    moi_primal_status::MOI.ResultStatusCode
    moi_dual_status::MOI.ResultStatusCode
    primal_sol::MasterPrimalSolution
    dual_sol::MasterDualSolution
end
is_infeasible(sol::MasterSolution) = sol.moi_termination_status == MOI.INFEASIBLE
is_unbounded(sol::MasterSolution) = sol.moi_termination_status == MOI.DUAL_INFEASIBLE || sol.moi_termination_status == MOI.INFEASIBLE_OR_UNBOUNDED
get_obj_val(sol::MasterSolution) = sol.primal_obj_value

get_primal_sol(sol::MasterSolution) = sol.primal_sol
get_dual_sol(sol::MasterSolution) = sol.dual_sol

is_better_primal_sol(::MasterPrimalSolution, ::Nothing) = true

function _populate_variable_values(model)
    variable_values = Dict{MOI.VariableIndex, Float64}()
    primal_status = MOI.get(model, MOI.PrimalStatus())
    
    if primal_status == MOI.FEASIBLE_POINT
        # Get all variables in the model
        variables = MOI.get(model, MOI.ListOfVariableIndices())
        
        # Retrieve primal value for each variable
        for var in variables
            variable_values[var] = MOI.get(model, MOI.VariablePrimal(), var)
        end
    end
    return variable_values
end

function _populate_constraint_duals(model)
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex}, Dict{Int64, Float64}}()
    dual_status = MOI.get(model, MOI.DualStatus())
    
    if dual_status == MOI.FEASIBLE_POINT
        # Get all constraint types present in the model
        constraint_types = MOI.get(model, MOI.ListOfConstraintTypesPresent())
        
        # For each constraint type, get the constraint indices and their dual values
        for (F, S) in constraint_types
            constraint_indices = MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
            
            if !isempty(constraint_indices)
                # Initialize inner dictionary for this constraint type
                constraint_type = typeof(first(constraint_indices))
                constraint_duals[constraint_type] = Dict{Int64, Float64}()
                
                # Get dual value for each constraint of this type
                for constraint_index in constraint_indices
                    dual_value = MOI.get(model, MOI.ConstraintDual(), constraint_index)
                    constraint_duals[constraint_type][constraint_index.value] = dual_value
                end
            end
        end
    end
    return constraint_duals
end

function optimize_master_lp_problem!(master, ::DantzigWolfeColGenImpl)
    MOI.optimize!(moi_master(master))
    
    # Get objective value
    obj_value = MOI.get(moi_master(master), MOI.ObjectiveValue())
    # Get variable primal values
    variable_values = _populate_variable_values(moi_master(master))
    primal_sol = MasterPrimalSolution(obj_value, variable_values)

    # Get dual objective value
    dual_obj_value = MOI.get(moi_master(master), MOI.DualObjectiveValue())
    # Get constraint dual values
    constraint_duals = _populate_constraint_duals(moi_master(master))
    dual_sol = MasterDualSolution(dual_obj_value, constraint_duals)
    return MasterSolution(
        MOI.get(moi_master(master), MOI.TerminationStatus()),
        MOI.get(moi_master(master), MOI.PrimalStatus()),
        MOI.get(moi_master(master), MOI.DualStatus()),
        primal_sol,
        dual_sol
    )
end

struct ProjectedIpPrimalSol end

function check_primal_ip_feasibility!(::MasterPrimalSolution, ::DantzigWolfeColGenImpl, ::MixedPhase1and2)
    return ProjectedIpPrimalSol(), false
end

function update_inc_primal_sol!(::DantzigWolfeColGenImpl, ::Nothing, ::ProjectedIpPrimalSol)
    
end





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