# Master

struct MasterPrimalSolution
    obj_value::Float64
    variable_values::Dict{MOI.VariableIndex,Float64}
end

struct MasterDualSolution
    obj_value::Float64
    constraint_duals::Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}
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
get_obj_val(sol::MasterSolution) = sol.primal_sol.obj_value

get_primal_sol(sol::MasterSolution) = sol.primal_sol
get_dual_sol(sol::MasterSolution) = sol.dual_sol

is_better_primal_sol(::MasterPrimalSolution, ::Nothing) = true

function _populate_variable_values(model)
    variable_values = Dict{MOI.VariableIndex,Float64}()
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
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
    dual_status = MOI.get(model, MOI.DualStatus())

    if dual_status == MOI.FEASIBLE_POINT
        # Get all constraint types present in the model
        constraint_types = MOI.get(model, MOI.ListOfConstraintTypesPresent())

        # For each constraint type, get the constraint indices and their dual values
        for (F, S) in constraint_types
            constraint_indices = MOI.get(model, MOI.ListOfConstraintIndices{F,S}())

            if !isempty(constraint_indices)
                # Initialize inner dictionary for this constraint type
                constraint_type = typeof(first(constraint_indices))
                constraint_duals[constraint_type] = Dict{Int64,Float64}()

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

# Implementation is OK
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


# Implementation is OK
function update_master_constrs_dual_vals!(::DantzigWolfeColGenImpl, ::MasterDualSolution)
    # We do not support non-robust cuts.
    return nothing
end

# Reduced costs

struct ReducedCosts
    values::Dict{Any,Dict{MOI.VariableIndex,Float64}}
end


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
                if haskey(mast_dual_sol.constraint_duals, constraint_type)
                    constraint_dict = mast_dual_sol.constraint_duals[constraint_type]
                    if haskey(constraint_dict, constraint_value)
                        dual_value = constraint_dict[constraint_value]
                        dual_contribution += coeff * dual_value
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
    dual_bound::Float64 # TODO: understand what we return here
    primal_sols::Vector{PricingPrimalSolution}
end

is_infeasible(sol::PricingSolution) = sol.is_infeasible
is_unbounded(sol::PricingSolution) = sol.is_unbounded
get_primal_sols(sol::PricingSolution) = sol.primal_sols
get_primal_bound(sol::PricingSolution) = sol.primal_bound
get_dual_bound(sol::PricingSolution) = sol.dual_bound


struct PricingPrimalMoiSolution
    subproblem_id::Any  # Subproblem that generated this solution
    obj_value::Float64  # This is the reduced cost
    variable_values::Dict{MOI.VariableIndex,Float64}
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

function optimize_pricing_problem!(context::DantzigWolfeColGenImpl, sp_id::Any, pricing_sp::PricingSubproblem, ::SubproblemMoiOptimizer, ::MasterDualSolution, stab_changes_mast_dual_sol)
    MOI.optimize!(moi_pricing_sp(pricing_sp))

    # Get objective value
    primal_obj_value = MOI.get(moi_pricing_sp(pricing_sp), MOI.ObjectiveValue())

    # Determine if this solution has an improving reduced cost
    # For minimization: negative reduced cost is improving
    # For maximization: positive reduced cost is improving
    is_improving = if is_minimization(context)
        primal_obj_value < 0
    else
        primal_obj_value > 0
    end

    # Get variable primal values
    variable_values = _populate_variable_values(moi_pricing_sp(pricing_sp))
    primal_sol = PricingPrimalMoiSolution(sp_id, primal_obj_value, variable_values, is_improving)

    moi_termination_status = MOI.get(moi_pricing_sp(pricing_sp), MOI.TerminationStatus())

    is_infeasible = moi_termination_status == MOI.INFEASIBLE
    is_unbounded = moi_termination_status == MOI.DUAL_INFEASIBLE || moi_termination_status == MOI.INFEASIBLE_OR_UNBOUNDED

    return PricingSolution(
        is_infeasible,
        is_unbounded,
        primal_obj_value,
        primal_obj_value, # exact phase so primal bound == dual bound
        [primal_sol]
    )
end

function _convexity_contrib(impl::DantzigWolfeColGenImpl, sep_mast_dual_sol::MasterDualSolution)
    reformulation = get_reform(impl)
    convexity_contribution = 0.0
    
    # Process convexity upper bound constraints (≤)
    for (sp_id, conv_constraint_ref) in reformulation.convexity_constraints_ub
        constraint_index = JuMP.index(conv_constraint_ref)
        constraint_type = typeof(constraint_index)
        constraint_value = constraint_index.value
        master_backend = JuMP.backend(RK.master(reformulation))
        constraint_set = MOI.get(master_backend, MOI.ConstraintSet(), constraint_index)
        rhs = constraint_set.upper
        
        if haskey(sep_mast_dual_sol.constraint_duals, constraint_type)
            constraint_dict = sep_mast_dual_sol.constraint_duals[constraint_type]
            if haskey(constraint_dict, constraint_value)
                dual_value = constraint_dict[constraint_value]
                convexity_contribution += rhs * dual_value
            end
        end
    end
    
    # Process convexity lower bound constraints (≥)
    for (sp_id, conv_constraint_ref) in reformulation.convexity_constraints_lb
        constraint_index = JuMP.index(conv_constraint_ref)
        constraint_type = typeof(constraint_index)
        constraint_value = constraint_index.value
        master_backend = JuMP.backend(RK.master(reformulation))
        constraint_set = MOI.get(master_backend, MOI.ConstraintSet(), constraint_index)
        rhs = constraint_set.lower
        
        if haskey(sep_mast_dual_sol.constraint_duals, constraint_type)
            constraint_dict = sep_mast_dual_sol.constraint_duals[constraint_type]
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
    # Contribution = reduced_cost * multiplicity, where multiplicity depends on reduced cost sign
    reformulation = get_reform(impl)
    master_backend = JuMP.backend(RK.master(reformulation))
    subprob_contribution = 0.0
    
    for (sp_id, reduced_cost) in sps_db
        multiplicity = 0.0
        
        # Determine multiplicity based on reduced cost sign
        if reduced_cost < 0  # Improving reduced cost: use upper multiplicity
            if haskey(reformulation.convexity_constraints_ub, sp_id)
                constraint_index = JuMP.index(reformulation.convexity_constraints_ub[sp_id])
                constraint_set = MOI.get(master_backend, MOI.ConstraintSet(), constraint_index)
                multiplicity = constraint_set.upper
            end
        else  # Non-improving reduced cost: use lower multiplicity
            if haskey(reformulation.convexity_constraints_lb, sp_id)
                constraint_index = JuMP.index(reformulation.convexity_constraints_lb[sp_id])
                constraint_set = MOI.get(master_backend, MOI.ConstraintSet(), constraint_index)
                multiplicity = constraint_set.lower
            end
        end
        
        subprob_contribution += reduced_cost * multiplicity
    end
    
    return subprob_contribution
end

function compute_dual_bound(impl::DantzigWolfeColGenImpl, ::MixedPhase1and2, sps_db::Dict{Int64,Float64}, mast_dual_sol::MasterDualSolution)
    master_lp_obj_val = mast_dual_sol.obj_value - _convexity_contrib(impl, mast_dual_sol)
    
    sp_contrib = _subprob_contrib(impl, sps_db)
    
    # additional master variables are missing.
    
    return master_lp_obj_val + sp_contrib 
end

function _compute_original_column_cost(column::PricingPrimalMoiSolution, original_cost_mapping::RK.OriginalCostMapping)
    # Compute the original cost of the column using costs from the compact formulation
    # This is ∑(c_i * x_i) where c_i are original variable costs and x_i are solution values
    original_cost = 0.0
    for (var_index, var_value) in column.variable_values
        if haskey(original_cost_mapping, var_index)
            original_cost += original_cost_mapping[var_index] * var_value
        end
    end
    return original_cost
end

function _compute_master_constraint_membership(
    column::PricingPrimalMoiSolution, 
    coupling_mapping::RK.CouplingConstraintMapping,
    reformulation::RK.DantzigWolfeReformulation
)
    constraint_coeffs = Dict{MOI.ConstraintIndex, Float64}()
    sp_id = column.subproblem_id
    
    # Compute coupling constraint memberships (A * x for each constraint)
    for (var_index, var_value) in column.variable_values
        coefficients = RK.get_variable_coefficients(coupling_mapping, var_index)
        for (constraint_type, constraint_value, coeff) in coefficients
            constraint_ref = constraint_type(constraint_value)
            constraint_coeffs[constraint_ref] = get(constraint_coeffs, constraint_ref, 0.0) + coeff * var_value
        end
    end
    
    # Add convexity constraint membership (coefficient = 1.0)
    if haskey(reformulation.convexity_constraints_ub, sp_id)
        conv_constraint_ref = JuMP.index(reformulation.convexity_constraints_ub[sp_id])
        constraint_coeffs[conv_constraint_ref] = 1.0
    end
    if haskey(reformulation.convexity_constraints_lb, sp_id)
        conv_constraint_ref = JuMP.index(reformulation.convexity_constraints_lb[sp_id])
        constraint_coeffs[conv_constraint_ref] = 1.0
    end
    
    return constraint_coeffs
end

function insert_columns!(context::DantzigWolfeColGenImpl, ::MixedPhase1and2, columns_to_insert::PricingPrimalMoiSolutionToInsert)
    master = get_master(context)
    master_moi = moi_master(master)
    reformulation = get_reform(context)
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
            reformulation
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



