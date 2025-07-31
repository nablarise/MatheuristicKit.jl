struct DantzigWolfeColGenImpl
    reformulation::RK.DantzigWolfeReformulation
    eq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}, Tuple{MOI.VariableIndex, MOI.VariableIndex}}
    leq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}, MOI.VariableIndex}
    geq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}, MOI.VariableIndex}
    
    function DantzigWolfeColGenImpl(reformulation::RK.DantzigWolfeReformulation)
        eq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}, Tuple{MOI.VariableIndex, MOI.VariableIndex}}()
        leq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}, MOI.VariableIndex}()
        geq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}, MOI.VariableIndex}()
        return new(reformulation, eq_art_vars, leq_art_vars, geq_art_vars)
    end
end

## Reformulation API
get_master(impl::DantzigWolfeColGenImpl) = RK.master(impl.reformulation)
get_reform(impl::DantzigWolfeColGenImpl) = impl.reformulation
is_minimization(impl::DantzigWolfeColGenImpl) = JuMP.objective_sense(get_master(impl)) != JuMP.MAX_SENSE
get_pricing_subprobs(impl::DantzigWolfeColGenImpl) = RK.subproblems(impl.reformulation)


struct ColGenPhaseIterator end

struct MixedPhase1and2 
    artificial_var_cost::Float64
    
    MixedPhase1and2(artificial_var_cost::Float64 = 1e6) = new(artificial_var_cost)
end

struct ColGenStageIterator end
struct ExactStage end

struct NoStabilization end


new_phase_iterator(::DantzigWolfeColGenImpl) = ColGenPhaseIterator()
initial_phase(::ColGenPhaseIterator) = MixedPhase1and2()
new_stage_iterator(::DantzigWolfeColGenImpl) = ColGenStageIterator()
initial_stage(::ColGenStageIterator) = ExactStage()


stop_colgen(::DantzigWolfeColGenImpl, ::Nothing) = false



## Stabilization
setup_stabilization!(::DantzigWolfeColGenImpl, ::JuMP.Model) = NoStabilization()

function setup_reformulation!(context::DantzigWolfeColGenImpl, phase::MixedPhase1and2)
    reform = context.reformulation
    master_jump = RK.master(reform)
    master = JuMP.backend(master_jump)  # Get the MOI backend from JuMP model
    
    # Determine cost sign based on optimization sense (large positive cost penalizes artificial variables)
    sense = MOI.get(master, MOI.ObjectiveSense())
    cost = sense == MOI.MIN_SENSE ? phase.artificial_var_cost : -phase.artificial_var_cost
    
    # Higher cost for convexity constraints (10x the regular cost)
    convexity_cost = 10.0 * cost
    
    # Get convexity constraint references from the reformulation  
    # Convert JuMP constraint references to MOI constraint indices
    convexity_leq_refs = Set(JuMP.index(ref) for ref in values(reform.convexity_constraints_ub))
    convexity_geq_refs = Set(JuMP.index(ref) for ref in values(reform.convexity_constraints_lb))
    
    # Get all equality constraints in the master problem
    eq_constraints = MOI.get(master, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}())
    
    # Add artificial variables for each equality constraint: ax = b becomes ax + s⁺ - s⁻ = b
    for constraint_ref in eq_constraints
        # Add positive artificial variable (s⁺)
        s_pos = add_variable!(master; 
            lower_bound=0.0, 
            constraint_coeffs=Dict(constraint_ref => 1.0),
            objective_coeff=cost
        )
        
        # Add negative artificial variable (s⁻)  
        s_neg = add_variable!(master;
            lower_bound=0.0,
            constraint_coeffs=Dict(constraint_ref => -1.0),
            objective_coeff=cost
        )
        
        # Store in tracking dictionary
        context.eq_art_vars[constraint_ref] = (s_pos, s_neg)
    end
    
    # Get all less-than-or-equal constraints in the master problem
    leq_constraints = MOI.get(master, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}())
    
    # Add artificial variables for each ≤ constraint: ax ≤ b becomes ax + s = b where s ≥ 0
    for constraint_ref in leq_constraints
        # Determine if this is a convexity constraint
        is_convexity = constraint_ref in convexity_leq_refs
        constraint_cost = is_convexity ? convexity_cost : cost
        
        # For ax ≤ b, we only need one artificial variable with positive coefficient
        # This allows the constraint to be violated upwards (ax can exceed b)
        s_pos = add_variable!(master;
            lower_bound=0.0,
            constraint_coeffs=Dict(constraint_ref => 1.0),
            objective_coeff=constraint_cost
        )
        
        # Store in tracking dictionary
        context.leq_art_vars[constraint_ref] = s_pos
    end
    
    # Get all greater-than-or-equal constraints in the master problem
    geq_constraints = MOI.get(master, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}())
    
    # Add artificial variables for each ≥ constraint: ax ≥ b becomes ax - s = b where s ≥ 0  
    for constraint_ref in geq_constraints
        # Determine if this is a convexity constraint
        is_convexity = constraint_ref in convexity_geq_refs
        constraint_cost = is_convexity ? convexity_cost : cost
        
        # For ax ≥ b, we need one artificial variable with negative coefficient
        # This allows the constraint to be violated downwards (ax can be less than b)
        s_neg = add_variable!(master;
            lower_bound=0.0,
            constraint_coeffs=Dict(constraint_ref => -1.0),
            objective_coeff=constraint_cost
        )
        
        # Store in tracking dictionary
        context.geq_art_vars[constraint_ref] = s_neg
    end
end

function setup_context!(context::DantzigWolfeColGenImpl, ::MixedPhase1and2)
    # I don't know what I should do.
end


##### column generation phase

function stop_colgen_phase(context::DantzigWolfeColGenImpl, ::MixedPhase1and2, colgen_iter_output, incumbent_dual_bound, ip_primal_sol, iteration)
    return iteration > 10
end

function before_colgen_iteration(::DantzigWolfeColGenImpl, ::MixedPhase1and2)
    return nothing
end

struct ColGenIterationOutput end

colgen_iteration_output_type(::DantzigWolfeColGenImpl) = ColGenIterationOutput

struct MasterSolution end
is_infeasible(::MasterSolution) = false
is_unbounded(::MasterSolution) = false

struct MasterPrimalSolution end
get_primal_sol(::MasterSolution) = MasterPrimalSolution()
is_better_primal_sol(::MasterPrimalSolution, ::Nothing) = true


function optimize_master_lp_problem!(master::JuMP.Model, ::DantzigWolfeColGenImpl)
    #JuMP.optimize!(master)
    return MasterSolution()
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

function update_stabilization_after_master_optim!(::NoStabilization, ::MixedPhase1and2, ::MasterDualSolution)
    # nothing to do.
    return false
end

struct SetOfColumns end
set_of_columns(::DantzigWolfeColGenImpl) = SetOfColumns()

function get_stab_dual_sol(::NoStabilization, ::MixedPhase1and2, dual_sol::MasterDualSolution)
    return dual_sol
end

struct ReducedCosts end

function compute_reduced_costs!(context::DantzigWolfeColGenImpl, phase::MixedPhase1and2, mast_dual_sol::MasterDualSolution)
    return ReducedCosts()
end

function update_reduced_costs!(::DantzigWolfeColGenImpl, ::MixedPhase1and2, ::ReducedCosts)
    # compute reduced costs.
    # update reducted costs in subproblems.
end


function compute_sp_init_db(::DantzigWolfeColGenImpl, ::JuMP.Model)

end

function compute_sp_init_pb(::DantzigWolfeColGenImpl, ::JuMP.Model)

end

struct PriceAllSubproblemsStrategy 
    collection
end
get_pricing_strategy(impl::DantzigWolfeColGenImpl, ::MixedPhase1and2) = PriceAllSubproblemsStrategy(get_pricing_subprobs(impl))
pricing_strategy_iterate(impl::PriceAllSubproblemsStrategy) = iterate(impl.collection)
pricing_strategy_iterate(impl::PriceAllSubproblemsStrategy, state) = iterate(impl.collection, state)

struct SubproblemOptimizer end
get_pricing_subprob_optimizer(stage::ExactStage, sp_to_solve::JuMP.Model) = SubproblemOptimizer()

struct PricingSolution end

is_infeasible(::PricingSolution) = false
is_unbounded(::PricingSolution) = false

function optimize_pricing_problem!(::DantzigWolfeColGenImpl, ::JuMP.Model, ::SubproblemOptimizer, ::MasterDualSolution, stab_changes_mast_dual_sol)
    @assert !stab_changes_mast_dual_sol
    return PricingSolution()
end

struct PricingPrimalSolution end
get_primal_sols(::PricingSolution) = [PricingPrimalSolution(), PricingPrimalSolution()]
push_in_set!(::DantzigWolfeColGenImpl, ::SetOfColumns, ::PricingPrimalSolution) = true

get_primal_bound(::PricingSolution) = nothing
get_dual_bound(::PricingSolution) = nothing

function compute_dual_bound(impl::DantzigWolfeColGenImpl, ::MixedPhase1and2, sps_db::Dict{Int64, Nothing}, generated_columns::SetOfColumns, sep_mast_dual_sol::MasterDualSolution)
    return 0.0
end

function update_stabilization_after_pricing_optim!(::NoStabilization, ::DantzigWolfeColGenImpl, ::SetOfColumns, ::JuMP.Model, ::Float64, ::MasterDualSolution)
    return nothing
end

check_misprice(::NoStabilization, ::SetOfColumns, ::MasterDualSolution) = false

function insert_columns!(::DantzigWolfeColGenImpl, ::MixedPhase1and2, ::SetOfColumns)
    return 0
end

update_stabilization_after_iter!(::NoStabilization, ::MasterDualSolution) = nothing

get_obj_val(::MasterSolution) = 0.0


function new_iteration_output(::Type{<:ColGenIterationOutput}, 
    min_sense,
    mlp,
    db,
    nb_new_cols,
    new_cut_in_master,
    infeasible_master,
    unbounded_master,
    infeasible_subproblem,
    unbounded_subproblem,
    time_limit_reached,
    master_lp_primal_sol,
    master_ip_primal_sol,
    master_lp_dual_sol,
)
    return ColGenIterationOutput()
end

get_dual_bound(::ColGenIterationOutput) = 0.0

function after_colgen_iteration(
    impl::DantzigWolfeColGenImpl, 
    phase::MixedPhase1and2, 
    stage::ExactStage, 
    colgen_iterations::Int64, 
    stab::NoStabilization, 
    ip_primal_sol::Nothing, 
    colgen_iter_output::ColGenIterationOutput
) 
    # Do nothing
end

is_better_dual_bound(
    ::DantzigWolfeColGenImpl, 
    dual_bound::Float64,
    incumbent_dual_bound::Float64
) = false


struct ColGenPhaseOutput end
colgen_phase_output_type(::DantzigWolfeColGenImpl) = ColGenPhaseOutput

function new_phase_output(
    ::Type{<:ColGenPhaseOutput}, 
    min_sense, 
    phase, 
    stage, 
    colgen_iter_output::ColGenIterationOutput, 
    iteration, 
    inc_dual_bound
)
    return ColGenPhaseOutput()
end

function next_phase(::ColGenPhaseIterator, ::MixedPhase1and2, ::ColGenPhaseOutput)
    return nothing
end

function next_stage(::ColGenStageIterator, ::ExactStage, ::ColGenPhaseOutput)
    return nothing
end

struct ColGenOutput end
colgen_output_type(::DantzigWolfeColGenImpl) = ColGenOutput

function new_output(::Type{ColGenOutput}, ::ColGenPhaseOutput)
    println("colgen end")
    return ColGenOutput()
end