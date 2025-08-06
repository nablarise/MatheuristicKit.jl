struct PricingSubproblem{MoiModel}
    moi_model::MoiModel
    coupling_constr_mapping::RK.CouplingConstraintMapping
    original_cost_mapping::RK.OriginalCostMapping
end

moi_pricing_sp(pricing_sp::PricingSubproblem) = pricing_sp.moi_model

# Provider types for production use
struct ReformulationMasterProvider
    reformulation::RK.DantzigWolfeReformulation
    eq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},Tuple{MOI.VariableIndex,MOI.VariableIndex}}
    leq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}},MOI.VariableIndex}
    geq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}},MOI.VariableIndex}
end

struct ReformulationPricingSubprobsProvider
    reformulation::RK.DantzigWolfeReformulation
end


struct DantzigWolfeColGenImpl{M,P}
    master_provider::M           # Master + convexity + optimization sense + artificial vars
    pricing_subprobs_provider::P # Contains all mapping objects (coupling_constr_mapping, original_cost_mapping)

    function DantzigWolfeColGenImpl(reformulation::RK.DantzigWolfeReformulation)
        # Assert optimizer is attached (should be validated upstream)
        master_backend = JuMP.backend(RK.master(reformulation))
        @assert master_backend.optimizer !== nothing "Master must have optimizer attached"

        # Create artificial variable tracking dictionaries
        eq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},Tuple{MOI.VariableIndex,MOI.VariableIndex}}()
        leq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}},MOI.VariableIndex}()
        geq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}},MOI.VariableIndex}()

        # Create master provider that contains all master-related data
        master_provider = ReformulationMasterProvider(reformulation, eq_art_vars, leq_art_vars, geq_art_vars)

        # Create pricing subproblems provider
        pricing_subprobs_provider = ReformulationPricingSubprobsProvider(reformulation)

        return new{typeof(master_provider),typeof(pricing_subprobs_provider)}(master_provider, pricing_subprobs_provider)
    end

    # Constructor for testing with custom providers
    function DantzigWolfeColGenImpl(master_provider::M, pricing_subprobs_provider::P) where {M,P}
        return new{M,P}(master_provider, pricing_subprobs_provider)
    end
end

struct Master{MoiModel,Cu,Cl}
    moi_master::MoiModel
    convexity_constraints_ub::Cu
    convexity_constraints_lb::Cl
    eq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},Tuple{MOI.VariableIndex,MOI.VariableIndex}}
    leq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}},MOI.VariableIndex}
    geq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}},MOI.VariableIndex}
end

moi_master(master::Master) = master.moi_master

## Reformulation API
get_master(impl::DantzigWolfeColGenImpl) = get_master(impl.master_provider)
get_reform(impl::DantzigWolfeColGenImpl) = get_reform(impl.master_provider)
is_minimization(impl::DantzigWolfeColGenImpl) = is_minimization(impl.master_provider)
get_pricing_subprobs(impl::DantzigWolfeColGenImpl) = get_pricing_subprobs(impl.pricing_subprobs_provider)

# Provider interface methods for ReformulationMasterProvider
get_master(provider::ReformulationMasterProvider) = Master(
    JuMP.backend(RK.master(provider.reformulation)),
    Dict{Int64,MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}}(
        sp_id => JuMP.index(jump_ref) for (sp_id, jump_ref) in provider.reformulation.convexity_constraints_ub
    ),
    Dict{Int64,MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}}(
        sp_id => JuMP.index(jump_ref) for (sp_id, jump_ref) in provider.reformulation.convexity_constraints_lb
    ),
    provider.eq_art_vars,
    provider.leq_art_vars,
    provider.geq_art_vars
)

get_reform(provider::ReformulationMasterProvider) = provider.reformulation
is_minimization(provider::ReformulationMasterProvider) = MOI.get(JuMP.backend(RK.master(provider.reformulation)), MOI.ObjectiveSense()) != MOI.MAX_SENSE

# Provider interface methods for ReformulationPricingSubprobsProvider
function get_pricing_subprobs(provider::ReformulationPricingSubprobsProvider)
    subproblems_dict = Dict{Any,PricingSubproblem}()

    for (sp_id, jump_subproblem) in RK.subproblems(provider.reformulation)
        # Extract MOI backend (preserving its concrete type)
        moi_model = JuMP.backend(jump_subproblem)

        # Extract RK mappings from JuMP model extensions
        coupling_constr_mapping = jump_subproblem.ext[:dw_coupling_constr_mapping]
        original_cost_mapping = jump_subproblem.ext[:dw_sp_var_original_cost]

        # Create PricingSubproblem with type-stable MOI model template
        pricing_subproblem = PricingSubproblem(
            moi_model,
            coupling_constr_mapping,
            original_cost_mapping
        )

        subproblems_dict[sp_id] = pricing_subproblem
    end

    return subproblems_dict
end


struct ColGenPhaseIterator end

struct MixedPhase1and2
    artificial_var_cost::Float64
    convexity_artificial_var_cost::Float64

    function MixedPhase1and2(artificial_var_cost::Float64=10000.0, convexity_artificial_var_cost::Float64=10000.0)
        return new(artificial_var_cost, convexity_artificial_var_cost)
    end
end

struct ColGenStageIterator end
struct ExactStage end

struct NoStabilization end


new_phase_iterator(::DantzigWolfeColGenImpl) = ColGenPhaseIterator()
initial_phase(::ColGenPhaseIterator) = MixedPhase1and2()
new_stage_iterator(::DantzigWolfeColGenImpl) = ColGenStageIterator()
initial_stage(::ColGenStageIterator) = ExactStage()


stop_colgen(::DantzigWolfeColGenImpl, ::Nothing) = false



function setup_reformulation!(context::DantzigWolfeColGenImpl, phase::MixedPhase1and2)
    setup_reformulation!(context.master_provider, phase)
end

function setup_reformulation!(provider::ReformulationMasterProvider, phase::MixedPhase1and2)
    reform = provider.reformulation
    master_jump = RK.master(reform)
    master = JuMP.backend(master_jump)  # Get the MOI backend from JuMP model

    # Determine cost sign based on optimization sense (large positive cost penalizes artificial variables)
    sense = MOI.get(master, MOI.ObjectiveSense())
    cost = sense == MOI.MIN_SENSE ? phase.artificial_var_cost : -phase.artificial_var_cost

    # Cost for convexity constraints (configurable)
    convexity_cost = sense == MOI.MIN_SENSE ? phase.convexity_artificial_var_cost : -phase.convexity_artificial_var_cost

    # Get convexity constraint references from the reformulation  
    # Convert JuMP constraint references to MOI constraint indices
    convexity_leq_refs = Set(JuMP.index(ref) for ref in values(reform.convexity_constraints_ub))
    convexity_geq_refs = Set(JuMP.index(ref) for ref in values(reform.convexity_constraints_lb))

    # Get all equality constraints in the master problem
    eq_constraints = MOI.get(master, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}())

    # Add artificial variables for each equality constraint: ax = b becomes ax + s⁺ - s⁻ = b
    for constraint_idx in eq_constraints
        # Add positive artificial variable (s⁺)
        s_pos = add_variable!(master;
            lower_bound=0.0,
            constraint_coeffs=Dict(constraint_idx => 1.0),
            objective_coeff=cost,
            name="s⁺[$(constraint_idx.value)]"
        )

        # Add negative artificial variable (s⁻)  
        s_neg = add_variable!(master;
            lower_bound=0.0,
            constraint_coeffs=Dict(constraint_idx => -1.0),
            objective_coeff=cost,
            name="s⁻[$(constraint_idx.value)]"
        )

        # Store in tracking dictionary
        provider.eq_art_vars[constraint_idx] = (s_pos, s_neg)
    end

    # Get all less-than-or-equal constraints in the master problem
    leq_constraints = MOI.get(master, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}())

    # Add artificial variables for each ≤ constraint: ax ≤ b becomes ax - s <= b where s ≥ 0
    for constraint_idx in leq_constraints
        is_convexity = constraint_idx in convexity_leq_refs
        constraint_cost = is_convexity ? convexity_cost : cost

        s_neg = add_variable!(master;
            lower_bound=0.0,
            constraint_coeffs=Dict(constraint_idx => -1.0),
            objective_coeff=constraint_cost,
            name="s[$(constraint_idx.value)]"
        )

        provider.leq_art_vars[constraint_idx] = s_neg
    end

    # Get all greater-than-or-equal constraints in the master problem
    geq_constraints = MOI.get(master, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}())

    # Add artificial variables for each ≥ constraint: ax ≥ b becomes ax + s >= b where s ≥ 0  
    for constraint_idx in geq_constraints
        is_convexity = constraint_idx in convexity_geq_refs
        constraint_cost = is_convexity ? convexity_cost : cost

        s_pos = add_variable!(master;
            lower_bound=0.0,
            constraint_coeffs=Dict(constraint_idx => 1.0),
            objective_coeff=constraint_cost,
            name="s[$(constraint_idx.value)]"
        )

        provider.geq_art_vars[constraint_idx] = s_pos
    end
end


##### column generation phase

function stop_colgen_phase(context::DantzigWolfeColGenImpl, ::MixedPhase1and2, colgen_iter_output, incumbent_dual_bound, ip_primal_sol, iteration)
    return iteration > 10
end

struct ColGenIterationOutput
    master_lp_obj::Union{Float64,Nothing}
    dual_bound::Union{Float64,Nothing}
    nb_columns_added::Int64
    master_lp_primal_sol::Any
    master_ip_primal_sol::Any
end

colgen_iteration_output_type(::DantzigWolfeColGenImpl) = ColGenIterationOutput



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
    return ColGenIterationOutput(mlp, db, nb_new_cols, master_lp_dual_sol, master_ip_primal_sol)
end

get_dual_bound(output::ColGenIterationOutput) = output.dual_bound

function after_colgen_iteration(
    impl::DantzigWolfeColGenImpl,
    phase::MixedPhase1and2,
    stage::ExactStage,
    colgen_iterations::Int64,
    stab::NoStabilization,
    ip_primal_sol::Nothing,
    colgen_iter_output::ColGenIterationOutput
)
    # Log iteration information
    print("Iter $colgen_iterations | ")
    print("Cols: $(colgen_iter_output.nb_columns_added) | ")

    # Dual bound
    if !isnothing(colgen_iter_output.dual_bound)
        print("DB: $(round(colgen_iter_output.dual_bound, digits=2)) | ")
    else
        print("DB: N/A | ")
    end

    # LP master objective
    if !isnothing(colgen_iter_output.master_lp_obj)
        print("LP: $(round(colgen_iter_output.master_lp_obj, digits=2)) | ")
    else
        print("LP: N/A | ")
    end

    # IP primal bound (always Nothing in this signature, but show structure for completeness)
    print("IP: N/A")

    println()
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