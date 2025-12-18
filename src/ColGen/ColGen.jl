
module ColGen

using MathOptInterface, ReformulationKit, JuMP
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const RK = ReformulationKit

include("helpers.jl")
include("coluna.jl")
include("moi_solutions.jl")
include("dw_colgen.jl")
include("callback_impl.jl")  # Default callback implementation
include("master_optimization.jl")
include("reduced_costs.jl")
include("pricing_optimization.jl")
include("dual_bounds.jl")
include("column_insertion.jl")
include("ip_management.jl")
include("dw_colgen_iteration.jl")
include("dw_stabilization.jl")

# Export helper functions
export add_variable!, add_constraint!

# Export callback interface and default implementation
export AbstractColumnGenerationCallbacks, MappingBasedCallbacks
export compute_column_cost, compute_column_coefficients, compute_reduced_costs


#### reformulation API
function get_master end
function get_reform end
function is_minimization end
function get_pricing_subprobs end


#### formulation manipulation API


function run_column_generation(reformulation)
    # Validate optimizer is attached before proceeding
    master_moi = JuMP.backend(RK.master(reformulation))
    if MOIU.state(master_moi) == MOIU.NO_OPTIMIZER
        throw(ErrorException(
            """
            No optimizer attached to the master problem.
            Please attach an optimizer to the master model before running column generation.
            Example: JuMP.set_optimizer(ReformulationKit.master(reformulation), HiGHS.Optimizer)
            """
        ))
    end

    # Validate required callback object is present in subproblems
    for (sp_id, jump_subproblem) in RK.subproblems(reformulation)
        if !haskey(jump_subproblem.ext, :dw_colgen_callbacks)
            throw(ErrorException(
                """
                Missing :dw_colgen_callbacks in subproblem $sp_id extension dictionary.
                This extension should be set by ReformulationKit.dantzig_wolfe_decomposition().

                The callback object must implement the AbstractColumnGenerationCallbacks interface:
                - compute_column_cost(callbacks, subproblem_id, solution) -> Float64
                - compute_column_coefficients(callbacks, subproblem_id, solution, master_info) -> Dict
                - compute_reduced_costs(callbacks, subproblem_id, master_dual_solution) -> Dict

                Example:
                    callbacks = MyCallbacks()  # Must be <: AbstractColumnGenerationCallbacks
                    subproblem.ext[:dw_colgen_callbacks] = callbacks

                Please ensure you're using a compatible version of ReformulationKit.
                """
            ))
        end

        # # Validate that the callback object implements the correct interface
        # callbacks = jump_subproblem.ext[:dw_colgen_callbacks]
        # if !isa(callbacks, AbstractColumnGenerationCallbacks)
        #     throw(ErrorException(
        #         """
        #         Invalid callback object type in subproblem $sp_id.

        #         Expected: <: AbstractColumnGenerationCallbacks
        #         Got:      $(typeof(callbacks))

        #         The callback object must implement the AbstractColumnGenerationCallbacks interface.
        #         See the MatheuristicKit documentation for details.
        #         """
        #     ))
        # end
    end

    context = DantzigWolfeColGenImpl(reformulation)
    ip_primal_sol = nothing
    run!(context, ip_primal_sol)
end

end