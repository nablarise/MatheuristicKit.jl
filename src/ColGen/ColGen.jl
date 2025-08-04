
module ColGen

using MathOptInterface, ReformulationKit, JuMP
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const RK = ReformulationKit

include("helpers.jl")
include("coluna.jl")
include("dw_colgen.jl")
include("dw_colgen_iteration.jl")
include("dw_stabilization.jl")

# Export helper functions
export add_variable!, add_constraint!


#### reformulation API
function get_master end
function get_reform end
function is_minimization end
function get_pricing_subprobs end

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
    
    context = DantzigWolfeColGenImpl(reformulation)
    @show get_reform(context).subproblems
    ip_primal_sol = nothing
    run!(context, ip_primal_sol)
end

end