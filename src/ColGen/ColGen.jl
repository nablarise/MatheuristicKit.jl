
module ColGen

using MathOptInterface, ReformulationKit, JuMP
const MOI = MathOptInterface
const RK = ReformulationKit

include("coluna.jl")
include("impl.jl")


#### reformulation API
function get_master end
function get_reform end
function is_minimization end
function get_pricing_subprobs end


function run_column_generation(reformulation)
    context = ColGenDefaultImplementation(reformulation)
    ip_primal_sol = nothing
    run!(context, ip_primal_sol)
end


end