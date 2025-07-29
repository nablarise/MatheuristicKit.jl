
module ColGen

using MathOptInterface
const MOI = MathOptInterface

include("coluna.jl")
include("impl.jl")


function run_column_generation(master, subproblems)
    context = ColGenDefaultImplementation()
    ip_primal_sol = nothing
    run!(context, ip_primal_sol)
end


end