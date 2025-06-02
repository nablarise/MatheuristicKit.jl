module Branching

using MathOptInterface

const MOI = MathOptInterface
const ColId = Int
const RowId = Int

# Branching rules

# Provide an order for variables, the first one is selected for branching.
abstract type AbstractOrder end


struct NaturalOrder <: AbstractOrder end
struct LeastFractionalOrder <: AbstractOrder end
struct MostFractionalOrder <: AbstractOrder end


struct Candidate{ListOfNodes}
    children::ListOfNodes # if necessary (e.g. strong branching)
end

function get_candidates(generic_model, order::AbstractOrder)
    return nothing
end




# natural order : use variable index
# variable locks
# objective
# reduced costs
# pseudo costs
# matrix related ?
# random



end