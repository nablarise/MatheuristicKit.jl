module TreeSearch

using DataStructures

##################################################
# Interface to implement.
##################################################

# abstract search strategy
abstract type AbstractSearchStrategy end

# (composition of decision tree)
abstract type AbstractSearchSpace end

# problem
abstract type AbstractProblem end

search(strategy, search_space, problem) = nothing
new_root(search_space, problem) = nothing
stop(search_space, problem, untreated_nodes) = nothing
children(search_space, problem, current_node) = nothing
output(search_space, problem) = nothing

##################################################
# Implemented search strategies.
##################################################

include("depth_first_search.jl")
include("best_first_search.jl")
include("breadth_first_search.jl")
include("limited_discrepancy_search.jl")
include("beam_search.jl")

end