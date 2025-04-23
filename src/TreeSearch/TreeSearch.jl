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

"""
    search(strategy, search_space, problem)

Execute a tree search algorithm using the specified strategy on the given search space and problem.
This is the main entry point for all tree search algorithms.

# Arguments
- `strategy`: The search strategy to use (e.g., DepthFirstSearchStrategy, BestFirstSearchStrategy)
- `search_space`: The search space representing the decision tree structure
- `problem`: The problem instance to be solved

# Returns
The solution found by the search algorithm, as determined by the `output` function
"""
search(strategy, search_space, problem) = nothing

"""
    new_root(search_space, problem)

Create and return the root node of the search tree for the given problem.

# Arguments
- `search_space`: The search space representing the decision tree structure
- `problem`: The problem instance to be solved

# Returns
The root node of the search tree
"""
new_root(search_space, problem) = nothing

"""
    stop(search_space, problem, untreated_nodes)

Determine whether the search algorithm should terminate.

# Arguments
- `search_space`: The search space representing the decision tree structure
- `problem`: The problem instance to be solved
- `untreated_nodes`: The collection of nodes that have not yet been processed

# Returns
`true` if the search should stop, `false` otherwise
"""
stop(search_space, problem, untreated_nodes) = nothing

"""
    children(search_space, problem, current_node)

Generate and return the child nodes of the current node in the search tree.

# Arguments
- `search_space`: The search space representing the decision tree structure
- `problem`: The problem instance to be solved
- `current_node`: The node whose children are to be generated

# Returns
An iterable collection of child nodes
"""
children(search_space, problem, current_node) = nothing

"""
    output(search_space, problem)

Extract and return the final solution from the search space after the search is complete.

# Arguments
- `search_space`: The search space representing the decision tree structure
- `problem`: The problem instance to be solved

# Returns
The solution found by the search algorithm
"""
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