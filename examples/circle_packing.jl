# Circle Packing Optimization Example - Packing Circles into Circular Containers

using JuMP
using SCIP
using MadNLP
using Ipopt
using LinearAlgebra
using Random
using JSON3
using NablaMatheuristicKit

const NMK = NablaMatheuristicKit

"""
    CirclePackingProblem

A structure to hold the data for a circle packing problem.

# Fields
- `num_containers::Int`: Number of circular containers
- `container_radii::Vector{Float64}`: Radius of each container
- `num_items::Int`: Number of items (circles)
- `item_radii::Vector{Float64}`: Radius of each item
- `item_rewards::Vector{Float64}`: Reward value of each item
"""
struct CirclePackingProblem
    num_containers::Int
    container_radii::Vector{Float64}
    num_items::Int
    item_radii::Vector{Float64}
    item_rewards::Vector{Float64}

    function CirclePackingProblem(
        num_containers::Int,
        container_radii::Vector{Float64},
        num_items::Int,
        item_radii::Vector{Float64},
        item_rewards::Vector{Float64}
    )
        # Validate inputs
        @assert num_containers == length(container_radii) "Number of containers must match container radii array"
        @assert num_items == length(item_radii) == length(item_rewards) "Number of items must match radii and rewards arrays"
        @assert all(container_radii .> 0) "Container radii must be positive"
        @assert all(item_radii .> 0) "Item radii must be positive"

        new(num_containers, container_radii, num_items, item_radii, item_rewards)
    end
end

"""
    create_sample_problem()

Create a sample circle packing problem with 3 circular containers and 15 items.

# Returns
- `CirclePackingProblem`: A sample problem instance
"""
function create_sample_problem()
    # Define 3 circular containers with different radii
    num_containers = 2
    container_radii = [20.0, 25.0]
    #container_radii = [20.0, 15.0]

    # Define 15 items with different radii and rewards
    num_items = 15
    #num_items = 7
    Random.seed!(42)  # For reproducibility

    # Generate random radii between 5 and 15
    item_radii = round.(5.0 .+ 10.0 .* rand(num_items))

    @show container_radii
    @show item_radii

    # Generate rewards somewhat proportional to the area of the circle
    # with some randomness
    item_rewards = [π * r^2 * (0.8 + 0.4 * rand()) for r in item_radii]

    return CirclePackingProblem(
        num_containers,
        container_radii,
        num_items,
        item_radii,
        item_rewards
    )
end

"""
    build_jump_model(problem::CirclePackingProblem)

Build a JuMP model for the circle packing problem with continuous variables.

# Arguments
- `problem::CirclePackingProblem`: The problem instance

# Returns
- `JuMP.Model`: The JuMP model
"""
function build_jump_model(problem::CirclePackingProblem)
    model = Model(Ipopt.Optimizer)

    # Extract problem data for convenience
    num_containers = problem.num_containers
    container_radii = problem.container_radii
    num_items = problem.num_items
    item_radii = problem.item_radii
    item_rewards = problem.item_rewards

    # Decision variables
    items = 1:num_items
    containers = 1:num_containers

    # x[i,c], y[i,c]: Position of item i in container c (if placed)
    # These are relative to the center of the container
    @variable(model, x[items, containers])
    @variable(model, y[items, containers])

    # z[i,c]: 1 if item i is placed in container c, 0 otherwise
    @variable(model, 0 <= α[items, containers] <= 1, Bin)

    # Objective: Maximize total reward of placed items
    @objective(model, Max, sum(item_rewards[i] * α[i, c] for i in items, c in containers))

    # Constraints
    @constraint(model, covering[i in items], sum(α[i, c] for c in containers) <= 1)

    @constraint(model, item_fits_container[i in items, c in containers],
        x[i, c]^2 + y[i, c]^2 <= α[i, c] * (container_radii[c] - item_radii[c])^2
    )

    item_pairs = [(i, j) for i in items for j in items if i < j]

    @constraint(model, non_overlap[(i, j) in item_pairs, c in containers],
        (x[i, c] - x[j, c])^2 + (y[i, c] - y[j, c])^2 >= α[i, c] * α[j, c] * (item_radii[i] + item_radii[j])^2
    )

    @constraint(model, pos_x_lb[i in items, c in containers],
        x[i, c] >= -α[i, c] * (container_radii[c] - item_radii[i])
    )

    @constraint(model, pos_x_ub[i in items, c in containers],
        x[i, c] <= α[i, c] * (container_radii[c] - item_radii[i])
    )

    @constraint(model, pos_y_lb[i in items, c in containers],
        y[i, c] >= -α[i, c] * (container_radii[c] - item_radii[i])
    )

    @constraint(model, pos_y_ub[i in items, c in containers],
        y[i, c] <= α[i, c] * (container_radii[c] - item_radii[i])
    )

    return model
end


################################################

mutable struct CirclePackingSearchSpace <: NMK.TreeSearch.AbstractSearchSpace
    node_limit::Int
    nb_nodes_evaluated::Int
    problem::CirclePackingProblem
    model::JuMP.Model
    helper::NMK.MathOptState.DomainChangeTrackerHelper
    root_state
    tracker
    var_names::Dict{MOI.VariableIndex, String}
    # For DOT file generation
    dot_nodes::Dict{Int, String}  # node_id => node label
    dot_edges::Vector{Tuple{Int, Int, String}}  # (parent_id, child_id, edge label)
    next_node_id::Int
end

function CirclePackingSearchSpace(
    problem::CirclePackingProblem, 
    model::JuMP.Model, 
    helper::NMK.MathOptState.DomainChangeTrackerHelper,
    root_state,
    tracker; 
    node_limit=1000
)
    var_names = Dict{MOI.VariableIndex, String}()
    for var in JuMP.all_variables(model)
        var_names[JuMP.index(var)] = JuMP.name(var)
    end
    
    # Initialize DOT file generation structures
    dot_nodes = Dict{Int, String}()
    dot_edges = Vector{Tuple{Int, Int, String}}()
    
    return CirclePackingSearchSpace(
        node_limit, 0, problem, model, helper, root_state, tracker, 
        var_names, dot_nodes, dot_edges, 1
    )
end

struct CirclePackingNode
    id::Int
    depth::Int
    value_guess::Float64
    state
end

struct CirclePackingResult
    node_id::Int
    assignement::Matrix{Float64} # (container_id, item_id)
    obj_value::Float64
    integral::Bool
end

function CirclePackingResult(node_id::Int, model::JuMP.Model, problem::CirclePackingProblem)
    assignment = zeros(problem.num_items, problem.num_containers)
    α = model[:α]
    for container_id in 1:problem.num_containers, item_id in 1:problem.num_items
        assignment[item_id, container_id] = value(α[item_id, container_id])
    end

    obj_value = MOI.get(model, MOI.ObjectiveValue())
    integral = all(val -> abs(val - round(val)) < 1e-5, assignment)

    return CirclePackingResult(
        node_id,
        assignment,
        obj_value,
        integral
    )
end

##########
##########
#########

# 1 is the best value (meaning the value x = round(x) +/- 1/2)
# 0 is the worst (meaning x = round(x))
function most_fractional_value(space, var_id)
    val = MOI.get(JuMP.backend(space.model), MOI.VariablePrimal(), var_id)
    return 2 * abs(round(val) - val)
end

NMK.TreeSearch.new_root(space::CirclePackingSearchSpace) = CirclePackingNode(1, 0, 0.0, space.root_state)
NMK.TreeSearch.stop(space::CirclePackingSearchSpace, untreated_nodes) = space.nb_nodes_evaluated >= space.node_limit || isempty(untreated_nodes)

function NMK.TreeSearch.children(space::CirclePackingSearchSpace, current::CirclePackingNode)
    println("--- node depth = $(current.depth)")
    NMK.MathOptState.apply_change!(JuMP.backend(space.model), NMK.MathOptState.forward(current.state), space.helper)
    space.nb_nodes_evaluated += 1
    depth = current.depth + 1
    
    # Assign a unique ID to the current node if it doesn't have one already
    current_node_id = current.id
    if current_node_id == 0
        current_node_id = space.next_node_id
        space.next_node_id += 1
    end
    
    # Create node label for DOT file

    MOI.set(space.model, MOI.Silent(), true)
    optimize!(space.model)
    @show JuMP.termination_status(space.model)

    obj_value = round(MOI.get(space.model, MOI.ObjectiveValue()), digits=2)
    node_label = "Node $(current_node_id)\nDepth: $(current.depth)\nObj: $(obj_value)"
    space.dot_nodes[current_node_id] = node_label


    term_status = JuMP.termination_status(space.model)
    if term_status == MOI.INFEASIBLE || term_status == MOI.LOCALLY_INFEASIBLE
        # Update node label to indicate infeasibility
        space.dot_nodes[current_node_id] = "Node $(current_node_id)\nDepth: $(current.depth)\nINFEASIBLE"
        
        NMK.MathOptState.apply_change!(JuMP.backend(space.model), NMK.MathOptState.backward(current.state), space.helper)
        return CirclePackingNode[]
    end

    result = CirclePackingResult(current_node_id, space.model, space.problem)
    branching_scores = [(var_id, most_fractional_value(space, var_id)) for var_id in space.helper.original_binary_vars]
  
    best_candidate_score, best_candidate_pos = findmax(Iterators.map(elem -> last(elem), branching_scores))
    best_candidate_var_id = first(branching_scores[best_candidate_pos])

    @show best_candidate_var_id, best_candidate_score
    children = CirclePackingNode[]

    if best_candidate_score > 1e-3
        # Get variable name for better labeling
        var_name = get(space.var_names, best_candidate_var_id, string(best_candidate_var_id))
        
        left_lb_forward_changes = [NMK.MathOptState.LowerBoundVarChange(best_candidate_var_id, 1)]
        left_ub_forward_changes = NMK.MathOptState.UpperBoundVarChange[]
        left_local_forward_change = NMK.MathOptState.DomainChangeDiff(left_lb_forward_changes, left_ub_forward_changes)
        left_forward_diff = NMK.MathOptState.merge_forward_change_diff(NMK.MathOptState.forward(current.state), left_local_forward_change)

        left_lb_backward_changes = [NMK.MathOptState.LowerBoundVarChange(best_candidate_var_id, 0)]
        left_ub_backward_changes = NMK.MathOptState.UpperBoundVarChange[]
        left_local_backward_change = NMK.MathOptState.DomainChangeDiff(left_lb_backward_changes, left_ub_backward_changes)
        left_backward_diff = NMK.MathOptState.merge_backward_change_diff(NMK.MathOptState.backward(current.state), left_local_backward_change)

        left_state = NMK.MathOptState.new_state(space.tracker, left_forward_diff, left_backward_diff)

        right_lb_forward_changes = NMK.MathOptState.LowerBoundVarChange[]
        right_ub_forward_changes = [NMK.MathOptState.UpperBoundVarChange(best_candidate_var_id, 0)]
        right_local_forward_change = NMK.MathOptState.DomainChangeDiff(right_lb_forward_changes, right_ub_forward_changes)
        right_forward_diff = NMK.MathOptState.merge_forward_change_diff(NMK.MathOptState.forward(current.state), right_local_forward_change)

        right_lb_backward_changes = NMK.MathOptState.LowerBoundVarChange[]
        right_ub_backward_changes = [NMK.MathOptState.UpperBoundVarChange(best_candidate_var_id, 1)]
        right_local_backward_change = NMK.MathOptState.DomainChangeDiff(right_lb_backward_changes, right_ub_backward_changes)
        right_backward_diff = NMK.MathOptState.merge_backward_change_diff(NMK.MathOptState.backward(current.state), right_local_backward_change)

        right_state = NMK.MathOptState.new_state(space.tracker, right_forward_diff, right_backward_diff)

        # Create child nodes with unique IDs
        left_node_id = space.next_node_id
        space.next_node_id += 1
        
        right_node_id = space.next_node_id
        space.next_node_id += 1
        
        # Create children with a branching algorithm.
        children = CirclePackingNode[
            CirclePackingNode(left_node_id, depth, 0, left_state),
            CirclePackingNode(right_node_id, depth, 0, right_state)
        ]
        
        # Add edges to DOT file representation
        push!(space.dot_edges, (current_node_id, left_node_id, "$(var_name) = 1"))
        push!(space.dot_edges, (current_node_id, right_node_id, "$(var_name) = 0"))
        
        # Generate DOT file after each branching
        generate_dot_file(space)
    else
        # Update node label to indicate leaf node (optimal solution)
        space.dot_nodes[current_node_id] = "Node $(current_node_id)\nDepth: $(current.depth)\nObj: $(obj_value)\nOPTIMAL"
        generate_dot_file(space)
    end

    NMK.MathOptState.apply_change!(JuMP.backend(space.model), NMK.MathOptState.backward(current.state), space.helper)

    return children
end

function NMK.TreeSearch.output(space::CirclePackingSearchSpace, untreated_nodes)
    # Generate final DOT file
    generate_dot_file(space)
    return nothing
end

# Function to generate DOT file representation of the decision tree
function generate_dot_file(space::CirclePackingSearchSpace)
    dot_file_path = joinpath(dirname(@__FILE__), "circle_packing_tree.dot")
    open(dot_file_path, "w") do f
        # Write DOT file header
        write(f, "digraph CirclePackingDecisionTree {\n")
        write(f, "    // Graph settings\n")
        write(f, "    graph [rankdir=TB, fontname=\"Arial\", splines=true];\n")
        write(f, "    node [shape=box, style=\"rounded,filled\", fillcolor=lightblue, fontname=\"Arial\"];\n")
        write(f, "    edge [fontname=\"Arial\"];\n\n")
        
        # Write nodes
        for (node_id, label) in space.dot_nodes
            # Set different colors for different node types
            if occursin("INFEASIBLE", label)
                write(f, "    node$(node_id) [label=\"$(label)\", fillcolor=lightcoral];\n")
            elseif occursin("OPTIMAL", label)
                write(f, "    node$(node_id) [label=\"$(label)\", fillcolor=lightgreen];\n")
            else
                write(f, "    node$(node_id) [label=\"$(label)\"];\n")
            end
        end
        
        write(f, "\n")
        
        # Write edges
        for (parent_id, child_id, label) in space.dot_edges
            write(f, "    node$(parent_id) -> node$(child_id) [label=\"$(label)\"];\n")
        end
        
        # Add note about the branching strategy
        write(f, "\n    // Notes about the branching process\n")
        write(f, "    note1 [shape=note, fillcolor=lightyellow, label=\"Branching Strategy:\nMost fractional binary variable\"];\n")
        write(f, "    note2 [shape=note, fillcolor=lightyellow, label=\"Search Strategy:\nBest-first search based on\nnode.value_guess\"];\n")
        
        # Close the graph
        write(f, "}\n")
    end
    
    println("DOT file generated at: ", dot_file_path)
    return dot_file_path
end

struct BestValueSearchStrategy <: NMK.TreeSearch.AbstractBestFirstSearchStrategy end

function NMK.TreeSearch.get_priority(::BestValueSearchStrategy, space::CirclePackingSearchSpace, node::CirclePackingNode)
    return node.value_guess
end

"""
    solve_circle_packing(problem::CirclePackingProblem; time_limit_seconds::Float64=60.0)

Solve the circle packing problem using JuMP and return the solution.

# Arguments
- `problem::CirclePackingProblem`: The problem instance
- `time_limit_seconds::Float64=60.0`: Time limit for the solver in seconds

# Returns
- `Dict`: A dictionary containing the solution information
"""
function solve_circle_packing(problem::CirclePackingProblem; time_limit_seconds::Float64=600.0)
    model = build_jump_model(problem)
    tracker = NMK.MathOptState.DomainChangeTracker()
    helper = NMK.MathOptState.transform_model!(tracker, JuMP.backend(model))

    original_state = NMK.MathOptState.root_state(tracker, JuMP.backend(model))
    relaxed_state = NMK.MathOptState.relax_integrality!(JuMP.backend(model), helper)
    NMK.MathOptState.recover_state!(JuMP.backend(model), original_state, relaxed_state, helper)

    root_state = NMK.MathOptState.root_state(tracker, JuMP.backend(model))

    space = CirclePackingSearchSpace(problem, model, helper, root_state, tracker; node_limit=1000)

    strategy = BestValueSearchStrategy()
    result = NMK.TreeSearch.search(strategy, space)
end

# Main execution
function main()
    println("Creating circle packing problem...")
    problem = create_sample_problem()

    println("Solving circle packing problem...")
    solution = solve_circle_packing(problem, time_limit_seconds=600.0)
    @show solution
end

# Run the example
main()
