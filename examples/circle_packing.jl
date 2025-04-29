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
    num_containers = 3
    container_radii = [40.0, 30.0, 45.0]
    #container_radii = [20.0, 15.0]

    # Define 15 items with different radii and rewards
    num_items = 12
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
    @variable(model, 0 <= α[items, containers] <= 1)

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
end

function CirclePackingSearchSpace(problem::CirclePackingProblem, model::JuMP.Model; node_limit=1000)
    return CirclePackingSearchSpace(node_limit, 0, problem, model)
end

struct CirclePackingNode
    id::Int
    depth::Int
    value_guess::Float64
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

NMK.TreeSearch.new_root(space::CirclePackingSearchSpace) = CirclePackingNode(0, 0, 0.0)
NMK.TreeSearch.stop(space::CirclePackingSearchSpace, untreated_nodes) = space.nb_nodes_evaluated >= space.node_limit || isempty(untreated_nodes)

function NMK.TreeSearch.children(space::CirclePackingSearchSpace, current::CirclePackingNode)
    space.nb_nodes_evaluated += 1
    depth = current.depth + 1

    optimize!(space.model)
    result = CirclePackingResult(current.id, space.model, space.problem)

    @show result

    # Create children with a branching algorithm.
    return CirclePackingNode[]
end

function NMK.TreeSearch.output(space::CirclePackingSearchSpace, untreated_nodes)
    return nothing
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

    space = CirclePackingSearchSpace(
        problem, model; node_limit=100
    )

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
