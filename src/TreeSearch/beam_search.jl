struct BeamSearchStrategy{S} <: AbstractSearchStrategy
    inner_strategy::S # search strategy.
    max_width::Int
end

mutable struct BeamSearchSpace{S}
    inner_space::S
    current_depth::Int
    max_width::Int
end

struct BeamSearchNode{N}
    inner_node::N
    depth::Int
    width::Int
end

function new_root(space::BeamSearchSpace)
    inner = new_root(space.inner_space)
    return BeamSearchNode(inner, 1, 1)
end

stop(space::BeamSearchSpace, open_nodes) = stop(space.inner_space, open_nodes)
output(space::BeamSearchSpace, open_nodes) = output(space.inner_space, open_nodes)

function get_priority(strategy::BeamSearchStrategy, space::BeamSearchSpace, node::BeamSearchNode)
    inner_priority = get_priority(strategy.inner_strategy, space.inner_space, node.inner_node)
    return inner_priority
end

function children(space::BeamSearchSpace, current::BeamSearchNode)
    return Iterators.map(
        child -> BeamSearchNode(child, current.depth + 1, current.width),
        children(space.inner_space, current.inner_node)
    )
end

function search(strategy::BeamSearchStrategy, search_space)
    space = BeamSearchSpace(search_space, 1, strategy.max_width)
    return _search(strategy, space)
end

function _search(strategy::BeamSearchStrategy, search_space::BeamSearchSpace)
    root_node = new_root(search_space)
    current_depth_queue = PriorityQueue{typeof(root_node),Float64}()
    enqueue!(current_depth_queue, root_node, get_priority(strategy, search_space, root_node))

    nb_nodes_evaluated_at_current_depth = 0
    next_depth_queue = PriorityQueue{typeof(root_node),Float64}()
    while !stop(search_space, current_depth_queue) && !isempty(current_depth_queue)
        current = dequeue!(current_depth_queue)
        for child in children(search_space, current)
            enqueue!(next_depth_queue, child, get_priority(strategy, search_space, child))
        end
        nb_nodes_evaluated_at_current_depth += 1

        # if we have evaluated the maximum number of allowed nodes at the current depth,
        # we move to the next depth.
        # In certain case, we may not have enough nodes to evaluate at the current depth to reach the
        # maximum width, in this case we move to the next depth when the current queue is empty.
        if nb_nodes_evaluated_at_current_depth >= search_space.max_width || isempty(current_depth_queue)
            current_depth_queue = next_depth_queue
            next_depth_queue = PriorityQueue{typeof(root_node),Float64}()
            nb_nodes_evaluated_at_current_depth = 0
        end
    end
    return output(search_space, current_depth_queue)
end