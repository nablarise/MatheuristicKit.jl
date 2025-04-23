struct BreadthFirstSearchStrategy <: AbstractSearchStrategy end

function search(strategy::BreadthFirstSearchStrategy, search_space)
    root_node = new_root(search_space)
    queue = Deque{typeof(root_node)}()
    push!(queue, root_node)
    # it is important to call `stop()` function first, as it may update `space`
    while !stop(search_space, queue) && !isempty(queue)
        current = popfirst!(queue)
        for child in children(search_space, current)
            push!(queue, child)
        end
    end
    return output(search_space, queue)
end