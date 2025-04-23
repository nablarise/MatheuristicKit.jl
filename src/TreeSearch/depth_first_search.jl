struct DepthFirstSearchStrategy <: AbstractSearchStrategy
    node_limit::Int
end

function DepthFirstSearchStrategy(;
    node_limit=1000
)
    return DepthFirstSearchStrategy(node_limit)
end

function search(strategy::DepthFirstSearchStrategy, search_space)
    root_node = new_root(search_space)
    stack = Stack{typeof(root_node)}()
    push!(stack, root_node)
    # it is important to call `stop()` function first, as it may update `space`
    while !stop(search_space, stack) && !isempty(stack)
        current = pop!(stack)
        for child in Iterators.reverse(children(search_space, current))
            push!(stack, child)
        end
    end
    return output(search_space, stack)
end