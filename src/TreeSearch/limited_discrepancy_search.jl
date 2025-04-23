struct LimitedDiscrepancySearchStrategy{T} <: AbstractSearchStrategy
    inner_search_strategy::T
    max_discrepancy::Int
end

struct LimitedDiscrepancySpace{S}
    inner_space::S
    max_discrepancy::Int
end

struct LimitedDiscrepancyNode{N}
    inner_node::N
    discrepancy::Int
end

function new_root(space::LimitedDiscrepancySpace)
    inner = new_root(space.inner_space)
    d = space.max_discrepancy
    return LimitedDiscrepancyNode(inner, d)
end

stop(space::LimitedDiscrepancySpace, open_nodes) = stop(space.inner_space, open_nodes)
output(space::LimitedDiscrepancySpace, open_nodes) = output(space.inner_space, open_nodes)

function children(space::LimitedDiscrepancySpace, current::LimitedDiscrepancyNode{N}) where {N}
    lds_children = LimitedDiscrepancyNode{N}[]
    inner_children = children(space.inner_space, current.inner_node)
    for (i, child) in Iterators.reverse(enumerate(inner_children))
        if current.discrepancy > 0
            d = current.discrepancy - i + 1
            if d <= 0
                # TODO: Best to break but problem with reverse in DFS or BFS (need to check what's going on here or change DFS)
                # TODO: The reverse may not work with a best first search (need to test)
                continue
            end
            pushfirst!(lds_children, LimitedDiscrepancyNode(child, d))
        end
    end
    return lds_children
end

function search(strategy::LimitedDiscrepancySearchStrategy, search_space)
    space = LimitedDiscrepancySpace(search_space, strategy.max_discrepancy)
    return search(strategy.inner_search_strategy, space)
end