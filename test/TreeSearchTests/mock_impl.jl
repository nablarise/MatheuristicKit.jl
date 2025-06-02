mutable struct MockSearchSpace <: MK.TreeSearch.AbstractSearchSpace
    node_limit::Int # number of evaluated nodes
    nb_nodes_evaluated::Int
    last_evaluated_node_id::Int
    last_generated_node_id::Int
    nb_children_per_node::Int
    max_depth::Int
    test_function::Function
end

function MockSearchSpace(f; node_limit=6, nb_children_per_node=2, max_depth=10)
    return MockSearchSpace(node_limit, 0, 0, 1, nb_children_per_node, max_depth, f)
end

_new_node_id!(space::MockSearchSpace) = space.last_generated_node_id += 1

struct MockNode
    id::Int
    depth::Int
end

struct MockResult{T}
    last_evaluated_node_id::Int
    nb_evaluated_node::Int
    open_nodes::T
end

MK.TreeSearch.new_root(space::MockSearchSpace) = MockNode(1, 1)
MK.TreeSearch.stop(space::MockSearchSpace, untreated_nodes) = space.nb_nodes_evaluated >= space.node_limit

function MK.TreeSearch.children(space::MockSearchSpace, current::MockNode)
    @test space.test_function(space, current)
    space.nb_nodes_evaluated += 1
    space.last_evaluated_node_id = current.id
    depth = current.depth + 1
    if depth <= space.max_depth
        return [MockNode(_new_node_id!(space), depth) for _ in 1:space.nb_children_per_node]
    end
    return MockNode[]
end

function MK.TreeSearch.output(space::MockSearchSpace, untreated_nodes)
    return MockResult(
        space.last_evaluated_node_id,
        space.nb_nodes_evaluated,
        untreated_nodes
    )
end
