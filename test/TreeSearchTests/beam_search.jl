struct MockBeamSearchStrategy <: MK.TreeSearch.AbstractBestFirstSearchStrategy
    scores::Dict{Int,Int}
end
MK.TreeSearch.get_priority(strategy::MockBeamSearchStrategy, ::MockSearchSpace, node::MockNode) = strategy.scores[node.id]

function test_beam_search_strategy()
    ##########################
    # breadth first search (best)
    # * evaluated nodes
    #
    #                1
    #                *
    #           /    |         \
    #          2     3          4
    #                *          *       (3, 4)
    #             /  |  \     / | \
    #            5   6   7   8  9  10
    #                    *          *        (10, 7)
    #                  / | \      / | \
    #                 14 15 16   11 12 13
    #                  *             *       (12, 14)

    previous_node_id = Dict(
        1 => 0,
        3 => 1,
        4 => 3,
        10 => 4,
        7 => 10,
        12 => 7,
        14 => 12,
    )

    function test_function(space, current)
        return previous_node_id[current.id] == space.last_evaluated_node_id
    end

    scores = Dict(
        1 => -1,
        #
        2 => 0,
        3 => -5,
        4 => -4,
        #
        5 => -8,
        6 => -8,
        7 => -10,
        8 => -6,
        9 => -2,
        10 => -12,
        #
        11 => -1,
        12 => -30,
        13 => -2,
        14 => -15,
        15 => -7,
        16 => 0,
    )

    strategy = MK.TreeSearch.BeamSearchStrategy(MockBeamSearchStrategy(scores), 2) # beam width.
    search_space = MockSearchSpace(test_function; node_limit=100, nb_children_per_node=3, max_depth=4)
    output = MK.TreeSearch.search(strategy, search_space)

    return
end
