struct BestFirstSearchStrategy <: NMK.TreeSearch.AbstractBestFirstSearchStrategy
    scores::Dict{Int,Int} # node_id -> score
end

NMK.TreeSearch.get_priority(strategy::BestFirstSearchStrategy, ::MockSearchSpace, node::MockNode) = -strategy.scores[node.id]

function test_best_first_search()
    ##########################
    #  [evalued node <score>]
    #
    #           [1<1>]
    #        /          \
    #    [2<10>]        [3<5>]
    #   /      \        /     \
    # 4<2>   [5<7>]  [10<9>]  11<6>
    #        /   \     /   \
    #   [6<10>] 7<1> 12<2> 13<5>
    #   /    \
    # 8<3>  9<4>
    #
    # Order of visits: 1 -> 2 -> 5 -> 6 -> 3 -> 10

    previous_node_id = Dict(
        1 => 0,
        2 => 1,
        3 => 6,
        5 => 2,
        6 => 5,
        10 => 3,
    )

    function test_function(space, current)
        return previous_node_id[current.id] == space.last_evaluated_node_id
    end

    scores = Dict(
        1 => 1,
        2 => 10,
        3 => 5,
        4 => 2,
        5 => 7,
        6 => 10,
        7 => 1,
        8 => 3,
        9 => 4,
        10 => 9,
        11 => 6,
        12 => 2,
        13 => 5,
    )
    strategy = BestFirstSearchStrategy(scores)
    search_space = MockSearchSpace(test_function; node_limit=6)
    output = NMK.TreeSearch.search(strategy, search_space)

    @test output.last_evaluated_node_id == 10
    @test output.nb_evaluated_node == 6

    @test dequeue!(output.open_nodes).id == 11
    @test dequeue!(output.open_nodes).id == 13
    @test dequeue!(output.open_nodes).id == 9
    @test dequeue!(output.open_nodes).id == 8
    @test dequeue!(output.open_nodes).id == 12
    @test dequeue!(output.open_nodes).id == 4
    @test dequeue!(output.open_nodes).id == 7
    @test isempty(output.open_nodes)
    return
end
