# Test with a best first search as inner strategy.

function test_limited_discrepancy_search1()

    ##########################
    #
    #               [1]
    #      /         |        \
    #     2          3         4
    #     *          *         *
    #  /  |  \    /  |  \    /  |  \
    # 5   6   7  17  18  19 29  30  31
    # *   *   *  *   *       *
    # A   B   C  D   E       F
    #
    # A: 8*, 9*, 10*
    # B: 11*, 12*, 13
    # C: 14*, 15, 16
    # D: 20*, 21*, 22
    # E: 23*, 24, 25
    # F: 32*, 33, 34
    #
    # # Test function should check:
    #

    previous_node_id = Dict(
        1 => 0,
        2 => 1,
        5 => 2,
        8 => 5,
        9 => 8,
        10 => 9,
        6 => 10,
        11 => 6,
        12 => 11,
        7 => 12,
        14 => 7,
        3 => 14,
        17 => 3,
        20 => 17,
        21 => 20,
        18 => 21,
        23 => 18,
        4 => 23,
        26 => 4,
        29 => 26
    )

    function test_function(space, current)
        return previous_node_id[current.id] == space.last_evaluated_node_id
    end

    strategy = NMK.TreeSearch.LimitedDiscrepancySearchStrategy(
        NMK.TreeSearch.DepthFirstSearchStrategy(),
        3 # max discrepancy.
    )
    search_space = MockSearchSpace(test_function; node_limit=100, nb_children_per_node=3, max_depth=4)
    output = NMK.TreeSearch.search(strategy, search_space)

    @test output.last_evaluated_node_id == 29
    @test output.nb_evaluated_node == 20

    @test isempty(output.open_nodes)
    return
end

function test_limited_discrepancy_search2()
    # TODO
end
