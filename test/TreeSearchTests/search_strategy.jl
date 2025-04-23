function test_depth_first_search()
    ##########################
    #              [1]
    #              / \
    #            [2]   3
    #            / \
    #          [4]   5
    #          / \
    #        [6]   7
    #        / \
    #      [8]   9
    #     /  \
    #   [10]  11
    #   /  \
    #  12  13


    function test_function(space, current)
        expected_diff = current.id <= 2 ? 1 : 2
        return current.id - space.last_evaluated_node_id == expected_diff
    end

    strategy = NMK.TreeSearch.DepthFirstSearchStrategy()
    search_space = MockSearchSpace(test_function)
    output = NMK.TreeSearch.search(strategy, search_space)
    @test output.last_evaluated_node_id == 10
    @test output.nb_evaluated_node == 6

    @test pop!(output.open_nodes).id == 12
    @test pop!(output.open_nodes).id == 13
    @test pop!(output.open_nodes).id == 11
    @test pop!(output.open_nodes).id == 9
    @test pop!(output.open_nodes).id == 7
    @test pop!(output.open_nodes).id == 5
    @test pop!(output.open_nodes).id == 3
    @test isempty(output.open_nodes)

    return
end
