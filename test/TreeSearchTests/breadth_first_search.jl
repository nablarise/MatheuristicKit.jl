function test_breadth_first_search()
    ##########################
    #          [1]
    #        /     \
    #     [2]       [3]
    #    /   \     /   \
    #  [4]   [5]  6     7
    #  / \   / \
    # 8  9  10 11

    function test_function(space, current)
        expected_diff = 1
        return current.id - space.last_evaluated_node_id == expected_diff
    end

    strategy = MK.TreeSearch.BreadthFirstSearchStrategy()
    search_space = MockSearchSpace(test_function; node_limit=5)
    output = MK.TreeSearch.search(strategy, search_space)
    @test output.last_evaluated_node_id == 5
    @test output.nb_evaluated_node == 5

    @test popfirst!(output.open_nodes).id == 6
    @test popfirst!(output.open_nodes).id == 7
    @test popfirst!(output.open_nodes).id == 8
    @test popfirst!(output.open_nodes).id == 9
    @test popfirst!(output.open_nodes).id == 10
    @test popfirst!(output.open_nodes).id == 11
    @test isempty(output.open_nodes)

    return
end
