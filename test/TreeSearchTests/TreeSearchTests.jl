module TreeSearchTests

using Test, MatheuristicKit, Revise, DataStructures
const MK = MatheuristicKit

include("mock_impl.jl")
include("search_strategy.jl")
include("breadth_first_search.jl")
include("best_first_search.jl")
include("limited_discrepancy_search.jl")
include("beam_search.jl")

function run()
    @testset "TreeSearchTests" begin
        test_depth_first_search()
        test_breadth_first_search()
        test_best_first_search()
        test_limited_discrepancy_search1()
        test_limited_discrepancy_search2()
        test_beam_search_strategy()
    end
end
end # end module
