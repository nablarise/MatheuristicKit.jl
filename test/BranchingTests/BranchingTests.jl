module BranchingTests

using Test, MatheuristicKit, JuMP, GLPK
const MK = MatheuristicKit

# pour le strong branching, je dois pouvoir évaluer des noeuds avec un algo.
# génrer les candidats
# génrer le noeud correspondant à chaque candidat
# les évaluer avec un algo
# les classer

function test_most_fractional_branching()
    w = [1.0, 2.0, 3.0]
    c = [1.0, 2.0, 3.0]
    q = 4.0
    model = JuMP.Model(GLPK.Optimizer)
    
    I = 1:3
    @variable(model, 0 <= x[i in I] <= 1)
    @objective(model, Max, sum(c[i] * x[i] for i in I))
    @constraint(model, sum(w[i] * x[i] for i in I) <= q)
    optimize!(model)
    @show JuMP.value.(x)

    branching = MK.Branching.MostFractionalOrder()
    candidates = MK.Branching.get_candidates(model, branching)
    @show candidates

end

function run()
    @testset "BranchingTests" begin
        test_most_fractional_branching()
    end 
end

end