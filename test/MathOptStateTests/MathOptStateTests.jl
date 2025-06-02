module MathOptStateTests

using JuMP, MathOptInterface, Combinatorics, GLPK
# =====================================================================
# Example: Problem-specific Model and Solver Invocation
# =====================================================================
using GLPK
using Test
using MatheuristicKit

const MK = MatheuristicKit
const SingleVarBoundChange = Tuple{JuMP.GenericVariableRef,Float64}

function DomainChangeDiff(
    lb_changes::Vector{SingleVarBoundChange},
    ub_changes::Vector{SingleVarBoundChange}
)
    lower_bounds = Dict(JuMP.index(var).value => MK.MathOptState.LowerBoundVarChange(JuMP.index(var), lb) for (var, lb) in lb_changes)
    upper_bounds = Dict(JuMP.index(var).value => MK.MathOptState.UpperBoundVarChange(JuMP.index(var), ub) for (var, ub) in ub_changes)
    return MK.MathOptState.DomainChangeDiff(lower_bounds, upper_bounds)
end

function test_math_opt_state1()
    model = Model(GLPK.Optimizer)
    @variable(model, x >= 0, Int)
    @objective(model, Min, x)
    @constraint(model, x <= 8)

    #  Consider the following nodes : 

    # 1 ( 0 <= x)
    # 2 ( 0 <= x <= 4)
    # 3 ( 3 <= x <= 7)
    # 4 ( 2 <= x <= 8)
    # 5 ( -Inf <= x <= Inf)
    # 6 ( 9 <= x <= Inf)

    expected_has_lb = [true, true, true, true, false, true]
    expected_has_ub = [false, true, true, true, false, false]
    expected_lb_value = [0, 0, 3, 2, -Inf, 9]
    expected_ub_value = [Inf, 4, 7, 8, Inf, Inf]

    # we want to create the following treesearch

    # 1 -> 2 -> 3 
    # |--> 4 -> 5 -> 6

    var_id = JuMP.index(x)
    col_id = JuMP.index(x).value

    ##############################
    ## Node 1
    ##############################
    math_opt_state = MK.MathOptState.DomainChangeTracker()
    helper = MK.MathOptState.transform_model!(math_opt_state, JuMP.backend(model))

    state1 = MK.MathOptState.root_state(math_opt_state, JuMP.backend(model))

    @test JuMP.lower_bound(x) == 0
    @test !JuMP.has_upper_bound(x)

    ## Prepare node 2
    forward_local_change2 = DomainChangeDiff(SingleVarBoundChange[], SingleVarBoundChange[(x, 4)])
    forward_change_diff2 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state1), forward_local_change2)

    ub = JuMP.has_upper_bound(x) ? JuMP.upper_bound(x) : Inf
    backward_local_change2 = DomainChangeDiff(SingleVarBoundChange[], SingleVarBoundChange[(x, ub)])
    backward_change_diff2 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state1), backward_local_change2)

    state2 = MK.MathOptState.new_state(math_opt_state, forward_change_diff2, backward_change_diff2)

    ## Prepare node 4
    forward_local_change4 = DomainChangeDiff(SingleVarBoundChange[(x, 2)], SingleVarBoundChange[(x, 8)])
    forward_change_diff4 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state1), forward_local_change4)

    lb = JuMP.has_lower_bound(x) ? JuMP.lower_bound(x) : -Inf
    ub = JuMP.has_upper_bound(x) ? JuMP.upper_bound(x) : Inf
    backward_local_change4 = DomainChangeDiff(SingleVarBoundChange[(x, lb)], SingleVarBoundChange[(x, ub)])
    backward_change_diff4 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state1), backward_local_change4)

    state4 = MK.MathOptState.new_state(math_opt_state, forward_change_diff4, backward_change_diff4)

    ##############################
    ## Node 2
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state1, state2, helper)

    @test JuMP.lower_bound(x) == 0
    @test JuMP.upper_bound(x) == 4

    ## Prepare node 3
    forward_local_change3 = DomainChangeDiff(SingleVarBoundChange[(x, 3)], SingleVarBoundChange[(x, 7)])
    forward_change_diff3 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state2), forward_local_change3)

    lb = JuMP.has_lower_bound(x) ? JuMP.lower_bound(x) : -Inf
    ub = JuMP.has_upper_bound(x) ? JuMP.upper_bound(x) : Inf
    backward_local_change3 = DomainChangeDiff(SingleVarBoundChange[(x, lb)], SingleVarBoundChange[(x, ub)])
    backward_change_diff3 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state2), backward_local_change3)

    state3 = MK.MathOptState.new_state(math_opt_state, forward_change_diff3, backward_change_diff3)

    ##############################
    ## Node 3
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state2, state3, helper)

    @test JuMP.lower_bound(x) == 3
    @test JuMP.upper_bound(x) == 7

    # No child

    ##############################
    ## Node 4
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state3, state4, helper)

    @test JuMP.lower_bound(x) == 2
    @test JuMP.upper_bound(x) == 8

    forward_local_change5 = DomainChangeDiff(SingleVarBoundChange[(x, -Inf)], SingleVarBoundChange[(x, Inf)])
    forward_change_diff5 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state4), forward_local_change5)

    lb = JuMP.has_lower_bound(x) ? JuMP.lower_bound(x) : -Inf
    ub = JuMP.has_upper_bound(x) ? JuMP.upper_bound(x) : Inf
    backward_local_change5 = DomainChangeDiff(SingleVarBoundChange[(x, lb)], SingleVarBoundChange[(x, ub)])
    backward_change_diff5 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state4), backward_local_change5)

    state5 = MK.MathOptState.new_state(math_opt_state, forward_change_diff5, backward_change_diff5)

    ##############################
    ## Node 5
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state4, state5, helper)

    @test JuMP.lower_bound(x) == -Inf
    @test JuMP.upper_bound(x) == Inf

    forward_local_change6 = DomainChangeDiff(SingleVarBoundChange[(x, 9)], SingleVarBoundChange[])
    forward_change_diff6 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state5), forward_local_change6)

    lb = JuMP.has_lower_bound(x) ? JuMP.lower_bound(x) : -Inf
    backward_local_change6 = DomainChangeDiff(SingleVarBoundChange[(x, lb)], SingleVarBoundChange[])
    backward_change_diff6 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state5), backward_local_change6)

    state6 = MK.MathOptState.new_state(math_opt_state, forward_change_diff6, backward_change_diff6)

    ##############################
    ## Node 6
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state5, state6, helper)

    @test JuMP.lower_bound(x) == 9
    @test JuMP.upper_bound(x) == Inf

    #########################################################################
    # Then we move from node to node and make sure the formulation is correct
    # We try all the sequence combinations.
    #########################################################################
    prev_state = state6
    states = [state1, state2, state3, state4, state5, state6]
    for permutation in permutations([1, 2, 3, 4, 5, 6])
        for node_id in permutation
            next_state = states[node_id]
            MK.MathOptState.recover_state!(JuMP.backend(model), prev_state, next_state, helper)

            @test JuMP.has_lower_bound(x)
            @test JuMP.lower_bound(x) == expected_lb_value[node_id]
           
            @test JuMP.has_upper_bound(x)
            @test JuMP.upper_bound(x) == expected_ub_value[node_id]
            
            prev_state = next_state
        end
    end
end

##################################################################

const SingleCutRhsChange = Tuple{JuMP.ConstraintRef,Float64}
function CutRhsChangeDiff(cut_rhs::Vector{SingleCutRhsChange})
    cut_rhs_changes = Dict{MK.MathOptState.RowId,MK.MathOptState.CutRhsChange}()
    for (constr, rhs) in cut_rhs
        cut_rhs_changes[JuMP.index(constr).value] = MK.MathOptState.CutRhsChange(JuMP.index(constr), rhs)
    end
    return MK.MathOptState.CutRhsChangeDiff(cut_rhs_changes)
end

function test_math_opt_state2()
    model = Model(GLPK.Optimizer)
    @variable(model, x >= 0, Int)
    @variable(model, y >= 0, Int)
    @variable(model, z >= 0, Int)
    @objective(model, Min, x + y + z)
    #@constraint(model, c1, 3x + 3y + 2z >= 1)
    #@constraint(model, c2, 3x + 2y + 2z >= 1)
    #@constraint(model, c3, 2x + 2y + 2z >= 1)
    #@constraint(model, c4, 2x + 2y + z >= 1
    #@constraint(model, c5, x + 2y + 2z >= 1)
    #@constraint(model, c6, x + y + 2z >= 1)
    #@constraint(model, c7, x + y + z >= 1)
    #@constraint(model, c8, 4x + 3y + 4z >= 1)
    #@constraint(model, c9, 3x + 4y + 3z >= 1)
    #@constraint(model, c10, 4x + 4y + 4z >= 1)
    #@constraint(model, c11, 5x + 4y + 5z >= 1)

    # we want to create the following treesearch
    #      | -> 7 -> 8
    # 1 -> 2 -> 3 -> 
    # |--> 4 -> 5 -> 6 

    # 1 s.t. c8
    # 2 s.t. c1, c11
    # 3 s.t. c3, c7
    # 4 s.t. c4
    # 5 s.t. c5, c10
    # 6 s.t. c6
    # 7 s.t. c2 c9
    #

    expected_rhs = Dict{MK.MathOptState.RowId,Vector{Float64}}(
        #        1     2     3     4     5     6     7     8     9    10    11  
        1 => [-1e5, -1e5, -1e5, -1e5, -1e5, -1e5, -1e5, 1, -1e5, -1e5, -1e5],
        2 => [1, -1e5, -1e5, -1e5, -1e5, -1e5, -1e5, 1, -1e5, -1e5, 1],
        3 => [1, -1e5, 1, -1e5, -1e5, -1e5, 1, 1, -1e5, -1e5, 1],
        4 => [-1e5, -1e5, -1e5, 1, -1e5, -1e5, -1e5, 1, -1e5, -1e5, -1e5],
        5 => [-1e5, -1e5, -1e5, 1, 1, -1e5, -1e5, 1, -1e5, 1, -1e5],
        6 => [-1e5, -1e5, -1e5, 1, 1, 1, -1e5, 1, -1e5, 1, -1e5],
        7 => [1, 1, -1e5, -1e5, -1e5, -1e5, -1e5, 1, 1, -1e5, 1])

    ##############################
    ## Node 1
    ##############################
    math_opt_state = MK.MathOptState.CutsTracker()

    state1 = MK.MathOptState.root_state(math_opt_state, JuMP.backend(model))

    @constraint(model, c8, 4x + 3y + 4z >= 1)

    # udpate state 1.
    forward_local_change1 = CutRhsChangeDiff(SingleCutRhsChange[(c8, 1)])
    forward_change_diff1 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state1), forward_local_change1)

    backward_local_change1 = CutRhsChangeDiff(SingleCutRhsChange[(c8, -1e5)])
    backward_change_diff1 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state1), backward_local_change1)
    state1 = MK.MathOptState.new_state(math_opt_state, forward_change_diff1, backward_change_diff1)

    ## Prepare node 2
    # The real code to run:
    forward_local_change2 = MK.MathOptState.CutRhsChangeDiff()
    forward_change_diff2 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state1), forward_local_change2)

    backward_local_change2 = MK.MathOptState.CutRhsChangeDiff()
    backward_change_diff2 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state1), backward_local_change2)

    # but it does nothing, we replace it by state2 = deepcopy(state1)
    state2 = MK.MathOptState.new_state(math_opt_state, forward_change_diff2, backward_change_diff2)

    ## Prepare node 4
    state4 = deepcopy(state1)

    ##############################
    ## Node 2
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state1, state2, nothing)

    @constraint(model, c1, 3x + 3y + 2z >= 1)
    @constraint(model, c11, 5x + 4y + 5z >= 1)

    # udpate state 2.
    forward_local_change2 = CutRhsChangeDiff(SingleCutRhsChange[(c1, 1), (c11, 1)])
    forward_change_diff2 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state2), forward_local_change2)

    backward_local_change2 = CutRhsChangeDiff(SingleCutRhsChange[(c1, -1e5), (c11, -1e5)])
    backward_change_diff2 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state2), backward_local_change2)
    state2 = MK.MathOptState.new_state(math_opt_state, forward_change_diff2, backward_change_diff2)

    ## Prepare node 3
    state3 = deepcopy(state2)

    ## Prepare node 7
    state7 = deepcopy(state3)

    ##############################
    ## Node 3
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state2, state3, nothing)

    @constraint(model, c3, 2x + 2y + 2z >= 1)
    @constraint(model, c7, x + y + z >= 1)

    # udpate state 3.
    forward_local_change3 = CutRhsChangeDiff(SingleCutRhsChange[(c3, 1), (c7, 1)])
    forward_change_diff3 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state3), forward_local_change3)

    backward_local_change3 = CutRhsChangeDiff(SingleCutRhsChange[(c3, -1e5), (c7, -1e5)])
    backward_change_diff3 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state3), backward_local_change3)
    state3 = MK.MathOptState.new_state(math_opt_state, forward_change_diff3, backward_change_diff3)

    # No child

    ##############################
    ## Node 4
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state3, state4, nothing)

    @constraint(model, c4, 2x + 2y + z >= 1)

    # Update state 4
    forward_local_change4 = CutRhsChangeDiff(SingleCutRhsChange[(c4, 1)])
    forward_change_diff4 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state4), forward_local_change4)

    backward_local_change4 = CutRhsChangeDiff(SingleCutRhsChange[(c4, -1e5)])
    backward_change_diff4 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state4), backward_local_change4)
    state4 = MK.MathOptState.new_state(math_opt_state, forward_change_diff4, backward_change_diff4)

    # Prepare node 5
    state5 = deepcopy(state4)

    ##############################
    ## Node 5
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state4, state5, nothing)

    @constraint(model, c5, x + 2y + 2z >= 1)
    @constraint(model, c10, 4x + 4y + 4z >= 1)

    # Update state 5
    forward_local_change5 = CutRhsChangeDiff(SingleCutRhsChange[(c5, 1), (c10, 1)])
    forward_change_diff5 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state5), forward_local_change5)

    backward_local_change5 = CutRhsChangeDiff(SingleCutRhsChange[(c5, -1e5), (c10, -1e5)])
    backward_change_diff5 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state5), backward_local_change5)
    state5 = MK.MathOptState.new_state(math_opt_state, forward_change_diff5, backward_change_diff5)

    # Prepare node 6
    state6 = deepcopy(state5)

    ##############################
    ## Node 6
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state5, state6, nothing)

    @constraint(model, c6, x + y + 2z >= 1)

    # Update state 6
    forward_local_change6 = CutRhsChangeDiff(SingleCutRhsChange[(c6, 1)])
    forward_change_diff6 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state6), forward_local_change6)

    backward_local_change6 = CutRhsChangeDiff(SingleCutRhsChange[(c6, -1e5)])
    backward_change_diff6 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state6), backward_local_change6)
    state6 = MK.MathOptState.new_state(math_opt_state, forward_change_diff6, backward_change_diff6)

    # No child

    ##############################
    ## Node 7
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state6, state7, nothing)

    @constraint(model, c2, 3x + 2y + 2z >= 1)
    @constraint(model, c9, 4x + 3y + 4z >= 1)

    # Update state 7
    forward_local_change7 = CutRhsChangeDiff(SingleCutRhsChange[(c2, 1), (c9, 1)])
    forward_change_diff7 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state7), forward_local_change7)

    backward_local_change7 = CutRhsChangeDiff(SingleCutRhsChange[(c2, -1e5), (c9, -1e5)])
    backward_change_diff7 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state7), backward_local_change7)
    state7 = MK.MathOptState.new_state(math_opt_state, forward_change_diff7, backward_change_diff7)

    # No child

    #########################################################################   
    # Then we move from node to node and make sure the formulation is correct
    # We try all the sequence combinations.
    #########################################################################
    prev_state = state7
    states = [state1, state2, state3, state4, state5, state6, state7]
    for permutation in permutations([1, 2, 3, 4, 5, 6, 7])
        for node_id in permutation
            next_state = states[node_id]
            MK.MathOptState.recover_state!(JuMP.backend(model), prev_state, next_state, nothing)

            @test JuMP.normalized_rhs(c1) == expected_rhs[node_id][1]
            @test JuMP.normalized_rhs(c2) == expected_rhs[node_id][2]
            @test JuMP.normalized_rhs(c3) == expected_rhs[node_id][3]
            @test JuMP.normalized_rhs(c4) == expected_rhs[node_id][4]
            @test JuMP.normalized_rhs(c5) == expected_rhs[node_id][5]
            @test JuMP.normalized_rhs(c6) == expected_rhs[node_id][6]
            @test JuMP.normalized_rhs(c7) == expected_rhs[node_id][7]
            @test JuMP.normalized_rhs(c8) == expected_rhs[node_id][8]
            @test JuMP.normalized_rhs(c9) == expected_rhs[node_id][9]
            @test JuMP.normalized_rhs(c10) == expected_rhs[node_id][10]
            @test JuMP.normalized_rhs(c11) == expected_rhs[node_id][11]

            prev_state = next_state
        end
    end
end

##################################################################

FixVarChange(var::JuMP.VariableRef, value) = MK.MathOptState.FixVarChange(JuMP.index(var), value)
UnfixVarChange(var::JuMP.VariableRef, lower_bound, upper_bound) = MK.MathOptState.UnfixVarChange(JuMP.index(var), lower_bound, upper_bound)

function test_math_opt_state3()
    model = Model(GLPK.Optimizer)
    @variable(model, 0 <= x <= 1)
    @variable(model, 0 <= y <= 1)

    @objective(model, Min, x + y)

    ### Tree is
    ##          1
    ##     2          3     <-- fix x 
    ##  4    5     6     7  <--- fix y

    expected_x_is_fixed = [false, true, true, true, true, true, true]
    expected_y_is_fixed = [false, false, false, true, true, true, true]
    expected_x_fix_value = [NaN, 0, 1, 0, 0, 1, 1]
    expected_y_fix_value = [NaN, NaN, NaN, 0, 1, 0, 1]

    ##############################
    ## Node 1
    ##############################
    math_opt_state = MK.MathOptState.FixVarChangeTracker()
    helper = MK.MathOptState.transform_model!(math_opt_state, JuMP.backend(model))

    state1 = MK.MathOptState.root_state(math_opt_state, JuMP.backend(model))

    ## Prepare node 2
    forward_local_change2 = MK.MathOptState.FixVarChangeDiff([FixVarChange(x, 0)], MK.MathOptState.UnfixVarChange[])
    forward_change_diff2 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state1), forward_local_change2)

    backward_local_change2 = MK.MathOptState.FixVarChangeDiff(MK.MathOptState.FixVarChange[], [UnfixVarChange(x, 0, 1)])
    backward_change_diff2 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state1), backward_local_change2)

    state2 = MK.MathOptState.new_state(math_opt_state, forward_change_diff2, backward_change_diff2)

    ## Prepare node 3
    forward_local_change3 = MK.MathOptState.FixVarChangeDiff([FixVarChange(x, 1)], MK.MathOptState.UnfixVarChange[])
    forward_change_diff3 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state1), forward_local_change3)

    backward_local_change3 = MK.MathOptState.FixVarChangeDiff(MK.MathOptState.FixVarChange[], [UnfixVarChange(x, 0, 1)])
    backward_change_diff3 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state1), backward_local_change3)

    state3 = MK.MathOptState.new_state(math_opt_state, forward_change_diff3, backward_change_diff3)

    ##############################
    ## Node 2
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state1, state2, helper)

    ## Prepare node 4
    forward_local_change4 = MK.MathOptState.FixVarChangeDiff([FixVarChange(y, 0)], MK.MathOptState.UnfixVarChange[])
    forward_change_diff4 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state2), forward_local_change4)

    backward_local_change4 = MK.MathOptState.FixVarChangeDiff(MK.MathOptState.FixVarChange[], [UnfixVarChange(y, 0, 1)])
    backward_change_diff4 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state2), backward_local_change4)

    state4 = MK.MathOptState.new_state(math_opt_state, forward_change_diff4, backward_change_diff4)

    ## Prepare node 5
    forward_local_change5 = MK.MathOptState.FixVarChangeDiff([FixVarChange(y, 1)], MK.MathOptState.UnfixVarChange[])
    forward_change_diff5 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state2), forward_local_change5)

    backward_local_change5 = MK.MathOptState.FixVarChangeDiff(MK.MathOptState.FixVarChange[], [UnfixVarChange(y, 0, 1)])
    backward_change_diff5 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state2), backward_local_change5)

    state5 = MK.MathOptState.new_state(math_opt_state, forward_change_diff5, backward_change_diff5)

    ##############################
    ## Node 3
    ##############################
    MK.MathOptState.recover_state!(JuMP.backend(model), state2, state3, helper)

    ## Prepare node 4
    forward_local_change6 = MK.MathOptState.FixVarChangeDiff([FixVarChange(y, 0)], MK.MathOptState.UnfixVarChange[])
    forward_change_diff6 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state3), forward_local_change6)

    backward_local_change6 = MK.MathOptState.FixVarChangeDiff(MK.MathOptState.FixVarChange[], [UnfixVarChange(y, 0, 1)])
    backward_change_diff6 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state3), backward_local_change6)

    state6 = MK.MathOptState.new_state(math_opt_state, forward_change_diff6, backward_change_diff6)

    ## Prepare node 5
    forward_local_change7 = MK.MathOptState.FixVarChangeDiff([FixVarChange(y, 1)], MK.MathOptState.UnfixVarChange[])
    forward_change_diff7 = MK.MathOptState.merge_forward_change_diff(MK.MathOptState.forward(state3), forward_local_change7)

    backward_local_change7 = MK.MathOptState.FixVarChangeDiff(MK.MathOptState.FixVarChange[], [UnfixVarChange(y, 0, 1)])
    backward_change_diff7 = MK.MathOptState.merge_backward_change_diff(MK.MathOptState.backward(state3), backward_local_change7)

    state7 = MK.MathOptState.new_state(math_opt_state, forward_change_diff7, backward_change_diff7)

    #########################################################################   
    # Then we move from node to node and make sure the formulation is correct
    # We try all the sequence combinations.
    #########################################################################
    prev_state = state7
    states = [state1, state2, state3, state4, state5, state6, state7]
    for permutation in permutations([1, 2, 3, 4, 5, 6, 7])

        for node_id in permutation
            next_state = states[node_id]

            MK.MathOptState.recover_state!(JuMP.backend(model), prev_state, next_state, helper)

            @test JuMP.is_fixed(x) == expected_x_is_fixed[node_id]
            @test JuMP.is_fixed(y) == expected_y_is_fixed[node_id]

            if expected_x_is_fixed[node_id]
                @test JuMP.fix_value(x) == expected_x_fix_value[node_id]
            else
                @test JuMP.lower_bound(x) == 0
                @test JuMP.upper_bound(x) == 1
            end

            if expected_y_is_fixed[node_id]
                @test JuMP.fix_value(y) == expected_y_fix_value[node_id]
            else
                @test JuMP.lower_bound(y) == 0
                @test JuMP.upper_bound(y) == 1
            end

            prev_state = next_state
        end
    end
end

function test_domain_change_tracker_continuous()
    model = Model(GLPK.Optimizer)
    @variable(model, x >= 0)
    @variable(model, 0 <= y <= 3)

    @objective(model, Min, x + y)

    tracker = MK.MathOptState.DomainChangeTracker()
    helper = MK.MathOptState.transform_model!(tracker, JuMP.backend(model))

    @test JuMP.index(x) in keys(helper.map_lb)
    @test JuMP.index(y) in keys(helper.map_lb)
    @test length(helper.map_lb) == 2

    @test JuMP.index(y) in keys(helper.map_ub)
    @test length(helper.map_ub) == 1

    @test isempty(helper.map_eq)
    @test isempty(helper.map_binary)
    @test isempty(helper.map_integer)
    @test isempty(helper.original_binary_vars)
    @test isempty(helper.original_integer_vars)
    return
end

function test_domain_change_tracker_binary()
    model = Model(GLPK.Optimizer)
    @variable(model, x, Bin)
    @variable(model, 0 <= y <= 1, Bin)

    @objective(model, Min, x + y)

    tracker = MK.MathOptState.DomainChangeTracker()
    helper = MK.MathOptState.transform_model!(tracker, JuMP.backend(model))

    @test JuMP.index(y) in keys(helper.map_lb)
    @test length(helper.map_lb) == 1

    @test JuMP.index(y) in keys(helper.map_ub)
    @test length(helper.map_ub) == 1

    @test isempty(helper.map_eq)
    @test JuMP.index(x) in keys(helper.map_binary)
    @test JuMP.index(y) in keys(helper.map_binary)
    @test isempty(helper.map_integer)

    @test JuMP.index(x) in helper.original_binary_vars
    @test JuMP.index(y) in helper.original_binary_vars
    @test isempty(helper.original_integer_vars)

    # What happens when we relax ?
    undo_relax = JuMP.relax_integrality(model)

    @test !MOI.is_valid(JuMP.backend(model), helper.map_binary[JuMP.index(x)])
    @test !MOI.is_valid(JuMP.backend(model), helper.map_binary[JuMP.index(y)])

    @test MOI.is_valid(JuMP.backend(model), helper.map_lb[JuMP.index(y)])
    @test MOI.is_valid(JuMP.backend(model), helper.map_ub[JuMP.index(y)])
    return
end

function test_domain_change_tracker_integer()
    model = Model(GLPK.Optimizer)
    @variable(model, x >= 0, Int)
    @variable(model, 0 <= y <= 9, Int)

    @objective(model, Min, x + y)

    tracker = MK.MathOptState.DomainChangeTracker()
    helper = MK.MathOptState.transform_model!(tracker, JuMP.backend(model))

    @test JuMP.index(x) in keys(helper.map_lb)
    @test JuMP.index(y) in keys(helper.map_lb)
    @test length(helper.map_lb) == 2

    @test JuMP.index(y) in keys(helper.map_ub)
    @test length(helper.map_ub) == 1

    @test isempty(helper.map_eq)
    @test isempty(helper.map_binary)
    @test JuMP.index(x) in keys(helper.map_integer)
    @test JuMP.index(y) in keys(helper.map_integer)

    @test JuMP.index(x) in helper.original_integer_vars
    @test JuMP.index(y) in helper.original_integer_vars
    @test isempty(helper.original_binary_vars)

    # What happens when we relax ?
    undo_relax = JuMP.relax_integrality(model)

    @test !MOI.is_valid(JuMP.backend(model), helper.map_integer[JuMP.index(x)])
    @test !MOI.is_valid(JuMP.backend(model), helper.map_integer[JuMP.index(y)])
    return
end

function test_relax_integrality()
    model = JuMP.Model()
    @variable(model, 1 <= w <= 5)
    @variable(model, x, Bin)
    @variable(model, 0 <= y <= 2, Bin)
    @variable(model, z >= 2, Int)

    tracker = MK.MathOptState.DomainChangeTracker()
    helper = MK.MathOptState.transform_model!(tracker, JuMP.backend(model))

    integral_state = MK.MathOptState.root_state(tracker, JuMP.backend(model))
    integrality_relaxation_state = MK.MathOptState.relax_integrality!(JuMP.backend(model), helper)

    relax_var_2 = false
    relax_var_3 = false
    relax_var_4 = false

    for change in integrality_relaxation_state.forward_diff.integrality_changes
        if change.var_id.value == 2
            @test change.change == MK.MathOptState.relax_zero_one
            relax_var_2 = true
        elseif change.var_id.value == 3
            @test change.change == MK.MathOptState.relax_zero_one
            relax_var_3 = true
        elseif change.var_id.value == 4
            @test change.change == MK.MathOptState.relax_integrality
            relax_var_4 = true
        end
    end

    @test relax_var_2
    @test relax_var_3
    @test relax_var_4

    restrict_var_2 = false
    restrict_var_3 = false
    restrict_var_4 = false

    for change in integrality_relaxation_state.backward_diff.integrality_changes
        if change.var_id.value == 2
            @test change.change == MK.MathOptState.restrict_zero_one
            restrict_var_2 = true
        elseif change.var_id.value == 3
            @test change.change == MK.MathOptState.restrict_zero_one
            restrict_var_3 = true
        elseif change.var_id.value == 4
            @test change.change == MK.MathOptState.restrict_integrality
            restrict_var_4 = true
        end
    end

    @test restrict_var_2
    @test restrict_var_3
    @test restrict_var_4

    MK.MathOptState.recover_state!(JuMP.backend(model), integral_state, integrality_relaxation_state, helper)

    @test !JuMP.is_binary(w)
    @test !JuMP.is_integer(w)
    @test JuMP.has_lower_bound(w) && JuMP.lower_bound(w) == 1
    @test JuMP.has_upper_bound(w) && JuMP.upper_bound(w) == 5
    
    @test !JuMP.is_binary(x)
    @test !JuMP.is_integer(x)
    @test JuMP.has_lower_bound(x) && JuMP.lower_bound(x) == 0
    @test JuMP.has_upper_bound(x) && JuMP.upper_bound(x) == 1

    @test !JuMP.is_binary(y)
    @test !JuMP.is_integer(y)
    @test JuMP.has_lower_bound(y) && JuMP.lower_bound(y) == 0
    @test JuMP.has_upper_bound(y) && JuMP.upper_bound(y) == 1

    @test !JuMP.is_binary(z)
    @test !JuMP.is_integer(z)
    @test JuMP.has_lower_bound(z) && JuMP.lower_bound(z) == 2
    @test !JuMP.has_upper_bound(z)

    MK.MathOptState.recover_state!(JuMP.backend(model), integrality_relaxation_state, integral_state, helper)

    @test !JuMP.is_binary(w)
    @test !JuMP.is_integer(w)
    @test JuMP.has_lower_bound(w) && JuMP.lower_bound(w) == 1
    @test JuMP.has_upper_bound(w) && JuMP.upper_bound(w) == 5
    
    @test JuMP.is_binary(x)
    @test !JuMP.is_integer(x)
    @test JuMP.has_lower_bound(x) && JuMP.lower_bound(x) == 0
    @test JuMP.has_upper_bound(x) && JuMP.upper_bound(x) == 1

    @test JuMP.is_binary(y)
    @test !JuMP.is_integer(y)
    @test JuMP.has_lower_bound(y) && JuMP.lower_bound(y) == 0
    @test JuMP.has_upper_bound(y) && JuMP.upper_bound(y) == 1

    @test !JuMP.is_binary(z)
    @test JuMP.is_integer(z)
    @test JuMP.has_lower_bound(z) && JuMP.lower_bound(z) == 2
    @test !JuMP.has_upper_bound(z)
end

function run()
    @testset "MathOptStateTests" begin
        test_math_opt_state1()
        test_math_opt_state2()
        test_math_opt_state3()
        test_domain_change_tracker_continuous()
        test_domain_change_tracker_binary()
        test_domain_change_tracker_integer()
        test_relax_integrality()
    end
end
end # end module