module MathOptState

using MathOptInterface

const MOI = MathOptInterface
const ColId = Int
const RowId = Int

struct ModelState{T}
    forward_diff::T
    backward_diff::T
end

backward(model_state::ModelState) = model_state.backward_diff
forward(model_state::ModelState) = model_state.forward_diff

function recover_state!(model, prev_state::ModelState, next_state::ModelState, helper)
    apply_change!(model, backward(prev_state), helper)
    apply_change!(model, forward(next_state), helper)
end

abstract type AbstractAtomicChange end

function restore_changes!(model, changes::Dict{ColId,C}) where {C<:AbstractAtomicChange}
    for change in values(changes)
        restore_change!(model, change)
    end
    return
end

"""
    restore_change!(model, change::AbstractAtomicChange)
"""
function apply_change! end


"""
Type to implement the interface.
"""
abstract type AbstractMathOptStateTracker end

"""

"""
abstract type AbstractMathOptStateDiff end

"""
function merge_forward_change_diff(parent_forward_diff::DomainChangeDiff, local_forward_change::DomainChangeDiff)
"""
function merge_forward_change_diff end

"""
function merge_backward_change_diff(parent_backward_diff::DomainChangeDiff, local_backward_change::DomainChangeDiff)
"""
function merge_backward_change_diff end

"""
root_state(math_opt_state_tracker, model)
"""
function root_state end

"""
helper(math_opt_state_tracker, model)
"""
function helper end

"""
transform_model!(math_opt_state_tracker, model)

Useful if you need to transform some models to make the state tracking easier.
Should not change the original model !!
"""
function transform_model! end

"""
new_state(math_opt_state_tracker, backward, forward)
"""
function new_state end


### Default Implementations
include("var_bounds_state.jl")
include("cut_rhs_state.jl")
include("fixed_var_state.jl")

end # module MathOptState