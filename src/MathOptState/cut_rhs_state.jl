
struct CutRhsChange <: AbstractAtomicChange
    constr_id::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}
    rhs::Float64
end

function apply_change!(backend, change::CutRhsChange, helper)
    MOI.set(backend, MOI.ConstraintSet(), change.constr_id, MOI.GreaterThan(change.rhs))
    return
end
struct CutRhsChangeDiff <: AbstractMathOptStateDiff
    cut_rhs::Dict{RowId,CutRhsChange}
end

CutRhsChangeDiff() = CutRhsChangeDiff(Dict{RowId,CutRhsChange}())

function merge_forward_change_diff(parent_forward_diff::CutRhsChangeDiff, local_forward_change::CutRhsChangeDiff)
    child_cut_rhs_changes = copy(parent_forward_diff.cut_rhs)

    for (row_id, change) in local_forward_change.cut_rhs
        child_cut_rhs_changes[row_id] = change
    end
    return CutRhsChangeDiff(child_cut_rhs_changes)
end

function merge_backward_change_diff(parent_backward_diff::CutRhsChangeDiff, local_backward_change::CutRhsChangeDiff)
    # take a look at the diff of the children node (ChangeBuffer)
    child_cut_rhs_changes = copy(local_backward_change.cut_rhs)

    for (row_id, change) in parent_backward_diff.cut_rhs
        child_cut_rhs_changes[row_id] = change
    end
    return CutRhsChangeDiff(child_cut_rhs_changes)
end

function apply_change!(backend, diff::CutRhsChangeDiff, helper)
    for change in values(diff.cut_rhs)
        apply_change!(backend, change, helper)
    end
    return
end

struct CutsTracker <: AbstractMathOptStateTracker end

root_state(::CutsTracker, backend) = ModelState(CutRhsChangeDiff(), CutRhsChangeDiff())
new_state(::CutsTracker, backward::CutRhsChangeDiff, forward::CutRhsChangeDiff) = ModelState(backward, forward)
