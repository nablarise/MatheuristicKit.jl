
struct DomainChangeTrackerHelper
    map_lb::Dict{MOI.VariableIndex, MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}}}
    map_ub::Dict{MOI.VariableIndex, MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}}}
    map_eq::Dict{MOI.VariableIndex, MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}}}
    function DomainChangeTrackerHelper()
        return new(
            Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}}(),
            Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}}(),
            Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}}}()
        )
    end
end

_register_constraints!(helper, vi, ci) = nothing

function _register_constraints!(helper, vi::F, ci::MOI.ConstraintIndex{F,S}) where {F<:MOI.VariableIndex,S<:MOI.GreaterThan}
    helper.map_lb[vi] = ci
end

function _register_constraints!(helper, vi::F, ci::MOI.ConstraintIndex{F,S}) where {F<:MOI.VariableIndex,S<:MOI.LessThan}
    helper.map_ub[vi] = ci
end

function _register_constraints!(helper, vi::F, ci::MOI.ConstraintIndex{F,S}) where {F<:MOI.VariableIndex,S<:MOI.EqualTo}
    helper.map_eq[vi] = ci
end

struct VarBoundsChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    lb::Float64
    ub::Float64
end

struct LowerBoundVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    new_lb::Float64
end

function apply_change!(backend, change::LowerBoundVarChange, helper::DomainChangeTrackerHelper)
    @assert !haskey(helper.map_eq, change.var_id)
    ci = get(helper.map_lb, change.var_id, nothing)
    if isnothing(ci)
        new_ci = MOI.add_constraint(backend, change.var_id, MOI.GreaterThan(change.new_lb))
        helper.map_lb[change.var_id] = new_ci
    else
        MOI.set(backend, MOI.ConstraintSet(), ci, MOI.GreaterThan(change.new_lb))
    end
    return
end
struct UpperBoundVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    new_ub::Float64
end

function apply_change!(backend, change::UpperBoundVarChange, helper::DomainChangeTrackerHelper)
    @assert !haskey(helper.map_eq, change.var_id)
    ci = get(helper.map_ub, change.var_id, nothing)

    if isnothing(ci)
       new_ci = MOI.add_constraint(backend, change.var_id, MOI.LessThan(change.new_ub))
       helper.map_ub[change.var_id] = new_ci
    else
        MOI.set(backend, MOI.ConstraintSet(), ci, MOI.LessThan(change.new_ub))
    end
    return
end

struct DomainChangeDiff <: AbstractMathOptStateDiff
    lower_bounds::Dict{ColId,LowerBoundVarChange}
    upper_bounds::Dict{ColId,UpperBoundVarChange}
end

DomainChangeDiff() = DomainChangeDiff(
    Dict{ColId,LowerBoundVarChange}(),
    Dict{ColId,UpperBoundVarChange}()
)

function merge_forward_change_diff(parent_forward_diff::DomainChangeDiff, local_forward_change::DomainChangeDiff)
    child_lb_changes = copy(parent_forward_diff.lower_bounds)
    child_ub_changes = copy(parent_forward_diff.upper_bounds)

    for (col_id, change) in local_forward_change.lower_bounds
        child_lb_changes[col_id] = change
    end
    for (col_id, change) in local_forward_change.upper_bounds
        child_ub_changes[col_id] = change
    end
    return DomainChangeDiff(child_lb_changes, child_ub_changes)
end

function merge_backward_change_diff(parent_backward_diff::DomainChangeDiff, local_backward_change::DomainChangeDiff)
    # take a look at the diff of the children node (ChangeBuffer)
    child_lb_changes = copy(local_backward_change.lower_bounds)
    child_ub_changes = copy(local_backward_change.upper_bounds)

    for (col_id, change) in parent_backward_diff.lower_bounds
        child_lb_changes[col_id] = change
    end
    for (col_id, change) in parent_backward_diff.upper_bounds
        child_ub_changes[col_id] = change
    end
    return DomainChangeDiff(child_lb_changes, child_ub_changes)
end

function apply_change!(backend, diff::DomainChangeDiff, helper)
    for change in values(diff.lower_bounds)
        apply_change!(backend, change, helper)
    end
    for change in values(diff.upper_bounds)
        apply_change!(backend, change, helper)
    end
    return
end


struct DomainChangeTracker <: AbstractMathOptStateTracker end

root_state(::DomainChangeTracker, backend) = ModelState(DomainChangeDiff(), DomainChangeDiff())
new_state(::DomainChangeTracker, backward::DomainChangeDiff, forward::DomainChangeDiff) = ModelState(backward, forward)

function transform_model!(::DomainChangeTracker, backend)
    helper = DomainChangeTrackerHelper()
    for (F, S) in MOI.get(backend, MOI.ListOfConstraintTypesPresent())
        if F == MOI.VariableIndex
            for ci in MOI.get(backend, MOI.ListOfConstraintIndices{F,S}())
                vi = MOI.get(backend, MOI.ConstraintFunction(), ci)
                _register_constraints!(helper, vi, ci)
            end
        end
    end
    return helper
end