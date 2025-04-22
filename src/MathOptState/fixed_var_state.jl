
struct FixVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    value::Float64
end

function apply_change!(backend, change::FixVarChange, helper::DomainChangeTrackerHelper)
    @assert !haskey(helper.map_eq, change.var_id) 
    if haskey(helper.map_lb, change.var_id)
        MOI.delete(backend, helper.map_lb[change.var_id])
        delete!(helper.map_lb, change.var_id)
    end
    if haskey(helper.map_ub, change.var_id)
        MOI.delete(backend, helper.map_ub[change.var_id])
        delete!(helper.map_ub, change.var_id)
    end
    eq_ci = MOI.add_constraint(backend, change.var_id, MOI.EqualTo(change.value))
    helper.map_eq[change.var_id] = eq_ci
    return
end

struct UnfixVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    lower_bound::Float64
    upper_bound::Float64
end

function apply_change!(backend, change::UnfixVarChange, helper::DomainChangeTrackerHelper)
    if haskey(helper.map_eq, change.var_id)
        MOI.delete(backend, helper.map_eq[change.var_id])
        delete!(helper.map_eq, change.var_id)
        lb_ci = MOI.add_constraint(backend, change.var_id, MOI.GreaterThan(change.lower_bound))
        ub_ci = MOI.add_constraint(backend, change.var_id, MOI.LessThan(change.upper_bound))
        helper.map_lb[change.var_id] = lb_ci
        helper.map_ub[change.var_id] = ub_ci
    end
    return
end

struct FixVarChangeDiff <: AbstractMathOptStateDiff
    fix_vars::Dict{ColId,FixVarChange}
    unfix_vars::Dict{ColId,UnfixVarChange}
end

FixVarChangeDiff() = FixVarChangeDiff(Dict{ColId,FixVarChange}(), Dict{ColId,UnfixVarChange}())

function FixVarChangeDiff(fix_var_changes::Vector{FixVarChange}, unfix_var_changes::Vector{UnfixVarChange})
    fix_vars = Dict{ColId,FixVarChange}(change.var_id.value => change for change in fix_var_changes)
    unfix_vars = Dict{ColId,UnfixVarChange}(change.var_id.value => change for change in unfix_var_changes)
    return FixVarChangeDiff(fix_vars, unfix_vars)
end

function merge_forward_change_diff(parent_forward_diff::FixVarChangeDiff, local_forward_change::FixVarChangeDiff)
    child_fix_vars = copy(parent_forward_diff.fix_vars)
    for (col_id, change) in local_forward_change.fix_vars
        child_fix_vars[col_id] = change
    end

    child_unfix_vars = copy(parent_forward_diff.unfix_vars)
    for (col_id, change) in local_forward_change.unfix_vars
        child_unfix_vars[col_id] = change
    end
    return FixVarChangeDiff(child_fix_vars, child_unfix_vars)
end

function merge_backward_change_diff(parent_backward_diff::FixVarChangeDiff, local_backward_change::FixVarChangeDiff)
    # take a look at the diff of the children node (ChangeBuffer)
    child_fix_vars = copy(local_backward_change.fix_vars)
    for (col_id, change) in parent_backward_diff.fix_vars
        child_fix_vars[col_id] = change
    end

    child_unfix_vars = copy(local_backward_change.unfix_vars)
    for (col_id, change) in parent_backward_diff.unfix_vars
        child_unfix_vars[col_id] = change
    end
    return FixVarChangeDiff(child_fix_vars, child_unfix_vars)
end

function apply_change!(backend, diff::FixVarChangeDiff, helper::DomainChangeTrackerHelper)
    for change in values(diff.fix_vars)
        apply_change!(backend, change, helper)
    end
    for change in values(diff.unfix_vars)
        apply_change!(backend, change, helper)
    end
    return
end

struct FixVarChangeTracker <: AbstractMathOptStateTracker end

root_state(::FixVarChangeTracker, backend) = ModelState(FixVarChangeDiff(), FixVarChangeDiff())
new_state(::FixVarChangeTracker, backward::FixVarChangeDiff, forward::FixVarChangeDiff) = ModelState(backward, forward)

function transform_model!(::FixVarChangeTracker, backend)
    helper = DomainChangeTrackerHelper()
    # Get the MOI backend of the JuMP backend
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