
"""
    FixVarChange <: AbstractAtomicChange

Represents a change that fixes a variable to a specific value.

# Fields
- `var_id::MOI.VariableIndex`: The index of the variable to fix
- `value::Float64`: The value to fix the variable to
"""
struct FixVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    value::Float64
end

"""
    apply_change!(backend, change::FixVarChange, helper::DomainChangeTrackerHelper)

Apply a fix variable change to an optimization model.

# Arguments
- `backend`: The optimization model backend
- `change::FixVarChange`: The fix variable change to apply
- `helper::DomainChangeTrackerHelper`: Helper object with constraint mappings

# Note
- Asserts that the variable is not already fixed
- Removes any existing lower and upper bound constraints
- Adds a new equality constraint fixing the variable to the specified value
"""
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

"""
    UnfixVarChange <: AbstractAtomicChange

Represents a change that unfixes a variable and resets bounds.

# Fields
- `var_id::MOI.VariableIndex`: The index of the variable to unfix
- `lower_bound::Float64`: The initial lower bound of the variable
- `upper_bound::Float64`: The initial upper bound of the variable
"""
struct UnfixVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    lower_bound::Float64
    upper_bound::Float64
end

"""
    apply_change!(backend, change::UnfixVarChange, helper::DomainChangeTrackerHelper)

Apply an unfix variable change to an optimization model.

# Arguments
- `backend`: The optimization model backend
- `change::UnfixVarChange`: The unfix variable change to apply
- `helper::DomainChangeTrackerHelper`: Helper object with constraint mappings

# Note
- Only takes action if the variable is currently fixed (has an equality constraint)
- Removes the existing equality constraint
- Restore initial lower and upper bound constraints
"""
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

"""
    FixVarChangeDiff <: AbstractMathOptStateDiff

Represents a collection of changes to variable fixations.

# Fields
- `fix_vars::Dict{ColId,FixVarChange}`: Dictionary of fix variable changes indexed by column ID
- `unfix_vars::Dict{ColId,UnfixVarChange}`: Dictionary of unfix variable changes indexed by column ID
"""
struct FixVarChangeDiff <: AbstractMathOptStateDiff
    fix_vars::Dict{ColId,FixVarChange}
    unfix_vars::Dict{ColId,UnfixVarChange}
end

"""
    FixVarChangeDiff()

Create an empty fix variable change difference with no changes.

# Returns
A new empty `FixVarChangeDiff`.
"""
FixVarChangeDiff() = FixVarChangeDiff(Dict{ColId,FixVarChange}(), Dict{ColId,UnfixVarChange}())

"""
    FixVarChangeDiff(fix_var_changes::Vector{FixVarChange}, unfix_var_changes::Vector{UnfixVarChange})

Create a fix variable change difference from vectors of fix and unfix changes.

# Arguments
- `fix_var_changes::Vector{FixVarChange}`: Vector of fix variable changes
- `unfix_var_changes::Vector{UnfixVarChange}`: Vector of unfix variable changes

# Returns
A new `FixVarChangeDiff` containing the specified changes.
"""
function FixVarChangeDiff(fix_var_changes::Vector{FixVarChange}, unfix_var_changes::Vector{UnfixVarChange})
    fix_vars = Dict{ColId,FixVarChange}(change.var_id.value => change for change in fix_var_changes)
    unfix_vars = Dict{ColId,UnfixVarChange}(change.var_id.value => change for change in unfix_var_changes)
    return FixVarChangeDiff(fix_vars, unfix_vars)
end

"""
    merge_forward_change_diff(parent_forward_diff::FixVarChangeDiff, local_forward_change::FixVarChangeDiff)

Merge a local forward fix variable change difference into a parent forward fix variable change difference.

# Arguments
- `parent_forward_diff::FixVarChangeDiff`: The parent forward difference
- `local_forward_change::FixVarChangeDiff`: The local forward difference to merge

# Returns
A new `FixVarChangeDiff` that combines both differences, with local changes taking precedence
when there are conflicts.
"""
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

"""
    merge_backward_change_diff(parent_backward_diff::FixVarChangeDiff, local_backward_change::FixVarChangeDiff)

Merge a parent backward fix variable change difference into a local backward fix variable change difference.

# Arguments
- `parent_backward_diff::FixVarChangeDiff`: The parent backward difference
- `local_backward_change::FixVarChangeDiff`: The local backward difference

# Returns
A new `FixVarChangeDiff` that combines both differences, with parent changes taking precedence
when there are conflicts.
"""
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

"""
    apply_change!(backend, diff::FixVarChangeDiff, helper::DomainChangeTrackerHelper)

Apply all fix variable changes in a difference to an optimization model.

# Arguments
- `backend`: The optimization model backend
- `diff::FixVarChangeDiff`: The fix variable change difference to apply
- `helper::DomainChangeTrackerHelper`: Helper object with constraint mappings

# Returns
Nothing.
"""
function apply_change!(backend, diff::FixVarChangeDiff, helper::DomainChangeTrackerHelper)
    for change in values(diff.fix_vars)
        apply_change!(backend, change, helper)
    end
    for change in values(diff.unfix_vars)
        apply_change!(backend, change, helper)
    end
    return
end

"""
    FixVarChangeTracker <: AbstractMathOptStateTracker

Tracker for changes to variable fixations in an optimization model.
"""
struct FixVarChangeTracker <: AbstractMathOptStateTracker end

"""
    root_state(::FixVarChangeTracker, backend)

Create the root state for fix variable change tracking.

# Arguments
- `::FixVarChangeTracker`: The fix variable change tracker
- `backend`: The optimization model backend

# Returns
A `ModelState` with empty forward and backward differences.
"""
root_state(::FixVarChangeTracker, backend) = ModelState(FixVarChangeDiff(), FixVarChangeDiff())

"""
    new_state(::FixVarChangeTracker, backward::FixVarChangeDiff, forward::FixVarChangeDiff)

Create a new model state with the given backward and forward fix variable change differences.

# Arguments
- `::FixVarChangeTracker`: The fix variable change tracker
- `backward::FixVarChangeDiff`: The backward difference for the new state
- `forward::FixVarChangeDiff`: The forward difference for the new state

# Returns
A new `ModelState` with the specified differences.
"""
new_state(::FixVarChangeTracker, backward::FixVarChangeDiff, forward::FixVarChangeDiff) = ModelState(backward, forward)

"""
    transform_model!(::FixVarChangeTracker, backend)

Transform a model for fix variable change tracking by creating a helper that maps
variables to their bound constraints.

# Arguments
- `::FixVarChangeTracker`: The fix variable change tracker
- `backend`: The optimization model backend

# Returns
A `DomainChangeTrackerHelper` with mappings between variables and their bound constraints.
"""
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