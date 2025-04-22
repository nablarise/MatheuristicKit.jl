# MathOptState

MathOptState is a Julia module for efficient state management in `MathOptInterface` models, particularly for tree search algorithms like branch-and-bound.

## Overview

Tree search algorithms offer a structured approach to exploring a problem's solution space. When developing matheuristics based on tree search algorithm on top of `MathOptInterface` models, navigating between model states is often required. Rebuilding the model from scratch at each node is computationally costly. `MathOptState` provides an interface (and some implementations) to implement efficient transitions between states by tracking and applying only the minimal set of necessary changes.

## Key Concepts

### ModelState

The core concept is the `ModelState` structure, which encapsulates the changes required to restore a specific model state:

```julia
struct ModelState{T}
    forward_diff::T
    backward_diff::T
end
```

- `backward_diff`: modifications to undo in order to revert the model to the root formulation
- `forward_diff`: modifications to reapply in order to reconstruct the formulation corresponding to the current state

Restoring from the root at every node can become inefficient in deep search trees. To mitigate this, one can designate checkpoint nodes at regular depths (e.g., every k levels). When switching between nodes, the algorithm first identifies the nearest common checkpoint ancestor and restores the model from there. This strategy strikes a practical balance between memory usage and computational efficiency (this is not implemented in this package yet!).

### State Transitions

To transition between states, use the `recover_state!` function:

```julia
recover_state!(backend, prev_state, next_state, helper)
```

This applies the backward difference of the previous state (to revert to root) and then the forward difference of the next state (to advance to the target state).

## Supported Change Types

MathOptState provides implementations for tracking several types of model changes:

1. **Variable Bounds Changes** (`DomainChangeTracker`)
   - Modify lower and upper bounds of variables

2. **Cut RHS Changes** (`CutsTracker`)
   - Modify the right-hand side of cut constraints

3. **Variable Fixation** (`FixVarChangeTracker`)
   - Fix variables to specific values
   - Unfix variables and restore their original bounds

## Usage Example

```julia
using JuMP, MathOptInterface, GLPK
using NablaMatheuristicKit

const MOI = MathOptInterface
const NMK = NablaMatheuristicKit

# Create a model
model = Model(GLPK.Optimizer)
@variable(model, 0 <= x <= 3)
@objective(model, Min, x)

# Create a state tracker
tracker = NMK.MathOptState.DomainChangeTracker()

# Transform the model for tracking (registers existing constraints)
helper = NMK.MathOptState.transform_model!(tracker, JuMP.backend(model))

# Get the root state
root_state = NMK.MathOptState.root_state(tracker, JuMP.backend(model))

optimize!(model)
@show JuMP.objective_value(model)
@show JuMP.value(x)

# Make changes to the model and create a new state
# In this new state, we want 2 <= x <= 3

# Forward change to move from the current state to the new state
lb_changes = Dict(JuMP.index(x).value => NMK.MathOptState.LowerBoundVarChange(JuMP.index(x), 2.0))
ub_changes = Dict()
forward_local_change = NMK.MathOptState.DomainChangeDiff(lb_changes, ub_changes)

# Merge with the forward change of the root state (local changes == global changes in this special case)
forward_change = NMK.MathOptState.merge_forward_change_diff(NMK.MathOptState.forward(root_state), forward_local_change)

# Backward change to move from the new state to the current state
lb_changes = Dict(JuMP.index(x).value => NMK.MathOptState.LowerBoundVarChange(JuMP.index(x), 0.0))
ub_changes = Dict()
backward_local_change = NMK.MathOptState.DomainChangeDiff(lb_changes, ub_changes)

# Merge with the backward change of the root state (local changes == global changes in this special case)
backward_change = NMK.MathOptState.merge_backward_change_diff(NMK.MathOptState.backward(root_state), backward_local_change)

# Create the new state
new_state = NMK.MathOptState.new_state(tracker, forward_change, backward_change)

# Navigate between states
NMK.MathOptState.recover_state!(JuMP.backend(model), root_state, new_state, helper)
optimize!(model)
@show JuMP.objective_value(model)
@show JuMP.value(x)

NMK.MathOptState.recover_state!(JuMP.backend(model), new_state, root_state, helper)
optimize!(model)
@show JuMP.objective_value(model)
@show JuMP.value(x)
```

## Extending MathOptState

To implement a new type of change tracking:

1. Define a concrete subtype of `AbstractAtomicChange`
2. Implement a corresponding difference type as a subtype of `AbstractMathOptStateDiff`
3. Create a tracker type as a subtype of `AbstractMathOptStateTracker`
4. Implement the required interface methods:
   - `apply_change!`
   - `merge_forward_change_diff`
   - `merge_backward_change_diff`
   - `root_state`
   - `new_state`
   - `transform_model!` (if needed)
