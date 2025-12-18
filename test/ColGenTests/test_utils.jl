# Test utilities for ColGen module
# Contains mock types and helper functions for testing

using MathOptInterface, ReformulationKit, JuMP
using MatheuristicKit.ColGen
const MOI = MathOptInterface
const RK = ReformulationKit
const MK = MatheuristicKit

# Mock callback implementation for testing
# struct MockCallbacks <: MK.ColGen.AbstractColumnGenerationCallbacks
#     cost_per_var::Dict{MOI.VariableIndex, Float64}
#     coeffs_per_var::Dict{MOI.VariableIndex, Vector{Tuple{MOI.ConstraintIndex, Float64}}}
# end

# # Default constructor with empty mappings
# MockCallbacks() = MockCallbacks(
#     Dict{MOI.VariableIndex, Float64}(),
#     Dict{MOI.VariableIndex, Vector{Tuple{MOI.ConstraintIndex, Float64}}}()
# )

# function MK.ColGen.compute_column_cost(callbacks::MockCallbacks, sp_id, solution)
#     cost = 0.0
#     for (var_idx, value) in solution.variable_values
#         cost += get(callbacks.cost_per_var, var_idx, 0.0) * value
#     end
#     return cost
# end

# function MK.ColGen.compute_column_coefficients(callbacks::MockCallbacks, sp_id, solution, master_info)
#     coeffs = Dict{MOI.ConstraintIndex, Float64}()
#     for (var_idx, value) in solution.variable_values
#         if haskey(callbacks.coeffs_per_var, var_idx)
#             for (constr_idx, coeff) in callbacks.coeffs_per_var[var_idx]
#                 coeffs[constr_idx] = get(coeffs, constr_idx, 0.0) + coeff * value
#             end
#         end
#     end
#     return coeffs
# end

# function MK.ColGen.compute_reduced_costs(callbacks::MockCallbacks, sp_id, master_dual_sol)
#     # For testing, return the original costs as reduced costs
#     # (assuming zero duals or already incorporated)
#     return callbacks.cost_per_var
# end

# Mock types for testing
struct MockMaster
    is_minimization::Bool
    convexity_constraints_ub::Dict{Any, Any}
    convexity_constraints_lb::Dict{Any, Any}
    eq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}, Tuple{MOI.VariableIndex, MOI.VariableIndex}}
    leq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}, MOI.VariableIndex}
    geq_art_vars::Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}, MOI.VariableIndex}
    
    function MockMaster(is_minimization::Bool = true,
                       convexity_ub::Dict{Any, Any} = Dict{Any, Any}(),
                       convexity_lb::Dict{Any, Any} = Dict{Any, Any}())
        eq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}, Tuple{MOI.VariableIndex, MOI.VariableIndex}}()
        leq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}, MOI.VariableIndex}()
        geq_art_vars = Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}, MOI.VariableIndex}()
        return new(is_minimization, convexity_ub, convexity_lb, eq_art_vars, leq_art_vars, geq_art_vars)
    end
end

struct MockPricingSubprobs
    subprobs::Dict{Any, MK.ColGen.PricingSubproblem}
    
    function MockPricingSubprobs(subprobs::Dict{Any, MK.ColGen.PricingSubproblem} = Dict{Any, MK.ColGen.PricingSubproblem}())
        return new(subprobs)
    end
end

# Mock provider interface methods
MK.ColGen.get_master(mock::MockMaster) = MK.ColGen.Master(
    nothing,  # moi_master - not needed for most tests
    mock.convexity_constraints_ub,
    mock.convexity_constraints_lb,
    mock.eq_art_vars,
    mock.leq_art_vars,
    mock.geq_art_vars
)

MK.ColGen.get_reform(mock::MockMaster) = nothing  # Not needed for most tests
MK.ColGen.is_minimization(mock::MockMaster) = mock.is_minimization
MK.ColGen.get_pricing_subprobs(mock::MockPricingSubprobs) = mock.subprobs

# Testing factory function
function create_for_testing(;
    is_minimization = true,
    convexity_ub = Dict{Any, Any}(),
    convexity_lb = Dict{Any, Any}(),
    pricing_subprobs = Dict{Any, MK.ColGen.PricingSubproblem}()
)
    master_provider = MockMaster(is_minimization, convexity_ub, convexity_lb)
    subprobs_provider = MockPricingSubprobs(pricing_subprobs)
    return MK.ColGen.DantzigWolfeColGenImpl(master_provider, subprobs_provider)
end