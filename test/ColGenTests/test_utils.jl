# Test utilities for ColGen module
# Contains mock types and helper functions for testing

using MathOptInterface, ReformulationKit, JuMP
using MatheuristicKit.ColGen
const MOI = MathOptInterface
const RK = ReformulationKit
const MK = MatheuristicKit

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