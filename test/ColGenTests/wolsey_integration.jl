# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct WolseyTestData
    linking_constraint::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
    convexity_lb::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}
    convexity_ub::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
    mc_vars::Vector{MOI.VariableIndex}
end

struct SimpleMasterProvider
    master::MatheuristicKit.ColGen.Master
end

MatheuristicKit.ColGen.get_master(provider::SimpleMasterProvider) = provider.master
MatheuristicKit.ColGen.is_minimization(provider::SimpleMasterProvider) = MOI.get(MatheuristicKit.ColGen.moi_master(provider.master), MOI.ObjectiveSense()) != MOI.MAX_SENSE
function create_test_mappings(subproblem_moi, test_data)
    x_vars = MOI.get(subproblem_moi, MOI.ListOfVariableIndices())
    original_costs = [6.0, 7.0, 4.0, 3.0, 2.0]
    master_weights = [5.0, 8.0, 6.0, 4.0, 2.0]
    
    cost_map = Dict{MOI.VariableIndex, Float64}()
    for (i, var) in enumerate(x_vars)
        cost_map[var] = original_costs[i]
    end
    original_cost_mapping = ReformulationKit.OriginalCostMapping(cost_map)
    
    coeffs = Dict{MOI.VariableIndex, Vector{Tuple{DataType, Int64, Float64}}}()
    linking_constraint_type = typeof(test_data.linking_constraint)
    linking_constraint_value = test_data.linking_constraint.value
    
    for (i, var) in enumerate(x_vars)
        coeffs[var] = [(linking_constraint_type, linking_constraint_value, master_weights[i])]
    end
    coupling_mapping = ReformulationKit.CouplingConstraintMapping(coeffs)
    
    return coupling_mapping, original_cost_mapping
end

struct WolseyTestPricingProvider
    subproblem_moi::Any
    test_data::WolseyTestData
end

function MatheuristicKit.ColGen.get_pricing_subprobs(provider::WolseyTestPricingProvider)
    coupling_mapping, original_cost_mapping = create_test_mappings(provider.subproblem_moi, provider.test_data)
    
    pricing_subproblem = MatheuristicKit.ColGen.PricingSubproblem(
        provider.subproblem_moi,
        coupling_mapping,
        original_cost_mapping
    )
    
    return Dict{Any, Any}(1 => pricing_subproblem)
end

function setup_wolsey_master_colgen()
    master_model = Model(HiGHS.Optimizer)
    set_silent(master_model)
    
    @variable(master_model, mc1 >= 0)
    @variable(master_model, mc2 >= 0)
    @variable(master_model, mc3 >= 0)
    
    @constraint(master_model, linking, 5*mc1 + 8*mc2 + 6*mc3 <= 8)
    @constraint(master_model, convexity_lb, mc1 + mc2 + mc3 >= 1)
    @constraint(master_model, convexity_ub, mc1 + mc2 + mc3 <= 1)
    
    @objective(master_model, Max, 6*mc1 + 7*mc2 + 4*mc3)
    
    optimizer = JuMP.backend(master_model)
    linking_constraint = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(linking.index.value)
    convexity_lb_constraint = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}(convexity_lb.index.value)
    convexity_ub_constraint = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(convexity_ub.index.value)
    
    mc_vars = [MOI.VariableIndex(mc1.index.value), MOI.VariableIndex(mc2.index.value), MOI.VariableIndex(mc3.index.value)]
    
    convexity_constraints = Dict{Int64, Any}(1 => convexity_ub_constraint)
    convexity_constraints_lb = Dict{Int64, Any}(1 => convexity_lb_constraint)
    
    master = MatheuristicKit.ColGen.Master(
        optimizer,
        convexity_constraints,
        convexity_constraints_lb,
        Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}, Tuple{MOI.VariableIndex, MOI.VariableIndex}}(),
        Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}, MOI.VariableIndex}(),
        Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}, MOI.VariableIndex}()
    )
    
    test_data = WolseyTestData(
        linking_constraint,
        convexity_lb_constraint,
        convexity_ub_constraint,
        mc_vars
    )
    
    return master, test_data
end

function setup_wolsey_subproblem()
    subproblem_model = Model(HiGHS.Optimizer)
    set_silent(subproblem_model)
    
    @variable(subproblem_model, 0 <= x[1:5] <= 1, Bin)
    
    capacity_coeffs = [7.0, 8.0, 6.0, 3.0, 3.0]
    @constraint(subproblem_model, capacity, sum(capacity_coeffs[i] * x[i] for i in 1:5) <= 10)
    
    original_costs = [6.0, 7.0, 4.0, 3.0, 2.0]
    @objective(subproblem_model, Max, sum(original_costs[i] * x[i] for i in 1:5))
    
    optimizer = JuMP.backend(subproblem_model)
    x_vars = [MOI.VariableIndex(x[i].index.value) for i in 1:5]
    capacity_constraint = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(capacity.index.value)
    
    return optimizer, x_vars, capacity_constraint, original_costs
end

function _get_constraint_dual(dual_sol, constraint_ref)
    constraint_type = typeof(constraint_ref)
    constraint_value = constraint_ref.value
    
    if haskey(dual_sol.constraint_duals, constraint_type)
        constraint_dict = dual_sol.constraint_duals[constraint_type]
        if haskey(constraint_dict, constraint_value)
            return constraint_dict[constraint_value]
        end
    end
    return 0.0
end

function test_wolsey_manual_column_generation()
    original_costs = [6.0, 7.0, 4.0, 3.0, 2.0]
    master_weights = [5.0, 8.0, 6.0, 4.0, 2.0]
    
    master, test_data = setup_wolsey_master_colgen()
    subproblem_moi, x_vars, _, _ = setup_wolsey_subproblem()
    
    pricing_provider = WolseyTestPricingProvider(subproblem_moi, test_data)
    master_provider = SimpleMasterProvider(master)
    context = MatheuristicKit.ColGen.DantzigWolfeColGenImpl(master_provider, pricing_provider)
    
    result_1 = MatheuristicKit.ColGen.optimize_master_lp_problem!(master, context)
    @test !MatheuristicKit.ColGen.is_infeasible(result_1)
    @test !MatheuristicKit.ColGen.is_unbounded(result_1)
    
    primal_sol_1 = MatheuristicKit.ColGen.get_primal_sol(result_1)
    dual_sol_1 = MatheuristicKit.ColGen.get_dual_sol(result_1)
    
    @test primal_sol_1.variable_values[test_data.mc_vars[1]] ≈ 0.0 atol=1e-6
    @test primal_sol_1.variable_values[test_data.mc_vars[2]] ≈ 1.0 atol=1e-6
    @test primal_sol_1.variable_values[test_data.mc_vars[3]] ≈ 0.0 atol=1e-6
    @test primal_sol_1.obj_value ≈ 7.0 atol=1e-6
    
    linking_dual_1 = _get_constraint_dual(dual_sol_1, test_data.linking_constraint)
    convexity_lb_dual_1 = _get_constraint_dual(dual_sol_1, test_data.convexity_lb)
    convexity_ub_dual_1 = _get_constraint_dual(dual_sol_1, test_data.convexity_ub)
    
    @test linking_dual_1 ≈ -1/3 atol=1e-6
    @test convexity_lb_dual_1 ≈ 0.0 atol=1e-6
    @test convexity_ub_dual_1 ≈ -13/3 atol=1e-6
    
    phase = MatheuristicKit.ColGen.MixedPhase1and2()
    reduced_costs_result_1 = MatheuristicKit.ColGen.compute_reduced_costs!(context, phase, dual_sol_1)
    
    subproblem_reduced_costs = reduced_costs_result_1.values[1]
    reduced_costs_1 = [subproblem_reduced_costs[var] for var in x_vars]
    
    expected_reduced_costs_1 = [
        original_costs[1] - linking_dual_1 * master_weights[1], 
        original_costs[2] - linking_dual_1 * master_weights[2], 
        original_costs[3] - linking_dual_1 * master_weights[3], 
        original_costs[4] - linking_dual_1 * master_weights[4], 
        original_costs[5] - linking_dual_1 * master_weights[5]
    ]
    @test reduced_costs_1 ≈ expected_reduced_costs_1 atol=1e-6
    
    MatheuristicKit.ColGen.update_reduced_costs!(context, phase, reduced_costs_result_1)
    
    pricing_sp = MatheuristicKit.ColGen.get_pricing_subprobs(context)[1]
    sp_optimizer = MatheuristicKit.ColGen.SubproblemMoiOptimizer()
    pricing_result_1 = MatheuristicKit.ColGen.optimize_pricing_problem!(context, 1, pricing_sp, sp_optimizer, dual_sol_1, false)
    
    primal_sols_1 = MatheuristicKit.ColGen.get_primal_sols(pricing_result_1)
    subproblem_sol_1_values = [primal_sols_1[1].variable_values[var] for var in x_vars]
    subproblem_sol_1_objective = primal_sols_1[1].obj_value
    
    @test subproblem_sol_1_values[1] ≈ 1.0 atol=1e-6
    @test subproblem_sol_1_values[2] ≈ 0.0 atol=1e-6
    @test subproblem_sol_1_values[3] ≈ 0.0 atol=1e-6
    @test subproblem_sol_1_values[4] ≈ 1.0 atol=1e-6
    @test subproblem_sol_1_values[5] ≈ 0.0 atol=1e-6
    @test subproblem_sol_1_objective ≈ 12.0 atol=1e-6 
    
    generated_columns_1 = MatheuristicKit.ColGen.set_of_columns(context)
    
    for primal_sol in primal_sols_1
        result = MatheuristicKit.ColGen.push_in_set!(generated_columns_1, primal_sol)
        @test result == true
    end
    
    cols_inserted_1 = MatheuristicKit.ColGen.insert_columns!(context, phase, generated_columns_1)
    @test cols_inserted_1 == 1

    result_2 = MatheuristicKit.ColGen.optimize_master_lp_problem!(master, context)
    @test !MatheuristicKit.ColGen.is_infeasible(result_2)
    @test !MatheuristicKit.ColGen.is_unbounded(result_2)
    
    primal_sol_2 = MatheuristicKit.ColGen.get_primal_sol(result_2)
    dual_sol_2 = MatheuristicKit.ColGen.get_dual_sol(result_2)
    
    master = MatheuristicKit.ColGen.get_master(context)
    all_vars = MOI.get(MatheuristicKit.ColGen.moi_master(master), MOI.ListOfVariableIndices())
    
    @test length(all_vars) == 4
    @test primal_sol_2.obj_value > 7.0
    
    linking_dual_2 = _get_constraint_dual(dual_sol_2, test_data.linking_constraint)
    convexity_lb_dual_2 = _get_constraint_dual(dual_sol_2, test_data.convexity_lb)
    convexity_ub_dual_2 = _get_constraint_dual(dual_sol_2, test_data.convexity_ub)
    
    @test linking_dual_2 ≈ -0.75 atol=1e-6
    @test convexity_lb_dual_2 ≈ 0.0 atol=1e-6
    @test convexity_ub_dual_2 ≈ -2.25 atol=1e-6
    
    reduced_costs_result_2 = MatheuristicKit.ColGen.compute_reduced_costs!(context, phase, dual_sol_2)
    
    subproblem_reduced_costs_2 = reduced_costs_result_2.values[1]
    reduced_costs_2 = [subproblem_reduced_costs_2[var] for var in x_vars]
    
    linking_dual_raw_2 = -0.75
    expected_reduced_costs_2 = [
        6.0 - linking_dual_raw_2 * 5.0, 
        7.0 - linking_dual_raw_2 * 8.0, 
        4.0 - linking_dual_raw_2 * 6.0, 
        3.0 - linking_dual_raw_2 * 4.0, 
        2.0 - linking_dual_raw_2 * 2.0
    ]
    @test reduced_costs_2 ≈ expected_reduced_costs_2 atol=1e-6 
    
    MatheuristicKit.ColGen.update_reduced_costs!(context, phase, reduced_costs_result_2)
    
    pricing_result_2 = MatheuristicKit.ColGen.optimize_pricing_problem!(context, 1, pricing_sp, sp_optimizer, dual_sol_2, false)
    
    primal_sols_2 = MatheuristicKit.ColGen.get_primal_sols(pricing_result_2)
    subproblem_sol_2_values = [primal_sols_2[1].variable_values[var] for var in x_vars]
    subproblem_sol_2_obj = primal_sols_2[1].obj_value
    
    @test subproblem_sol_2_values[1] ≈ 1.0 atol=1e-6
    @test subproblem_sol_2_values[2] ≈ 0.0 atol=1e-6
    @test subproblem_sol_2_values[3] ≈ 0.0 atol=1e-6
    @test subproblem_sol_2_values[4] ≈ 1.0 atol=1e-6
    @test subproblem_sol_2_values[5] ≈ 0.0 atol=1e-6
    @test subproblem_sol_2_obj ≈ 15.75 atol=1e-6 
    
    generated_columns_2 = MatheuristicKit.ColGen.set_of_columns(context)
    
    for primal_sol in primal_sols_2
        result = MatheuristicKit.ColGen.push_in_set!(generated_columns_2, primal_sol)
        @test result == true
    end
    
    cols_inserted_2 = MatheuristicKit.ColGen.insert_columns!(context, phase, generated_columns_2)
    @test cols_inserted_2 == 1
end

function test_wolsey_integration()
    @testset "[wolsey_integration] Manual Wolsey Column Generation" begin
        test_wolsey_manual_column_generation()
    end
end