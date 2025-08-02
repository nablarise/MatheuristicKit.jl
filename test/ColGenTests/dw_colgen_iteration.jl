# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_optimize_master_lp_primal_integration()
    # Simple integration test: create minimal LP and test that optimization works
    master_model = Model(GLPK.Optimizer)

    @variable(master_model, x >= 1.0)
    @constraint(master_model, x <= 5.0)
    @objective(master_model, Min, x)

    # Create minimal reformulation 
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )

    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    MK.ColGen.setup_reformulation!(context, MK.ColGen.MixedPhase1and2())

    # Test that optimization returns proper MasterSolution with dual solution
    master_solution = MK.ColGen.optimize_master_lp_problem!(MK.ColGen.get_master(context), context)

    @test master_solution isa MK.ColGen.MasterSolution

    primal_solution = MK.ColGen.get_primal_sol(master_solution)
    @test primal_solution isa MK.ColGen.MasterPrimalSolution
    @test primal_solution.obj_value == 1.0
    @test primal_solution.variable_values[JuMP.index(x)] == 1.0
end

function test_optimize_master_lp_dual_integration()
    # Simple integration test: create minimal LP and test that optimization works
    master_model = Model(GLPK.Optimizer)

    @variable(master_model, x >= 0)
    @variable(master_model, y >= 1)
    @constraint(master_model, cstr1, x <= 5.0)
    @constraint(master_model, cstr2, x + y == 5)
    @objective(master_model, Min, x + 3y)

    #

    # Create minimal reformulation 
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict{Any,Model}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )

    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    MK.ColGen.setup_reformulation!(context, MK.ColGen.MixedPhase1and2())

    # Test that optimization returns proper MasterSolution with dual solution
    master_solution = MK.ColGen.optimize_master_lp_problem!(MK.ColGen.get_master(context), context)

    @test master_solution isa MK.ColGen.MasterSolution

    dual_solution = MK.ColGen.get_dual_sol(master_solution)
    @test dual_solution isa MK.ColGen.MasterDualSolution
    @test dual_solution.obj_value == 7.0

    @test dual_solution.constraint_duals[MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}][JuMP.index(JuMP.LowerBoundRef(x)).value] == 0
    @test dual_solution.constraint_duals[MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}][JuMP.index(JuMP.LowerBoundRef(y)).value] == 2
    @test dual_solution.constraint_duals[MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}][JuMP.index(cstr1).value] == 0
    @test dual_solution.constraint_duals[MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}][JuMP.index(cstr2).value] == 1
end

function test_reduced_costs_computation_basic()
    # Test scenario:
    # - 1 subproblem with 5 variables
    # - 3 master constraints (≥, ≤, ==)
    # - Known coefficient matrix A and costs c
    # - Verify: reduced_costs = c - y^T × A
    
    # Test data:
    # Original costs c
    original_costs = [10.0, 15.0, 8.0, 20.0, 12.0]
    
    # Dual values y (3 constraints: ≥, ≤, ==)
    dual_values = [2.0, 1.5, 3.0]
    
    # Coefficient matrix A (3×5):
    A = [
        1.0  2.0  0.0  1.5  0.5;   # constraint 1 (≥)
        0.5  0.0  1.0  2.0  1.0;   # constraint 2 (≤)
        2.0  1.0  0.5  0.0  1.5    # constraint 3 (==)
    ]
    
    # Expected reduced costs = c - y^T × A
    expected_reduced_costs = original_costs - (dual_values' * A)'
    
    # Create mock MOI variable indices
    var_indices = [MOI.VariableIndex(i) for i in 1:5]
    
    # Create CouplingConstraintMapping with known coefficients
    coupling_mapping = RK.CouplingConstraintMapping()
    
    # Define constraint types (matching what MasterDualSolution would use)
    geq_constraint_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}
    leq_constraint_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
    eq_constraint_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
    

    # Add coefficients manually to the coupling mapping data structure
    # (bypassing the JuMP constraint reference requirement for this test)
    for (var_idx, var_index) in enumerate(var_indices)
        coefficients_for_var = Vector{Tuple{DataType, Int64, Float64}}()
        
        # Add coefficient for constraint 1 (≥) if non-zero
        if A[1, var_idx] != 0.0
            push!(coefficients_for_var, (geq_constraint_type, 1, A[1, var_idx]))
        end
        
        # Add coefficient for constraint 2 (≤) if non-zero
        if A[2, var_idx] != 0.0
            push!(coefficients_for_var, (leq_constraint_type, 2, A[2, var_idx]))
        end
        
        # Add coefficient for constraint 3 (==) if non-zero
        if A[3, var_idx] != 0.0
            push!(coefficients_for_var, (eq_constraint_type, 3, A[3, var_idx]))
        end
        
        coupling_mapping.data[var_index] = coefficients_for_var
    end
    
    # Create OriginalCostMapping
    cost_mapping = RK.OriginalCostMapping()
    for (var_idx, var_index) in enumerate(var_indices)
        cost_mapping.data[var_index] = original_costs[var_idx]
    end
    
    # Create minimal reformulation and context with proper subproblem extensions
    master_model = Model(GLPK.Optimizer)
    subproblem_model = Model()
    
    # Add the required extensions to the subproblem model
    subproblem_model.ext[:dw_coupling_constr_mapping] = coupling_mapping
    subproblem_model.ext[:dw_sp_var_original_cost] = cost_mapping
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict(1 => subproblem_model),  # subproblem dict with extensions
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)

    # Create MasterDualSolution with known dual values
    constraint_duals = Dict{Type{<:MOI.ConstraintIndex}, Dict{Int64, Float64}}()
    constraint_duals[geq_constraint_type] = Dict(1 => dual_values[1])
    constraint_duals[leq_constraint_type] = Dict(2 => dual_values[2])
    constraint_duals[eq_constraint_type] = Dict(3 => dual_values[3])
    
    mast_dual_sol = MK.ColGen.MasterDualSolution(0.0, constraint_duals)
    
    reduced_costs = MK.ColGen.compute_reduced_costs!(context, MK.ColGen.MixedPhase1and2(), mast_dual_sol)

    for var_index in var_indices
        @test reduced_costs.values[1][var_index] ≈ expected_reduced_costs[var_index.value] rtol=1e-10
    end
end

function test_update_reduced_costs_basic()
    # Test that update_reduced_costs! properly sets objective coefficients in subproblem
    # Test scenario:
    # - 1 subproblem with 3 variables in JuMP model
    # - Known reduced costs values
    # - Verify objective coefficients are updated correctly in the subproblem's MOI backend
    
    # Test data
    reduced_costs_values = [5.5, -2.3, 8.7]
    
    # Create minimal mappings (required for PricingSubproblem)
    coupling_mapping = RK.CouplingConstraintMapping()
    cost_mapping = RK.OriginalCostMapping()
    
    # Create minimal reformulation with JuMP subproblem
    master_model = Model(GLPK.Optimizer)
    subproblem_model = Model(GLPK.Optimizer)
    
    # Add variables to the JuMP subproblem
    @variable(subproblem_model, x[1:3])
    
    # Set initial objective with zero coefficients
    @objective(subproblem_model, Min, 0*x[1] + 0*x[2] + 0*x[3])
    
    # Get the actual MOI variable indices from JuMP
    var_indices = [JuMP.index(x[i]) for i in 1:3]
    
    # Add the required extensions to the subproblem model
    subproblem_model.ext[:dw_coupling_constr_mapping] = coupling_mapping
    subproblem_model.ext[:dw_sp_var_original_cost] = cost_mapping
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict(1 => subproblem_model),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Create ReducedCosts with known values using the actual MOI variable indices
    sp_reduced_costs = Dict{MOI.VariableIndex, Float64}()
    for (i, var_index) in enumerate(var_indices)
        sp_reduced_costs[var_index] = reduced_costs_values[i]
    end
    reduced_costs = MK.ColGen.ReducedCosts(Dict(1 => sp_reduced_costs))
    
    # Call update_reduced_costs!
    MK.ColGen.update_reduced_costs!(context, MK.ColGen.MixedPhase1and2(), reduced_costs)
    
    # Verify that objective coefficients were updated correctly
    # Get the MOI backend of the subproblem
    moi_backend = JuMP.backend(subproblem_model)
    updated_obj_func = MOI.get(moi_backend, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    
    # Check that each variable has the correct coefficient in the objective
    var1_ok = false
    var2_ok = false
    var3_ok = false

    for term in updated_obj_func.terms
        if term.variable == MOI.VariableIndex(1)
            var1_ok = term.coefficient == 5.5
        end
        if term.variable == MOI.VariableIndex(2)
            var2_ok = term.coefficient == -2.3
        end
        if term.variable == MOI.VariableIndex(3)
            var3_ok = term.coefficient == 8.7
        end
    end
    @test var1_ok
    @test var2_ok
    @test var3_ok
end

function test_compute_original_column_cost_basic()
    # Test scenario:
    # - 5 variables with known costs and values (including one with cost 0)
    # - Expected result: 2.5*1.0 + (-1.0)*3.0 + 0.0*5.0 + 4.0*2.0 + 0.0*1.5 = 2.5 - 3.0 + 0.0 + 8.0 + 0.0 = 7.5
    
    # Test data
    variable_costs = [2.5, -1.0, 0.0, 4.0, 0.0]  # Known costs (includes variable with cost 0)
    variable_values = [1.0, 3.0, 5.0, 2.0, 1.5]  # Known values
    expected_cost = 7.5
    
    # Create mock MOI variable indices
    var_indices = [MOI.VariableIndex(i) for i in 1:5]
    
    # Create OriginalCostMapping with known costs
    cost_mapping = RK.OriginalCostMapping()
    for (i, var_index) in enumerate(var_indices)
        cost_mapping.data[var_index] = variable_costs[i]
    end
    
    # Create PricingPrimalMoiSolution with known variable values
    variable_values_dict = Dict{MOI.VariableIndex, Float64}()
    for (i, var_index) in enumerate(var_indices)
        variable_values_dict[var_index] = variable_values[i]
    end
    
    column = MK.ColGen.PricingPrimalMoiSolution(
        1,  # subproblem_id
        -5.0,  # obj_value (reduced cost, not used in this test)
        variable_values_dict
    )
    
    # Call the function under test
    result = MK.ColGen._compute_original_column_cost(column, cost_mapping)
    
    # Verify result matches expected mathematical computation
    @test result ≈ expected_cost rtol=1e-10
end

function test_compute_master_constraint_membership_basic()
    # Test scenario:
    # - 3 variables with known values: [1.0, 2.0, 1.5]
    # - 3 coupling constraints (≥, ≤, ==) with known coefficients
    # - Coefficient matrix A (3×3):
    #   constraint 1 (≥): [2.0, 1.0, 0.0] → membership = 2.0*1.0 + 1.0*2.0 + 0.0*1.5 = 4.0
    #   constraint 2 (≤): [1.0, 0.0, 3.0] → membership = 1.0*1.0 + 0.0*2.0 + 3.0*1.5 = 5.5
    #   constraint 3 (==): [0.5, 2.0, 1.0] → membership = 0.5*1.0 + 2.0*2.0 + 1.0*1.5 = 6.0
    # - Plus convexity constraints with coefficient 1.0
    
    # Test data
    variable_values = [1.0, 2.0, 1.5]
    A = [
        2.0  1.0  0.0;   # constraint 1 (≥)
        1.0  0.0  3.0;   # constraint 2 (≤)
        0.5  2.0  1.0    # constraint 3 (==)
    ]
    expected_memberships = [4.0, 5.5, 6.0]  # A * x
    
    # Create mock MOI variable indices
    var_indices = [MOI.VariableIndex(i) for i in 1:3]
    
    # Create CouplingConstraintMapping with known coefficients
    coupling_mapping = RK.CouplingConstraintMapping()
    
    # Define constraint types
    geq_constraint_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}
    leq_constraint_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
    eq_constraint_type = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
    
    # Add coefficients to coupling mapping
    for (var_idx, var_index) in enumerate(var_indices)
        coefficients_for_var = Vector{Tuple{DataType, Int64, Float64}}()
        
        # Add coefficient for constraint 1 (≥) if non-zero
        if A[1, var_idx] != 0.0
            push!(coefficients_for_var, (geq_constraint_type, 101, A[1, var_idx]))
        end
        
        # Add coefficient for constraint 2 (≤) if non-zero
        if A[2, var_idx] != 0.0
            push!(coefficients_for_var, (leq_constraint_type, 102, A[2, var_idx]))
        end
        
        # Add coefficient for constraint 3 (==) if non-zero
        if A[3, var_idx] != 0.0
            push!(coefficients_for_var, (eq_constraint_type, 103, A[3, var_idx]))
        end
        
        coupling_mapping.data[var_index] = coefficients_for_var
    end
    
    # Create mock reformulation with convexity constraints
    master_model = Model(GLPK.Optimizer)
    
    # Create JuMP variables for convexity constraints
    @variable(master_model, λ >= 0)
    conv_ub_constraint = @constraint(master_model, λ <= 1)
    conv_lb_constraint = @constraint(master_model, λ >= 0)
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict(1 => Model()),
        Dict(1 => conv_lb_constraint),  # convexity_constraints_lb
        Dict(1 => conv_ub_constraint)   # convexity_constraints_ub
    )
    
    # Create PricingPrimalMoiSolution with known variable values
    variable_values_dict = Dict{MOI.VariableIndex, Float64}()
    for (i, var_index) in enumerate(var_indices)
        variable_values_dict[var_index] = variable_values[i]
    end
    
    column = MK.ColGen.PricingPrimalMoiSolution(
        1,  # subproblem_id
        -2.0,  # obj_value (reduced cost, not used in this test)
        variable_values_dict
    )
    
    # Call the function under test
    result = MK.ColGen._compute_master_constraint_membership(column, coupling_mapping, reformulation)
    
    # Verify coupling constraint memberships
    geq_constraint_ref = geq_constraint_type(101)
    leq_constraint_ref = leq_constraint_type(102)
    eq_constraint_ref = eq_constraint_type(103)
    
    @test haskey(result, geq_constraint_ref)
    @test haskey(result, leq_constraint_ref)
    @test haskey(result, eq_constraint_ref)
    
    @test result[geq_constraint_ref] ≈ expected_memberships[1] rtol=1e-10  # 4.0
    @test result[leq_constraint_ref] ≈ expected_memberships[2] rtol=1e-10  # 5.5
    @test result[eq_constraint_ref] ≈ expected_memberships[3] rtol=1e-10   # 6.0
    
    # Verify convexity constraints have coefficient 1.0
    conv_ub_ref = JuMP.index(conv_ub_constraint)
    conv_lb_ref = JuMP.index(conv_lb_constraint)
    
    @test haskey(result, conv_ub_ref)
    @test haskey(result, conv_lb_ref)
    @test result[conv_ub_ref] ≈ 1.0 rtol=1e-10
    @test result[conv_lb_ref] ≈ 1.0 rtol=1e-10
    
    # Additional test: empty variable values should return only convexity constraints
    empty_column = MK.ColGen.PricingPrimalMoiSolution(
        1,  # subproblem_id
        0.0,  # obj_value
        Dict{MOI.VariableIndex, Float64}()  # empty variable values
    )
    empty_result = MK.ColGen._compute_master_constraint_membership(empty_column, coupling_mapping, reformulation)
    
    # Should only have convexity constraints with coefficient 1.0
    @test length(empty_result) == 2  # Only convexity constraints
    @test haskey(empty_result, conv_ub_ref)
    @test haskey(empty_result, conv_lb_ref)
    @test empty_result[conv_ub_ref] ≈ 1.0 rtol=1e-10
    @test empty_result[conv_lb_ref] ≈ 1.0 rtol=1e-10
end

function test_unit_solution()
    @testset "[solution] integration test" begin
        test_optimize_master_lp_primal_integration()
        test_optimize_master_lp_dual_integration()
        test_reduced_costs_computation_basic()
        test_update_reduced_costs_basic()
        test_compute_original_column_cost_basic()
        test_compute_master_constraint_membership_basic()
    end
end