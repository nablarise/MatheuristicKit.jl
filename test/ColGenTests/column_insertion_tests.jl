# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

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
        MK.ColGen.PrimalMoiSolution(-5.0, variable_values_dict),  # wrapped unified solution
        true  # is_improving (negative reduced cost for minimization)
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
        Dict(1 => JuMP.index(conv_lb_constraint)),  # convexity_constraints_lb
        Dict(1 => JuMP.index(conv_ub_constraint))   # convexity_constraints_ub
    )
    
    # Create PricingPrimalMoiSolution with known variable values
    variable_values_dict = Dict{MOI.VariableIndex, Float64}()
    for (i, var_index) in enumerate(var_indices)
        variable_values_dict[var_index] = variable_values[i]
    end
    
    column = MK.ColGen.PricingPrimalMoiSolution(
        1,  # subproblem_id
        MK.ColGen.PrimalMoiSolution(-2.0, variable_values_dict),  # wrapped unified solution
        true  # is_improving (negative reduced cost for minimization)
    )
    
    # Create a Master with the convexity constraints
    master = MK.ColGen.Master(
        nothing,  # moi_master not needed for this test
        reformulation.convexity_constraints_ub,
        reformulation.convexity_constraints_lb,
        Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}, Tuple{MOI.VariableIndex, MOI.VariableIndex}}(),
        Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}, MOI.VariableIndex}(),
        Dict{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}, MOI.VariableIndex}()
    )
    
    # Call the function under test
    result = MK.ColGen._compute_master_constraint_membership(column, coupling_mapping, master)
    
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
        MK.ColGen.PrimalMoiSolution(0.0, Dict{MOI.VariableIndex, Float64}()),  # wrapped unified solution
        false  # is_improving (zero reduced cost is not improving)
    )
    empty_result = MK.ColGen._compute_master_constraint_membership(empty_column, coupling_mapping, master)
    
    # Should only have convexity constraints with coefficient 1.0
    @test length(empty_result) == 2  # Only convexity constraints
    @test haskey(empty_result, conv_ub_ref)
    @test haskey(empty_result, conv_lb_ref)
    @test empty_result[conv_ub_ref] ≈ 1.0 rtol=1e-10
    @test empty_result[conv_lb_ref] ≈ 1.0 rtol=1e-10
end

function test_column_insertion_integration()
    # Test the full column insertion process
    master_model = Model(GLPK.Optimizer)
    @objective(master_model, Min, 0)
    
    # Create minimal subproblem with mappings
    coupling_mapping = RK.CouplingConstraintMapping()
    cost_mapping = RK.OriginalCostMapping()
    
    subproblem_model = Model()
    subproblem_model.ext[:dw_coupling_constr_mapping] = coupling_mapping
    subproblem_model.ext[:dw_sp_var_original_cost] = cost_mapping
    
    # Add some mock data to mappings
    var_idx = MOI.VariableIndex(1)
    coupling_mapping.data[var_idx] = []  # No coupling constraints for simplicity
    cost_mapping.data[var_idx] = 5.0  # Original cost
    
    reformulation = RK.DantzigWolfeReformulation(
        master_model,
        Dict(1 => subproblem_model),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    
    # Create a column to insert
    variable_values_dict = Dict(var_idx => 2.0)
    column = MK.ColGen.PricingPrimalMoiSolution(
        1,
        MK.ColGen.PrimalMoiSolution(-1.0, variable_values_dict),
        true
    )
    
    # Create column set with the column
    columns_to_insert = MK.ColGen.PricingPrimalMoiSolutionToInsert([column])
    
    # Test column insertion
    initial_vars = length(MOI.get(JuMP.backend(master_model), MOI.ListOfVariableIndices()))
    cols_inserted = MK.ColGen.insert_columns!(context, MK.ColGen.MixedPhase1and2(), columns_to_insert)
    final_vars = length(MOI.get(JuMP.backend(master_model), MOI.ListOfVariableIndices()))
    
    @test cols_inserted == 1
    @test final_vars == initial_vars + 1
end

function test_unit_column_insertion()
    @testset "[column_insertion] cost computation and insertion" begin
        test_compute_original_column_cost_basic()
        test_compute_master_constraint_membership_basic()
        test_column_insertion_integration()
    end
end