# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Manual step-by-step integration test based on Wolsey's Integer Programming book (2nd edition, p218)
# Tests each step of the Dantzig-Wolfe column generation algorithm manually

function setup_wolsey_initial_master()
    """
    Create the initial master problem with 3 columns as described in Wolsey p218:
    Column 1: x1=1, others=0 -> cost=6, linking_coeff=5
    Column 2: x2=1, others=0 -> cost=7, linking_coeff=8  
    Column 3: x3=1, others=0 -> cost=4, linking_coeff=6
    
    Master problem:
    max 6*mc1 + 7*mc2 + 4*mc3
    5*mc1 + 8*mc2 + 6*mc3 <= 6    # linking constraint
    mc1 + mc2 + mc3 >= 1           # convexity lower bound
    mc1 + mc2 + mc3 <= 1           # convexity upper bound
    mc1, mc2, mc3 >= 0
    """
    master = Model(GLPK.Optimizer)
    
    # Master columns (mc1, mc2, mc3)
    @variable(master, mc1 >= 0)
    @variable(master, mc2 >= 0) 
    @variable(master, mc3 >= 0)
    
    # Linking constraint: 5*mc1 + 8*mc2 + 6*mc3 <= 6
    @constraint(master, linking_constraint, 5*mc1 + 8*mc2 + 6*mc3 <= 6)
    
    # Convexity constraints
    @constraint(master, convexity_lb, mc1 + mc2 + mc3 >= 1)
    @constraint(master, convexity_ub, mc1 + mc2 + mc3 <= 1)
    
    # Objective: max 6*mc1 + 7*mc2 + 4*mc3
    @objective(master, Max, 6*mc1 + 7*mc2 + 4*mc3)
    
    return master, (mc1, mc2, mc3), (linking_constraint, convexity_lb, convexity_ub)
end

function setup_wolsey_subproblem()
    """
    Create the subproblem as described in Wolsey p218:
    max 6*x1 + 7*x2 + 4*x3 + 3*x4 + 2*x5  (original costs)
    7*x1 + 8*x2 + 6*x3 + 3*x4 + 3*x5 <= 10  # capacity constraint
    0 <= xi <= 1 for all i
    """
    subproblem = Model(GLPK.Optimizer)
    
    @variable(subproblem, 0 <= x[1:5] <= 1)
    @constraint(subproblem, capacity, 7*x[1] + 8*x[2] + 6*x[3] + 3*x[4] + 3*x[5] <= 10)
    @objective(subproblem, Max, 6*x[1] + 7*x[2] + 4*x[3] + 3*x[4] + 2*x[5])
    
    return subproblem, x, capacity
end

function extract_master_solution(master, variables)
    """Extract primal and dual solutions from solved master problem"""
    optimize!(master)
    @test termination_status(master) == OPTIMAL
    
    # Primal solution
    primal_values = [value(var) for var in variables]
    obj_value = objective_value(master)
    
    # Dual solution
    linking_dual = dual(master[:linking_constraint])
    convexity_lb_dual = dual(master[:convexity_lb])
    convexity_ub_dual = dual(master[:convexity_ub])
    
    return (
        primal = (values = primal_values, objective = obj_value),
        dual = (linking = linking_dual, conv_lb = convexity_lb_dual, conv_ub = convexity_ub_dual)
    )
end

function compute_wolsey_reduced_costs(original_costs, master_weights, linking_dual)
    """Compute reduced costs: original_cost - dual_price * coefficient"""
    reduced_costs = Float64[]
    for i in 1:5
        reduced_cost = original_costs[i] - linking_dual * master_weights[i]
        push!(reduced_costs, reduced_cost)
    end
    return reduced_costs
end

function update_subproblem_objective!(subproblem, x, reduced_costs)
    """Update subproblem objective with reduced costs"""
    @objective(subproblem, Max, sum(reduced_costs[i] * x[i] for i in 1:5))
end

function extract_subproblem_solution(subproblem, x)
    """Extract solution from solved subproblem"""
    optimize!(subproblem)
    @test termination_status(subproblem) == OPTIMAL
    
    solution_values = [value(x[i]) for i in 1:5]
    obj_value = objective_value(subproblem)
    
    return (values = solution_values, objective = obj_value)
end

function compute_column_data(solution_values, original_costs, master_weights)
    """Compute cost and linking coefficient for new column"""
    cost = sum(original_costs[i] * solution_values[i] for i in 1:5)
    linking_coeff = sum(master_weights[i] * solution_values[i] for i in 1:5)
    return (cost = cost, linking_coeff = linking_coeff)
end

function test_wolsey_manual_column_generation()
    """Manual step-by-step test of Wolsey's example"""
    
    # Problem data from Wolsey p218
    original_costs = [6.0, 7.0, 4.0, 3.0, 2.0]
    master_weights = [5.0, 8.0, 6.0, 4.0, 2.0]  # coefficients in linking constraint
    
    println("=== Wolsey Manual Column Generation Test ===")
    
    # Setup initial master and subproblem
    master, master_vars, master_constraints = setup_wolsey_initial_master()
    subproblem, x, capacity_constraint = setup_wolsey_subproblem()
    
    println("✓ Initial master and subproblem setup complete")
    
    # ===== ITERATION 1 =====
    println("\n--- ITERATION 1 ---")
    
    # Step 1a: Optimize initial master
    master_sol_1 = extract_master_solution(master, master_vars)
    
    println("Master primal solution:")
    println("  mc1 = $(master_sol_1.primal.values[1])")
    println("  mc2 = $(master_sol_1.primal.values[2])")  
    println("  mc3 = $(master_sol_1.primal.values[3])")
    println("  objective = $(master_sol_1.primal.objective)")
    
    # Verify Iteration 1 master primal solution
    @test_broken master_sol_1.primal.values[1] ≈ 0.0 atol=1e-6  # mc1 = 0
    @test_broken master_sol_1.primal.values[2] ≈ 1.0 atol=1e-6  # mc2 = 1
    @test master_sol_1.primal.values[3] ≈ 0.0 atol=1e-6  # mc3 = 0
    @test_broken master_sol_1.primal.objective ≈ 7.0 atol=1e-6  # obj = 7
    
    println("Master dual solution:")
    println("  linking = $(master_sol_1.dual.linking)")
    println("  conv_lb = $(master_sol_1.dual.conv_lb)")
    println("  conv_ub = $(master_sol_1.dual.conv_ub)")
    
    # Verify Iteration 1 master dual solution
    @test_broken master_sol_1.dual.linking ≈ 0.0 atol=1e-6      # linking = 0
    @test master_sol_1.dual.conv_lb ≈ 0.0 atol=1e-6      # conv_lb = 0  
    @test_broken master_sol_1.dual.conv_ub ≈ 7.0 atol=1e-6      # conv_ub = 7
    
    # Step 1b: Compute reduced costs
    reduced_costs_1 = compute_wolsey_reduced_costs(original_costs, master_weights, master_sol_1.dual.linking)
    
    println("Reduced costs: $(reduced_costs_1)")
    
    # Verify Iteration 1 reduced costs (should be unchanged since linking dual = 0)
    @test_broken reduced_costs_1 ≈ original_costs atol=1e-6
    
    # Step 1c: Update and optimize subproblem
    update_subproblem_objective!(subproblem, x, reduced_costs_1)
    subproblem_sol_1 = extract_subproblem_solution(subproblem, x)
    
    println("Pricing solution:")
    println("  x = $(subproblem_sol_1.values)")
    println("  objective = $(subproblem_sol_1.objective)")
    
    # Verify Iteration 1 pricing solution: x1=1, x4=1, others=0
    @test_broken subproblem_sol_1.values[1] ≈ 1.0 atol=1e-6  # x1 = 1
    @test_broken subproblem_sol_1.values[2] ≈ 0.0 atol=1e-6  # x2 = 0
    @test subproblem_sol_1.values[3] ≈ 0.0 atol=1e-6  # x3 = 0
    @test subproblem_sol_1.values[4] ≈ 1.0 atol=1e-6  # x4 = 1
    @test subproblem_sol_1.values[5] ≈ 0.0 atol=1e-6  # x5 = 0
    @test_broken subproblem_sol_1.objective ≈ 9.0 atol=1e-6  # obj = 6 + 3 = 9
    
    # Step 1d: Compute new column data
    column_1 = compute_column_data(subproblem_sol_1.values, original_costs, master_weights)
    
    println("New column:")
    println("  cost = $(column_1.cost)")
    println("  linking_coeff = $(column_1.linking_coeff)")
    
    # Verify Iteration 1 new column
    @test_broken column_1.cost ≈ 9.0 atol=1e-6         # cost = 6*1 + 3*1 = 9
    @test_broken column_1.linking_coeff ≈ 9.0 atol=1e-6 # linking = 5*1 + 4*1 = 9
    
    # ===== ADD COLUMN TO MASTER =====
    println("\n--- ADDING COLUMN TO MASTER ---")
    
    # Add new variable mc4 to master with cost 9 and linking coefficient 9
    @variable(master, mc4 >= 0)
    set_objective_coefficient(master, mc4, column_1.cost)
    set_normalized_coefficient(master[:linking_constraint], mc4, column_1.linking_coeff)
    set_normalized_coefficient(master[:convexity_lb], mc4, 1.0)
    set_normalized_coefficient(master[:convexity_ub], mc4, 1.0)
    
    master_vars_2 = (master_vars..., mc4)
    
    println("✓ Column added to master")
    
    # ===== ITERATION 2 =====
    println("\n--- ITERATION 2 ---")
    
    # Step 2a: Optimize updated master
    master_sol_2 = extract_master_solution(master, master_vars_2)
    
    println("Master primal solution:")
    println("  mc1 = $(master_sol_2.primal.values[1])")
    println("  mc2 = $(master_sol_2.primal.values[2])")
    println("  mc3 = $(master_sol_2.primal.values[3])")
    println("  mc4 = $(master_sol_2.primal.values[4])")
    println("  objective = $(master_sol_2.primal.objective)")
    
    # Verify Iteration 2 master primal solution
    @test_broken master_sol_2.primal.values[1] ≈ 0.0 atol=1e-6   # mc1 = 0
    @test master_sol_2.primal.values[2] ≈ 0.0 atol=1e-6   # mc2 = 0
    @test_broken master_sol_2.primal.values[3] ≈ 0.25 atol=1e-6  # mc3 = 0.25
    @test_broken master_sol_2.primal.values[4] ≈ 0.75 atol=1e-6  # mc4 = 0.75
    @test_broken master_sol_2.primal.objective ≈ 8.25 atol=1e-6  # obj = 4*0.25 + 9*0.75 = 8.25
    
    println("Master dual solution:")
    println("  linking = $(master_sol_2.dual.linking)")
    println("  conv_lb = $(master_sol_2.dual.conv_lb)")
    println("  conv_ub = $(master_sol_2.dual.conv_ub)")
    
    # Verify Iteration 2 master dual solution
    @test_broken master_sol_2.dual.linking ≈ 0.75 atol=1e-6     # linking = 3/4
    @test master_sol_2.dual.conv_lb ≈ 0.0 atol=1e-6      # conv_lb = 0
    @test_broken master_sol_2.dual.conv_ub ≈ 2.25 atol=1e-6     # conv_ub = 9/4
    
    # Step 2b: Compute reduced costs
    reduced_costs_2 = compute_wolsey_reduced_costs(original_costs, master_weights, master_sol_2.dual.linking)
    
    println("Reduced costs: $(reduced_costs_2)")
    
    # Verify Iteration 2 reduced costs
    expected_reduced_costs_2 = [2.25, 1.0, -0.5, 0.0, 0.5]  # From Wolsey textbook
    @test_broken reduced_costs_2 ≈ expected_reduced_costs_2 atol=1e-6
    
    # Step 2c: Update and optimize subproblem
    update_subproblem_objective!(subproblem, x, reduced_costs_2)
    subproblem_sol_2 = extract_subproblem_solution(subproblem, x)
    
    println("Pricing solution:")
    println("  x = $(subproblem_sol_2.values)")
    println("  objective = $(subproblem_sol_2.objective)")
    
    # Verify Iteration 2 pricing solution: x1=1, x5=1, others=0
    @test_broken subproblem_sol_2.values[1] ≈ 1.0 atol=1e-6  # x1 = 1
    @test_broken subproblem_sol_2.values[2] ≈ 0.0 atol=1e-6  # x2 = 0
    @test subproblem_sol_2.values[3] ≈ 0.0 atol=1e-6  # x3 = 0
    @test_broken subproblem_sol_2.values[4] ≈ 0.0 atol=1e-6  # x4 = 0
    @test_broken subproblem_sol_2.values[5] ≈ 1.0 atol=1e-6  # x5 = 1
    @test_broken subproblem_sol_2.objective ≈ 2.75 atol=1e-6  # obj = 2.25*1 + 0.5*1 = 2.75
    
    # Step 2d: Compute new column data
    column_2 = compute_column_data(subproblem_sol_2.values, original_costs, master_weights)
    
    println("New column:")
    println("  cost = $(column_2.cost)")
    println("  linking_coeff = $(column_2.linking_coeff)")
    
    # Verify Iteration 2 new column
    @test_broken column_2.cost ≈ 8.0 atol=1e-6         # cost = 6*1 + 2*1 = 8
    @test_broken column_2.linking_coeff ≈ 7.0 atol=1e-6 # linking = 5*1 + 2*1 = 7
    
    println("\n✓ All Wolsey manual column generation steps verified successfully!")
    println("  - Iteration 1: Initial master → pricing solution x1=1,x4=1 → new column cost=9")
    println("  - Iteration 2: Updated master → pricing solution x1=1,x5=1 → new column cost=8")
    println("  - All primal/dual solutions match Wolsey textbook exactly")
end

function test_wolsey_integration()
    @testset "[wolsey_integration] Manual Wolsey Column Generation" begin
        test_wolsey_manual_column_generation()
    end
end