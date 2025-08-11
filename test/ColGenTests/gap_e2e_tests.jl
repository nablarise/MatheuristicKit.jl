# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    create_gap_instance_data()

Create the GAP instance data as specified in the requirements:
- 3 machines, 5 jobs
- Machine capacities: [15, 15, 16]
- Job consumption and costs per machine as given in the problem description

Returns:
- machines: index set 1:3
- jobs: index set 1:5  
- capacities: vector [15, 15, 16]
- consumption: matrix [3×5] consumption[machine, job]
- cost: matrix [3×5] cost[machine, job]
"""
function create_gap_instance_data()
    machines = 1:3
    jobs = 1:5
    
    # Machine capacities
    capacities = [15, 15, 16]
    
    # Resource consumption matrix: consumption[machine, job]
    consumption = [
        10  4  5  9  1;  # machine 1
         5  6  1  4  3;  # machine 2
        10  7  2  2  3   # machine 3
    ]
    
    # Cost matrix: cost[machine, job]
    cost = [
        3  7  1  1  4;   # machine 1
        4  6  7  8  4;   # machine 2
        4  5  7  4  2    # machine 3
    ]
    
    return machines, jobs, capacities, consumption, cost
end

"""
Test 1: Classic GAP Formulation with >= master constraints

This test verifies the basic column generation algorithm with:
- Capacity constraints in subproblems: sum(consumption * assignment) <= capacity
- Assignment constraints in master: sum(assignment) >= 1 (each job assigned to at least one machine)
- Minimization objective
- Expected dual bound: 13.0

Goal: Establish baseline functionality and verify >= master constraint handling
"""
function test_gap_e2e_classic()
    # Get GAP instance data
    machines, jobs, capacities, consumption, cost = create_gap_instance_data()
    
    # Build JuMP model from scratch
    model = Model(GLPK.Optimizer)
    set_silent(model)
    
    # Decision variables: assignment[machine, job] = 1 if job assigned to machine
    @variable(model, assignment[machine in machines, job in jobs], Bin)
    
    # Capacity constraints (will be in subproblems after decomposition)
    @constraint(model, knapsack[machine in machines], 
        sum(consumption[machine, job] * assignment[machine, job] for job in jobs) <= capacities[machine])
        
    # Assignment constraints (will be in master after decomposition)
    @constraint(model, coverage[job in jobs], 
        sum(assignment[machine, job] for machine in machines) >= 1)
    
    # Minimization objective
    @objective(model, Min, 
        sum(cost[machine, job] * assignment[machine, job] for machine in machines, job in jobs))
    
    # Apply Dantzig-Wolfe reformulation using existing annotation function
    reformulation = RK.dantzig_wolfe_decomposition(model, dw_annotation)
    
    # Set optimizers for master and subproblems
    JuMP.set_optimizer(RK.master(reformulation), GLPK.Optimizer)
    MOI.set(RK.master(reformulation), MOI.Silent(), true)
    
    for (sp_id, sp_model) in RK.subproblems(reformulation)
        JuMP.set_optimizer(sp_model, GLPK.Optimizer)
        set_silent(sp_model)
    end
    
    # Run column generation
    result = MK.ColGen.run_column_generation(reformulation)

    @test result isa MK.ColGen.ColGenOutput
    @test result.master_lp_obj ≈ 13
    @test result.incumbent_dual_bound ≈ 13
end

"""
Test 2: GAP with Constant in Objective Function

This test verifies column generation handles constant terms correctly:
- Same constraint structure as Test 1
- Objective function includes constant term +300
- Expected dual bound: 313.0 (13 + 300)

Goal: Ensure constant terms in objective are properly handled during decomposition
"""
function test_gap_e2e_with_constant()
    # Get GAP instance data
    machines, jobs, capacities, consumption, cost = create_gap_instance_data()
    
    # Build JuMP model from scratch
    model = Model(GLPK.Optimizer)
    set_silent(model)
    
    # Decision variables
    @variable(model, assignment[machine in machines, job in jobs], Bin)
    
    # Capacity constraints (subproblems)
    @constraint(model, knapsack[machine in machines], 
        sum(consumption[machine, job] * assignment[machine, job] for job in jobs) <= capacities[machine])
        
    # Assignment constraints (master)
    @constraint(model, coverage[job in jobs], 
        sum(assignment[machine, job] for machine in machines) >= 1)
    
    # Minimization objective WITH CONSTANT TERM (+2)
    @objective(model, Min, 
        300.0 + sum(cost[machine, job] * assignment[machine, job] for machine in machines, job in jobs))
    
    # Apply reformulation using existing annotation function
    reformulation = RK.dantzig_wolfe_decomposition(model, dw_annotation)
    
    # Set optimizers
    JuMP.set_optimizer(RK.master(reformulation), GLPK.Optimizer)
    MOI.set(RK.master(reformulation), MOI.Silent(), true)
    
    for (sp_id, sp_model) in RK.subproblems(reformulation)
        JuMP.set_optimizer(sp_model, GLPK.Optimizer)
        set_silent(sp_model)
    end
    
    # Run column generation
    result = MK.ColGen.run_column_generation(reformulation)

    @test result isa MK.ColGen.ColGenOutput
    @test result.master_lp_obj ≈ 313
    @test result.incumbent_dual_bound ≈ 313
end

"""
Test 3: Maximize Negative Cost (Max objective sense)

This test verifies column generation works with maximization problems:
- Same constraint structure as Test 1
- Maximize negative cost coefficients (equivalent to minimize positive costs)
- Expected dual bound: -13.0

Goal: Verify both MIN and MAX objective senses are supported
"""
function test_gap_e2e_maximize_negative()
    # Get GAP instance data
    machines, jobs, capacities, consumption, cost = create_gap_instance_data()
    
    # Build JuMP model from scratch
    model = Model(GLPK.Optimizer)
    set_silent(model)
    
    # Decision variables
    @variable(model, assignment[machine in machines, job in jobs], Bin)
    
    # Capacity constraints (subproblems)
    @constraint(model, knapsack[machine in machines], 
        sum(consumption[machine, job] * assignment[machine, job] for job in jobs) <= capacities[machine])
        
    # Assignment constraints (master)
    @constraint(model, coverage[job in jobs], 
        sum(assignment[machine, job] for machine in machines) >= 1)
    
    # MAXIMIZATION objective with NEGATIVE costs
    @objective(model, Max, 
        sum(-cost[machine, job] * assignment[machine, job] for machine in machines, job in jobs))
    
    # Apply reformulation using existing annotation function
    reformulation = RK.dantzig_wolfe_decomposition(model, dw_annotation)
    
    # Set optimizers
    JuMP.set_optimizer(RK.master(reformulation), GLPK.Optimizer)
    MOI.set(RK.master(reformulation), MOI.Silent(), true)
    
    for (sp_id, sp_model) in RK.subproblems(reformulation)
        JuMP.set_optimizer(sp_model, GLPK.Optimizer)
        set_silent(sp_model)
    end
    
    # Run column generation
    result = MK.ColGen.run_column_generation(reformulation)
    
    @test result isa MK.ColGen.ColGenOutput
    @test result.master_lp_obj ≈ -13
    @test result.incumbent_dual_bound ≈ -13
end

"""
Test 4: Equality Master Constraints

This test verifies column generation handles == constraints in master:
- Capacity constraints in subproblems: sum(consumption * assignment) <= capacity
- Assignment constraints in master: sum(assignment) == 1 (exactly one assignment per job)
- Minimization objective

Goal: Verify == master constraint handling and dual value computation
"""
function test_gap_e2e_equality_constraints()
    # Get GAP instance data
    machines, jobs, capacities, consumption, cost = create_gap_instance_data()
    
    # Build JuMP model from scratch
    model = Model(GLPK.Optimizer)
    set_silent(model)
    
    # Decision variables
    @variable(model, assignment[machine in machines, job in jobs], Bin)
    
    # Capacity constraints (subproblems)
    @constraint(model, knapsack[machine in machines], 
        sum(consumption[machine, job] * assignment[machine, job] for job in jobs) <= capacities[machine])
        
    # EQUALITY assignment constraints (master): exactly one assignment per job
    @constraint(model, coverage[job in jobs], 
        sum(assignment[machine, job] for machine in machines) == 1)
    
    # Minimization objective
    @objective(model, Min, 
        sum(cost[machine, job] * assignment[machine, job] for machine in machines, job in jobs))
    
    # Apply reformulation using existing annotation function
    reformulation = RK.dantzig_wolfe_decomposition(model, dw_annotation)
    
    # Set optimizers
    JuMP.set_optimizer(RK.master(reformulation), GLPK.Optimizer)
    MOI.set(RK.master(reformulation), MOI.Silent(), true)
    
    for (sp_id, sp_model) in RK.subproblems(reformulation)
        JuMP.set_optimizer(sp_model, GLPK.Optimizer)
        set_silent(sp_model)
    end
    
    # Run column generation
    result = MK.ColGen.run_column_generation(reformulation)
   
    @test result isa MK.ColGen.ColGenOutput
    @test result.master_lp_obj ≈ 13
    @test result.incumbent_dual_bound ≈ 13
end

"""
Test 5: Less-Than-Equal Master Constraints

This test verifies column generation handles <= master constraints correctly:
- Capacity constraints in subproblems: sum(consumption * assignment) <= capacity  
- Reformulated assignment constraints in master: sum(-1*assignment) <= -1
  (mathematically equivalent to sum(assignment) >= 1 but uses <= form)
- Minimization objective

Goal: Verify <= master constraint handling, dual value signs, and reduced cost computation
This ensures the algorithm correctly processes different constraint orientations.
"""
function test_gap_e2e_leq_master_constraints()
    # Get GAP instance data
    machines, jobs, capacities, consumption, cost = create_gap_instance_data()
    
    # Build JuMP model from scratch
    model = Model(GLPK.Optimizer)
    set_silent(model)
    
    # Decision variables
    @variable(model, assignment[machine in machines, job in jobs], Bin)
    
    # Capacity constraints (subproblems)
    @constraint(model, knapsack[machine in machines], 
        sum(consumption[machine, job] * assignment[machine, job] for job in jobs) <= capacities[machine])
        
    # LESS-THAN-EQUAL assignment constraints (master): sum(-1*assignment) <= -1
    # This is mathematically equivalent to sum(assignment) >= 1 but tests <= constraint handling
    @constraint(model, coverage[job in jobs], 
        sum(-1 * assignment[machine, job] for machine in machines) <= -1)
    
    # Minimization objective
    @objective(model, Min, 
        sum(cost[machine, job] * assignment[machine, job] for machine in machines, job in jobs))
    
    # Apply reformulation using existing annotation function
    reformulation = RK.dantzig_wolfe_decomposition(model, dw_annotation)
    
    # Set optimizers
    JuMP.set_optimizer(RK.master(reformulation), GLPK.Optimizer)
    MOI.set(RK.master(reformulation), MOI.Silent(), true)
    
    for (sp_id, sp_model) in RK.subproblems(reformulation)
        JuMP.set_optimizer(sp_model, GLPK.Optimizer)
        set_silent(sp_model)
    end
    
    # Run column generation
    result = MK.ColGen.run_column_generation(reformulation)
   
    @test result isa MK.ColGen.ColGenOutput
    @test result.master_lp_obj ≈ 13
    @test result.incumbent_dual_bound ≈ 13
end


"""
    create_gap_instance_data_with_penalties()

Create GAP instance data specifically designed for penalty variable testing:
- 3 machines, 5 jobs (same structure as standard GAP)
- Machine capacities: [15, 15, 16]
- Modified cost matrix with higher values to better test penalty trade-offs
- Job-specific penalty costs: [5, 100, 100, 5] where low penalties (jobs 1,5) 
  may favor leaving jobs unassigned vs high assignment costs

Returns:
- machines: index set 1:3
- jobs: index set 1:5  
- capacities: vector [15, 15, 16]
- consumption: matrix [3×5] consumption[machine, job] (same as standard)
- cost: matrix [3×5] cost[machine, job] (scaled up 10x from standard)
- penalties: vector [5, 100, 100, 5] penalty cost for leaving each job unassigned
"""
function create_gap_instance_data_with_penalties()
    machines = 1:3
    jobs = 1:5
    
    # Machine capacities
    capacities = [15, 15, 16]
    
    # Resource consumption matrix: consumption[machine, job]
    consumption = [
        10  4  5  9  1;  # machine 1
         5  6  1  4  3;  # machine 2
        10  7  2  2  3   # machine 3
    ]
    
    # Cost matrix: cost[machine, job]
    cost = [
        30  70  10  10  40;   # machine 1
        40  60  70  80  40;   # machine 2
        40  50  70  40  20    # machine 3
    ]

    penalties = [5, 100, 100, 100, 5]
    
    return machines, jobs, capacities, consumption, cost, penalties
end

"""
Test 6: GAP with Penalty Variables for Unassigned Jobs

This test verifies column generation handles mixed master/subproblem variables:
- Capacity constraints in subproblems: sum(consumption * assignment) <= capacity
- Assignment variables decomposed to subproblems
- Penalty variables (unassigned) remain in master problem
- Modified coverage constraints: sum(assignment) + unassigned >= 1
- Objective penalizes unassigned jobs with high penalty (100 per job)
- Expected dual bound: 13.0 (penalty high enough that all jobs assigned)

Goal: Verify mixed variable decomposition and master-only penalty variables
"""
function test_gap_e2e_with_penalty()
    # Get GAP instance data
    machines, jobs, capacities, consumption, cost, penalty_cost = create_gap_instance_data_with_penalties()
    
    # Build JuMP model from scratch
    model = Model(GLPK.Optimizer)
    set_silent(model)
    
    # Decision variables: assignment[machine, job] = 1 if job assigned to machine
    @variable(model, assignment[machine in machines, job in jobs], Bin)
    
    # Penalty variables: unassigned[job] = 1 if job is not assigned (MASTER VARIABLE)
    @variable(model, unassigned[job in jobs], Bin)
    
    # Capacity constraints (will be in subproblems after decomposition)
    @constraint(model, knapsack[machine in machines], 
        sum(consumption[machine, job] * assignment[machine, job] for job in jobs) <= capacities[machine])
        
    # Modified assignment constraints (will be in master after decomposition)
    # Each job must be either assigned to a machine OR marked as unassigned
    @constraint(model, coverage[job in jobs], 
        sum(assignment[machine, job] for machine in machines) + unassigned[job] >= 1)
    
    # Minimization objective with penalty for unassigned jobs
    @objective(model, Min, 
        sum(cost[machine, job] * assignment[machine, job] for machine in machines, job in jobs) +
        sum(penalty_cost[job] * unassigned[job] for job in jobs))
    
    # Apply Dantzig-Wolfe reformulation using existing annotation function
    reformulation = RK.dantzig_wolfe_decomposition(model, dw_annotation_with_penalty)
    
    # Set optimizers for master and subproblems
    JuMP.set_optimizer(RK.master(reformulation), GLPK.Optimizer)
    MOI.set(RK.master(reformulation), MOI.Silent(), true)
    
    for (sp_id, sp_model) in RK.subproblems(reformulation)
        JuMP.set_optimizer(sp_model, GLPK.Optimizer)
        set_silent(sp_model)
    end
    
    # Run column generation
    result = MK.ColGen.run_column_generation(reformulation)

    @test result isa MK.ColGen.ColGenOutput
    @test result.master_lp_obj ≈ 80 
    @test result.incumbent_dual_bound ≈ 80
end

"""
Main test function that runs all GAP E2E tests
"""
function test_gap_e2e_all()
    @testset "[GAP E2E] Test 1: Classic formulation" begin
        test_gap_e2e_classic()
    end
    
    @testset "[GAP E2E] Test 2: With constant in objective" begin
        test_gap_e2e_with_constant()
    end
    
    @testset "[GAP E2E] Test 3: Maximize negative cost" begin
        test_gap_e2e_maximize_negative()
    end
    
    @testset "[GAP E2E] Test 4: Equality master constraints" begin
        test_gap_e2e_equality_constraints()
    end
    
    @testset "[GAP E2E] Test 5: <= master constraints" begin
        test_gap_e2e_leq_master_constraints()
    end
    
    @testset "[GAP E2E] Test 6: With penalty variables" begin
        test_gap_e2e_with_penalty()
    end
end