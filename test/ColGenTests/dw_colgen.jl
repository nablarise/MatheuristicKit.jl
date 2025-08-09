# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_setup_reformulation_with_artificial_variables()
    # Create a JuMP master problem with different constraint types
    master = Model(GLPK.Optimizer)
    
    # Variables: x1, x2
    @variable(master, x1 >= 0)
    @variable(master, x2 >= 0)
    
    # Regular constraints
    @constraint(master, eq_constraint, x1 + x2 == 5.0)          # Equality constraint  
    @constraint(master, leq_constraint, x1 + x2 <= 10.0)        # ≤ constraint
    @constraint(master, geq_constraint, x1 - x2 >= 2.0)         # ≥ constraint
    
    # Convexity constraints (these should get higher cost artificial variables)
    @constraint(master, conv_leq_constraint, 0 <= 3.0)         # Convexity ≤ constraint
    @constraint(master, conv_geq_constraint, 0 >= 0.0)         # Convexity ≥ constraint
    
    # Set objective: minimize x1 + 2*x2
    @objective(master, Min, x1 + 2*x2)
    
    # Create RK.DantzigWolfeReformulation with convexity constraints
    subproblems = Dict{Any, Model}()  # Empty subproblems
    
    # Map convexity constraints (simulate what ReformulationKit would do)
    convexity_constraints_lb = Dict(:subproblem1 => conv_geq_constraint)  # ≥ constraint
    convexity_constraints_ub = Dict(:subproblem1 => conv_leq_constraint)  # ≤ constraint
    
    reformulation = RK.DantzigWolfeReformulation(
        master, 
        subproblems, 
        convexity_constraints_lb, 
        convexity_constraints_ub
    )
    
    # Create context and phase
    context = MK.ColGen.DantzigWolfeColGenImpl(reformulation)
    phase = MK.ColGen.MixedPhase1and2(1000.0, 10000.0)  # Regular cost = 1000.0, Convexity cost = 10000.0
    
    # Get master MOI backend for verification
    master_moi = JuMP.backend(master)
    
    # Call setup_reformulation! - this should add artificial variables
    MK.ColGen.setup_reformulation!(context, phase)
    
    # Get the master provider to access artificial variables
    master_provider = context.master_provider
    
    # Verify artificial variables were stored in tracking dictionaries
    @test length(master_provider.eq_art_vars) == 1    # 1 equality constraint
    @test length(master_provider.leq_art_vars) == 2   # 2 ≤ constraints (regular + convexity)
    @test length(master_provider.geq_art_vars) == 2   # 2 ≥ constraints (regular + convexity)
    
    # Get constraint references to verify specific mappings
    eq_constraint_ref = JuMP.constraint_ref_with_index(master, MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(1))
    
    # Test that equality constraint has 2 artificial variables
    eq_constraint_moi_ref = JuMP.index(eq_constraint_ref)
    @test haskey(master_provider.eq_art_vars, eq_constraint_moi_ref)
    s_pos, s_neg = master_provider.eq_art_vars[eq_constraint_moi_ref]
    @test s_pos isa MOI.VariableIndex
    @test s_neg isa MOI.VariableIndex
    @test s_pos != s_neg
        
    # Verify coefficients in the equality constraint
    eq_constraint_func = MOI.get(master_moi, MOI.ConstraintFunction(), eq_constraint_moi_ref)
    terms_dict = Dict(term.variable => term.coefficient for term in eq_constraint_func.terms)
    @test terms_dict[s_pos] == 1.0   # Positive artificial variable
    @test terms_dict[s_neg] == -1.0  # Negative artificial variable

    # Verify coefficients of artificial variables in inequality constraints
    # ≤ constraint: x1 + x2 <= 10.0 should become x1 + x2 + s_leq <= 10.0
    leq_constraint_ref = JuMP.index(leq_constraint)
    leq_art_var = master_provider.leq_art_vars[leq_constraint_ref]
    leq_constraint_func = MOI.get(master_moi, MOI.ConstraintFunction(), leq_constraint_ref)
    leq_terms_dict = Dict(term.variable => term.coefficient for term in leq_constraint_func.terms)
    @test leq_terms_dict[leq_art_var] == -1.0  # Should be -1.0 for ≤ constraints
    
    # ≥ constraint: x1 - x2 >= 2.0 should become x1 - x2 - s_geq >= 2.0
    geq_constraint_ref = JuMP.index(geq_constraint)
    geq_art_var = master_provider.geq_art_vars[geq_constraint_ref]
    geq_constraint_func = MOI.get(master_moi, MOI.ConstraintFunction(), geq_constraint_ref)
    geq_terms_dict = Dict(term.variable => term.coefficient for term in geq_constraint_func.terms)
    @test geq_terms_dict[geq_art_var] == +1.0  # Should be +1.0 for ≥ constraints
    
    # Verify objective function includes artificial variables with correct costs
    obj_func = MOI.get(master_moi, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    obj_terms_dict = Dict(term.variable => term.coefficient for term in obj_func.terms)
    
    # Regular artificial variables should have cost = 1000.0
    @test obj_terms_dict[s_pos] == 1000.0
    @test obj_terms_dict[s_neg] == 1000.0
    
    # Verify convexity constraint artificial variables have higher cost (10000.0)
    conv_leq_ref = JuMP.index(conv_leq_constraint)
    conv_geq_ref = JuMP.index(conv_geq_constraint)
    
    @test haskey(master_provider.leq_art_vars, conv_leq_ref)
    @test haskey(master_provider.geq_art_vars, conv_geq_ref)
    
    conv_leq_art_var = master_provider.leq_art_vars[conv_leq_ref]
    conv_geq_art_var = master_provider.geq_art_vars[conv_geq_ref]
    
    @test obj_terms_dict[conv_leq_art_var] == 10000.0  # 10x higher cost
    @test obj_terms_dict[conv_geq_art_var] == 10000.0  # 10x higher cost
    
    # Verify coefficients of artificial variables in convexity constraints
    # Convexity ≤ constraint should have artificial variable with coefficient -1.0
    conv_leq_constraint_func = MOI.get(master_moi, MOI.ConstraintFunction(), conv_leq_ref)
    conv_leq_terms_dict = Dict(term.variable => term.coefficient for term in conv_leq_constraint_func.terms)
    @test conv_leq_terms_dict[conv_leq_art_var] == -1.0 
    
    # Convexity ≥ constraint should have artificial variable with coefficient +1.0
    conv_geq_constraint_func = MOI.get(master_moi, MOI.ConstraintFunction(), conv_geq_ref)
    conv_geq_terms_dict = Dict(term.variable => term.coefficient for term in conv_geq_constraint_func.terms)
    @test conv_geq_terms_dict[conv_geq_art_var] == +1.0
end

function test_dw_colgen()
    @testset "[dw_colgen] setup_reformulation! with artificial variables" begin
        test_setup_reformulation_with_artificial_variables()
    end
end