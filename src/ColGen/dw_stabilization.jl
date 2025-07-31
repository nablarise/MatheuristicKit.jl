# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

setup_stabilization!(::DantzigWolfeColGenImpl, master) = nothing
update_stabilization_after_master_optim!(::NoStabilization, phase, ::MasterDualSolution) = false
get_stab_dual_sol(::NoStabilization, phase, dual_sol::MasterDualSolution) = dual_sol
update_stabilization_after_pricing_optim!(::NoStabilization, ::DantzigWolfeColGenImpl, ::SetOfColumns, ::JuMP.Model, ::Float64, ::MasterDualSolution) = nothing
check_misprice(::NoStabilization, ::SetOfColumns, ::MasterDualSolution) = false
update_stabilization_after_iter!(::NoStabilization, ::MasterDualSolution) = nothing
