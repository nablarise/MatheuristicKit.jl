# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary



setup_stabilization!(::DantzigWolfeColGenImpl, master) = NoStabilization()
update_stabilization_after_master_optim!(::NoStabilization, phase, ::MasterDualSolution) = false
get_stab_dual_sol(::NoStabilization, phase, dual_sol::MasterDualSolution) = dual_sol
update_stabilization_after_pricing_optim!(::NoStabilization, ::DantzigWolfeColGenImpl, _, _, _, _) = nothing
check_misprice(::NoStabilization, _, _) = false
update_stabilization_after_iter!(::NoStabilization, ::MasterDualSolution) = nothing
