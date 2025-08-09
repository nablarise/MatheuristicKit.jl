# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct ProjectedIpPrimalSol end

function check_primal_ip_feasibility!(::MasterPrimalSolution, ::DantzigWolfeColGenImpl, ::MixedPhase1and2)
    return ProjectedIpPrimalSol(), false
end

function update_inc_primal_sol!(::DantzigWolfeColGenImpl, ::Nothing, ::ProjectedIpPrimalSol)

end