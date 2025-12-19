# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# This file has been refactored into focused test modules:
# - master_optimization_tests.jl: Master problem optimization tests
# - reduced_costs_tests.jl: Reduced cost computation and update tests
# - pricing_optimization_tests.jl: Pricing problem optimization and strategy tests
# - dual_bounds_tests.jl: Dual bound computation tests
# - column_insertion_tests.jl: Column cost computation and insertion tests
# - ip_management_tests.jl: Integer programming management tests
#
# All test functionality has been moved to these specialized modules.
# The `test_unit_solution()` function below is preserved for backward compatibility.

