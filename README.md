
# Chicken Farm Harvest Plan Optimization System

A production-ready Python-based optimization system for planning chicken harvests across multiple farms. The system generates globally optimal daily harvest schedules that maximize total meat output while adhering to complex constraints.

## Features

- **Global Optimization**: Uses mixed-integer programming with PuLP solver
- **SOLID Architecture**: Clean, maintainable, and extensible code structure
- **Multi-Objective Optimization**: Maximizes meat output while minimizing unharvested stock
- **Comprehensive CSV Export**: Generates 8 detailed analysis files
- **Flexible Configuration**: Easily configurable constraints and parameters
- **Production Ready**: Robust error handling and logging
- **Age Expansion Scenarios**: Advanced optimization with increased harvest age for unharvested stock

## Quick Start

```python
from harvest.models import SlaughterHouseConfig, MarketConfig, ServiceConfig
from harvest.main import PoultryHarvestOptimizationService

# Configure optimization parameters
sh_config = SlaughterHouseConfig(
    min_weight=1.55, max_weight=1.70, min_stock=30000,
    max_stock=30000, max_pct_per_house=0.3
)

market_config = MarketConfig(
    min_weight=1.90, max_weight=2.20, min_stock=99900,
    max_stock=100000, max_pct_per_house=1.0,
    tolerance_step=1000, max_tolerance=10000
)

service_config = ServiceConfig(
    duration_days=40, max_harvest_date='2025-05-30', cull_adjustment=0.03
)

# Run optimization
service = PoultryHarvestOptimizationService(service_config)
service.sh_config = sh_config
service.market_config = market_config

result = service.run_full_optimization("growth_data.csv", "output")
```

## Age Expansion Optimization

The system now includes an advanced age expansion scenario that:

1. **Identifies Unharvested Stock**: After the initial market optimization, identifies farms and houses with remaining unharvested stock
2. **Selective Age Extension**: Increases the maximum harvest age by 3 days for only those specific farms and houses
3. **Re-optimization**: Runs another market optimization pass on the age-expanded data
4. **Combined Results**: Integrates the age expansion results with the main optimization output

This scenario is particularly useful when:
- Initial optimization leaves some stock unharvested due to age constraints
- You want to maximize harvest potential without affecting the entire dataset
- You need to balance between optimal timing and maximum yield

The age expansion results are exported to a separate `output_age_expansion` directory alongside the other optimization scenarios.

## Installation

```bash
pip install -r requirements.txt
```

1. Core Domain Model & System Logic
1.1. Farm & House Structure
Farms: 9 total farms.
Houses: 12 houses per farm
House Area: Each house has a fixed area of 1680,m2.
Placement Strategy: Chickens are placed in houses in staggered groups to ensure continuous supply.
1.2. Harvest Allocation Types
Slaughterhouse: Targets chickens within a specific weight range.
Market: Targets chickens within a different, higher weight range.
Culls: A separate, non-optional allocation.
1.3. Input Data
The system will ingest a growth_data.csv file with the columns:
farm, date, house, age, expected_mortality, expected_stock, expected_mortality_rate, avg_weight, farm_house_key
expected_stock is the number of live chickens at the start of a given day.
avg_weight is in kilograms.
For each farm_house_key there is approximately 40 days the stock should be updated for every row for each allocation
1.4. Optimization Objectives (Multi-Objective)
The primary objective for the solver is to Maximize: Total kilograms of meat harvested across all farms for the entire cycle. Secondary objectives can be modeled as costs or constraints to:
Minimize the number of harvest days.
Minimize unharvested stock.
1.5. Algorithm & Global Optimization Logic
Shift from Greedy to Global Optimization: Due to the Overall Cycle Allocation Targets, a simple day-by-day greedy algorithm is insufficient. Decisions for one house are dependent on decisions for all other houses. Therefore, the system must be implemented as a global optimization model.
Model Formulation: The problem must be formulated as a Mixed-Integer Programming (MIP) or Constraint Programming (CP) problem. A dedicated optimization solver (e.g., PuLP with a CBC/GLPK solver) is required to find the optimal solution.
Non negotiables: we must minimize the unharvested stock only for impossible allocation that we can leave unharvested stock but it would be best to override the  capacity constraint but this should be a last resort
Decision Variables: The model will have decision variables representing the quantity of chickens to harvest, for example: harvest_quantity[house][day][allocation_type].
Objective Function: The solver's primary goal will be to maximize the sum of harvest_quantity * avg_weight across all decision variables.
Constraints: All system constraints described in Section 2 must be encoded as mathematical constraints that the solver must satisfy.

2. System Constraints (User-Configurable)
All constraints will be fed into the optimization model.
2.1. Per-Harvest Constraints (Local Constraints)
These apply to each individual harvest transaction.
Weight Range: Chickens must be within the specified min/max weight for the allocation type.
Daily Capacity: The sum of all harvests on a given day cannot exceed the daily capacity for that allocation type.
Per-Harvest Percentage: A single harvest event cannot take more than a set percentage (e.g., 70%) of the house's current stock. This allows for multiple harvests from one house.
Density: The post-harvest density must remain above a minimum threshold (this can be a soft constraint).
2.2. Timing Constraints
House-Specific Maximum Age (Hard Deadline): No harvest can occur for a house if the chickens' age exceeds max_age_days.
Rule: Harvest Date ≤ House Placement Date + max_age_days
Overall Cycle Horizon (Planning Deadline): Defines the end of the simulation. The optimizer will not plan any activity beyond this date.
Rule: Planning End Date = First Placement Date of the Cycle + max_age_days + cleaning_days + safety_days
Facility Closures: No harvests can be scheduled for a given allocation type on its specified closure days (weekdays or holidays).
2.3. Overall Cycle Allocation Targets (Global Constraint)
This is a new, holistic constraint that governs the entire plan.
Rule: The total number of chickens allocated to a type (e.g., slaughterhouse) over the entire cycle, divided by the total number of harvested chickens, must equal the target percentage.
Example: Total Slaughterhouse Quantity / Total Harvested Quantity ≈ 0.30
Distinction: This is different from the per-harvest percentage. A single house might allocate 50% of its birds to the slaughterhouse, and another only 10%, as long as the entire plan balances to meet the overall 30% cycle target.

3. Code Structure & Design Requirements
3.1. Required Design
Model Builder Class: A dedicated class responsible for taking the input data and config, and building the mathematical model for the solver (defining variables, constraints, and the objective function).
Solver Class: A class that interfaces with the chosen solver library (e.g., ortools.sat.python.cp_model), runs the optimization, and checks the solution status.
Solution Parser Class: A class that takes the raw result from the solver and translates it into the structured data objects (HarvestEntry, RejectionEntry, etc.) needed for the CSV exporters.
SOLID Principles: The code should still adhere to SOLID principles for clarity and maintainability.
3.2. Class Structure (SOLID Examples)
Single Responsibility:
OptimizationModelBuilder: Only builds the solver model.
HarvestPlanSolver: Only executes the solver and returns the result.
SolutionProcessor: Only processes a valid solution into the required output formats.
Dependency Inversion: High-level modules should depend on abstractions. For instance, the main HarvestOptimizer class would coordinate these three components.

4. Input Parameter Structure (config object)
Python
config = {
    "farms": {
        "count": 9, "houses_per_farm": 12, "house_area_m2": 1680
    },
    "weight_ranges": {
        "slaughterhouse": {"min": 1.55, "max": 1.70},
        "market": {"min": 1.90, "max": 2.20},
        "cull_weight": 1.2
    },
    "capacities": {
        "slaughterhouse_daily": 20000, "market_daily": 15000
    },
    "constraints": {
        "min_density_kg_m2": 25,
        "max_per_harvest_percentage": 0.70, # Max % of a house's stock in one go
        "cull_percentage": 0.02
    },
    "overall_allocation_targets": { # New Global Constraint
        "slaughterhouse_percent": 0.30,
        "market_percent": 0.70
    },
    "timing": {
        "max_age_days": 40, "cleaning_days": 15, "safety_days": 3
    },
    "holidays": {
        "weekly_closures": {
            "slaughterhouse": ["Thursday"], "market": ["Friday"]
        },
        "national_holidays": [
            "2025-01-07", "2025-01-25", "2025-04-20", "2025-04-21", 
            "2025-05-01", "2025-06-30", "2025-07-23"
        ]
    }
}



5. Output Requirements: Multi-File CSV Export
The output requirements remain the same. The SolutionProcessor will be responsible for generating the data lists needed by the CSVExporter.
5.1. Required CSV File Schemas
(Schemas for the 8 CSVs: harvest_plan_master, harvest_plan_slaughterhouse, harvest_plan_market, harvest_plan_culls, summary_houses_farms, unharvested_stock, rejection_reasons, and daily_harvest_summary remain exactly as specified previously).
5.2. CSV Export Implementation
(The class structure for CSVExporter and HarvestPlanExporter, the main function generate_harvest_plan_csvs, and the CSV naming/configuration remain exactly as specified previously).

6. Additional Technical Requirements
(Requirements for Error Handling, Logging, Testing, Performance, and Documentation remain exactly as specified previously).
Final Deliverable
The expected deliverable is a complete Python project that implements the global optimization model described. It must correctly use an optimization library like Google OR-Tools to solve the harvest plan, satisfying all local and global constraints. The solution must then be processed and exported into the eight specified CSV files.


