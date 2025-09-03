"""
Data models and configuration classes for the harvest optimization system.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional
from enum import Enum


class HarvestType(Enum):
    """Enumeration of harvest allocation types."""
    SLAUGHTERHOUSE = "SH"
    MARKET = "Market"
    CULL = "Cull"


@dataclass
class FarmConfig:
    """Configuration for farm structure."""
    count: int = 9
    houses_per_farm: int = 12
    house_area_m2: int = 1680


@dataclass
class WeightRange:
    """Weight range configuration for harvest types."""
    min: float
    max: float


@dataclass
class WeightRanges:
    """Weight ranges for different harvest types."""
    slaughterhouse: WeightRange
    market: WeightRange
    cull_weight: float


@dataclass
class Capacities:
    """Daily capacity constraints."""
    slaughterhouse_daily: int
    market_daily: int


@dataclass
class Constraints:
    """Harvest and density constraints."""
    min_density_kg_m2: float
    max_per_harvest_percentage: float
    cull_percentage: float


@dataclass
class OverallAllocationTargets:
    """Global allocation targets for the entire cycle."""
    slaughterhouse_percent: float
    market_percent: float


@dataclass
class Timing:
    """Timing constraints and parameters."""
    max_age_days: int
    cleaning_days: int
    safety_days: int


@dataclass
class WeeklyClosures:
    """Weekly closure days for different harvest types."""
    slaughterhouse: List[str]
    market: List[str]


@dataclass
class Holidays:
    """Holiday and closure configuration."""
    weekly_closures: WeeklyClosures
    national_holidays: List[str]


@dataclass
class OptimizationConfig:
    """Complete configuration for the optimization system."""
    farms: FarmConfig
    weight_ranges: WeightRanges
    capacities: Capacities
    constraints: Constraints
    overall_allocation_targets: OverallAllocationTargets
    timing: Timing
    holidays: Holidays

    @classmethod
    def create_default(cls) -> 'OptimizationConfig':
        """Create a default configuration matching the README specifications."""
        return cls(
            farms=FarmConfig(),
            weight_ranges=WeightRanges(
                slaughterhouse=WeightRange(min=1.55, max=1.70),
                market=WeightRange(min=1.90, max=2.20),
                cull_weight=1.2
            ),
            capacities=Capacities(
                slaughterhouse_daily=20000,
                market_daily=15000
            ),
            constraints=Constraints(
                min_density_kg_m2=25,
                max_per_harvest_percentage=0.70,
                cull_percentage=0.02
            ),
            overall_allocation_targets=OverallAllocationTargets(
                slaughterhouse_percent=0.30,
                market_percent=0.70
            ),
            timing=Timing(
                max_age_days=40,
                cleaning_days=15,
                safety_days=3
            ),
            holidays=Holidays(
                weekly_closures=WeeklyClosures(
                    slaughterhouse=["Thursday"],
                    market=["Friday"]
                ),
                national_holidays=[
                    "2025-01-07", "2025-01-25", "2025-04-20", "2025-04-21",
                    "2025-05-01", "2025-06-30", "2025-07-23"
                ]
            )
        )


@dataclass
class SlaughterHouseConfig:
    """Configuration specific to slaughterhouse harvesting."""
    min_weight: float
    max_weight: float
    min_stock: int
    max_stock: int
    max_pct_per_house: float
    optimizer_type: str = "base"  # Options: "base", "weight", "pct"


@dataclass
class MarketConfig:
    """Configuration specific to market harvesting."""
    min_weight: float
    max_weight: float
    min_stock: int
    max_stock: int
    max_pct_per_house: float
    tolerance_step: int
    max_tolerance: int


@dataclass
class ServiceConfig:
    """Service-level configuration."""
    duration_days: int
    max_harvest_date: str
    cull_adjustment: float
    cleaning_days: int = 15
    safety_days: int = 3
    culls_avg_weight: float = 1.2
    daily_mortality_rate: float = 0.001,  # 0.1% daily mortality rate for stock recalculation
    enable_stock_recalculation: bool = True  # Enable/disable expected stock recalculation after combined harvest


@dataclass
class HarvestEntry:
    """Represents a single harvest transaction."""
    farm: str
    date: datetime
    house: str
    age: int
    harvest_type: HarvestType
    harvest_stock: int
    expected_stock: int
    avg_weight: float
    net_meat: float
    expected_mortality: int
    expected_mortality_rate: float


@dataclass
class RejectionEntry:
    """Represents rejected/unharvested stock."""
    farm: str
    house: str
    date: datetime
    age: int
    expected_stock: int
    avg_weight: float
    rejection_reason: str


@dataclass
class SummaryEntry:
    """Summary statistics for farms/houses."""
    farm: str
    house: Optional[str]
    total_harvest_stock: int
    total_net_meat: float
    harvest_days: int
    unique_houses: int


@dataclass
class DailySummary:
    """Daily harvest summary."""
    date: datetime
    harvest_type: HarvestType
    total_stock: int
    total_net_meat: float
    houses_harvested: int