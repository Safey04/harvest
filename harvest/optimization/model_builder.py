"""
Model builder for the harvest optimization problem.
"""

import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpContinuous, PULP_CBC_CMD
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

from ..models import OptimizationConfig, HarvestType

logger = logging.getLogger(__name__)


class OptimizationModelBuilder:
    """
    Responsible for building the mathematical optimization model.
    Follows Single Responsibility Principle - only builds solver models.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize the model builder with configuration.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.problem = None
        self.variables = {}
        self.constraints = {}
    
    def flag_ready_avg_weight(
        self, 
        df: pd.DataFrame, 
        min_weight: float, 
        max_weight: float, 
        harvest_type: str
    ) -> pd.DataFrame:
        """
        Flag rows that meet weight requirements for harvest type using V5.1 logic.
        
        Args:
            df: Input DataFrame
            min_weight: Minimum weight threshold
            max_weight: Maximum weight threshold  
            harvest_type: Type of harvest ('SH' or 'Market')
            
        Returns:
            DataFrame with readiness flags added
        """
        from .slaughterhouse_optimizer_v5 import flag_ready_avg_weight as flag_ready_v5
        
        return flag_ready_v5(df, min_weight, max_weight, harvest_type)
    
    def flag_ready_daily_stock(
        self, 
        df: pd.DataFrame, 
        min_stock: int, 
        max_stock: int, 
        harvest_type: str
    ) -> pd.DataFrame:
        """
        Flag days that meet stock requirements for harvest type using V5.1 logic.
        
        Args:
            df: Input DataFrame with weight readiness flags
            min_stock: Minimum daily stock threshold
            max_stock: Maximum daily stock threshold
            harvest_type: Type of harvest ('SH' or 'Market')
            
        Returns:
            DataFrame with daily stock flags added
        """
        from .slaughterhouse_optimizer_v5 import flag_ready_daily_stock as flag_stock_v5
        
        return flag_stock_v5(df, min_stock, max_stock, harvest_type, min_per_house=2000)

    def flag_bypass_capacity(
        self,
        df: pd.DataFrame,
        min_slaughterhouse_weight: float,
        max_slaughterhouse_weight: float,
        unharvested_stock: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Flag rows that meet of unharvested stock for bypass constraint.
        """
        df = df.copy()
        # Ensure consistent date dtype for merge
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        unharvested = unharvested_stock.copy()
        unharvested['date'] = pd.to_datetime(unharvested['date']).dt.normalize()

        # Keep minimal columns from unharvested and rename to avoid collisions
        cols_to_keep = [c for c in ['farm', 'house'] if c in unharvested.columns]
        unharvested = unharvested[cols_to_keep]
        unharvested['bypass_capacity'] = 1

        # Merge and compute bypass flag when there is unharvested stock and weight is within bounds
        merged = df.merge(unharvested, on=['farm', 'house'], how='left')

        
        within_weight = merged['avg_weight'].between(min_slaughterhouse_weight, max_slaughterhouse_weight)
        merged['bypass_capacity'] = (within_weight).astype(int)
        merged.to_csv('merged_capacity.csv', index=False)
   

        return merged
    
    def flag_bypass_constraint(
        self,
        df: pd.DataFrame,
        min_slaughterhouse_weight: float,
        max_slaughterhouse_weight: float,
        unharvested_stock: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Flag rows that meet of unharvested stock for bypass constraint.
        """
        df = df.copy()
        # Ensure consistent date dtype for merge
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        unharvested = unharvested_stock.copy()
        unharvested['date'] = pd.to_datetime(unharvested['date']).dt.normalize()

        # Keep minimal columns from unharvested and rename to avoid collisions
        cols_to_keep = [c for c in ['farm', 'house'] if c in unharvested.columns]
        unharvested = unharvested[cols_to_keep]
        unharvested['bypass_constraint'] = 1

        # Merge and compute bypass flag when there is unharvested stock and weight is within bounds
        merged = df.merge(unharvested, on=['farm', 'house'], how='left')

        within_weight = merged['avg_weight'].between(min_slaughterhouse_weight - 0.1, max_slaughterhouse_weight + 0.1)
        merged['bypass_constraint'] = (within_weight).astype(int)
        merged.to_csv('merged.csv', index=False)

        return merged

    
    
    def distribute_slaughterhouse_uniform_pct(
        self, 
        df_day: pd.DataFrame, 
        min_total_stock: int = 30000,
        max_pct_per_house: float = 0.3
    ) -> pd.DataFrame:
        """
        Distribute slaughterhouse harvest using the new V5.1 optimizer.
        This function now delegates to the enhanced optimizer for better vacation day handling.
        
        Args:
            df_day: DataFrame for a specific day
            min_total_stock: Minimum total stock to harvest
            max_pct_per_house: Maximum percentage per house
            
        Returns:
            DataFrame with harvest allocations
        """
        from .slaughterhouse_optimizer_v5 import SH_min_houses_uniform_extra
        
        return SH_min_houses_uniform_extra(
            df_day=df_day,
            min_total_stock=min_total_stock,
            max_pct_per_house=max_pct_per_house,
            min_per_house=2000
        )
    
    def build_market_model(
        self,
        df_day: pd.DataFrame,
        max_total_stock: int = 100000,
        min_total_stock: int = 99900,
        max_pct_per_house: float = 1.0,
        capacity_penalty: float = 0,  # Big penalty for exceeding capacity
        bypass_capacity: int = 0
    ) -> LpProblem:
        """
        Build optimization model for market harvest using iterative relaxation of
        the minimum stock constraint (tolerance loop), mirroring the attached logic.
        
        Args:
            df_day: DataFrame for a specific day
            max_total_stock: Maximum total stock to harvest (soft constraint with penalty)
            min_total_stock: Minimum total stock to harvest
            max_pct_per_house: Maximum percentage per house
            capacity_penalty: Penalty coefficient for exceeding max capacity
            
        Returns:
            PuLP optimization problem
        """
        df = df_day.copy()
        df = df[df['ready_Market'] == 1].reset_index(drop=True)
        
        if df.empty:
            logger.warning("No eligible rows for market harvest")
            return None
        
        df['farm_house_key'] = df['farm'].astype(str) + "_" + df['house'].astype(str)
        valid_indices = df.index.tolist()
        
        # Tolerance-based relaxation of minimum total stock (attached logic)
        tolerance_step = 1000
        max_tolerance = 10000
        
        feasible_prob = None
        feasible_vars = None
        
        for tolerance in range(0, max_tolerance + tolerance_step, tolerance_step):
            adjusted_min_stock = max(min_total_stock - tolerance, 0)
            prob = LpProblem("MarketHarvest_MaximizeMeat", LpMaximize)
            
            # Decision variables per house up to allowed percentage
            harvest_vars = {}
            for i in valid_indices:
                max_harvest = df.loc[i, 'expected_stock'] * max_pct_per_house
                harvest_vars[i] = LpVariable(
                    f"harvest_{i}",
                    lowBound=0,
                    upBound=max_harvest,
                    cat=LpContinuous
                )
            
            # Objective: maximize total net meat
            prob += lpSum([harvest_vars[i] * df.loc[i, 'avg_weight'] for i in valid_indices])
            
            # Capacity constraints (with optional bypass)
            total_harvest = lpSum([harvest_vars[i] for i in valid_indices])
            capacity_limit = max_total_stock
            prob += total_harvest <= capacity_limit
            if adjusted_min_stock > 0:
                prob += total_harvest >= adjusted_min_stock
            
            # Solve to check feasibility under current tolerance
            status = prob.solve(PULP_CBC_CMD(msg=False))
            if status == 1:
                logger.info(f"Market model feasible with min_total_stock = {adjusted_min_stock}")
                feasible_prob = prob
                feasible_vars = harvest_vars
                break
            else:
                logger.warning(
                    f"Market model infeasible with min_total_stock = {adjusted_min_stock}; relaxing further"
                )
        else:
            # Final fallback — no lower bound constraint at all
            logger.error("All attempts failed — building model without min_total_stock constraint")
            prob = LpProblem("MarketHarvest_NoMinBound", LpMaximize)
            harvest_vars = {}
            for i in valid_indices:
                max_harvest = df.loc[i, 'expected_stock'] * max_pct_per_house
                harvest_vars[i] = LpVariable(
                    f"harvest_{i}",
                    lowBound=0,
                    upBound=max_harvest,
                    cat=LpContinuous
                )
            prob += lpSum([df.loc[i, 'avg_weight'] for i in valid_indices])
            total_harvest = lpSum([harvest_vars[i] for i in valid_indices])
            capacity_limit = max_total_stock + 20000 if bypass_capacity == 1 else max_total_stock
            prob += total_harvest <= capacity_limit
            status = prob.solve(PULP_CBC_CMD(msg=False))
            if status == 1:
                feasible_prob = prob
                feasible_vars = harvest_vars
            else:
                logger.error("Final fallback optimization failed for market model")
                return None
        
        # Store for later use
        self.problem = feasible_prob
        self.variables = feasible_vars
        self.data = df
        
        return feasible_prob
    
    def get_model_variables(self) -> Dict[str, Any]:
        """Get the model variables for solution processing."""
        return self.variables
    
    def get_model_data(self) -> pd.DataFrame:
        """Get the data used in model building."""
        return self.data if hasattr(self, 'data') else pd.DataFrame()
