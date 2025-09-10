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
            
            # Objective: maximize metric with fallbacks: profit_per_bird -> avg_weight -> 1
            if 'profit_per_bird' in df.columns:
                objective_metric = df['profit_per_bird'].copy()
            else:
                objective_metric = None

            if objective_metric is None or objective_metric.isna().all():
                if 'avg_weight' in df.columns:
                    objective_metric = df['avg_weight'].copy()
                else:
                    objective_metric = pd.Series(1.0, index=df.index)
            else:
                if 'avg_weight' in df.columns:
                    objective_metric = objective_metric.fillna(df['avg_weight'])
                objective_metric = objective_metric.fillna(1.0)

            prob += lpSum([
                harvest_vars[i] * objective_metric.loc[i]
                for i in valid_indices
            ])
            
            # Capacity constraints (with optional bypass)
            total_harvest = lpSum([harvest_vars[i] for i in valid_indices])
            capacity_limit = max_total_stock
            prob += total_harvest <= capacity_limit
            if adjusted_min_stock > 0:
                prob += total_harvest >= adjusted_min_stock
            
            # Priority constraint: ensure minimum priority houses are always selected
            # Find minimum priority value (highest priority)
            min_priority = df['priority'].min() if not df.empty else 1
            min_priority_indices = df[df['priority'] == min_priority].index.tolist()
            
            # Add constraint: at least one minimum priority house must be harvested
            if min_priority_indices:
                prob += lpSum([harvest_vars[i] for i in min_priority_indices]) >= 1
            
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
            # Objective: maximize metric with fallbacks: profit_per_bird -> avg_weight -> 1
            if 'profit_per_bird' in df.columns:
                objective_metric = df['profit_per_bird'].copy()
            else:
                objective_metric = None

            if objective_metric is None or objective_metric.isna().all():
                if 'avg_weight' in df.columns:
                    objective_metric = df['avg_weight'].copy()
                else:
                    objective_metric = pd.Series(1.0, index=df.index)
            else:
                if 'avg_weight' in df.columns:
                    objective_metric = objective_metric.fillna(df['avg_weight'])
                objective_metric = objective_metric.fillna(1.0)

            prob += lpSum([
                harvest_vars[i] * objective_metric.loc[i]
                for i in valid_indices
            ])
            
            # Capacity constraints
            total_harvest = lpSum([harvest_vars[i] for i in valid_indices])
            capacity_limit = max_total_stock + 20000 if bypass_capacity == 1 else max_total_stock
            prob += total_harvest <= capacity_limit
            
            # Priority constraint: ensure minimum priority houses are always selected
            min_priority = df['priority'].min() if not df.empty else 1
            min_priority_indices = df[df['priority'] == min_priority].index.tolist()
            
            # Add constraint: at least one minimum priority house must be harvested
            if min_priority_indices:
                prob += lpSum([harvest_vars[i] for i in min_priority_indices]) >= 1
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