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
        Distribute slaughterhouse harvest using uniform percentage approach.
        This ensures each house gets exactly one allocation and respects max_pct_per_house.
        
        Args:
            df_day: DataFrame for a specific day
            min_total_stock: Minimum total stock to harvest
            max_pct_per_house: Maximum percentage per house
            
        Returns:
            DataFrame with harvest allocations
        """
        df = df_day.reset_index(drop=True).copy()
        df['farm_house_key'] = df['farm'].astype(str) + "_" + df['house'].astype(str)
        
        # Filter only valid rows
        df = df[(df['avg_weight'] > 0) & (df['expected_stock'] > 0)]
        if df.empty:
            logger.warning("No valid houses available for slaughterhouse harvest")
            return pd.DataFrame()
        
        # Sort houses by increasing avg_weight (lowest weight first for slaughterhouse preference)
        df = df.sort_values(by='avg_weight').reset_index(drop=True)
        
        # Binary search to find the best uniform percentage that meets min_total_stock
        low = 0.0
        high = max_pct_per_house
        best_pct = 0.0
        
        # Binary search to find the best uniform percentage (same for all) that stays within target
        for _ in range(100):
            mid = (low + high) / 2
            total_harvest = (df['expected_stock'] * mid).sum()
            
            if total_harvest < min_total_stock:
                low = mid
            else:
                best_pct = mid
                high = mid
            
            if abs(total_harvest - min_total_stock) < 1:
                break
        
        # Apply the uniform percentage to all houses
        df['harvest_pct'] = best_pct
        df['harvest_stock'] = (df['expected_stock'] * df['harvest_pct']).clip(
            upper=(df['expected_stock'] * max_pct_per_house)
        )
        df['harvest_stock'] = df['harvest_stock'].astype(int)
        df['net_meat'] = df['harvest_stock'] * df['avg_weight']
        df['selected'] = (df['harvest_stock'] > 0).astype(int)
        df['harvest_type'] = 'SH'
        
        # Adjust total if needed to exactly match min_total_stock
        harvest_total = df['harvest_stock'].sum()
        if harvest_total > min_total_stock:
            overflow = harvest_total - min_total_stock
            # Start removing from highest-weight houses (least preferred)
            for idx in df.sort_values(by='avg_weight', ascending=False).index:
                if overflow <= 0:
                    break
                remove_qty = min(df.at[idx, 'harvest_stock'], overflow)
                df.at[idx, 'harvest_stock'] -= remove_qty
                df.at[idx, 'net_meat'] = df.at[idx, 'harvest_stock'] * df.at[idx, 'avg_weight']
                df.at[idx, 'selected'] = int(df.at[idx, 'harvest_stock'] > 0)
                overflow -= remove_qty
        
        # Return only selected harvests with proper column structure
        result_df = df[df['harvest_stock'] > 0].copy()
        if not result_df.empty:
            # Ensure proper column structure for consistency with other parts of the system
            result_columns = [
                'farm', 'date', 'house', 'age', 'expected_mortality', 'expected_stock',
                'expected_mortality_rate', 'avg_weight', 'selected', 'harvest_stock',
                'net_meat', 'harvest_type'
            ]
            
            # Rename columns to match expected format if needed
            column_mapping = {
                'expected_mortality': 'expected_mortality',
                'expected_mortality_rate': 'expected_mortality_rate'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in result_df.columns and new_col not in result_df.columns:
                    result_df = result_df.rename(columns={old_col: new_col})
            
            # Select only the required columns that exist
            existing_columns = [col for col in result_columns if col in result_df.columns]
            result_df = result_df[existing_columns]
        
        return result_df
    
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
