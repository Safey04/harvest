"""
Model builder for the harvest optimization problem.
"""

import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, lpSum, LpBinary, LpContinuous, PULP_CBC_CMD
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
    
    def build_profit_market_model(
        self,
        df_day: pd.DataFrame,
        max_total_stock: int = 100000,
        min_total_stock: int = 99900,
        max_pct_per_house: float = 1.0,
        leftover_threshold: int = 2000
    ) -> LpProblem:
        """
        Build optimization model for market harvest with profit maximization focus.
        LP objective: minimize sum(cost_i * x_i)
        s.t. sum(x_i) = H, 0 <= x_i <= expected_stock_i
        Priority: lower 'priority' -> lower 'profit_loss' -> higher 'avg_weight'
        After solving, harvest any leftover <= leftover_threshold entirely, allowing overshoot only from this rule.
        
        Args:
            df_day: DataFrame for a specific day
            max_total_stock: Maximum total stock to harvest (target harvest amount H)
            min_total_stock: Minimum total stock to harvest (not used in new logic)
            max_pct_per_house: Maximum percentage per house (not used in new logic)
            leftover_threshold: Threshold for harvesting small leftovers entirely
            
        Returns:
            PuLP optimization problem with solved variables
        """
        # Filter for ready market rows
        df = df_day.copy()
        df = df[df['ready_Market'] == 1].reset_index(drop=True)
        
        if df.empty:
            logger.warning("No eligible rows for profit market harvest")
            return None
        
        # Check required columns
        need = {"farm", "house", "expected_stock", "priority", "profit_loss", "avg_weight"}
        miss = need - set(df.columns)
        if miss:
            logger.error(f"Missing required columns for profit harvest: {sorted(miss)}")
            return None

        H = int(max_total_stock)
        total_available = int(df["expected_stock"].sum())
        H = max(0, min(H, total_available))

        # Costs from stable sort by the required priority keys
        order_df = df.sort_values(["priority", "profit_loss", "avg_weight"], ascending=[True, True, False])
        order = order_df.index.to_list()
        cost = pd.Series({i: k + 1.0 for k, i in enumerate(order)}, dtype=float)

        # Solve LP; fallback to greedy in the same order if PuLP is unavailable
        harvest_vars = {}
        prob = None
        
        try:
            import pulp
            x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=float(df.at[i, "expected_stock"])) for i in df.index}
            prob = pulp.LpProblem("HarvestLP", pulp.LpMinimize)
            prob += pulp.lpSum(cost[i] * x[i] for i in df.index)
            prob += pulp.lpSum(x[i] for i in df.index) == float(H)
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            df["harvest_stock"] = [
                int(round(max(0.0, min(df.at[i, "expected_stock"], (pulp.value(x[i]) or 0.0))))) for i in df.index
            ]
            # Store variables for compatibility with solution processor
            harvest_vars = x
            logger.info(f"Profit harvest LP solved optimally")
        except Exception as e:
            logger.warning(f"LP solver failed ({e}), using greedy fallback")
            df["harvest_stock"] = 0
            remaining = H
            for i in order:
                if remaining <= 0:
                    break
                take = int(min(df.at[i, "expected_stock"], remaining))
                df.at[i, "harvest_stock"] = take
                remaining -= take
            
            # Create variables with greedy values for solution processor compatibility
            for i in df.index:
                harvest_vars[i] = LpVariable(f"x_{i}", lowBound=0, upBound=float(df.at[i, "expected_stock"]))
                harvest_vars[i].varValue = float(df.at[i, "harvest_stock"])

        # Base available
        df["available_stock"] = df["expected_stock"] - df["harvest_stock"]

        # Only overshoot via leftover rule
        mask_small_leftover = (df["available_stock"] > 0) & (df["available_stock"] <= leftover_threshold)
        df.loc[mask_small_leftover, "harvest_stock"] += df.loc[mask_small_leftover, "available_stock"]
        df.loc[mask_small_leftover, "available_stock"] = 0
        
        # Update variable values to reflect leftover harvesting for solution processor compatibility
        for i in df.index:
            if i in harvest_vars:
                harvest_vars[i].varValue = float(df.at[i, "harvest_stock"])

        logger.info(f"Profit harvest completed with total harvest: {df['harvest_stock'].sum()}")
        
        # Store for later use by solution processor
        self.problem = prob
        self.variables = harvest_vars
        self.data = df
        
        return prob
    
    def get_model_variables(self) -> Dict[str, Any]:
        """Get the model variables for solution processing."""
        return self.variables
    
    def get_model_data(self) -> pd.DataFrame:
        """Get the data used in model building."""
        return self.data if hasattr(self, 'data') else pd.DataFrame()