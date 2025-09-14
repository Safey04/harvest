"""
Main orchestrator for the harvest optimization system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any, Callable
import logging
import os
from datetime import datetime, timedelta

from .models import OptimizationConfig, SlaughterHouseConfig, MarketConfig, ServiceConfig
from .data_loader import DataLoader
from .optimization.model_builder import OptimizationModelBuilder
from .optimization.solver import HarvestPlanSolver
from .optimization.solution_processor import SolutionProcessor
from .export.csv_exporter import HarvestPlanExporter

logger = logging.getLogger(__name__)


def join_with_harvest_stock(df_original, market_harvest_df, removed_df=None, sh_plan=None):
    df = df_original.copy()
    mkt = market_harvest_df.copy()
    mkt.to_csv("mkt.csv", index=False)

    if 'harvest_type' not in mkt.columns:
        mkt['harvest_type'] = 'Market'
    else:
        mkt = mkt[mkt['harvest_type'] == 'Market']

    # Normalize dtypes
    for c in ["farm", "house"]:
        df[c] = df[c].astype(str)
        mkt[c] = mkt[c].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    mkt["date"] = pd.to_datetime(mkt["date"])

    mkt = mkt.groupby(["farm", "house", "date"], as_index=False)["harvest_stock"].sum()

    # Drop original rows before the first market date for the same farm+house
    min_mkt = (
        mkt.groupby(["farm", "house"], as_index=False)["date"]
           .min()
           .rename(columns={"date": "min_mkt_date"})
    )
    df = df.merge(min_mkt, on=["farm", "house"], how="left")
    # df = df[(df["min_mkt_date"].isna()) | (df["date"] >= df["min_mkt_date"])].drop(columns=["min_mkt_date"])

    # Left join only harvest_stock
    mkt_h = mkt[["farm", "house", "date", "harvest_stock"]].drop_duplicates(["farm", "house", "date"], keep="last")
    out = df.merge(mkt_h, on=["farm", "house", "date"], how="left")
    

    # Handle dates not in mkt but in df by fetching from removed_df
    if removed_df is not None and sh_plan is not None and not removed_df.empty and not sh_plan.empty:
        # Find dates that are in df but not in mkt
        df_dates = set(zip(df['farm'], df['house'], df['date']))
        mkt_dates = set(zip(mkt['farm'], mkt['house'], mkt['date']))
        missing_dates = mkt_dates - df_dates
        
        if missing_dates:
            # Convert removed_df to same format
            removed_df_copy = removed_df.copy().drop(columns=["start_date","end_date","last_date","max_harvest_date","removal_reason","farm_house_key"], errors='ignore')
            for c in ["farm", "house"]:
                removed_df_copy[c] = removed_df_copy[c].astype(str)
            removed_df_copy["date"] = pd.to_datetime(removed_df_copy["date"])
            removed_df_copy.to_csv("removed_df_copy.csv", index=False)
            
            # Filter removed_df to only include missing dates
            missing_data = []
            for farm, house, date in missing_dates:
                removed_rows = removed_df_copy[
                    (removed_df_copy['farm'] == farm) & 
                    (removed_df_copy['house'] == house) & 
                    (removed_df_copy['date'] == date)
                ]
                
                if not removed_rows.empty:
                    # Get harvest_stock from sh_plan for this farm+house+date
                    sh_harvest = sh_plan[
                        (sh_plan['farm'] == farm) & 
                        (sh_plan['house'] == house) & 
                        (sh_plan['date'] == date)
                    ]['harvest_stock'].sum()
                    
                    # Get harvest_stock from market_harvest_df for this farm+house+date
                    mkt_harvest = mkt[
                        (mkt['farm'] == farm) & 
                        (mkt['house'] == house) & 
                        (mkt['date'] == date)
                    ]['harvest_stock'].sum()
                    
                    # Subtract both harvest_stock values from expected_stock
                    for _, row in removed_rows.iterrows():
                        row_copy = row.copy()
                        row_copy['expected_stock'] = max(0, row_copy['expected_stock'] - sh_harvest)
                        # Add harvest_stock column with the total harvest amount
                        row_copy['harvest_stock'] = mkt_harvest
                        missing_data.append(row_copy)
            
            if missing_data:
                # Add missing data to output
                missing_df = pd.DataFrame(missing_data)
                out = pd.concat([out, missing_df], ignore_index=True)
                
                # Sort by farm, house, date
                out = out.sort_values(['farm', 'house', 'date']).reset_index(drop=True)
                out.to_csv("out.csv", index=False)

    return out


def propagate_harvest(df, market_harvest_df, finalize_threshold=2000, removed_df=None, sh_plan=None, unharvested_df=None,min_weight=1.9,max_weight=2.2):

    x = join_with_harvest_stock(market_harvest_df, df, removed_df, sh_plan)
    
    x["date"] = pd.to_datetime(x["date"])
    x["expected_stock"] = x["expected_stock"].astype(int)
    x["expected_mortality_rate"] = x["expected_mortality_rate"].fillna(0.0).astype(float)
    x["harvest_stock"] = x["harvest_stock"].fillna(0).round().astype(int)
    x = x.sort_values(["farm", "house", "date"]).reset_index(drop=True)
    x.to_csv("x.csv", index=False)


    x["Available Stock"] = 0
    rows_to_drop = []
    

    for (farm, house), idxs in x.groupby(["farm", "house"], sort=False).groups.items():
        idxs = sorted(idxs, key=lambda i: x.at[i, "date"])
        prev_available = None
        closed = False

        for idx in idxs:
            if closed:
                rows_to_drop.append(idx)
                continue

            # today's expected from yesterday's available, using today's mortality
            if prev_available is None:
                exp_today = int(x.at[idx, "expected_stock"])
            else:
                mr_today = float(x.at[idx, "expected_mortality_rate"])
                exp_today = int(round(prev_available * (1 - mr_today)))
                x.at[idx, "expected_stock"] = exp_today  # keep pre-harvest value
                x.at[idx, "net_meat"] = exp_today * float(x.at[idx, "avg_weight"])
                x.at[idx, "expected_mortality"] = abs(int(round(prev_available - exp_today)))

            # harvest and available
            hs = int(x.at[idx, "harvest_stock"])
            hs = max(0, min(hs, exp_today))  # cap to [0, expected]
            available = exp_today - hs

            if available < finalize_threshold:
                # finalize today: harvest all, keep expected_stock as pre-harvest
                x.at[idx, "harvest_stock"] = exp_today
                x.at[idx, "Available Stock"] = 0
                closed = True
                prev_available = 0
            else:
                x.at[idx, "harvest_stock"] = hs
                x.at[idx, "Available Stock"] = available
                prev_available = available

                
    x['harvest_type'] = 'Market'
    x = x.drop(index=rows_to_drop).reset_index(drop=True)
    x['net_meat'] = x['harvest_stock'] * x['avg_weight']
    harvest_df_progress = x
    x = x[x['harvest_stock'] > 0]

    # Create unharvested_df_left with corrected expected_stock calculation
    unharvested_df_left = harvest_df_progress[['farm', 'house', 'date', 'Available Stock', 'expected_mortality']].copy()
    
    # For each farm-house group, calculate the correct expected_stock
    # by looking ahead to future harvest events and adding back the mortality
    for (farm, house), group in unharvested_df_left.groupby(['farm', 'house']):
        group_sorted = group.sort_values('date').reset_index(drop=True)
        
        for i in range(len(group_sorted)):
            current_idx = group_sorted.index[i]
            
            # If this is the last date for this farm-house, use Available Stock as is
            if i == len(group_sorted) - 1:
                unharvested_df_left.at[current_idx, 'expected_stock'] = group_sorted.iloc[i]['Available Stock']
            else:
                # Look ahead to the next date's expected_stock and add back the mortality
                next_expected_stock = group_sorted.iloc[i + 1]['Available Stock']
                next_mortality = group_sorted.iloc[i + 1]['expected_mortality']
                corrected_expected_stock = next_expected_stock + next_mortality
                unharvested_df_left.at[current_idx, 'expected_stock'] = corrected_expected_stock
    
    unharvested_df_left = unharvested_df_left.drop(columns=['Available Stock'])

    unharvested_df = unharvested_df.drop(columns=['expected_mortality'])
    
    # Left merge unharvested_df with unharvested_df_left, prioritizing unharvested_df_left values for duplicate columns
    if unharvested_df is not None:
        # Perform left merge on common columns (farm, house, date)
        merge_columns = ['farm', 'house', 'date']
        unharvested_df = unharvested_df.merge(
            unharvested_df_left, 
            on=merge_columns, 
            how='left', 
            suffixes=('', '_left')
        )
        
        # For duplicate columns, use values from unharvested_df_left
        for col in unharvested_df_left.columns:
            if col in unharvested_df.columns and col != 'farm' and col != 'house' and col != 'date':
                if f'{col}_left' in unharvested_df.columns:
                    unharvested_df[col] = unharvested_df[f'{col}_left'].fillna(unharvested_df[col])
                    unharvested_df = unharvested_df.drop(columns=[f'{col}_left'])
    else:
        unharvested_df = unharvested_df_left

    return x.drop(columns=["Available Stock"]), harvest_df_progress, unharvested_df


class HarvestOptimizer:
    """
    Main orchestrator class that coordinates all optimization components.
    Follows Dependency Inversion Principle - depends on abstractions.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize the harvest optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.model_builder = OptimizationModelBuilder(config)
        self.solver = HarvestPlanSolver()
        self.solution_processor = SolutionProcessor()
        self.exporter = HarvestPlanExporter()
    
    def run_slaughterhouse_harvest_loop(
        self, 
        df_input: pd.DataFrame,
        start_date: str,
        min_weight: float,
        max_weight: float,
        min_stock: int,
        max_stock: int,
        num_days: int = 7,
        max_pct_per_house: float = 1.0,
        optimizer_type: str = "base"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run daily harvest loop for slaughterhouse using the new V5.2 optimizer.
        
        Args:
            df_input: Input DataFrame
            start_date: Start date for harvesting
            min_weight: Minimum weight threshold
            max_weight: Maximum weight threshold
            min_stock: Minimum stock threshold
            max_stock: Maximum stock threshold
            num_days: Number of days to run
            max_pct_per_house: Maximum percentage per house
            optimizer_type: Type of optimizer ("base", "weight", "pct")
            
        Returns:
            Tuple of (harvest results, updated DataFrame)
        """
        from .optimization.slaughterhouse_optimizer_v5 import (
            SH_run_daily_harvest_loop as sh_run_loop,
            get_optimizer_function
        )
        
        optimizer_fn = get_optimizer_function(optimizer_type)
        
        return sh_run_loop(
            df_input=df_input,
            optimizer_fn=optimizer_fn,
            start_date=start_date,
            min_weight=min_weight,
            max_weight=max_weight,
            min_stock=min_stock,
            max_stock=max_stock,
            num_days=num_days,
            max_pct_per_house=max_pct_per_house,
            min_per_house=2000
        )
    
    def run_multiple_slaughterhouse_starts(
        self,
        df_input: pd.DataFrame,
        min_weight: float,
        max_weight: float,
        min_stock: int,
        max_stock: int,
        max_pct_per_house: float = 1.0,
        optimizer_type: str = "base"
    ) -> Tuple[Dict[Any, Dict], Dict[Any, pd.DataFrame]]:
        """
        Run slaughterhouse harvest with multiple start dates using the new V5.2 optimizer.
        
        Args:
            df_input: Input DataFrame
            min_weight: Minimum weight threshold
            max_weight: Maximum weight threshold
            min_stock: Minimum stock threshold
            max_stock: Maximum stock threshold
            max_pct_per_house: Maximum percentage per house
            optimizer_type: Type of optimizer ("base", "weight", "pct")
            
        Returns:
            Tuple of (results by start day, updated DataFrames)
        """
        from .optimization.slaughterhouse_optimizer_v5 import (
            SH_run_multiple_harvest_starts as sh_run_multiple,
            get_optimizer_function
        )
        
        optimizer_fn = get_optimizer_function(optimizer_type)
        
        return sh_run_multiple(
            df_input=df_input,
            optimizer_fn=optimizer_fn,
            min_weight=min_weight,
            max_weight=max_weight,
            min_stock=min_stock,
            max_stock=max_stock,
            max_pct_per_house=max_pct_per_house,
            min_per_house=2000
        )
    

    
    def run_iterative_market_harvest(
        self,
        df_input: pd.DataFrame,
        max_total_stock: int = 100000,
        min_total_stock: int = 99000,
        tolerance_step: int = 1000,
        max_tolerance: int = 10000,
        harvest_type: str = 'Market',
        max_pct_per_house: float = 1.0,
        min_weight: float = 1.9,
        max_weight: float = 2.2,
        bypass_capacity: int = 0,
        bypass_constraint: int = 0,
        valid_days: pd.DataFrame = None,
        optimizer_type: str = "weight"

    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run iterative market harvest (backward from last date).
        
        Args:
            df_input: Input DataFrame
            max_total_stock: Maximum total stock
            min_total_stock: Minimum total stock
            tolerance_step: Tolerance step for relaxation
            max_tolerance: Maximum tolerance
            harvest_type: Type of harvest
            max_pct_per_house: Maximum percentage per house
            min_weight: Minimum weight
            max_weight: Maximum weight
            
        Returns:
            Tuple of (harvest plan, updated DataFrame)
        """
        df = df_input.copy()
        # Preserve bypass-capacity eligible rows if enabled; otherwise enforce ready_Market only

        df_flagged = self.model_builder.flag_ready_avg_weight(df, min_weight, max_weight, harvest_type)
        df = df_flagged[(df_flagged['ready_Market'] == 1)].copy()
            
        df['date'] = pd.to_datetime(df['date'])

        if valid_days is not None:
            if 'date' in valid_days.columns:
                valid_days = valid_days.copy()
                valid_days['date'] = pd.to_datetime(valid_days['date'])
        
        all_harvests = []
        
        if df.empty:
            logger.warning("No rows eligible for market harvest")
            return pd.DataFrame(), df
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        problem = None
        current_date = max_date
        while current_date >= min_date:
            day_name = current_date.day_name()
            if day_name == 'Friday':
                logger.info(f"Skipping {current_date.date()} (Friday)")
                df = df[df['date'] != current_date]
                current_date -= pd.Timedelta(days=1)
                continue
            
            logger.info(f"Harvesting from: {current_date.date()}")
            df_day = df[df['date'] == current_date].copy()

            # Exclude houses with insufficient expected stock for optimization
            df_day = df_day[df_day['expected_stock'] >= 1950].copy()
            if df_day.empty:
                logger.info(f"No houses with expected_stock >= 1950 on {current_date.date()} - skipping")
                current_date -= pd.Timedelta(days=1)
                continue

            # Build and solve market optimization, honoring per-day remaining capacity when provided
            problem = None
            if valid_days is None:
                if optimizer_type == "weight":
                    problem = self.model_builder.build_market_model(
                        df_day,
                        max_total_stock=max_total_stock,
                        min_total_stock=min_total_stock,
                        max_pct_per_house=max_pct_per_house,
                        bypass_capacity=bypass_capacity
                    )
                else:
                    problem = self.model_builder.build_profit_market_model(
                    df_day,
                    max_total_stock=max_total_stock,
                    min_total_stock=min_total_stock,
                    max_pct_per_house=max_pct_per_house,
                )
            else:
                vd_today = valid_days.loc[valid_days['date'] == current_date] if 'date' in valid_days.columns else pd.DataFrame()
                if not vd_today.empty:
                    remaining_capacity_value = int(vd_today['remaining_capacity'].iloc[0]) if 'remaining_capacity' in vd_today.columns else None
                    if remaining_capacity_value is not None:
                        if remaining_capacity_value == 0:
                            logger.info(f"No remaining capacity for {current_date.date()}, skipping day")
                            current_date -= pd.Timedelta(days=1)
                            continue
                        if remaining_capacity_value > 0:
                            logger.info(f"Remaining capacity for {current_date.date()}")
                            if optimizer_type == "weight":
                                problem = self.model_builder.build_market_model(
                                    df_day,
                                    max_total_stock=remaining_capacity_value,
                                    min_total_stock=2000,
                                    max_pct_per_house=max_pct_per_house,
                                    capacity_penalty=1000000,
                                    bypass_capacity=bypass_capacity
                                )
                            else:
                                problem = self.model_builder.build_profit_market_model(
                                    df_day,
                                    max_total_stock=remaining_capacity_value,
                                    min_total_stock=2000,
                                    max_pct_per_house=max_pct_per_house,
                                )

            
            if problem is None:
                logger.warning(f"Failed to build market model for {current_date.date()}")
                current_date -= pd.Timedelta(days=1)
                continue
            
            # Solve the market optimization problem
            # Note: No relaxation needed since we use penalty-based soft constraints
            status, obj_value = self.solver.solve_problem(problem)
            
            if not self.solver.is_optimal():
                logger.warning(f"Market optimization failed for {current_date.date()}")
                current_date -= pd.Timedelta(days=1)
                continue
            
            # Process solution
            variables = self.model_builder.get_model_variables()
            model_data = self.model_builder.get_model_data()
            
            harvest_df = self.solution_processor.process_market_solution(
                variables, model_data, harvest_type
            )
            
            if harvest_df.empty:
                logger.warning(f"No harvest possible on {current_date.date()}")
            else:

                harvest_df = harvest_df[harvest_df['harvest_stock'] >= 1950]
                
                df = self.solution_processor.update_stock_after_harvest_market(df, harvest_df)

                small_candidates = df[(df['date'] == current_date) & (df['expected_stock'] <= 1950)].copy()
                if not small_candidates.empty and {'farm', 'house'}.issubset(harvest_df.columns):
                    allocated_pairs = harvest_df[['farm', 'house']].drop_duplicates()
                    small_candidates = small_candidates.merge(
                        allocated_pairs,
                        on=['farm', 'house'],
                        how='inner'
                    )
                    total_small_stock = small_candidates['expected_stock'].sum()
                    if 0 < total_small_stock <= 5000:
                        cols_present = set(small_candidates.columns)
                        required_cols = {'farm', 'house', 'date', 'expected_stock'}
                        if required_cols.issubset(cols_present):
                            extra_df = small_candidates[['farm', 'house', 'date', 'expected_stock']].copy()
                            # Use avg_weight if available to compute net_meat
                            if 'avg_weight' in small_candidates.columns:
                                extra_df['avg_weight'] = small_candidates['avg_weight'].values
                            else:
                                # Fallback: use model_data default avg_weight per house if available
                                default_weight = None
                                if 'avg_weight' in model_data.get('parameters', {}):
                                    default_weight = model_data['parameters']['avg_weight']
                                extra_df['avg_weight'] = default_weight if default_weight is not None else min_weight
                            extra_df['harvest_stock'] = extra_df['expected_stock'].astype(int)
                            extra_df['harvest_type'] = harvest_type
                            extra_df['net_meat'] = extra_df['harvest_stock'] * extra_df['avg_weight']
                            extra_df = extra_df[['farm', 'house', 'date', 'harvest_type', 'harvest_stock', 'expected_stock', 'avg_weight', 'net_meat']]
                            df = self.solution_processor.update_stock_after_harvest_market(df, extra_df)
                            harvest_df = pd.concat([harvest_df, extra_df], ignore_index=True)

                            # Consolidate duplicate allocations for the same farm-house-date-harvest_type
                            group_keys = ['farm', 'house', 'date', 'harvest_type']
                            if set(group_keys).issubset(harvest_df.columns):
                                numeric_cols = [c for c in ['harvest_stock', 'net_meat'] if c in harvest_df.columns]
                                keep_cols = [c for c in harvest_df.columns if c not in numeric_cols]
                                # Compute weighted avg_weight if present
                                def _aggregate_group(g):
                                    out = g.iloc[0][keep_cols].copy()
                                    total_stock = g['harvest_stock'].sum() if 'harvest_stock' in g.columns else 0
                                    out['harvest_stock'] = total_stock
                                    if 'avg_weight' in g.columns and 'harvest_stock' in g.columns and total_stock > 0:
                                        weighted_avg = (g['avg_weight'] * g['harvest_stock']).sum() / total_stock
                                        out['avg_weight'] = weighted_avg
                                        out['net_meat'] = total_stock * weighted_avg
                                    elif 'net_meat' in g.columns:
                                        out['net_meat'] = g['net_meat'].sum()
                                    return out
                                harvest_df = (harvest_df
                                    .groupby(group_keys, as_index=False)
                                    .apply(_aggregate_group)
                                    .reset_index(drop=True))
            
                all_harvests.append(harvest_df)
            
            current_date -= pd.Timedelta(days=1)
        
        if all_harvests:
            final_harvest_plan = pd.concat(all_harvests, ignore_index=True)
        else:
            final_harvest_plan = pd.DataFrame()
        
        return final_harvest_plan, df
    
    def run_market_reverse_sweep(
        self,
        df_input: pd.DataFrame,
        max_total_stock: int = 100000,
        min_total_stock: int = 99000,
        tolerance_step: int = 1000,
        max_tolerance: int = 10000,
        harvest_type: str = 'Market',
        max_pct_per_house: float = 1.0,
        min_weight: float = 1.9,
        max_weight: float = 2.2,
        valid_days: pd.DataFrame = None,
        unharvested_stock: pd.DataFrame = None,
        bypass_capacity: int = 0,
        bypass_constraint: int = 0,
        optimizer_type: str = "weight",
        adjust_priority: Callable = None,
        feed_price: float = None,
        input_file_path_price: str = None

    ) -> Tuple[Dict[Any, Dict[str, pd.DataFrame]], Dict[Any, pd.DataFrame]]:
        """
        Run a reverse-sweep over the market window by progressively truncating the
        maximum date and re-running the iterative market harvest, then pick the best.
        
        Returns a tuple of (results_by_start_day, all_updated_dfs) where each entry
        in results_by_start_day has keys 'harvest' and 'updated_df'.
        """
        df_base = df_input.copy()
        
            
        df_flagged = self.model_builder.flag_ready_avg_weight(df_base, min_weight, max_weight, harvest_type)
        df_ready = df_flagged[(df_flagged['ready_Market'] == 1)].copy()
        
        df_ready = adjust_priority(df_ready, feed_price, input_file_path_price)
        df_ready.to_csv(f"df_ready_adjusted_priority.csv", index=False)
        
            

        df_ready['date'] = pd.to_datetime(df_ready['date'])
        
        results_by_start_day: Dict[Any, Dict[str, pd.DataFrame]] = {}
        all_updated_dfs: Dict[Any, pd.DataFrame] = {}
        
        if df_ready.empty:
            logger.warning("No eligible rows for market reverse sweep")
            return {}, {}
        
        full_max_date = df_ready['date'].max()
        full_min_date = df_ready['date'].min()
        current_max_date = full_max_date
        
        while current_max_date >= full_min_date:
            logger.info(f"Sweep window: {current_max_date.date()} → {full_min_date.date()}")
            df_subset = df_ready[df_ready['date'] <= current_max_date].copy()
            if bypass_capacity == 1:
                if df_subset['bypass_capacity'].sum() > 0:
                    bypass_capacity = 1
                else:
                    bypass_capacity = 0
            
            harvest_plan, updated_df = self.run_iterative_market_harvest(
                df_input=df_subset,
                max_total_stock=max_total_stock,
                min_total_stock=min_total_stock,
                tolerance_step=tolerance_step,
                max_tolerance=max_tolerance,
                harvest_type=harvest_type,
                max_pct_per_house=max_pct_per_house,
                min_weight=min_weight,
                max_weight=max_weight,
                bypass_capacity=bypass_capacity,
                bypass_constraint=bypass_constraint,
                valid_days=valid_days if valid_days is not None else None,
                optimizer_type=optimizer_type
            )
            
            results_by_start_day[current_max_date.date()] = {
                'harvest': harvest_plan,
                'updated_df': updated_df
            }
            all_updated_dfs[current_max_date.date()] = updated_df.copy()
            
            current_max_date -= pd.Timedelta(days=1)
        
        return results_by_start_day, all_updated_dfs
    
    def get_best_net_meat_plan(self, plan_dict: Dict,optimizer_type: str = None) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Find the best plan by total net meat using the new V5.1 optimizer.
        
        Args:
            plan_dict: Dictionary of plans by date
            
        Returns:
            Tuple of (best_date, best_harvest_df, best_updated_df, sorted_summary)
        """
        from .optimization.slaughterhouse_optimizer_v5 import get_best_harvest_stock_plan
        
        return get_best_harvest_stock_plan(plan_dict, optimizer_type)
    
    def calculate_culls_allocation(
        self, 
        df_input: pd.DataFrame, 
        cull_percentage: float = 0.03,
        culls_avg_weight: float = 1.2
    ) -> pd.DataFrame:
        """
        Calculate culls allocation for each house based on percentage of stock.
        
        Args:
            df_input: Input DataFrame with stock data
            cull_percentage: Percentage of stock to allocate as culls (e.g., 0.03 for 3%)
            culls_avg_weight: Average weight for culls in kg (default 1.2)
            
        Returns:
            DataFrame with culls allocation data
        """
        logger.info(f"Calculating culls allocation with {cull_percentage*100}% rate and {culls_avg_weight}kg avg weight")
        
        if df_input.empty:
            logger.warning("No data available for culls calculation")
            return pd.DataFrame()
        
        df = df_input.copy()
        
        # Ensure required columns exist
        required_columns = ['farm', 'house', 'date', 'age', 'expected_stock']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for culls calculation: {missing_columns}")
            return pd.DataFrame()
        
        # Calculate culls for each house
        culls_data = []
        
        # Group by house to calculate culls for each house
        for (farm, house), house_data in df.groupby(['farm', 'house']):
            # Use the latest date/age entry for each house
            latest_entry = house_data.sort_values('date').iloc[-2]
            
            # Calculate culls stock (percentage of expected stock)
            culls_stock = int(round(latest_entry['expected_stock'] * cull_percentage))
            
            if culls_stock > 0:  # Only include houses with culls
                culls_entry = {
                    'farm': farm,
                    'date': latest_entry['date'],
                    'house': house,
                    'age': latest_entry['age'],
                    'harvest_type': 'Culls',
                    'harvest_stock': culls_stock,
                    'expected_stock': latest_entry['expected_stock'],
                    'avg_weight': culls_avg_weight,
                    'net_meat': culls_stock * culls_avg_weight
                }
                culls_data.append(culls_entry)
        
        if not culls_data:
            logger.warning("No culls allocation calculated")
            return pd.DataFrame()
        
        culls_df = pd.DataFrame(culls_data)

        df_input['expected_stock'] = (1 - cull_percentage) * df_input['expected_stock'] 
        
        
        # Log summary
        total_culls_stock = culls_df['harvest_stock'].sum()
        total_culls_meat = culls_df['net_meat'].sum()
        houses_with_culls = len(culls_df)
        
        logger.info(f"Culls allocation calculated:")
        logger.info(f"  - Total culls stock: {total_culls_stock:,}")
        logger.info(f"  - Total culls meat: {total_culls_meat:,.1f} kg")
        logger.info(f"  - Houses with culls: {houses_with_culls}")
        
        return culls_df,df_input



            
class PoultryHarvestOptimizationService:
    """
    High-level service class that provides the main API for harvest optimization.
    """
    
    def __init__(self, service_config: ServiceConfig):
        """
        Initialize the optimization service.
        
        Args:
            service_config: Service configuration
        """
        self.service_config = service_config
        self.sh_config: Optional[SlaughterHouseConfig] = None
        self.market_config: Optional[MarketConfig] = None
        
        # Create default optimization config
        self.optimization_config = OptimizationConfig.create_default()
        
        # Initialize optimizer
        self.optimizer = HarvestOptimizer(self.optimization_config)
    
    def run_full_optimization(
        self, 
        input_file_path: str, 
        input_file_path_price: str,
        feed_price: float,
        output_dir: str = "output"
    ) -> Dict[str, Any]:
        """
        Run the complete harvest optimization process.
        
        Args:
            input_file_path: Path to input CSV file
            output_dir: Output directory for results
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting full harvest optimization")
        
        # Step 1: Load and preprocess data
        ready_df, removed_df = self.optimizer.data_loader.preprocess_data(
            input_file_path,
            input_file_path_price,
            feed_price,
            duration_days=self.service_config.duration_days,
            max_harvest_date=self.service_config.max_harvest_date,
            cleaning_days=self.service_config.cleaning_days,
            safety_days=self.service_config.safety_days,
            cull_percentage=self.service_config.cull_adjustment
        )

        culls_plan, ready_df = self.optimizer.calculate_culls_allocation(
            ready_df,
            cull_percentage=self.service_config.cull_adjustment,
            culls_avg_weight=self.service_config.culls_avg_weight
        )

        
        # Step 2: Run slaughterhouse optimization with three scenarios
        if self.sh_config is None:
            logger.info("Skipping slaughterhouse optimization - configuration not set")
            best_sh_plan = pd.DataFrame()
            best_sh_updated = ready_df.copy()
            sh_scenarios = {}
        else:
            from .optimization.slaughterhouse_optimizer_v5 import run_three_scenario_optimization
            
            # Run three-scenario optimization
            sh_scenarios = run_three_scenario_optimization(
                df_input=ready_df,
                min_weight=self.sh_config.min_weight,
                max_weight=self.sh_config.max_weight,
                min_stock=self.sh_config.min_stock,
                max_stock=self.sh_config.max_stock,
                max_pct_per_house=self.sh_config.max_pct_per_house,
                min_per_house=2000
            )
            
            # Use weight scenario as the best plan
            if 'weight' in sh_scenarios and sh_scenarios['weight']['plan_df'] is not None and not sh_scenarios['weight']['plan_df'].empty:
                logger.info("Using weight scenario as best slaughterhouse plan")
                best_sh_plan = sh_scenarios['weight']['plan_df'].copy()
                best_sh_updated = sh_scenarios['weight']['updated_df'].copy()
                
                # Clean up columns
                columns_to_drop = ["final_pct_per_house", "cap_pct_per_house", "opportunity", "selected"]
                existing_columns = [col for col in columns_to_drop if col in best_sh_plan.columns]
                best_sh_plan.drop(columns=existing_columns, inplace=True)
            else:
                logger.warning("Weight scenario not found or empty - using base scenario as fallback")
                if 'base' in sh_scenarios and sh_scenarios['base']['plan_df'] is not None and not sh_scenarios['base']['plan_df'].empty:
                    best_sh_plan = sh_scenarios['base']['plan_df'].copy()
                    best_sh_updated = sh_scenarios['base']['updated_df'].copy()
                    
                    # Clean up columns
                    columns_to_drop = ["final_pct_per_house", "cap_pct_per_house", "opportunity", "selected"]
                    existing_columns = [col for col in columns_to_drop if col in best_sh_plan.columns]
                    best_sh_plan.drop(columns=existing_columns, inplace=True)
                else:
                    logger.warning("No valid slaughterhouse scenario found")
                    best_sh_plan = pd.DataFrame()
                    best_sh_updated = ready_df.copy()

            # Export scenario results
            for scenario_name, scenario_data in sh_scenarios.items():
                if scenario_data['plan_df'] is not None and not scenario_data['plan_df'].empty:
                    scenario_data['plan_df'].to_csv(f"best_sh_plan_{scenario_name}.csv", index=False)
                    scenario_data['updated_df'].to_csv(f"best_sh_updated_{scenario_name}.csv", index=False)
            
            best_sh_updated.to_csv("best_sh_updated.csv", index=False)
            best_sh_plan.to_csv("best_sh_plan.csv", index=False)
            best_sh_updated = self.optimizer.data_loader.adjust_priority_column(best_sh_updated, feed_price, input_file_path_price)
            best_sh_updated.to_csv("best_sh_updated_adjusted_priority.csv", index=False)

            
            # Apply propagate_harvest to base_sh_updated for all scenarios
            base_sh_updated = best_sh_updated.copy()
            recursive_sh_updated = best_sh_updated.copy()
            expanded_sh_updated = best_sh_updated.copy()
            combined_sh_updated = best_sh_updated.copy()
            age_expansion_sh_updated = best_sh_updated.copy()
    

            # Export slaughterhouse summary        
            
            if best_sh_plan is None or best_sh_plan.empty:
                logger.warning("No valid slaughterhouse harvest plan found")
                best_sh_plan = pd.DataFrame()
                best_sh_updated = ready_df.copy()
            
            base_sh_plan = best_sh_plan.copy()
            recursive_sh_plan = best_sh_plan.copy()
            expanded_sh_plan = best_sh_plan.copy()
            combined_sh_plan = best_sh_plan.copy()
            age_expansion_sh_plan = best_sh_plan.copy()



        # Step 3: Run market optimization on remaining stock (reverse sweep + best selection)
        if self.market_config is None:
            logger.info("Skipping market optimization - configuration not set")
            market_harvest = pd.DataFrame()
            market_updated = best_sh_updated.copy()

        else:
            results_by_start_day_M, all_updated_dfs_M = self.optimizer.run_market_reverse_sweep(
                df_input=best_sh_updated,
                max_total_stock=self.market_config.max_stock,
                min_total_stock=self.market_config.min_stock,
                tolerance_step=self.market_config.tolerance_step,
                max_tolerance=self.market_config.max_tolerance,
                harvest_type='Market',
                max_pct_per_house=self.market_config.max_pct_per_house,
                min_weight=self.market_config.min_weight,
                max_weight=self.market_config.max_weight,
                optimizer_type=self.market_config.optimizer_type,
                adjust_priority=self.optimizer.data_loader.adjust_priority_column,
                feed_price=feed_price,
                input_file_path_price=input_file_path_price

            )

            M_best_date, market_harvest, market_updated, _ = self.optimizer.get_best_net_meat_plan(results_by_start_day_M,optimizer_type=self.market_config.optimizer_type)

            market_updated_pairs = market_updated[['farm', 'house']].drop_duplicates()
            market_harvest_pairs = market_harvest[['farm', 'house']].drop_duplicates()
            best_sh_pairs = best_sh_plan[['farm', 'house']].drop_duplicates()

            # Apply propagate_harvest to market_updated for all scenarios
            market_harvest, market_harvest_progress, market_updated = propagate_harvest(market_harvest.copy(), base_sh_updated, finalize_threshold=2000, unharvested_df=market_updated)

            unique_pairs = ready_df[['farm', 'house']].drop_duplicates()


            
            # Find farm-house pairs that are in ready_df but NOT in either sh_harvested_pairs OR market_updated_pairs
            # First, combine both harvested datasets
            all_harvested_pairs = pd.concat([market_updated_pairs, market_harvest_pairs]).drop_duplicates()

            
            # Perform anti-join to get pairs in ready_df but not in any harvested data
            not_in_unharvested = unique_pairs.merge(all_harvested_pairs, on=['farm', 'house'], how='left', indicator=True)
            not_in_unharvested = not_in_unharvested[not_in_unharvested['_merge'] == 'left_only'].drop(columns=['_merge'])

            ready_df_not_in_unharvested = ready_df[ready_df['farm'].isin(not_in_unharvested['farm']) & ready_df['house'].isin(not_in_unharvested['house'])]
            
            # Subtract previously allocated harvest_stock (from market_harvest and best_sh_plan)
            alloc_sources = []
            if 'harvest_stock' in best_sh_plan.columns and not best_sh_plan.empty:
                alloc_sources.append(best_sh_plan[['farm', 'house', 'harvest_stock']])
            if 'harvest_stock' in market_harvest.columns and not market_harvest.empty:
                alloc_sources.append(market_harvest[['farm', 'house', 'harvest_stock']])

            if alloc_sources:
                alloc_df = pd.concat(alloc_sources, ignore_index=True)
                alloc_sum = (
                    alloc_df
                    .groupby(['farm', 'house'], as_index=False)['harvest_stock']
                    .sum()
                    .rename(columns={'harvest_stock': 'allocated_stock'})
                )
                ready_df_not_in_unharvested = ready_df_not_in_unharvested.merge(
                    alloc_sum, on=['farm', 'house'], how='left'
                )
                ready_df_not_in_unharvested['allocated_stock'] = ready_df_not_in_unharvested['allocated_stock'].fillna(0)
                ready_df_not_in_unharvested['expected_stock'] = (
                    ready_df_not_in_unharvested['expected_stock'] - ready_df_not_in_unharvested['allocated_stock']
                ).clip(lower=0)
                ready_df_not_in_unharvested = ready_df_not_in_unharvested.drop(columns=['allocated_stock'])
            
            

            expanded_updated_stock_initial = market_harvest.copy()
            combined_updated_stock_initial = market_harvest.copy()
            age_expansion_updated_stock_initial = market_harvest.copy()
            recursive_updated_stock_initial = market_harvest.copy()
            
            recursive_unharvested =  pd.concat([ready_df_not_in_unharvested, market_updated])
            expanded_unharvested = pd.concat([ready_df_not_in_unharvested, market_updated])
            combined_unharvested = pd.concat([ready_df_not_in_unharvested, market_updated])
            age_expansion_unharvested = pd.concat([ready_df_not_in_unharvested, market_updated])

            market_adjusted_priority = self.optimizer.data_loader.adjust_priority_column(combined_unharvested, feed_price, input_file_path_price)
            market_adjusted_priority.to_csv("market_adjusted_priority.csv", index=False)

        # Step 4: Calculate culls allocation
        

                
        # Step 6: Export CSV files BEFORE recursive optimization
        pre_recursive_exporter = HarvestPlanExporter(f"harvest_results/{output_dir}_base", "base")
        pre_recursive_harvest_plans = []
        if not best_sh_plan.empty:
            pre_recursive_harvest_plans.append(best_sh_plan)
        if not market_harvest.empty:
            pre_recursive_harvest_plans.append(market_harvest)
        if not culls_plan.empty:
            pre_recursive_harvest_plans.append(culls_plan)
        
        pre_recursive_full_plan = pd.concat(pre_recursive_harvest_plans, ignore_index=True) if pre_recursive_harvest_plans else pd.DataFrame()
        
        # Export files to the base output directory
        pre_recursive_files = pre_recursive_exporter.generate_harvest_plan_csvs(
            harvest_df=pre_recursive_full_plan,
            unharvested_df=market_updated,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=market_harvest,
            cull_df=culls_plan,
            harvest_df_progress=market_harvest_progress
        )
        logger.info(f"Exported base optimization files to {output_dir}: {list(pre_recursive_files.keys())}")
        
        
        if self.market_config is not None:

            # Run age expansion optimization for unharvested stock (+3 days max_harvest_age)
            age_expansion_market_harvest, age_expansion_updated_stock = self._run_unharvested_age_expansion_market_optimization(
                age_expansion_updated_stock_initial, age_expansion_unharvested, best_sh_plan, removed_df, culls_plan, feed_price, input_file_path_price, output_dir
            )

            # Apply propagate_harvest to age expansion results
            age_expansion_market_harvest, age_expansion_market_progress, age_expansion_updated_stock = propagate_harvest(age_expansion_market_harvest.copy(), age_expansion_sh_updated, finalize_threshold=2000, removed_df=removed_df, sh_plan=best_sh_plan, unharvested_df=age_expansion_updated_stock)

            # Run recursive optimization (separate from expanded)
            recursive_market_harvest, recursive_updated_stock = self._run_recursive_market_optimization(
                recursive_updated_stock_initial, recursive_unharvested, feed_price, input_file_path_price
            )

            # Apply propagate_harvest to recursive results
            recursive_market_harvest, recursive_market_progress, recursive_updated_stock = propagate_harvest(recursive_market_harvest.copy(), recursive_sh_updated, finalize_threshold=2000, unharvested_df=recursive_updated_stock)

            # Run expanded optimization (separate from recursive)
            expanded_market_harvest, expanded_updated_stock = self._run_expanded_market_optimization(
                expanded_updated_stock_initial, expanded_unharvested, expanded_sh_plan, culls_plan, feed_price, input_file_path_price, output_dir
            )

            # Apply propagate_harvest to expanded results
            expanded_market_harvest, expanded_market_progress, expanded_updated_stock = propagate_harvest(expanded_market_harvest.copy(),expanded_sh_updated, finalize_threshold=2000, unharvested_df=expanded_updated_stock)

            
            # Run combined recursive + expanded optimization with dynamic capacity
            combined_market_harvest, combined_updated_stock = self._run_combined_market_optimization(
                combined_updated_stock_initial, combined_unharvested, combined_sh_plan, culls_plan, removed_df, feed_price, input_file_path_price, output_dir
            )

            # Apply propagate_harvest to combined results
            combined_market_harvest, combined_market_progress, combined_updated_stock = propagate_harvest(combined_market_harvest.copy(), combined_sh_updated, finalize_threshold=3000, removed_df=removed_df, sh_plan=best_sh_plan, unharvested_df=combined_updated_stock)

            # Apply propagate_harvest to expanded results
            expanded_market_harvest = pd.concat([market_harvest, expanded_market_harvest], ignore_index=True) if not market_harvest.empty and not expanded_market_harvest.empty else pd.DataFrame()
            expanded_market_harvest, expanded_market_progress, expanded_updated_stock = propagate_harvest(expanded_market_harvest.copy(),expanded_sh_updated, finalize_threshold=2000, unharvested_df=expanded_updated_stock)
        
        # Step 8: Create recursive-only harvest plan (base + recursive)
        recursive_harvest_plans = []
        if not best_sh_plan.empty:
            recursive_harvest_plans.append(best_sh_plan)
        if not recursive_market_harvest.empty:
            recursive_harvest_plans.append(recursive_market_harvest)
        if not culls_plan.empty:
            recursive_harvest_plans.append(culls_plan)
        
        recursive_full_harvest_plan = pd.concat(recursive_harvest_plans, ignore_index=True) if recursive_harvest_plans else pd.DataFrame()
        
        # Step 9: Create expanded harvest plan (base + expanded market combined)
        expanded_harvest_plans = []
        if not best_sh_plan.empty:
            expanded_harvest_plans.append(best_sh_plan)
        if not expanded_market_harvest.empty:
            expanded_harvest_plans.append(expanded_market_harvest)
        if not culls_plan.empty:
            expanded_harvest_plans.append(culls_plan)
        
        expanded_full_harvest_plan = pd.concat(expanded_harvest_plans, ignore_index=True) if expanded_harvest_plans else pd.DataFrame()
        
        
        # Step 9.5: Create combined harvest plan (base + combined market harvest with dynamic capacity)
        combined_harvest_plans = []
        if not best_sh_plan.empty:
            combined_harvest_plans.append(best_sh_plan)
        if not combined_market_harvest.empty:
            combined_harvest_plans.append(combined_market_harvest)
        if not culls_plan.empty:
            combined_harvest_plans.append(culls_plan)
        
        combined_full_harvest_plan = pd.concat(combined_harvest_plans, ignore_index=True).drop_duplicates(subset=['farm', 'house', 'date', 'age','harvest_type','harvest_stock']) if combined_harvest_plans else pd.DataFrame()
        
        # Step 9.6: Create age expansion harvest plan (base + age expansion market harvest)
        age_expansion_harvest_plans = []
        if not best_sh_plan.empty:
            age_expansion_harvest_plans.append(best_sh_plan)
        if not age_expansion_market_harvest.empty:
            age_expansion_harvest_plans.append(age_expansion_market_harvest)
        if not culls_plan.empty:
            age_expansion_harvest_plans.append(culls_plan)
        
        age_expansion_full_harvest_plan = pd.concat(age_expansion_harvest_plans, ignore_index=True).drop_duplicates(subset=['farm', 'house', 'date', 'age','harvest_type','harvest_stock']) if age_expansion_harvest_plans else pd.DataFrame()
        
        # Step 10: Export CSV files AFTER recursive optimization (recursive results only)
        recursive_output_dir = f"harvest_results/{output_dir}_capacity_increased"
        recursive_exporter = HarvestPlanExporter(recursive_output_dir, "capacity_increased")
        recursive_files = recursive_exporter.generate_harvest_plan_csvs(
            harvest_df=recursive_full_harvest_plan,
            unharvested_df=recursive_updated_stock,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=recursive_market_harvest,
            cull_df=culls_plan,
            harvest_df_progress=recursive_market_progress
        )

        logger.info(f"Exported recursive optimization files to {recursive_output_dir}: {list(recursive_files.keys())}")
        
        # Step 11: Export CSV files AFTER expanded optimization (expanded results only)
        expanded_output_dir = f"harvest_results/{output_dir}_weight_expanded"
        expanded_exporter = HarvestPlanExporter(expanded_output_dir, "weight_expanded")
        expanded_files = expanded_exporter.generate_harvest_plan_csvs(
            harvest_df=expanded_full_harvest_plan,
            unharvested_df=expanded_updated_stock,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=expanded_market_harvest,
            cull_df=culls_plan,
            harvest_df_progress=expanded_market_progress
        )
        logger.info(f"Exported expanded optimization files to {expanded_output_dir}: {list(expanded_files.keys())}")
        
        # Step 12: Export CSV files for combined optimization (combined results)
        combined_output_dir = f"harvest_results/{output_dir}_combined"
        combined_exporter = HarvestPlanExporter(combined_output_dir, "combined")
        combined_files = combined_exporter.generate_harvest_plan_csvs(
            harvest_df=combined_full_harvest_plan,
            unharvested_df=combined_updated_stock,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=combined_market_harvest,
            cull_df=culls_plan,
            harvest_df_progress=combined_market_progress
        )
        logger.info(f"Exported combined optimization files to {combined_output_dir}: {list(combined_files.keys())}")
        
        # Step 13: Export CSV files for age expansion optimization (age expansion results)
        age_expansion_output_dir = f"harvest_results/{output_dir}_age_expansion"
        age_expansion_exporter = HarvestPlanExporter(age_expansion_output_dir, "age_expansion")
        age_expansion_files = age_expansion_exporter.generate_harvest_plan_csvs(
            harvest_df=age_expansion_full_harvest_plan,
            unharvested_df=age_expansion_updated_stock,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=age_expansion_market_harvest,
            cull_df=culls_plan,
            harvest_df_progress=age_expansion_market_progress
        )
        logger.info(f"Exported age expansion optimization files to {age_expansion_output_dir}: {list(age_expansion_files.keys())}")
        
        # Combine exported files information
        exported_files = {
            "base": pre_recursive_files,
            "recursive": recursive_files,
            "expanded": expanded_files,
            "combined": combined_files,
            "age_expansion": age_expansion_files
        }
        
        # Step 12: Prepare results
        results = {
            "status": "success",
        }
        
        logger.info("Full harvest optimization completed successfully")
        return results

    def _run_expanded_market_optimization(
        self, 
        initial_market_harvest: pd.DataFrame, 
        initial_unharvested_stock: pd.DataFrame,
        sh_harvest: pd.DataFrame,
        culls_harvest: pd.DataFrame,
        feed_price: float,
        input_file_path_price: str,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run expanded market optimization with wider weight range and capacity checking.
        Runs in parallel with recursive optimization from the same base unharvested stock.
        
        Args:
            initial_market_harvest: Initial market harvest (before recursive optimization)
            initial_unharvested_stock: Initial unharvested stock (before recursive optimization)
            output_dir: Output directory for exports
            sh_harvest: Slaughterhouse harvest results to account for allocated stock
            culls_harvest: Culls harvest results to account for allocated stock
            
        Returns:
            Tuple of (additional market harvest, remaining unharvested stock)
        """
        logger.info("Starting expanded market optimization (parallel to recursive)...")
        
        if initial_unharvested_stock.empty:
            logger.info("No unharvested stock - skipping expanded optimization")
            return pd.DataFrame(), initial_unharvested_stock
        
        if self.market_config is None:
            logger.info("No market config - skipping expanded optimization")
            return pd.DataFrame(), initial_unharvested_stock
        
        # Step 1: Expand market weight range by ±0.05
        expanded_min_weight = self.market_config.min_weight - 0.05 * self.market_config.min_weight
        expanded_max_weight = self.market_config.max_weight + 0.05 * self.market_config.max_weight
        
        logger.info(f"Expanded weight range: {expanded_min_weight:.2f} - {expanded_max_weight:.2f} (original: {self.market_config.min_weight:.2f} - {self.market_config.max_weight:.2f})")
        
        # Step 2: Refetch data for farm/house combinations in unharvested stock with expanded weight range
        # We need to get the original data for these farm/house combinations and reapply expanded weight filtering
        if hasattr(self.optimizer, 'data_loader') and hasattr(self.optimizer.data_loader, 'last_preprocessed_data'):
            # Get farm/house combinations from unharvested stock
            farm_house_combinations = initial_unharvested_stock[['farm', 'house']].drop_duplicates()
            
            # Get the original preprocessed data 
            original_data = self.optimizer.data_loader.last_preprocessed_data.copy()
            
            # Filter to only the farm/house combinations that have unharvested stock
            original_data = original_data.merge(
                farm_house_combinations, 
                on=['farm', 'house'], 
                how='inner'
            )
            
            # Calculate total allocated stock per farm/house from all harvest types
            # Need to get all harvest results that have been allocated so far
            all_allocated_harvests = []
            
            # Add slaughterhouse harvest if it exists
            if sh_harvest is not None and not sh_harvest.empty:
                all_allocated_harvests.append(sh_harvest[['farm', 'house', 'harvest_stock']])
            
            # Add culls harvest if it exists
            if culls_harvest is not None and not culls_harvest.empty:
                all_allocated_harvests.append(culls_harvest[['farm', 'house', 'harvest_stock']])
            
            # Add initial market harvest if it exists
            if not initial_market_harvest.empty:
                all_allocated_harvests.append(initial_market_harvest[['farm', 'house', 'harvest_stock']])
            
            # Calculate total allocated stock per farm/house
            if all_allocated_harvests:
                allocated_df = pd.concat(all_allocated_harvests, ignore_index=True)
                total_allocated = allocated_df.groupby(['farm', 'house'])['harvest_stock'].sum().reset_index()
                total_allocated.rename(columns={'harvest_stock': 'total_allocated_stock'}, inplace=True)
                
                # Merge with original data to subtract allocated stock
                original_data = original_data.merge(
                    total_allocated, 
                    on=['farm', 'house'], 
                    how='left'
                )
                original_data['total_allocated_stock'] = original_data['total_allocated_stock'].fillna(0)
                
                # Update expected_stock to reflect remaining stock after allocations
                original_data['expected_stock'] = original_data['expected_stock'] - original_data['total_allocated_stock']
                original_data['expected_stock'] = original_data['expected_stock'].clip(lower=0) 
                
                # Remove rows with zero expected stock
                original_data = original_data[original_data['expected_stock'] > 0].copy()
                
                logger.info(f"Updated expected_stock after subtracting {total_allocated['total_allocated_stock'].sum()} allocated stock")
            else:
                logger.info("No previous allocations found - using original expected_stock")
            
            # Apply expanded weight range flagging to the original data
            expanded_flagged_stock = self.optimizer.model_builder.flag_ready_avg_weight(
                original_data, 
                expanded_min_weight, 
                expanded_max_weight, 
                'Market'
            )
            
            # Filter to only ready stock
            ready_expanded_stock = expanded_flagged_stock[expanded_flagged_stock['ready_Market'] == 1].copy()
            
            logger.info(f"Refetched {len(original_data)} rows for {len(farm_house_combinations)} farm/house combinations")
        else:
            # Fallback: use the unharvested stock directly with expanded flagging
            logger.warning("Could not access original data, using unharvested stock directly")
            expanded_flagged_stock = self.optimizer.model_builder.flag_ready_avg_weight(
                initial_unharvested_stock.copy(), 
                expanded_min_weight, 
                expanded_max_weight, 
                'Market'
            )
            ready_expanded_stock = expanded_flagged_stock[expanded_flagged_stock['ready_Market'] == 1].copy()
        
        if ready_expanded_stock.empty:
            logger.info("No stock ready with expanded weight range")
            return pd.DataFrame(), initial_unharvested_stock
        
        logger.info(f"Found {ready_expanded_stock['expected_stock'].sum()} stock ready with expanded weight range")
        
        # Step 3: Check for additional harvest opportunities
        # The expanded optimization should find new opportunities, not just fill existing capacity gaps
        logger.info("Expanded optimization will look for additional harvest opportunities beyond existing plans")
        
        # Step 4: Check daily capacity constraints before optimization
        # Calculate existing daily allocations from all harvest types
        daily_allocations = {}
        
        if not initial_market_harvest.empty:
            market_daily = initial_market_harvest.groupby('date')['harvest_stock'].sum()
            for date, allocation in market_daily.items():
                daily_allocations[date] = daily_allocations.get(date, 0) + allocation
        # Initialize to_remove as empty list
        to_remove_dates = []
        
        # Filter expanded stock to only include days where we have remaining capacity
        if daily_allocations:
            # Convert to DataFrame for easier manipulation
            daily_df = pd.DataFrame(list(daily_allocations.items()), columns=['date', 'existing_allocation'])
            daily_df['date'] = pd.to_datetime(daily_df['date'])

            # Build a full calendar of dates across the optimization horizon
            horizon_start = ready_expanded_stock['date'].min()
            horizon_end = ready_expanded_stock['date'].max()
            full_dates = pd.DataFrame({'date': pd.date_range(horizon_start, horizon_end, freq='D')})

            # Merge existing allocations onto full calendar; missing become 0
            daily_df_full = full_dates.merge(daily_df, on='date', how='left')
            daily_df_full['existing_allocation'] = daily_df_full['existing_allocation'].fillna(0)
            # Base remaining capacity
            daily_df_full['remaining_capacity'] = self.market_config.max_stock - daily_df_full['existing_allocation']

            # Prevent negatives
            daily_df_full['remaining_capacity'] = daily_df_full['remaining_capacity'].clip(lower=0)

            # Use all days (missing days implicitly have full remaining capacity)
            valid_days = daily_df_full

            if not valid_days.empty:
                # Filter ready_expanded_stock to only valid days
                ready_expanded_stock = ready_expanded_stock[ready_expanded_stock['date'].isin(valid_days['date'])].copy()
                logger.info(f"Filtered to {len(valid_days)} days with remaining capacity for expanded optimization")
            else:
                logger.info("No days with remaining capacity - skipping expanded optimization")
                return pd.DataFrame(), initial_unharvested_stock
        else:
            # No prior allocations; create a full calendar from the expanded stock horizon with full capacity
            horizon_start = ready_expanded_stock['date'].min()
            horizon_end = ready_expanded_stock['date'].max()
            valid_days = pd.DataFrame({'date': pd.date_range(horizon_start, horizon_end, freq='D')})
            valid_days['existing_allocation'] = 0
            valid_days['remaining_capacity'] = self.market_config.max_stock
            # Calculate tolerance bounds for removing dates near max capacity
            tolerance = 0.05 * self.market_config.max_stock
            lower_bound = self.market_config.max_stock - tolerance
            upper_bound = self.market_config.max_stock + tolerance
            
            # Find dates to remove (those with allocation between bounds)
            to_remove_mask = (
                (valid_days['existing_allocation'] >= lower_bound) & 
                (valid_days['existing_allocation'] <= upper_bound)
            )
            to_remove_dates = valid_days[to_remove_mask]['date'].tolist()
            
            valid_days = valid_days[valid_days['remaining_capacity'] > 0].copy()
            
            
        if ready_expanded_stock.empty:
            logger.info("No stock ready after capacity filtering")
            return pd.DataFrame(), initial_unharvested_stock

        ready_expanded_stock = self.optimizer.data_loader.adjust_priority_column(ready_expanded_stock, feed_price, input_file_path_price)
        # Remove dates that are near capacity limits
        if to_remove_dates:
            ready_expanded_stock = ready_expanded_stock[~ready_expanded_stock['date'].isin(to_remove_dates)]
        ready_expanded_stock.to_csv(f"weight_expanded_stock_adjusted_priority.csv", index=False)
        
        # Step 5: Run market optimization with expanded stock and capacity-aware limits
        expanded_harvest, remaining_unharvested = self.optimizer.run_iterative_market_harvest(
            ready_expanded_stock,
            max_total_stock=100000,
            min_total_stock=100000,
            tolerance_step=self.market_config.tolerance_step,
            max_tolerance=self.market_config.max_tolerance,
            max_pct_per_house=self.market_config.max_pct_per_house,
            min_weight=expanded_min_weight,
            max_weight=expanded_max_weight,
            valid_days=valid_days,
            optimizer_type=self.market_config.optimizer_type

        )
        
        if not expanded_harvest.empty:
            logger.info(f"Expanded optimization harvested {expanded_harvest['harvest_stock'].sum()} additional stock")
        else:
            logger.info("No additional harvest from expanded optimization")
        
        return expanded_harvest, remaining_unharvested


    def _run_recursive_market_optimization(
        self, 
        initial_market_harvest: pd.DataFrame, 
        unharvested_stock: pd.DataFrame,
        feed_price: float,
        input_file_path_price: str,
        concat: bool = True,
        combined: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run recursive market optimization on unharvested stock.
        
        Args:
            initial_market_harvest: Initial market harvest results
            unharvested_stock: DataFrame with unharvested stock
            output_dir: Output directory for intermediate exports
            
        Returns:
            Tuple of (combined market harvest, final unharvested stock)
        """
        logger.info("Checking for recursive market optimization...")
        
        # Check if we need to run recursive optimization
        if unharvested_stock.empty:
            logger.info("No unharvested stock - skipping recursive optimization")
            return initial_market_harvest, unharvested_stock
        
        # Count unique farm-house combinations in unharvested stock
        unharvested_stock['farm_house_key'] = (
            unharvested_stock['farm'].astype(str) + "_" + 
            unharvested_stock['house'].astype(str)
        )
        unique_farm_houses = unharvested_stock['farm_house_key'].nunique()
        total_unharvested_stock = unharvested_stock.drop_duplicates(subset=['farm', 'house'])['expected_stock'].sum()
        
        logger.info(f"Unharvested stock analysis: {unique_farm_houses} unique farm-houses, {total_unharvested_stock} total stock")
        
        if total_unharvested_stock <= 10:
            logger.info("Total unharvested stock <= 20000 - no recursive optimization needed")
            return initial_market_harvest, unharvested_stock
        
        
        # Start recursive optimization
        all_additional_harvests = []
        current_unharvested = unharvested_stock.copy()
        current_unharvested.to_csv(f"current_unharvested.csv", index=False)
        current_max_stock = 0.2 * self.market_config.max_stock
        starting_max_stock = self.market_config.max_stock
        iteration = 1
        
        logger.info(f"Starting recursive market optimization with initial max_stock={current_max_stock}")
        
        while True:
            logger.info(f"Recursive optimization iteration {iteration}, max_stock={current_max_stock}")
            
            # Check total remaining unharvested stock
            remaining_stock = current_unharvested.drop_duplicates(subset=['farm', 'house'])['expected_stock'].sum()
            logger.info(f"Remaining unharvested stock: {remaining_stock}")
            
            if remaining_stock <= 10:
                logger.info("Remaining unharvested stock <= 10000 - stopping recursive optimization")
                break
            
            # Update market config for this iteration
            iteration_config = MarketConfig(
                min_weight=self.market_config.min_weight - 0.05 * self.market_config.min_weight if combined == True else self.market_config.min_weight,
                max_weight=self.market_config.max_weight + 0.05 * self.market_config.min_weight if combined == True else self.market_config.max_weight,
                min_stock=100,
                max_stock=current_max_stock,
                max_pct_per_house=self.market_config.max_pct_per_house,
                tolerance_step=self.market_config.tolerance_step,
                max_tolerance=self.market_config.max_tolerance
            )
            
            current_unharvested = self.optimizer.data_loader.adjust_priority_column(current_unharvested, feed_price, input_file_path_price)
            current_unharvested.to_csv(f"capacity_increased_stock_adjusted_priority_iteration{iteration}.csv", index=False)
            # Run market optimization on current unharvested stock
            results_by_start_day, _ = self.optimizer.run_market_reverse_sweep(
                current_unharvested,
                max_total_stock=iteration_config.max_stock,
                min_total_stock=iteration_config.min_stock,
                tolerance_step=iteration_config.tolerance_step,
                max_tolerance=iteration_config.max_tolerance,
                max_pct_per_house=iteration_config.max_pct_per_house,
                min_weight=iteration_config.min_weight,
                max_weight=iteration_config.max_weight,
                optimizer_type=iteration_config.optimizer_type,
                adjust_priority=self.optimizer.data_loader.adjust_priority_column,
                feed_price=feed_price,
                input_file_path_price=input_file_path_price

            )
            
            best_date, additional_harvest, updated_unharvested, _ = self.optimizer.get_best_net_meat_plan(results_by_start_day,optimizer_type=iteration_config.optimizer_type)

            if updated_unharvested is not None and not updated_unharvested.empty:
                updated_unharvested.to_csv(f"updated_unharvested_{iteration}.csv", index=False)
            
            if additional_harvest is None or additional_harvest.empty:
                logger.info(f"No additional harvest found in iteration {iteration} - increasing max_stock")
                current_pct = iteration * 0.2
                current_max_stock = current_pct * starting_max_stock 
                iteration += 1
                
                # Prevent infinite loop - stop after reasonable number of iterations
                if iteration > 10:
                    logger.warning("Reached maximum recursive optimization iterations - stopping")
                    break
                continue
            
            # Add to accumulated harvests
            additional_harvest['iteration'] = iteration
            all_additional_harvests.append(additional_harvest)
            
            current_unharvested = updated_unharvested
            
            logger.info(f"Iteration {iteration} harvested {additional_harvest['harvest_stock'].sum()} stock")
            
            # Increase max_stock for next iteration
            current_pct = 0.2
            current_max_stock = current_pct * starting_max_stock
            iteration += 1
            
            # Prevent infinite loop
            if iteration > 10:
                logger.warning("Reached maximum recursive optimization iterations - stopping")
                break
        
        # Combine all market harvests
        combined_market_harvest = initial_market_harvest.copy()
        
        if all_additional_harvests:
            additional_combined = pd.concat(all_additional_harvests, ignore_index=True)
            logger.info(f"Total additional harvest from recursive optimization: {additional_combined['harvest_stock'].sum()} stock")
            
            if not initial_market_harvest.empty and concat == True:
                initial_market_harvest['iteration'] = 0
                combined_market_harvest = pd.concat([initial_market_harvest, additional_combined], ignore_index=True)
            else:
                combined_market_harvest = additional_combined
        else:
            logger.info("No additional harvest from recursive optimization")

        return combined_market_harvest, current_unharvested
    
    
    def _run_combined_market_optimization(
        self, 
        initial_market_harvest: pd.DataFrame, 
        initial_unharvested_stock: pd.DataFrame,
        sh_harvest: pd.DataFrame,
        culls_harvest: pd.DataFrame,
        removed_df: pd.DataFrame,
        feed_price: float,
        input_file_path_price: str,
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run combined market optimization: first age expansion, then iterative market harvest.
        
        Args:
            initial_market_harvest: Initial market harvest (before recursive optimization)
            initial_unharvested_stock: Initial unharvested stock (before recursive optimization)
            output_dir: Output directory for exports
            sh_harvest: Slaughterhouse harvest results to account for allocated stock
            culls_harvest: Culls harvest results to account for allocated stock
            
        Returns:
            Tuple of (additional market harvest, remaining unharvested stock)
        """
        logger.info("Starting combined market optimization: age expansion first, then iterative harvest...")
        
        if initial_unharvested_stock.empty:
            logger.info("No unharvested stock - skipping combined optimization")
            return pd.DataFrame(), initial_unharvested_stock
        
        if self.market_config is None:
            logger.info("No market config - skipping combined optimization")
            return pd.DataFrame(), initial_unharvested_stock
        
        # Step 1: Expand market weight range by ±0.05
        expanded_min_weight = self.market_config.min_weight - 0.05 * self.market_config.min_weight
        expanded_max_weight = self.market_config.max_weight + 0.05 * self.market_config.max_weight
        
        logger.info(f"Expanded weight range: {expanded_min_weight:.2f} - {expanded_max_weight:.2f} (original: {self.market_config.min_weight:.2f} - {self.market_config.max_weight:.2f})")
        
        # Step 2: Run age expansion optimization first with expanded weight range
        logger.info("Step 1: Running age expansion optimization with expanded weight range...")
        age_expansion_harvest, age_expansion_unharvested = self._run_unharvested_age_expansion_market_optimization(
            initial_market_harvest,
            initial_unharvested_stock,
            sh_harvest,
            removed_df,
            culls_harvest,
            feed_price,
            input_file_path_price,
            output_dir,
            age_expansion_days=3,
            min_weight=expanded_min_weight,
            max_weight=expanded_max_weight,
            combined=True

        )
        
        # Step 3: Run iterative market harvest on remaining unharvested stock
        logger.info("Step 2: Running iterative market harvest on remaining unharvested stock...")
        
        if age_expansion_unharvested.empty:
            logger.info("No remaining unharvested stock after age expansion - returning age expansion results")
            return age_expansion_harvest, age_expansion_unharvested
        
        # Refetch data for farm/house combinations in remaining unharvested stock with expanded weight range
        if hasattr(self.optimizer, 'data_loader') and hasattr(self.optimizer.data_loader, 'last_preprocessed_data'):
            # Get farm/house combinations from remaining unharvested stock
            farm_house_combinations = age_expansion_unharvested[['farm', 'house']].drop_duplicates()
            
            # Get the original preprocessed data 
            original_data = self.optimizer.data_loader.last_preprocessed_data.copy()
            
            # Filter to only the farm/house combinations that have unharvested stock
            original_data = original_data.merge(
                farm_house_combinations, 
                on=['farm', 'house'], 
                how='inner'
            )
            
            # Calculate total allocated stock per farm/house from all harvest types
            all_allocated_harvests = []
            
            # Add slaughterhouse harvest if it exists
            if sh_harvest is not None and not sh_harvest.empty:
                all_allocated_harvests.append(sh_harvest[['farm', 'house', 'harvest_stock']])
            
            # Add culls harvest if it exists
            if culls_harvest is not None and not culls_harvest.empty:
                all_allocated_harvests.append(culls_harvest[['farm', 'house', 'harvest_stock']])
            
            # Add age expansion harvest if it exists
            if not age_expansion_harvest.empty:
                all_allocated_harvests.append(age_expansion_harvest[['farm', 'house', 'harvest_stock']])
            
            # Calculate total allocated stock per farm/house
            if all_allocated_harvests:
                allocated_df = pd.concat(all_allocated_harvests, ignore_index=True)
                total_allocated = allocated_df.groupby(['farm', 'house'])['harvest_stock'].sum().reset_index()
                total_allocated.rename(columns={'harvest_stock': 'total_allocated_stock'}, inplace=True)
                
                # Merge with original data to subtract allocated stock
                original_data = original_data.merge(
                    total_allocated, 
                    on=['farm', 'house'], 
                    how='left'
                )
                original_data['total_allocated_stock'] = original_data['total_allocated_stock'].fillna(0)
                
                # Update expected_stock to reflect remaining stock after allocations
                original_data['expected_stock'] = original_data['expected_stock'] - original_data['total_allocated_stock']
                original_data['expected_stock'] = original_data['expected_stock'].clip(lower=0) 
                
                # Remove rows with zero expected stock
                original_data = original_data[original_data['expected_stock'] > 0].copy()
                
                logger.info(f"Updated expected_stock after subtracting {total_allocated['total_allocated_stock'].sum()} allocated stock")
            else:
                logger.info("No previous allocations found - using original expected_stock")
            
            # Apply expanded weight range flagging to the original data
            expanded_flagged_stock = self.optimizer.model_builder.flag_ready_avg_weight(
                original_data, 
                expanded_min_weight, 
                expanded_max_weight, 
                'Market'
            )
            
            # Filter to only ready stock
            ready_expanded_stock = expanded_flagged_stock[expanded_flagged_stock['ready_Market'] == 1].copy()
            
            logger.info(f"Refetched {len(original_data)} rows for {len(farm_house_combinations)} farm/house combinations")
        else:
            # Fallback: use the unharvested stock directly with expanded flagging
            logger.warning("Could not access original data, using unharvested stock directly")
            expanded_flagged_stock = self.optimizer.model_builder.flag_ready_avg_weight(
                age_expansion_unharvested.copy(), 
                expanded_min_weight, 
                expanded_max_weight, 
                'Market'
            )
            ready_expanded_stock = expanded_flagged_stock[expanded_flagged_stock['ready_Market'] == 1].copy()
        
        if ready_expanded_stock.empty:
            logger.info("No stock ready with expanded weight range for iterative harvest")
            return age_expansion_harvest, age_expansion_unharvested
        
        logger.info(f"Found {ready_expanded_stock['expected_stock'].sum()} stock ready with expanded weight range for iterative harvest")
        
        # Check daily capacity constraints before optimization
        daily_allocations = {}
        
        if not age_expansion_harvest.empty:
            market_daily = age_expansion_harvest.groupby('date')['harvest_stock'].sum()
            for date, allocation in market_daily.items():
                daily_allocations[date] = daily_allocations.get(date, 0) + allocation
        
        # Filter expanded stock to only include days where we have remaining capacity
        if daily_allocations:
            # Convert to DataFrame for easier manipulation
            daily_df = pd.DataFrame(list(daily_allocations.items()), columns=['date', 'existing_allocation'])
            daily_df['date'] = pd.to_datetime(daily_df['date'])

            # Build a full calendar of dates across the optimization horizon
            horizon_start = ready_expanded_stock['date'].min()
            horizon_end = ready_expanded_stock['date'].max()
            full_dates = pd.DataFrame({'date': pd.date_range(horizon_start, horizon_end, freq='D')})

            # Merge existing allocations onto full calendar; missing become 0
            daily_df_full = full_dates.merge(daily_df, on='date', how='left')
            daily_df_full['existing_allocation'] = daily_df_full['existing_allocation'].fillna(0)
            # Base remaining capacity
            daily_df_full['remaining_capacity'] = self.market_config.max_stock - daily_df_full['existing_allocation']
            # If prior allocations meet/exceed the daily max, allow extra 20,000 to reach 120,000 total
            daily_df_full.loc[
                daily_df_full['existing_allocation'] >= self.market_config.max_stock - 3000,
                'remaining_capacity'
            ] = 80000
            # Prevent negatives
            daily_df_full['remaining_capacity'] = daily_df_full['remaining_capacity'].clip(lower=0)

            # Use all days (missing days implicitly have full remaining capacity)
            valid_days = daily_df_full

            if not valid_days.empty:
                # Filter ready_expanded_stock to only valid days
                ready_expanded_stock = ready_expanded_stock[ready_expanded_stock['date'].isin(valid_days['date'])].copy()
                logger.info(f"Filtered to {len(valid_days)} days with remaining capacity for iterative harvest")
            else:
                logger.info("No days with remaining capacity - skipping iterative harvest")
                return age_expansion_harvest, age_expansion_unharvested
        else:
            # No prior allocations; create a full calendar from the expanded stock horizon with full capacity
            horizon_start = ready_expanded_stock['date'].min()
            horizon_end = ready_expanded_stock['date'].max()
            valid_days = pd.DataFrame({'date': pd.date_range(horizon_start, horizon_end, freq='D')})
            valid_days['existing_allocation'] = 0
            valid_days['remaining_capacity'] = self.market_config.max_stock
            valid_days = valid_days[valid_days['remaining_capacity'] > 0].copy()

        if ready_expanded_stock.empty:
            logger.info("No stock ready after capacity filtering for iterative harvest")
            return age_expansion_harvest, age_expansion_unharvested

        ready_expanded_stock = self.optimizer.data_loader.adjust_priority_column(ready_expanded_stock, feed_price, input_file_path_price)
        ready_expanded_stock.to_csv(f"combined_stock_adjusted_priority.csv", index=False)
        
        # Run iterative market harvest with expanded stock and capacity-aware limits
        iterative_harvest, remaining_unharvested = self.optimizer.run_iterative_market_harvest(
            ready_expanded_stock,
            max_total_stock=100000,
            min_total_stock=100000,
            tolerance_step=self.market_config.tolerance_step,
            max_tolerance=self.market_config.max_tolerance,
            max_pct_per_house=self.market_config.max_pct_per_house,
            min_weight=expanded_min_weight,
            max_weight=expanded_max_weight,
            valid_days=valid_days,
            optimizer_type=self.market_config.optimizer_type
        )

        # Combine age expansion and iterative harvest results
        if not iterative_harvest.empty:
            logger.info(f"Iterative harvest harvested {iterative_harvest['harvest_stock'].sum()} additional stock")
            combined_harvest = pd.concat([age_expansion_harvest, iterative_harvest], ignore_index=True)
        else:
            logger.info("No additional harvest from iterative optimization")
            combined_harvest = age_expansion_harvest

        return combined_harvest, remaining_unharvested
            
    def _run_unharvested_age_expansion_market_optimization(
        self, 
        initial_market_harvest: pd.DataFrame,
        unharvested_stock: pd.DataFrame,
        best_sh_plan: pd.DataFrame,
        removed_df: pd.DataFrame,
        culls_plan: pd.DataFrame,
        feed_price: float,
        input_file_path_price: str,
        output_dir: str,
        age_expansion_days: int = 3,
        min_weight: float = None,
        max_weight: float = None,
        combined: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run market optimization on unharvested stock with sequential age expansion.
        
        Args:
            initial_market_harvest: Initial market harvest results
            unharvested_stock: DataFrame with unharvested stock
            best_sh_plan: Best slaughterhouse plan
            removed_df: DataFrame with removed stock data
            culls_plan: Culls plan (unused in current implementation)
            output_dir: Output directory for intermediate exports
            age_expansion_days: Maximum additional days to expand (default 3)
            min_weight: Minimum weight constraint (if None, uses market_config.min_weight)
            max_weight: Maximum weight constraint (if None, uses market_config.max_weight)
            
        Returns:
            Tuple of (cumulative harvest, final unharvested stock)
        """
        logger.info(f"Running sequential market optimization with age expansion (+{age_expansion_days} days)")

        if unharvested_stock.empty:
            logger.info("No unharvested stock - skipping age expansion optimization")
            return initial_market_harvest, unharvested_stock

        # Use provided weight constraints or fall back to market config
        effective_min_weight = min_weight if min_weight is not None else self.market_config.min_weight
        effective_max_weight = max_weight if max_weight is not None else self.market_config.max_weight
        
        logger.info(f"Using weight range: {effective_min_weight:.2f} - {effective_max_weight:.2f}")

        # Initialize tracking variables
        cumulative_harvest = pd.concat([initial_market_harvest, best_sh_plan, culls_plan], ignore_index=True)
        current_unharvested = unharvested_stock.copy()

        cumulative_harvest['date'] = pd.to_datetime(cumulative_harvest['date'])
        
        base_max_date = cumulative_harvest['date'].max()
    

        # Sequential age expansion - Fixed range to include all days
        for expansion_day in range(1, age_expansion_days + 1):
            logger.info(f"Running iteration {expansion_day}: expanding max_harvest_age by +{expansion_day} days")

            unharvested_farm_houses = current_unharvested[['farm', 'house']].drop_duplicates()

            # Create age-expanded dataset for this iteration
            df_age_expanded = self._create_age_expanded_data(
                removed_df, unharvested_farm_houses, expansion_day, cumulative_harvest
            )
            
            if df_age_expanded.empty:
                logger.info(f"No age-expanded data for iteration {expansion_day}")
                continue

            # Combine with current unharvested stock
            combined_stock = pd.concat([df_age_expanded, current_unharvested], ignore_index=True)
            
            # Apply weight range flagging with effective weight constraints
            flagged_stock = self.optimizer.model_builder.flag_ready_avg_weight(
                combined_stock, 
                effective_min_weight, 
                effective_max_weight, 
                'Market'
            )
            
            ready_stock = flagged_stock[flagged_stock['ready_Market'] == 1].copy()
            
            
            if ready_stock.empty:
                logger.info(f"No stock ready with +{expansion_day} day expansion")
                continue
            
            logger.info(f"Iteration {expansion_day}: Found {ready_stock['expected_stock'].sum()} ready stock")
            
            # Get valid days considering capacity constraints
            max_date = base_max_date + pd.Timedelta(days=expansion_day)
            valid_days = self._get_valid_days_with_capacity(cumulative_harvest, ready_stock, max_date)
            
            if valid_days.empty:
                logger.info(f"No days with remaining capacity for iteration {expansion_day}")
                continue
            
            # Run optimization for this iteration
            best_harvest_df, best_updated_df = self._run_iteration_optimization(
                ready_stock, valid_days, expansion_day, output_dir, 
                feed_price, input_file_path_price,
                optimizer_type=self.market_config.optimizer_type,
                min_weight=effective_min_weight,
                max_weight=effective_max_weight,
                combined=combined
            )
            
            if best_harvest_df is not None and not best_harvest_df.empty:
                cumulative_harvest = pd.concat([cumulative_harvest, best_harvest_df]).drop_duplicates(subset=['farm', 'house', 'date', 'age','harvest_type','harvest_stock'], ignore_index=True)
                current_unharvested = best_updated_df.copy().drop_duplicates(subset=['farm', 'house', 'date', 'age'])
                logger.info(f"Iteration {expansion_day}: Added {best_harvest_df['harvest_stock'].sum()} stock")
            else:
                logger.info(f"Iteration {expansion_day}: No additional harvest found")

        logger.info(f"Sequential age expansion completed after {age_expansion_days} iterations")
        return cumulative_harvest, current_unharvested


    def _create_age_expanded_data(
        self, 
        removed_df: pd.DataFrame, 
        farm_houses: pd.DataFrame, 
        expansion_day: int, 
        cumulative_harvest: pd.DataFrame
    ) -> pd.DataFrame:
        """Create age-expanded dataset for a specific iteration."""
        
        # Filter removed_df to only include relevant farm-house combinations
        df_age_expanded = removed_df[
            removed_df['farm'].isin(farm_houses['farm']) & 
            removed_df['house'].isin(farm_houses['house'])
        ].copy()
        
        if df_age_expanded.empty:
            return df_age_expanded
        
        # Clean up columns
        columns_to_drop = ['start_date', 'end_date', 'last_date', 'max_harvest_date', 'removal_reason']
        df_age_expanded = df_age_expanded.drop(columns=[col for col in columns_to_drop if col in df_age_expanded.columns])
        
        # Apply age expansion
        max_age = self.service_config.duration_days + expansion_day
        df_age_expanded = df_age_expanded[df_age_expanded['age'] <= max_age]
        
        # Adjust expected_stock based on cumulative harvests
        if not cumulative_harvest.empty:
            df_age_expanded = self._adjust_expected_stock(df_age_expanded, cumulative_harvest)
        
        return df_age_expanded


    def _adjust_expected_stock(self, df_age_expanded: pd.DataFrame, cumulative_harvest: pd.DataFrame) -> pd.DataFrame:
        """Adjust expected stock by subtracting already harvested amounts."""
        
        # Group cumulative harvest by farm-house
        harvested_by_farm_house = cumulative_harvest.groupby(['farm', 'house'])['harvest_stock'].sum().reset_index()
        
        # Process each harvested farm-house combination
        for _, harvest_row in harvested_by_farm_house.iterrows():
            farm, house, harvest_amount = harvest_row['farm'], harvest_row['house'], harvest_row['harvest_stock']
            
            # Find matching rows in age-expanded data
            mask = (df_age_expanded['farm'] == farm) & (df_age_expanded['house'] == house)
            matching_indices = df_age_expanded[mask].index
            
            for idx in matching_indices:
                current_expected = df_age_expanded.loc[idx, 'expected_stock']
                new_expected = max(0, current_expected - harvest_amount)
                df_age_expanded.loc[idx, 'expected_stock'] = new_expected
        
        # Remove rows where expected_stock becomes 0 or negative
        df_age_expanded = df_age_expanded[df_age_expanded['expected_stock'] > 0].copy()
        
        return df_age_expanded

    def _get_valid_days_with_capacity(
        self, 
        cumulative_harvest: pd.DataFrame, 
        ready_stock: pd.DataFrame, 
        max_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Get valid days considering daily capacity constraints."""
        
        # Create date range for optimization horizon
        horizon_start = ready_stock['date'].min()
        horizon_end = min(ready_stock['date'].max(), max_date)
        full_dates = pd.date_range(horizon_start, horizon_end, freq='D')
        
        # Calculate existing daily allocations
        daily_allocations = {}
        if not cumulative_harvest.empty:
            daily_allocations = cumulative_harvest.groupby('date')['harvest_stock'].sum().to_dict()
        
        # Build valid days DataFrame
        valid_days_data = []
        for date in full_dates:
            existing_allocation = daily_allocations.get(date, 0)
            remaining_capacity = self.market_config.max_stock - existing_allocation
            
            if remaining_capacity > 0:
                valid_days_data.append({
                    'date': date,
                    'existing_allocation': existing_allocation,
                    'remaining_capacity': remaining_capacity
                })
        
        return pd.DataFrame(valid_days_data)


    def _run_iteration_optimization(
        self, 
        combined_stock: pd.DataFrame, 
        valid_days: pd.DataFrame, 
        expansion_day: int, 
        output_dir: str,
        feed_price: float,
        input_file_path_price: str,
        optimizer_type: str = "weight",
        min_weight: float = None,
        max_weight: float = None,
        combined: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run market optimization for a specific iteration."""
        
        # Use provided weight constraints or fall back to market config
        effective_min_weight = min_weight if min_weight is not None else self.market_config.min_weight
        effective_max_weight = max_weight if max_weight is not None else self.market_config.max_weight
        if combined:
            combined_stock.to_csv(f"{output_dir}/combined_stock_adjusted_priority_{expansion_day}.csv", index=False)
        else:
            combined_stock.to_csv(f"{output_dir}/age_expansion_stock_adjusted_priority_{expansion_day}.csv", index=False)

        combined_stock = combined_stock.drop_duplicates(subset=['farm', 'house', 'date'])
        combined_stock = self.optimizer.data_loader.adjust_priority_column(combined_stock, feed_price, input_file_path_price)


        # Run market optimization
        results_by_start_day, _ = self.optimizer.run_market_reverse_sweep(
            df_input=combined_stock,
            max_total_stock=self.market_config.max_stock,
            min_total_stock=self.market_config.min_stock,
            tolerance_step=self.market_config.tolerance_step,
            max_tolerance=self.market_config.max_tolerance,
            harvest_type='Market',
            max_pct_per_house=self.market_config.max_pct_per_house,
            min_weight=effective_min_weight,
            max_weight=effective_max_weight,
            valid_days=valid_days,
            optimizer_type=optimizer_type,
            adjust_priority=self.optimizer.data_loader.adjust_priority_column,
            feed_price=feed_price,
            input_file_path_price=input_file_path_price

        )

        best_date, best_harvest_df, best_updated_df, _ = self.optimizer.get_best_net_meat_plan(results_by_start_day,optimizer_type=optimizer_type)


        
        if best_harvest_df is not None and not best_harvest_df.empty:
            # Add iteration tracking
            best_harvest_df['age_expansion_iteration'] = expansion_day
        
        return best_harvest_df, best_updated_df


