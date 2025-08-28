"""
Main orchestrator for the harvest optimization system.
"""

import pandas as pd
from typing import Dict, Optional, Tuple, Any
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
        max_pct_per_house: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run daily harvest loop for slaughterhouse.
        
        Args:
            df_input: Input DataFrame
            start_date: Start date for harvesting
            min_weight: Minimum weight threshold
            max_weight: Maximum weight threshold
            min_stock: Minimum stock threshold
            max_stock: Maximum stock threshold
            num_days: Number of days to run
            max_pct_per_house: Maximum percentage per house
            
        Returns:
            Tuple of (harvest results, updated DataFrame)
        """
        df = df_input.copy()
        df_all = df.copy()
        full_results = []
        start_date = pd.to_datetime(start_date)
        
        for day_offset in range(num_days):
            current_date = start_date + pd.Timedelta(days=day_offset)
            logger.info(f"Running harvest for {current_date.date()}")
            
            # Step 1: Re-flag SH readiness
            df = self.model_builder.flag_ready_avg_weight(df, min_weight, max_weight, 'SH')
            df = self.model_builder.flag_ready_daily_stock(df, min_stock, max_stock, 'SH')
            in_df = df[(df['flag_day_sh'] == 1) & (df['ready_SH'] == 1)]
            
            if in_df.empty:
                logger.info("No eligible houses - skipping")
                continue
            
            # Step 2: Filter only current_date
            day_df = in_df[in_df['date'] == current_date].copy()
            if day_df.empty:
                logger.info(f"No houses to harvest on {current_date.date()} - skipping")
                continue

            #skip if thursday
            if current_date.dayofweek == 3:
                logger.info(f"Skipping {current_date.date()} - Thursday")
                continue
            
            # Step 3: Run uniform percentage distribution optimizer
            result = self.model_builder.distribute_slaughterhouse_uniform_pct(
                day_df, 
                min_total_stock=min_stock,
                max_pct_per_house=max_pct_per_house
            )
            
            if result.empty:
                logger.warning(f"No harvest solution for {current_date.date()}")
                continue
            
            # Step 5: Keep only current date results
            result = result[result['date'] == current_date]
            
            if result.empty:
                logger.warning(f"Solution doesn't match current date {current_date.date()}")
                continue
            
            # Step 6: Apply updates
            full_results.append(result)
            df_all = self.solution_processor.apply_harvest_updates(df_all, result)
            df = self.solution_processor.apply_harvest_updates_only_once(df, result)
        
        # Step 7: Combine final output
        harvest_df = pd.concat(full_results, ignore_index=True) if full_results else pd.DataFrame()
        return harvest_df, df_all
    
    def run_multiple_slaughterhouse_starts(
        self,
        df_input: pd.DataFrame,
        min_weight: float,
        max_weight: float,
        min_stock: int,
        max_stock: int,
        max_pct_per_house: float = 1.0
    ) -> Tuple[Dict[Any, Dict], Dict[Any, pd.DataFrame]]:
        """
        Run slaughterhouse harvest with multiple start dates.
        
        Args:
            df_input: Input DataFrame
            min_weight: Minimum weight threshold
            max_weight: Maximum weight threshold
            min_stock: Minimum stock threshold
            max_stock: Maximum stock threshold
            max_pct_per_house: Maximum percentage per house
            
        Returns:
            Tuple of (results by start day, updated DataFrames)
        """
        df_base = df_input.copy()
        df_base['date'] = pd.to_datetime(df_base['date'])
        
        results_by_start_day = {}
        all_updated_dfs = {}
        
        # Flag SH eligibility
        df_flagged = self.model_builder.flag_ready_avg_weight(df_base, min_weight, max_weight, 'SH')
        df_flagged = self.model_builder.flag_ready_daily_stock(df_flagged, min_stock, max_stock, 'SH')
        in_df = df_flagged[(df_flagged['flag_day_sh'] == 1) & (df_flagged['ready_SH'] == 1)]
        
        if in_df.empty:
            logger.warning("No eligible houses for harvest")
            return {}, {}
        
        in_start = in_df['date'].min()
        in_end = in_df['date'].max()
        date_range = pd.date_range(in_start, in_end, freq='1D')
        
        for current_start in date_range:
            logger.info(f"Running harvest loop starting at {current_start.date()}")
            
            df_restart = df_base.copy()
            
            harvest_df, updated_df = self.run_slaughterhouse_harvest_loop(
                df_input=df_restart,
                start_date=current_start,
                min_weight=min_weight,
                max_weight=max_weight,
                min_stock=min_stock,
                max_stock=max_stock,
                num_days=(in_end - in_start).days + 1,
                max_pct_per_house=max_pct_per_house
            )
            
            if harvest_df.empty:
                logger.warning(f"No harvest generated for start date {current_start.date()}")
                continue
            
            actual_start_date = harvest_df['date'].min().date()
            results_by_start_day[actual_start_date] = {
                'harvest': harvest_df,
                'updated_df': updated_df
            }
            all_updated_dfs[actual_start_date] = updated_df.copy()
        
        return results_by_start_day, all_updated_dfs
    
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
        valid_days: pd.DataFrame = None
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
                problem = self.model_builder.build_market_model(
                    df_day,
                    max_total_stock=max_total_stock,
                    min_total_stock=min_total_stock,
                    max_pct_per_house=max_pct_per_house,
                    bypass_capacity=bypass_capacity
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
                            problem = self.model_builder.build_market_model(
                                df_day,
                                max_total_stock=remaining_capacity_value,
                                min_total_stock=2000,
                                max_pct_per_house=max_pct_per_house,
                                capacity_penalty=1000000,
                                bypass_capacity=bypass_capacity
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
                            extra_df.to_csv(f"extra_df_{current_date.date()}.csv", index=False)

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
                            extra_df.to_csv(f"extra_df_{current_date.date()}.csv", index=False)
                            harvest_df = pd.concat([harvest_df, extra_df], ignore_index=True)
            
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
        bypass_constraint: int = 0
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
                valid_days=valid_days if valid_days is not None else None
            )
            
            results_by_start_day[current_max_date.date()] = {
                'harvest': harvest_plan,
                'updated_df': updated_df
            }
            all_updated_dfs[current_max_date.date()] = updated_df.copy()
            
            current_max_date -= pd.Timedelta(days=1)
        
        return results_by_start_day, all_updated_dfs
    
    def get_best_net_meat_plan(self, plan_dict: Dict) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Find the best plan by total net meat.
        
        Args:
            plan_dict: Dictionary of plans by date
            
        Returns:
            Tuple of (best_date, best_harvest_df, best_updated_df, sorted_summary)
        """
        summary = {}
        
        for date_key, data in plan_dict.items():
            harvest_df = data.get('harvest')
            if isinstance(harvest_df, pd.DataFrame) and not harvest_df.empty and 'net_meat' in harvest_df.columns:
                total_meat = harvest_df['net_meat'].sum()
                summary[date_key] = total_meat
        
        if not summary:
            logger.warning("No valid harvest plans with net_meat found")
            return None, None, None, {}
        
        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))
        best_date = next(iter(sorted_summary))
        best_harvest_df = plan_dict[best_date]['harvest']
        best_updated_df = plan_dict[best_date]['updated_df']
        
        return best_date, best_harvest_df, best_updated_df, sorted_summary
    
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
        
        # Log summary
        total_culls_stock = culls_df['harvest_stock'].sum()
        total_culls_meat = culls_df['net_meat'].sum()
        houses_with_culls = len(culls_df)
        
        logger.info(f"Culls allocation calculated:")
        logger.info(f"  - Total culls stock: {total_culls_stock:,}")
        logger.info(f"  - Total culls meat: {total_culls_meat:,.1f} kg")
        logger.info(f"  - Houses with culls: {houses_with_culls}")
        
        return culls_df


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

        
        # Step 2: Run slaughterhouse optimization
        if self.sh_config is None:
            logger.info("Skipping slaughterhouse optimization - configuration not set")
            best_sh_plan = pd.DataFrame()
            best_sh_updated = ready_df.copy()
        else:
            sh_results, sh_updated_dfs = self.optimizer.run_multiple_slaughterhouse_starts(
                ready_df,
                min_weight=self.sh_config.min_weight,
                max_weight=self.sh_config.max_weight,
                min_stock=self.sh_config.min_stock,
                max_stock=self.sh_config.max_stock,
                max_pct_per_house=self.sh_config.max_pct_per_house
            )
            
            # Get best slaughterhouse plan
            best_sh_date, best_sh_plan, best_sh_updated, sh_summary = self.optimizer.get_best_net_meat_plan(sh_results)
            best_sh_updated.to_csv(f"best_sh_updated.csv", index=False)

            
            # Export slaughterhouse summary        
            
            if best_sh_plan is None or best_sh_plan.empty:
                logger.warning("No valid slaughterhouse harvest plan found")
                best_sh_plan = pd.DataFrame()
                best_sh_updated = ready_df.copy()
            
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
                max_weight=self.market_config.max_weight
            )

            M_best_date, market_harvest, market_updated, _ = self.optimizer.get_best_net_meat_plan(results_by_start_day_M)

            unique_pairs = ready_df[['farm', 'house']].drop_duplicates()
            sh_harvested_pairs = best_sh_plan[['farm', 'house']].drop_duplicates()
            market_updated_pairs = market_updated[['farm', 'house']].drop_duplicates()
            market_harvest_pairs = market_harvest[['farm', 'house']].drop_duplicates()
            
            # Find farm-house pairs that are in ready_df but NOT in either sh_harvested_pairs OR market_updated_pairs
            # First, combine both harvested datasets
            all_harvested_pairs = pd.concat([sh_harvested_pairs, market_updated_pairs, market_harvest_pairs]).drop_duplicates()

            
            # Perform anti-join to get pairs in ready_df but not in any harvested data
            not_in_unharvested = unique_pairs.merge(all_harvested_pairs, on=['farm', 'house'], how='left', indicator=True)
            not_in_unharvested = not_in_unharvested[not_in_unharvested['_merge'] == 'left_only'].drop(columns=['_merge'])

            ready_df_not_in_unharvested = ready_df[ready_df['farm'].isin(not_in_unharvested['farm']) & ready_df['house'].isin(not_in_unharvested['house'])]

            expanded_updated_stock_initial = market_harvest.copy()
            combined_updated_stock_initial = market_harvest.copy()
            age_expansion_updated_stock_initial = market_harvest.copy()
            recursive_updated_stock_initial = market_harvest.copy()
            
            recursive_unharvested = pd.concat([ready_df_not_in_unharvested, market_updated])
            expanded_unharvested = pd.concat([ready_df_not_in_unharvested, market_updated])
            combined_unharvested = pd.concat([ready_df_not_in_unharvested, market_updated])
            age_expansion_unharvested = pd.concat([ready_df_not_in_unharvested, market_updated])

        # Step 4: Calculate culls allocation
        culls_plan = self.optimizer.calculate_culls_allocation(
            ready_df,
            cull_percentage=self.service_config.cull_adjustment,
            culls_avg_weight=self.service_config.culls_avg_weight
        )

                
        # Step 6: Export CSV files BEFORE recursive optimization
        pre_recursive_exporter = HarvestPlanExporter(f"harvest_results/{output_dir}_base")
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
            cull_df=culls_plan
        )
        logger.info(f"Exported base optimization files to {output_dir}: {list(pre_recursive_files.keys())}")
        
        
        if self.market_config is not None:
            # Run recursive optimization (separate from expanded)
            recursive_market_harvest, recursive_updated_stock, _ = self._run_recursive_market_optimization(
                recursive_updated_stock_initial, recursive_unharvested, output_dir
            )
            
            # Run expanded optimization (separate from recursive)
            expanded_market_harvest, expanded_updated_stock = self._run_expanded_market_optimization(
                expanded_updated_stock_initial, expanded_unharvested, expanded_sh_plan, culls_plan, output_dir
            )
            
            # Run combined recursive + expanded optimization with dynamic capacity
            combined_market_harvest, combined_updated_stock = self._run_combined_market_optimization(
                combined_updated_stock_initial, combined_unharvested,  combined_sh_plan, culls_plan, removed_df, output_dir
            )

            # Run age expansion optimization for unharvested stock (+3 days max_harvest_age)
            age_expansion_market_harvest, age_expansion_updated_stock = self._run_unharvested_age_expansion_market_optimization(
                age_expansion_updated_stock_initial, age_expansion_unharvested, best_sh_plan, removed_df, culls_plan, output_dir
            )
        
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
        if not market_harvest.empty:
            expanded_harvest_plans.append(market_harvest)
        if not expanded_market_harvest.empty:
            expanded_harvest_plans.append(expanded_market_harvest)
            
        if not culls_plan.empty:
            expanded_harvest_plans.append(culls_plan)
        
        expanded_full_harvest_plan = pd.concat(expanded_harvest_plans, ignore_index=True) if expanded_harvest_plans else pd.DataFrame()
        expanded_market_harvest = pd.concat([market_harvest, expanded_market_harvest], ignore_index=True) if not market_harvest.empty and not expanded_market_harvest.empty else pd.DataFrame()
        
        # Step 9.5: Create combined harvest plan (base + combined market harvest with dynamic capacity)
        combined_harvest_plans = []
        if not best_sh_plan.empty:
            combined_harvest_plans.append(best_sh_plan)
        if not combined_market_harvest.empty:
            combined_harvest_plans.append(combined_market_harvest)
        if not culls_plan.empty:
            combined_harvest_plans.append(culls_plan)
        
        combined_full_harvest_plan = pd.concat(combined_harvest_plans, ignore_index=True) if combined_harvest_plans else pd.DataFrame()
        
        # Step 9.6: Create age expansion harvest plan (base + age expansion market harvest)
        age_expansion_harvest_plans = []
        if not best_sh_plan.empty:
            age_expansion_harvest_plans.append(best_sh_plan)
        if not age_expansion_market_harvest.empty:
            age_expansion_harvest_plans.append(age_expansion_market_harvest)
        if not culls_plan.empty:
            age_expansion_harvest_plans.append(culls_plan)
        
        age_expansion_full_harvest_plan = pd.concat(age_expansion_harvest_plans, ignore_index=True) if age_expansion_harvest_plans else pd.DataFrame()
        
        # Step 10: Export CSV files AFTER recursive optimization (recursive results only)
        recursive_output_dir = f"harvest_results/{output_dir}_capacity_increased"
        recursive_exporter = HarvestPlanExporter(recursive_output_dir)
        recursive_files = recursive_exporter.generate_harvest_plan_csvs(
            harvest_df=recursive_full_harvest_plan,
            unharvested_df=recursive_updated_stock,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=recursive_market_harvest,
            cull_df=culls_plan
        )
        logger.info(f"Exported recursive optimization files to {recursive_output_dir}: {list(recursive_files.keys())}")
        
        # Step 11: Export CSV files AFTER expanded optimization (expanded results only)
        expanded_output_dir = f"harvest_results/{output_dir}_weight_expanded"
        expanded_exporter = HarvestPlanExporter(expanded_output_dir)
        expanded_files = expanded_exporter.generate_harvest_plan_csvs(
            harvest_df=expanded_full_harvest_plan,
            unharvested_df=expanded_updated_stock,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=expanded_market_harvest,
            cull_df=culls_plan
        )
        logger.info(f"Exported expanded optimization files to {expanded_output_dir}: {list(expanded_files.keys())}")
        
        # Step 12: Export CSV files for combined optimization (combined results)
        combined_output_dir = f"harvest_results/{output_dir}_combined"
        combined_exporter = HarvestPlanExporter(combined_output_dir)
        combined_files = combined_exporter.generate_harvest_plan_csvs(
            harvest_df=combined_full_harvest_plan,
            unharvested_df=combined_updated_stock,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=combined_market_harvest,
            cull_df=culls_plan
        )
        logger.info(f"Exported combined optimization files to {combined_output_dir}: {list(combined_files.keys())}")
        
        # Step 13: Export CSV files for age expansion optimization (age expansion results)
        age_expansion_output_dir = f"harvest_results/{output_dir}_age_expansion"
        age_expansion_exporter = HarvestPlanExporter(age_expansion_output_dir)
        age_expansion_files = age_expansion_exporter.generate_harvest_plan_csvs(
            harvest_df=age_expansion_full_harvest_plan,
            unharvested_df=age_expansion_updated_stock,
            rejection_df=removed_df,
            ready_df=ready_df,
            best_sh_plan=best_sh_plan,
            best_market_plan=age_expansion_market_harvest,
            cull_df=culls_plan
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
            "slaughterhouse_harvest": best_sh_plan,
            "base_market_harvest": market_harvest,
            "recursive_market_harvest": recursive_market_harvest,
            "expanded_market_harvest": expanded_market_harvest,
            "culls_harvest": culls_plan,
            "combined_market_harvest": combined_market_harvest,
            "age_expansion_market_harvest": age_expansion_market_harvest,
            "base_full_harvest_plan": pre_recursive_full_plan,
            "recursive_full_harvest_plan": recursive_full_harvest_plan,
            "expanded_full_harvest_plan": expanded_full_harvest_plan,
            "combined_full_harvest_plan": combined_full_harvest_plan,
            "age_expansion_full_harvest_plan": age_expansion_full_harvest_plan,
            "base_unharvested_stock": market_updated,
            "recursive_unharvested_stock": recursive_updated_stock,
            "expanded_unharvested_stock": expanded_updated_stock,
            "combined_unharvested_stock": combined_updated_stock,
            "age_expansion_unharvested_stock": age_expansion_updated_stock,
            "removed_rows": removed_df,
            "exported_files": exported_files,
            "summary": {
                "total_slaughterhouse_stock": best_sh_plan['harvest_stock'].sum() if not best_sh_plan.empty else 0,
                "total_slaughterhouse_meat": best_sh_plan['net_meat'].sum() if not best_sh_plan.empty else 0,
                "base_market_stock": market_harvest['harvest_stock'].sum() if not market_harvest.empty else 0,
                "base_market_meat": market_harvest['net_meat'].sum() if not market_harvest.empty else 0,
                "total_base_market_stock": market_harvest['harvest_stock'].sum() if not market_harvest.empty else 0,
                "total_base_market_meat": market_harvest['net_meat'].sum() if not market_harvest.empty else 0,
                "recursive_market_stock": recursive_market_harvest['harvest_stock'].sum() if not recursive_market_harvest.empty else 0,
                "recursive_market_meat": recursive_market_harvest['net_meat'].sum() if not recursive_market_harvest.empty else 0,
                "additional_expanded_market_stock": expanded_market_harvest['harvest_stock'].sum() if not expanded_market_harvest.empty else 0,
                "additional_expanded_market_meat": expanded_market_harvest['net_meat'].sum() if not expanded_market_harvest.empty else 0,
                "total_expanded_market_stock": (market_harvest['harvest_stock'].sum() + expanded_market_harvest['harvest_stock'].sum() if not market_harvest.empty else 0) if not expanded_market_harvest.empty else 0,
                "total_expanded_market_meat": (market_harvest['net_meat'].sum() + expanded_market_harvest['net_meat'].sum() if not market_harvest.empty else 0) if not expanded_market_harvest.empty else 0,
                "base_total_harvest_stock": pre_recursive_full_plan['harvest_stock'].sum() if not pre_recursive_full_plan.empty else 0,
                "base_total_harvest_meat": pre_recursive_full_plan['net_meat'].sum() if not pre_recursive_full_plan.empty else 0,
                "combined_market_stock": combined_market_harvest['harvest_stock'].sum() if not combined_market_harvest.empty else 0,
                "combined_market_meat": combined_market_harvest['net_meat'].sum() if not combined_market_harvest.empty else 0,
                "age_expansion_market_stock": age_expansion_market_harvest['harvest_stock'].sum() if not age_expansion_market_harvest.empty else 0,
                "age_expansion_market_meat": age_expansion_market_harvest['net_meat'].sum() if not age_expansion_market_harvest.empty else 0,
                "recursive_total_harvest_stock": recursive_full_harvest_plan['harvest_stock'].sum() if not recursive_full_harvest_plan.empty else 0,
                "recursive_total_harvest_meat": recursive_full_harvest_plan['net_meat'].sum() if not recursive_full_harvest_plan.empty else 0,
                "expanded_total_harvest_stock": expanded_full_harvest_plan['harvest_stock'].sum() if not expanded_full_harvest_plan.empty else 0,
                "expanded_total_harvest_meat": expanded_full_harvest_plan['net_meat'].sum() if not expanded_full_harvest_plan.empty else 0,
                "combined_total_harvest_stock": combined_full_harvest_plan['harvest_stock'].sum() if not combined_full_harvest_plan.empty else 0,
                "combined_total_harvest_meat": combined_full_harvest_plan['net_meat'].sum() if not combined_full_harvest_plan.empty else 0,
                "age_expansion_total_harvest_stock": age_expansion_full_harvest_plan['harvest_stock'].sum() if not age_expansion_full_harvest_plan.empty else 0,
                "age_expansion_total_harvest_meat": age_expansion_full_harvest_plan['net_meat'].sum() if not age_expansion_full_harvest_plan.empty else 0,
                "total_culls_stock": culls_plan['harvest_stock'].sum() if not culls_plan.empty else 0,
                "total_culls_meat": culls_plan['net_meat'].sum() if not culls_plan.empty else 0,
            }
        }
        
        logger.info("Full harvest optimization completed successfully")
        return results

    def _run_expanded_market_optimization(
        self, 
        initial_market_harvest: pd.DataFrame, 
        initial_unharvested_stock: pd.DataFrame,
        sh_harvest: pd.DataFrame,
        culls_harvest: pd.DataFrame,
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
            expanded_flagged_stock.to_csv("expanded_flagged_stock.csv", index=False)
            
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
                daily_df_full['existing_allocation'] >= self.market_config.max_stock,
                'remaining_capacity'
            ] = 20000
            # Prevent negatives
            daily_df_full['remaining_capacity'] = daily_df_full['remaining_capacity'].clip(lower=0)

            # Save diagnostics

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
            valid_days = valid_days[valid_days['remaining_capacity'] > 0].copy()
    
            

            
        if ready_expanded_stock.empty:
            logger.info("No stock ready after capacity filtering")
            return pd.DataFrame(), initial_unharvested_stock
        
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
            valid_days=valid_days
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
        output_dir: str,
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
            
            # Run market optimization on current unharvested stock
            results_by_start_day, _ = self.optimizer.run_market_reverse_sweep(
                current_unharvested,
                max_total_stock=iteration_config.max_stock,
                min_total_stock=iteration_config.min_stock,
                tolerance_step=iteration_config.tolerance_step,
                max_tolerance=iteration_config.max_tolerance,
                max_pct_per_house=iteration_config.max_pct_per_house,
                min_weight=iteration_config.min_weight,
                max_weight=iteration_config.max_weight
            )
            
            best_date, additional_harvest, updated_unharvested, _ = self.optimizer.get_best_net_meat_plan(results_by_start_day)

            # updated_unharvested.to_csv(f"updated_unharvested_{iteration}.csv", index=False)
            
            if additional_harvest is None or additional_harvest.empty:
                logger.info(f"No additional harvest found in iteration {iteration} - increasing max_stock")
                current_pct = iteration * 0.2
                current_max_stock = current_pct * starting_max_stock 
                iteration += 1
                
                # Prevent infinite loop - stop after reasonable number of iterations
                if iteration > 6:
                    logger.warning("Reached maximum recursive optimization iterations - stopping")
                    break
                continue
            
            # Add to accumulated harvests
            additional_harvest['iteration'] = iteration
            all_additional_harvests.append(additional_harvest)
            
            current_unharvested = updated_unharvested
            
            logger.info(f"Iteration {iteration} harvested {additional_harvest['harvest_stock'].sum()} stock")
            
            # Increase max_stock for next iteration
            current_pct = iteration * 0.2
            current_max_stock = current_pct * starting_max_stock
            iteration += 1
            
            # Prevent infinite loop
            if iteration > 6:
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
        
        logger.info(f"Recursive market optimization completed after {iteration-1} iterations")
        return combined_market_harvest, current_unharvested, iteration-1
    
    
    def _run_combined_market_optimization(
        self, 
        initial_market_harvest: pd.DataFrame, 
        initial_unharvested_stock: pd.DataFrame,
        sh_harvest: pd.DataFrame,
        culls_harvest: pd.DataFrame,
        removed_df: pd.DataFrame,
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
            expanded_flagged_stock.to_csv("expanded_flagged_stock.csv", index=False)
            
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
                daily_df_full['existing_allocation'] >= self.market_config.max_stock,
                'remaining_capacity'
            ] = 20000
            # Prevent negatives
            daily_df_full['remaining_capacity'] = daily_df_full['remaining_capacity'].clip(lower=0)

            # Save diagnostics

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
            valid_days = valid_days[valid_days['remaining_capacity'] > 0].copy()

            
        if ready_expanded_stock.empty:
            logger.info("No stock ready after capacity filtering")
            return pd.DataFrame(), initial_unharvested_stock
        
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
            valid_days=valid_days
        )

        if not expanded_harvest.empty:
            logger.info(f"Expanded optimization harvested {expanded_harvest['harvest_stock'].sum()} additional stock")
            expanded_harvest = pd.concat([initial_market_harvest, expanded_harvest], ignore_index=True)
        else:
            logger.info("No additional harvest from expanded optimization")
        
        #pass to age expansion
        expanded_harvest, remaining_unharvested = self._run_unharvested_age_expansion_market_optimization(
            expanded_harvest,
            remaining_unharvested,
            sh_harvest,
            removed_df,
            culls_harvest,
            output_dir,
            age_expansion_days=3
        )

        return expanded_harvest, remaining_unharvested
        

    
    def _run_unharvested_age_expansion_market_optimization(
        self, 
        initial_market_harvest: pd.DataFrame,
        unharvested_stock: pd.DataFrame,
        best_sh_plan: pd.DataFrame,
        removed_df: pd.DataFrame,
        culls_plan: pd.DataFrame,
        output_dir: str,
        age_expansion_days: int = 3
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
            
        Returns:
            Tuple of (cumulative harvest, final unharvested stock)
        """
        logger.info(f"Running sequential market optimization with age expansion (+{age_expansion_days} days)")

        if unharvested_stock.empty:
            logger.info("No unharvested stock - skipping age expansion optimization")
            return initial_market_harvest, unharvested_stock

        # Initialize tracking variables
        cumulative_harvest = pd.concat([initial_market_harvest, best_sh_plan], ignore_index=True)
        current_unharvested = unharvested_stock.copy()
        current_unharvested.to_csv(f"current_unharvested_0.csv", index=False)
        base_max_date = cumulative_harvest['date'].max()
        
        # Get unique farm-house combinations from unharvested stock
        unharvested_farm_houses = current_unharvested[['farm', 'house']].drop_duplicates()

        # Sequential age expansion - Fixed range to include all days
        for expansion_day in range(1, age_expansion_days + 1):
            logger.info(f"Running iteration {expansion_day}: expanding max_harvest_age by +{expansion_day} days")

            
            current_unharvested.to_csv(f"current_unharvested_{expansion_day}.csv", index=False)
            # Create age-expanded dataset for this iteration
            df_age_expanded = self._create_age_expanded_data(
                removed_df, unharvested_farm_houses, expansion_day, cumulative_harvest
            )
            
            if df_age_expanded.empty:
                logger.info(f"No age-expanded data for iteration {expansion_day}")
                continue

            # Combine with current unharvested stock
            combined_stock = pd.concat([df_age_expanded, current_unharvested], ignore_index=True)
            
            # Apply weight range flagging
            flagged_stock = self.optimizer.model_builder.flag_ready_avg_weight(
                combined_stock, 
                self.market_config.min_weight, 
                self.market_config.max_weight, 
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
                ready_stock, valid_days, expansion_day, output_dir
            )
            
            
            if best_harvest_df is not None and not best_harvest_df.empty:
                cumulative_harvest = pd.concat([cumulative_harvest[cumulative_harvest['harvest_type'] == 'Market'], best_harvest_df], ignore_index=True)
                current_unharvested = best_updated_df.copy()
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
        output_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run market optimization for a specific iteration."""
        
        # Filter stock to valid days only
        # filtered_stock = combined_stock[combined_stock['date'].isin(valid_days['date'])].copy()
        

        
        # Run market optimization
        results_by_start_day, _ = self.optimizer.run_market_reverse_sweep(
            df_input=combined_stock,
            max_total_stock=self.market_config.max_stock,
            min_total_stock=self.market_config.min_stock,
            tolerance_step=self.market_config.tolerance_step,
            max_tolerance=self.market_config.max_tolerance,
            harvest_type='Market',
            max_pct_per_house=self.market_config.max_pct_per_house,
            min_weight=self.market_config.min_weight,
            max_weight=self.market_config.max_weight,
            valid_days=valid_days
        )

        best_date, best_harvest_df, best_updated_df, _ = self.optimizer.get_best_net_meat_plan(results_by_start_day)
        
        if best_harvest_df is not None and not best_harvest_df.empty:
            # Add iteration tracking
            best_harvest_df['age_expansion_iteration'] = expansion_day
        
        return best_harvest_df, best_updated_df


