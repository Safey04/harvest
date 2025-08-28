"""
Data loading and preprocessing utilities for the harvest optimization system.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
import logging
from .models import MarketConfig, OptimizationConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of growth data."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize the data loader."""
        self.config = config
        # Extract commonly used configs for convenience
        self.market_config = MarketConfig(
            min_weight=config.weight_ranges.market.min,
            max_weight=config.weight_ranges.market.max,
            min_stock=99900,  # Default values
            max_stock=100000,
            max_pct_per_house=1.0,
            tolerance_step=1000,
            max_tolerance=10000
        )
        self.price_df = None
    
    def load_growth_data(self, file_path: str) -> pd.DataFrame:
        """
        Load growth data from CSV file.
        
        Args:
            file_path: Path to the growth data CSV file
            
        Returns:
            DataFrame with loaded growth data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
        
    def load_price_data(self, file_path: str) -> pd.DataFrame:
        """
        Load price data from CSV file.
        """
        self.price_df = pd.read_csv(file_path)
        return self.price_df
    
    def get_price(self, date: str, file_path_price: str) -> float:
        """
        Get price for a given date from the price data.
        """
        if self.price_df is None:
            self.price_df = self.load_price_data(file_path_price)
            df = self.price_df
        else: 
            df = self.price_df
        df['date'] = pd.to_datetime(df['date'])
        return df.loc[df['date'] == date, 'price'].values[0]

    def calculate_max_harvest_date(
        self,
        df: pd.DataFrame,
        duration_days: int = 40,
        cleaning_days: int = 15,
        safety_days: int = 3
    ) -> str:
        """
        Calculate the maximum harvest date automatically based on the first placement date.
        
        Formula: max_harvest_date = first_placement_date + duration_days + cleaning_days + safety_days
        First placement date is the earliest date where age=1 across all houses.
        
        Args:
            df: Input DataFrame with age and date columns
            duration_days: Maximum age days for harvesting
            cleaning_days: Days needed for cleaning after harvest cycle
            safety_days: Additional safety buffer days
            
        Returns:
            Maximum harvest date as string in YYYY-MM-DD format
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the earliest date where age is 1 (first placement date)
        age_1_dates = df[df['age'] == 1]['date']
        if age_1_dates.empty:
            raise ValueError("No records found with age=1 to determine first placement date")
        
        first_placement_date = age_1_dates.min()
        
        # Calculate max harvest date
        total_days = duration_days + cleaning_days
        max_harvest_date = first_placement_date + pd.Timedelta(days=total_days) 
        
        logger.info(f"Calculated max harvest date: {max_harvest_date.strftime('%Y-%m-%d')}")
        logger.info(f"  - First placement date: {first_placement_date.strftime('%Y-%m-%d')}")
        logger.info(f"  - Duration days: {duration_days}")
        logger.info(f"  - Cleaning days: {cleaning_days}")
        logger.info(f"  - Safety days: {safety_days}")
        logger.info(f"  - Total cycle days: {total_days}")
        
        return max_harvest_date.strftime('%Y-%m-%d')

    def add_house_dates_columns(
        self, 
        df: pd.DataFrame, 
        duration_days: int = 40, 
        max_harvest_date: str = None,
        cleaning_days: int = 15,
        safety_days: int = 3
    ) -> pd.DataFrame:
        """
        Add date-related columns to the DataFrame.
        
        Args:
            df: Input DataFrame
            duration_days: Days to add to start_date to compute last_date
            max_harvest_date: Maximum harvest date for all rows (if None, will be calculated automatically)
            cleaning_days: Days needed for cleaning (used for automatic calculation)
            safety_days: Safety buffer days (used for automatic calculation)
            
        Returns:
            DataFrame with added date columns
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate max_harvest_date automatically if not provided
        if max_harvest_date is None:
            max_harvest_date = self.calculate_max_harvest_date(df, duration_days, cleaning_days, safety_days)
        
        # Convert max_harvest_date to pd.Timestamp
        max_harvest_date = pd.to_datetime(max_harvest_date)
        
        # Get start and end dates per (farm, house) group
        start_dates = df.groupby(['farm', 'house'])['date'].transform('min')
        end_dates = df.groupby(['farm', 'house'])['date'].transform('max')
        
        df['start_date'] = start_dates
        df['end_date'] = end_dates
        df['last_date'] = df['start_date'] + pd.to_timedelta(duration_days-1, unit='D')
        df['max_harvest_date'] = max_harvest_date
        
        return df

    def add_feed_consumption_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add feed consumption column to the DataFrame.
        """
        feed_consumption_cumulative = [
            12, 28, 48, 73, 101, 132, 166,
            204, 247, 295, 346, 404, 467, 535,
            609, 688, 773, 864, 962, 1065, 1174,
            1289, 1409, 1535, 1667, 1805, 1948,
            2096, 2249, 2408, 2572, 2742, 2916,
            3095, 3278, 3465, 3657, 3852, 4051,
            4252, 4458, 4666, 4878, 5092, 5307,
            5526, 5747, 5969, 6194, 6419, 6646,
            6874, 7102, 7332, 7561, 7791
            ]
        df = df.copy()

        # from gram to kg
        feed_consumption_cumulative = [x/1000 for x in feed_consumption_cumulative]

        #add feed_consumption_cumulative column
        df['feed_consumption_cumulative'] = df['age'].apply(lambda x: feed_consumption_cumulative[x-1])

        #multiply by expected_stock
        df['feed_consumption'] = df['expected_stock'] * df['feed_consumption_cumulative']

        df = df.drop(columns=['feed_consumption_cumulative'])
        
        return df

    def add_price_column(self, df: pd.DataFrame, file_path_price: str) -> pd.DataFrame:
        """
        Add price column to the DataFrame.
        """
        df = df.copy()
        df['price'] = df['date'].apply(lambda x: self.get_price(x, file_path_price))
        return df

    def add_total_profit_column(self, df: pd.DataFrame, feed_price: float) -> pd.DataFrame:
        """
        Add total profit column to the DataFrame.
        """
        df = df.copy()
        df['feed_cost'] = df['feed_consumption'] * feed_price
    
        df['total_profit'] = (df['expected_stock'] * df['avg_weight'] * df['price']) - df['feed_cost']
        df['total_profit'] = df['total_profit'].round(0).astype(int)
        df['profit_per_bird'] = df['total_profit'] / df['expected_stock']
        df['profit_per_bird'] = df['profit_per_bird'].round(2)
        return df

    def add_profit_loss_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add meat loss column to the DataFrame.
        """
        df = df.copy()
        # Calculate meat loss for each farm-house group (day - next day )

        stock_loss = df.groupby(['farm', 'house'])['expected_stock'].diff().fillna(0).abs().astype(int)
        df['profit_loss'] = stock_loss * df["avg_weight"].fillna(0) * df['price']
        return df
    
    def add_priority_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add priority column to the DataFrame.
        """
        df = df.copy()
        # Prioritize days with age between 28 and 40 (inclusive) within each farm-house group,
        # ranking by highest total_profit first, without changing row order.
        # Lower numbers indicate higher priority.
        age_priority_mask = df['age'].between(28, 40)

        # Count prioritized rows per group to offset non-priority ranks
        prioritized_counts = df.groupby(['farm', 'house'])['age'].transform(lambda s: s.between(28, 40).sum())

        # Rank by total_profit (descending) within each group; 'first' preserves current order for ties
        prioritized_rank = (
            df.loc[age_priority_mask]
              .groupby(['farm', 'house'])['profit_per_bird']
              .rank(method='first', ascending=False)
              .astype('int64')
        )
        non_prioritized_rank = (
            df.loc[~age_priority_mask]
              .groupby(['farm', 'house'])['profit_per_bird']
              .rank(method='first', ascending=False)
              .astype('int64')
        )

        # Build final priority series: prioritized first, then non-prioritized offset by prioritized count
        priority_series = pd.Series(index=df.index, dtype='int64')
        priority_series.loc[age_priority_mask] = prioritized_rank
        priority_series.loc[~age_priority_mask] = (non_prioritized_rank + prioritized_counts.loc[~age_priority_mask]).astype('int64')

        df['priority'] = priority_series.astype(int)
        return df
    
    def remove_invalid_dates(self, df_with_date: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove rows where date > last_date or date > max_harvest_date, considering allocation type constraints.
        
        Args:
            df_with_date: DataFrame with date columns added
            
        Returns:
            Tuple of (cleaned DataFrame, removed rows DataFrame)
        """
        df = df_with_date.copy()
    
        
        # Boolean masks
        cond_date_gt_last = df['date'] > df['last_date']
        cond_date_gt_max = df['date'] > df['max_harvest_date'] 

        to_remove = cond_date_gt_last | cond_date_gt_max

        # Subset of rows to be removed
        removed_df = df[to_remove].copy()

        # Generate reasons only for removed rows
        reasons = []
        for gt_last, gt_max in zip(cond_date_gt_last[to_remove], cond_date_gt_max[to_remove]):
            reason = []
            if gt_last:
                reason.append("date > last_date")
            if gt_max:
                reason.append("date > max_harvest_date")
            reasons.append("; ".join(reason))

        removed_df['removal_reason'] = reasons

        # Cleaned DataFrame (rows to keep)
        cleaned_df = df[~to_remove].copy()

        return cleaned_df, removed_df
    
    def apply_cull_adjustment(self, df: pd.DataFrame, cull_percentage: float = 0.03) -> pd.DataFrame:
        """
        Apply cull adjustment to expected stock.
        
        Args:
            df: Input DataFrame
            cull_percentage: Percentage to remove for culls
            
        Returns:
            DataFrame with adjusted stock
        """
        df = df.copy()
        df['expected_stock'] = df['expected_stock'] * (1 - cull_percentage)
        df['expected_stock'] = df['expected_stock'].round(0).astype(int)
        
        logger.info(f"Applied {cull_percentage*100}% cull adjustment to expected stock")
        
        return df
    
    def preprocess_data(
        self, 
        file_path: str, 
        file_path_price: str,
        feed_price: float,
        duration_days: int = 40,
        max_harvest_date: str = None,
        cleaning_days: int = 15,
        safety_days: int = 3,
        cull_percentage: float = 0.03
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data preprocessing pipeline.
        
        Args:
            file_path: Path to growth data CSV
            duration_days: Maximum age days for harvesting
            max_harvest_date: Maximum harvest date (if None, will be calculated automatically)
            cleaning_days: Days needed for cleaning after harvest cycle
            safety_days: Additional safety buffer days
            cull_percentage: Cull adjustment percentage
            
        Returns:
            Tuple of (processed DataFrame, removed rows DataFrame)
        """
        # Load data
        df = self.load_growth_data(file_path)
        


        # # Add feed consumption column
        # ready_df = self.add_feed_consumption_column(df)

        # # Add price column
        # ready_df = self.add_price_column(ready_df, file_path_price)

        # # Add total profit column
        # ready_df = self.add_total_profit_column(ready_df, feed_price)

        # # Add meat loss column
        # ready_df = self.add_profit_loss_column(ready_df)
        
        # # Add priority column
        # ready_df = self.add_priority_column(ready_df)

        df_with_dates = self.add_house_dates_columns(df, duration_days, max_harvest_date, cleaning_days, safety_days)
        
        # Remove invalid dates
        cleaned_df, removed_df = self.remove_invalid_dates(df_with_dates)
        
        # Clean up temporary columns
        ready_df = cleaned_df.drop(columns=['start_date', 'end_date', 'last_date', 'max_harvest_date'])
        
        # Store the preprocessed data for later use (e.g., in expanded optimization)
        self.last_preprocessed_data = ready_df.copy()

        ready_df.to_csv('ready_df.csv', index=False)
        
        return ready_df, removed_df