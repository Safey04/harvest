"""
Solution processor for converting optimization results to structured data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

from ..models import HarvestEntry, HarvestType, RejectionEntry, SummaryEntry, DailySummary

logger = logging.getLogger(__name__)


class SolutionProcessor:
    """
    Responsible for processing optimization solutions into structured data objects.
    Follows Single Responsibility Principle - only processes solutions into output formats.
    """
    
    def __init__(self):
        """Initialize the solution processor."""
        self.last_harvest_entries = []
        self.last_rejection_entries = []
    
    def process_slaughterhouse_solution(
        self, 
        variables: Dict[str, Any], 
        data: pd.DataFrame,
        harvest_type: str = 'SH'
    ) -> pd.DataFrame:
        """
        Process slaughterhouse optimization solution into harvest DataFrame.
        
        Args:
            variables: Dictionary of optimization variables
            data: Original data used in optimization
            harvest_type: Type of harvest
            
        Returns:
            DataFrame with harvest results
        """
        results = []
        
        for idx, var in variables.items():
            if var.varValue and var.varValue > 0:
                row = data.loc[idx].copy()
                harvested = int(np.round(var.varValue))
                
                harvest_entry = {
                    'farm': row['farm'],
                    'date': row['date'],
                    'house': row['house'],
                    'age': row['age'],
                    'expected_mortality': row['expected_mortality'],
                    'expected_stock': row['expected_stock'],
                    'expected_mortality_rate': row['expected_mortality_rate'],
                    'avg_weight': row['avg_weight'],
                    'selected': 1,
                    'harvest_stock': harvested,
                    'net_meat': harvested * row['avg_weight'],
                    'harvest_type': harvest_type
                }
                # Include priority if available in the source data
                if 'priority' in row.index:
                    harvest_entry['priority'] = row['priority']
                results.append(harvest_entry)
        
        if results:
            result_df = pd.DataFrame(results)
            logger.info(f"Processed {len(results)} harvest entries for slaughterhouse")
            return result_df
        else:
            logger.warning("No valid harvest solution found")
            return pd.DataFrame()
    
    def process_market_solution(
        self, 
        variables: Dict[str, Any], 
        data: pd.DataFrame,
        harvest_type: str = 'Market'
    ) -> pd.DataFrame:
        """
        Process market optimization solution into harvest DataFrame.
        
        Args:
            variables: Dictionary of optimization variables
            data: Original data used in optimization
            harvest_type: Type of harvest
            
        Returns:
            DataFrame with harvest results
        """
        results = []
        
        for idx, var in variables.items():
            if var.varValue and var.varValue > 0:
                row = data.loc[idx].copy()
                harvested = int(np.round(var.varValue))
                
                harvest_entry = {
                    'farm': row['farm'],
                    'date': row['date'],
                    'house': row['house'],
                    'age': row['age'],
                    'expected_mortality': row['expected_mortality'],
                    'expected_stock': row['expected_stock'],
                    'expected_mortality_rate': row['expected_mortality_rate'],
                    'avg_weight': row['avg_weight'],
                    'harvest_stock': harvested,
                    'net_meat': harvested * row['avg_weight'],
                    'harvest_type': harvest_type,
                }
                # Include priority if available in the source data
                if 'priority' in row.index:
                    harvest_entry['priority'] = row['priority']
                results.append(harvest_entry)
        
        if results:
            result_df = pd.DataFrame(results)
            logger.info(f"Processed {len(results)} harvest entries for market")
            return result_df
        else:
            logger.warning("No valid market harvest solution found")
            return pd.DataFrame()
    
    def apply_harvest_updates(self, df_full: pd.DataFrame, harvest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update the main DataFrame after applying harvests.
        
        Args:
            df_full: Full DataFrame to update
            harvest_df: DataFrame with harvest results
            
        Returns:
            Updated DataFrame
        """
        df = df_full.copy()
        df['expected_stock'] = df['expected_stock'].astype(int)
        df['farm_house_key'] = df['farm'].astype(str) + "_" + df['house'].astype(str)
        df['date'] = pd.to_datetime(df['date'])
        harvest_df['date'] = pd.to_datetime(harvest_df['date'])
        
        for _, harvest in harvest_df.iterrows():
            farm = harvest['farm']
            house = harvest['house']
            key = f"{farm}_{house}"
            date = harvest['date']
            harvested = harvest['harvest_stock']
            prev_stock = harvest['expected_stock'] - harvested
            
            if harvested <= 0:
                continue
            
            # Drop current and previous rows for this house
            df = df[~((df['farm_house_key'] == key) & (df['date'] <= date))]
            
            # If fully harvested, drop all future rows
            if prev_stock <= 0:
                df = df[df['farm_house_key'] != key]
                continue
            
            # Otherwise, update future rows
            mask = (df['farm_house_key'] == key) & (df['date'] > date)
            future_rows = df[mask].sort_values(by='date')
            
            for idx, row in future_rows.iterrows():
                mortality_rate = row['expected_mortality']
                new_stock = prev_stock - mortality_rate
                new_stock_rounded = int(round(new_stock))
                
                if new_stock_rounded > 0:
                    df.at[idx, 'expected_stock'] = new_stock_rounded
                    prev_stock = new_stock
                else:
                    df = df.drop(index=idx)
                    break
        
        df = df[df['expected_stock'] > 0].copy()
        
        # Remove entire farm/house combinations if any day has 0 stock
        farm_house_with_zero = df[df['expected_stock'] == 0][['farm', 'house']].drop_duplicates()
        for _, row in farm_house_with_zero.iterrows():
            farm = row['farm']
            house = row['house']
            df = df[~((df['farm'] == farm) & (df['house'] == house))]
        
        return df
    
    def update_stock_after_harvest_market(
        self, 
        df_input: pd.DataFrame, 
        harvest_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Update stock after market harvest (different logic than slaughterhouse).
        
        Args:
            df_input: Input DataFrame
            harvest_df: Harvest results DataFrame
            
        Returns:
            Updated DataFrame
        """
        df = df_input.copy()
        
        # Convert datetime columns once
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        harvest_df['date'] = pd.to_datetime(harvest_df['date']).dt.normalize()
        
        # Filter out zero or negative harvest stock
        valid_harvest = harvest_df[harvest_df['harvest_stock'] > 0].copy()
        
        if valid_harvest.empty:
            return df
        
        # Create a mapping of (farm, house) to total harvest stock
        harvest_mapping = valid_harvest.groupby(['farm', 'house'])['harvest_stock'].sum()
        
        # Create a multi-index for efficient lookup
        df_index = df.set_index(['farm', 'house']).index
        harvest_index = harvest_mapping.index
        
        # Find matching farm/house combinations
        common_indices = df_index.intersection(harvest_index)
        
        if not common_indices.empty:
            # Update stock for matching combinations
            for farm, house in common_indices:
                mask = (df['farm'] == farm) & (df['house'] == house)
                df.loc[mask, 'expected_stock'] -= harvest_mapping.loc[(farm, house)]
        
        # Remove farm/house combinations with zero or negative stock
        zero_stock_mask = df['expected_stock'] <= 0
        if zero_stock_mask.any():
            # Get unique farm/house combinations with zero stock
            zero_farm_house = df[zero_stock_mask][['farm', 'house']].drop_duplicates()
            
            # Create a mask to exclude these combinations
            exclude_mask = df.set_index(['farm', 'house']).index.isin(zero_farm_house.set_index(['farm', 'house']).index)
            df = df[~exclude_mask]
        
        return df
    
    def generate_rejection_entries(
        self, 
        unharvested_df: pd.DataFrame, 
        reason: str = "Not harvested"
    ) -> List[RejectionEntry]:
        """
        Generate rejection entries for unharvested stock.
        
        Args:
            unharvested_df: DataFrame with unharvested stock
            reason: Reason for rejection
            
        Returns:
            List of RejectionEntry objects
        """
        rejection_entries = []
        
        for _, row in unharvested_df.iterrows():
            entry = RejectionEntry(
                farm=row['farm'],
                house=row['house'],
                date=pd.to_datetime(row['date']),
                age=row['age'],
                expected_stock=row['expected_stock'],
                avg_weight=row['avg_weight'],
                rejection_reason=reason
            )
            rejection_entries.append(entry)
        
        return rejection_entries
    
    def generate_summary_entries(self, harvest_df: pd.DataFrame) -> List[SummaryEntry]:
        """
        Generate summary entries from harvest data.
        
        Args:
            harvest_df: DataFrame with harvest results
            
        Returns:
            List of SummaryEntry objects
        """
        summary_entries = []
        
        # Farm-level summaries
        farm_summary = harvest_df.groupby('farm').agg({
            'harvest_stock': 'sum',
            'net_meat': 'sum',
            'date': 'nunique',
            'house': 'nunique'
        }).reset_index()
        
        for _, row in farm_summary.iterrows():
            entry = SummaryEntry(
                farm=row['farm'],
                house=None,
                total_harvest_stock=row['harvest_stock'],
                total_net_meat=row['net_meat'],
                harvest_days=row['date'],
                unique_houses=row['house']
            )
            summary_entries.append(entry)
        
        return summary_entries
    
    def generate_daily_summaries(self, harvest_df: pd.DataFrame) -> List[DailySummary]:
        """
        Generate daily summary entries.
        
        Args:
            harvest_df: DataFrame with harvest results
            
        Returns:
            List of DailySummary objects
        """
        daily_summaries = []
        
        daily_summary = harvest_df.groupby(['date', 'harvest_type']).agg({
            'harvest_stock': 'sum',
            'net_meat': 'sum',
            'house': 'nunique'
        }).reset_index()
        
        for _, row in daily_summary.iterrows():
            harvest_type = HarvestType.SLAUGHTERHOUSE if row['harvest_type'] == 'SH' else HarvestType.MARKET
            
            entry = DailySummary(
                date=pd.to_datetime(row['date']),
                harvest_type=harvest_type,
                total_stock=row['harvest_stock'],
                total_net_meat=row['net_meat'],
                houses_harvested=row['house']
            )
            daily_summaries.append(entry)
        
        return daily_summaries
    
    def apply_harvest_updates_only_once(self, df_full: pd.DataFrame, harvest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update the main DataFrame after applying harvests, ensuring each house is only updated once.
        This prevents multiple harvests from the same house by removing the house entirely after harvest.
        
        Args:
            df_full: Full DataFrame to update
            harvest_df: DataFrame with harvest results
            
        Returns:
            Updated DataFrame with harvested houses removed
        """
        df = df_full.copy()
        df['expected_stock'] = df['expected_stock'].astype(int)
        df['farm_house_key'] = df['farm'].astype(str) + "_" + df['house'].astype(str)
        df['date'] = pd.to_datetime(df['date'])
        harvest_df['date'] = pd.to_datetime(harvest_df['date'])
        
        harvested_keys = set()
        
        for _, harvest in harvest_df.iterrows():
            farm = harvest['farm']
            house = harvest['house']
            key = f"{farm}_{house}"
            date = harvest['date']
            harvested = harvest['harvest_stock']
            prev_stock = harvest['expected_stock'] - harvested
            
            if harvested <= 0:
                continue
            
            harvested_keys.add(key)
            # Drop current and previous rows for this house
            df = df[~((df['farm_house_key'] == key) & (df['date'] <= date))]
            
            # Update future rows with reduced stock
            mask = (df['farm_house_key'] == key) & (df['date'] > date)
            future_rows = df[mask].sort_values(by='date')
            
            for idx, row in future_rows.iterrows():
                mortality_rate = row.get('expected_mortality', 0)
                new_stock = prev_stock - mortality_rate
                new_stock_rounded = int(round(new_stock))
                
                if new_stock_rounded > 0:
                    df.at[idx, 'expected_stock'] = new_stock_rounded
                    prev_stock = new_stock
                else:
                    df = df.drop(index=idx)
                    break
        
        # Remove all remaining rows for harvested houses to ensure only one harvest per house
        for key in harvested_keys:
            df = df[df['farm_house_key'] != key]
        
        df = df[df['expected_stock'] > 0].copy()
        
        # Remove entire farm/house combinations if any day has 0 stock
        farm_house_with_zero = df[df['expected_stock'] == 0][['farm', 'house']].drop_duplicates()
        for _, row in farm_house_with_zero.iterrows():
            farm = row['farm']
            house = row['house']
            df = df[~((df['farm'] == farm) & (df['house'] == house))]
        
        return df