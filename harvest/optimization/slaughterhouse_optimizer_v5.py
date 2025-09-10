# -*- coding: utf-8 -*-
"""
Slaughterhouse Optimizer V5.1
Updated implementation with enhanced logic for handling vacation days and opportunity counting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


def get_optimizer_function(optimizer_type):
    """
    Get the appropriate optimizer function based on the optimizer type.
    
    Parameters:
    - optimizer_type (str): Type of optimizer ("base", "weight", "pct")
    
    Returns:
    - function: The appropriate optimizer function
    """
    if optimizer_type == "base":
        return SH_min_houses_uniform_extra_base
    elif optimizer_type == "weight":
        return SH_min_houses_uniform_extra  # Original function with weight expansion
    elif optimizer_type == "pct":
        return SH_min_houses_uniform_extra_pct
    else:
        raise ValueError(f"Invalid optimizer_type: {optimizer_type}. Must be 'base', 'weight', or 'pct'.")


def run_three_scenario_optimization(
    df_input,
    min_weight,
    max_weight,
    min_stock,
    max_stock,
    max_pct_per_house=0.3,
    min_per_house=2000
):
    """
    Run three-scenario optimization: base, weight expansion, and percentage escalation.
    
    Parameters:
    - df_input: Input DataFrame
    - min_weight: Minimum weight threshold
    - max_weight: Maximum weight threshold
    - min_stock: Minimum stock threshold
    - max_stock: Maximum stock threshold
    - max_pct_per_house: Maximum percentage per house
    - min_per_house: Minimum per house
    
    Returns:
    - dict: Results for each scenario with best plans
    """
    logger.info("Starting three-scenario slaughterhouse optimization")
    
    # Scenario 1: Base scenario
    logger.info("Running base scenario...")
    base_results, base_updated_dfs = SH_run_multiple_harvest_starts(
        df_input=df_input,
        optimizer_fn=SH_min_houses_uniform_extra_base,
        min_weight=min_weight,
        max_weight=max_weight,
        min_stock=min_stock,
        max_stock=max_stock,
        max_pct_per_house=max_pct_per_house,
        min_per_house=min_per_house
    )
    
    best_base_date, best_base_plan_df, best_base_updated_df, sorted_net_meat_dict_base = get_best_harvest_stock_plan(base_results)
    
    # Scenario 2: Weight expansion scenario
    logger.info("Running weight expansion scenario...")
    
    # Create SH_UN flag for unharvested houses from base scenario
    if best_base_plan_df is not None and not best_base_plan_df.empty:
        unique_pairs = df_input[['farm', 'house']].drop_duplicates()
        harvested_pairs = best_base_plan_df[['farm', 'house']].drop_duplicates()
        
        # Find missing pairs (unharvested)
        missing_pairs = unique_pairs.merge(harvested_pairs, on=['farm', 'house'], how='left', indicator=True)
        missing_pairs = missing_pairs[missing_pairs['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        # Create SH_UN flag
        missing_set = set(zip(missing_pairs['farm'], missing_pairs['house']))
        df_input['SH_UN'] = [
            1 if (farm, house) in missing_set else 0
            for farm, house in zip(df_input['farm'], df_input['house'])
        ]
    else:
        df_input['SH_UN'] = 0
    
    # Run weight expansion scenario
    weight_results, weight_updated_dfs = SH_run_multiple_harvest_starts(
        df_input=df_input,
        optimizer_fn=SH_min_houses_uniform_extra_base,  # Uses SH_UN flag for weight expansion
        min_weight=min_weight,
        max_weight=max_weight,
        min_stock=min_stock,
        max_stock=max_stock,
        max_pct_per_house=max_pct_per_house,
        min_per_house=min_per_house
    )
    
    best_sc_weight_date, best_sc_weight_plan_df, best_sc_weight_updated_df, sorted_net_meat_dict_sc_weight = get_best_harvest_stock_plan(weight_results)
    
    # Scenario 3: Percentage escalation scenario
    logger.info("Running percentage escalation scenario...")
    
    # Check for unfulfilled days in weight scenario
    if best_sc_weight_plan_df is not None and not best_sc_weight_plan_df.empty:
        daily = best_sc_weight_plan_df.groupby('date', as_index=False)['harvest_stock'].sum()
        pending_days = pd.DataFrame({
            'date': daily['date'],
            'pending_SH': (daily['harvest_stock'] < min_stock).astype(int),
            'pending_stock': (min_stock - daily['harvest_stock']).astype(int)
        })
        pending_days = pending_days[pending_days['pending_SH'] == 1]
        
        # Find houses that need NF flag
        if not pending_days.empty:
            pending_days_houses = best_sc_weight_plan_df.merge(
                pending_days[['date', 'pending_SH']],
                on='date',
                how='inner'
            )[['date', 'farm', 'house', 'pending_SH']]
            
            subset = pending_days_houses[pending_days_houses.index > 0]
            unique_pairs_NF = subset[['farm', 'house']].drop_duplicates()
            
            if len(unique_pairs_NF) == 1:
                NF_rows = subset
                
                # Add NF column
                df_input['NF'] = 0
                keys = set(zip(NF_rows["date"], NF_rows["farm"], NF_rows["house"]))
                df_input.loc[
                    df_input.set_index(["date", "farm", "house"]).index.isin(keys),
                    "NF"
                ] = 1
            else:
                df_input['NF'] = 0
        else:
            df_input['NF'] = 0
    else:
        df_input['NF'] = 0
    
    # Run percentage escalation scenario
    pct_results, pct_updated_dfs = SH_run_multiple_harvest_starts(
        df_input=df_input,
        optimizer_fn=SH_min_houses_uniform_extra_base,  # Uses NF flag for percentage escalation
        min_weight=min_weight,
        max_weight=max_weight,
        min_stock=min_stock,
        max_stock=max_stock,
        max_pct_per_house=max_pct_per_house,
        min_per_house=min_per_house
    )
    
    best_sc_pct_date, best_sc_pct_plan_df, best_sc_pct_updated_df, sorted_net_meat_dict_sc_pct = get_best_harvest_stock_plan(pct_results)
    
    # Return results for all scenarios
    return {
        'base': {
            'date': best_base_date,
            'plan_df': best_base_plan_df,
            'updated_df': best_base_updated_df,
            'summary': sorted_net_meat_dict_base
        },
        'weight': {
            'date': best_sc_weight_date,
            'plan_df': best_sc_weight_plan_df,
            'updated_df': best_sc_weight_updated_df,
            'summary': sorted_net_meat_dict_sc_weight
        },
        'pct': {
            'date': best_sc_pct_date,
            'plan_df': best_sc_pct_plan_df,
            'updated_df': best_sc_pct_updated_df,
            'summary': sorted_net_meat_dict_sc_pct
        }
    }


def add_house_dates_columns(df, duration_days=40, max_harvest_date=None):
    """
    Adds 'start_date', 'end_date', 'last_date', and 'max_harvest_date' columns to the input DataFrame
    without altering its row count. Calculations are based on each (farm, house) group.

    Parameters:
    - df (pd.DataFrame): Original DataFrame with columns 'farm', 'house', and 'date'.
    - duration_days (int): Days to add to start_date to compute last_date.
    - max_harvest_date (str or pd.Timestamp): Manually passed date for all rows.

    Returns:
    - pd.DataFrame: Original DataFrame with new date columns added.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Convert max_harvest_date to pd.Timestamp
    if max_harvest_date is not None:
        max_harvest_date = pd.to_datetime(max_harvest_date)

    # Get start and end dates per (farm, house) group
    start_dates = df.groupby(['farm', 'house'])['date'].transform('min')
    end_dates = df.groupby(['farm', 'house'])['date'].transform('max')

    df['start_date'] = start_dates
    df['end_date'] = end_dates
    df['last_date'] = df['start_date'] + pd.to_timedelta(duration_days, unit='D')
    df['max_harvest_date'] = max_harvest_date

    return df


def remove_invalid_dates(df_with_date):
    """
    Removes rows where date > last_date or date > max_harvest_date.
    Returns two DataFrames:
      1. Cleaned DataFrame with valid rows
      2. Removed rows with a 'removal_reason' column
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


def compute_ready_sh(row, min_weight, max_weight):
    """Compute SH readiness with expanded weight range for unharvested houses."""
    if row.get("SH_UN", 0) == 1:  # if SH_UN exists and equals 1
        return 1 if (min_weight - 0.05) <= row['avg_weight'] <= (max_weight + 0.05) else 0
    else:
        return 1 if min_weight <= row['avg_weight'] <= max_weight else 0


def flag_ready_avg_weight(
    df,
    min_weight,
    max_weight,
    harvest_type='SH'
):
    """Flag rows that meet weight requirements for harvest type."""
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    if harvest_type == 'SH':
        df['ready_SH'] = df.apply(compute_ready_sh, axis=1, args=(min_weight, max_weight))
    elif harvest_type == 'Market':
        df['ready_Market'] = df['avg_weight'].apply(lambda x: 1 if min_weight <= x <= max_weight else 0)
    else:
        raise ValueError("Invalid harvest_type. Use 'SH' or 'Market'.")

    return df


def flag_ready_daily_stock(
    df,
    min_stock,
    max_stock,
    harvest_type='SH',
    min_per_house=2000
):
    """Flag days that meet stock requirements for harvest type."""
    df['date'] = pd.to_datetime(df['date'])

    if harvest_type == 'SH':
        readySH_df = df[df['ready_SH'] == 1]
        stock_sum_by_date = readySH_df.groupby('date')['expected_stock'].sum()
        flagged_dates = stock_sum_by_date[stock_sum_by_date >= min_per_house].index
        df['flag_day_sh'] = df['date'].isin(flagged_dates).astype(int)
    elif harvest_type == 'Market':
        readyM_df = df[df['ready_Market'] == 1]
        stock_mkt_by_date = readyM_df.groupby('date')['expected_stock'].sum()
        flagged_mkt_dates = stock_mkt_by_date[stock_mkt_by_date >= min_stock].index
        df['flag_day_M'] = df['date'].isin(flagged_mkt_dates).astype(int)
    else:
        raise ValueError("Invalid harvest_type. Use 'SH' or 'Market'.")

    return df


def apply_harvest_updates(df_full, harvest_df):
    """Apply harvest updates to the full DataFrame."""
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

        # ✅ If fully harvested, drop all future rows
        if prev_stock <= 0:
            df = df[df['farm_house_key'] != key]
            continue

        # Otherwise, update future rows
        mask = (df['farm_house_key'] == key) & (df['date'] > date)
        future_rows = df[mask].sort_values(by='date')

        for idx, row in future_rows.iterrows():
            mortality_rate = row.get('expected_mortality_rate', 0)
            new_stock = prev_stock * (1 - mortality_rate)
            new_stock_rounded = int(round(new_stock))

            if new_stock_rounded > 0:
                df.at[idx, 'expected_stock'] = new_stock_rounded
                df.at[idx, 'expected_mortality'] = abs(int(round(prev_stock - new_stock_rounded)))
                prev_stock = new_stock
            else:
                df = df.drop(index=idx)
                break

    df = df[df['expected_stock'] > 0].copy()
    return df


def apply_harvest_updates_only_once(df_full, harvest_df):
    """Apply harvest updates only once to avoid double-counting."""
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
        df = df[~((df['farm_house_key'] == key) & (df['date'] <= date))]

        mask = (df['farm_house_key'] == key) & (df['date'] > date)
        future_rows = df[mask].sort_values(by='date')

        for idx, row in future_rows.iterrows():
            mortality_rate = row.get('expected_mortality_rate', 0)
            new_stock = prev_stock * (1 - mortality_rate)
            new_stock_rounded = int(round(new_stock))

            if new_stock_rounded > 0:
                df.at[idx, 'expected_stock'] = new_stock_rounded
                prev_stock = new_stock
            else:
                df = df.drop(index=idx)
                break

    for key in harvested_keys:
        df = df[df['farm_house_key'] != key]

    df = df[df['expected_stock'] > 0].copy()
    return df


def SH_run_daily_harvest_loop(df_input, optimizer_fn, start_date,
                              min_weight, max_weight,
                              min_stock, max_stock,
                              num_days=7, max_pct_per_house=0.3, min_per_house=2000):
    """Run daily harvest loop for slaughterhouse optimization."""
    df = df_input.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_all = df.copy()
    full_results = []
    start_date = pd.to_datetime(start_date)
    loop_start = start_date.normalize()

    for day_offset in range(num_days):
        current_date = start_date + pd.Timedelta(days=day_offset)
        cur_day = current_date.normalize()

        # Skip Thursdays
        if current_date.weekday() == 3:
            logger.info(f"⏩ Skipping harvest on {current_date.date()} (Thursday).")
            continue

        logger.info(f"🔁 Running harvest for {current_date.date()}")

        # Re-flag SH readiness on the evolving df
        df = flag_ready_avg_weight(df, min_weight, max_weight, 'SH')
        df = flag_ready_daily_stock(df, min_stock, max_stock, 'SH', min_per_house)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Current eligible set
        in_df = df[(df['flag_day_sh'] == 1) & (df['ready_SH'] == 1)].copy()
        if in_df.empty:
            logger.info("⛔ No eligible houses at all — skipping.")
            continue

        in_df['date'] = pd.to_datetime(in_df['date'], errors='coerce')
        in_df = in_df.sort_values(['farm', 'house', 'date'], ascending=[True, True, False])

        # Opportunity counting that skips Thursdays
        is_not_thu = in_df['date'].dt.dayofweek != 3
        in_df.loc[:, 'opportunity'] = (
            is_not_thu.astype(int)
            .groupby([in_df['farm'], in_df['house']])
            .cumsum()
            .where(is_not_thu, other=pd.NA)
            .astype('Int64')
        )

        # Today's rows (normalize compare)
        day_df = in_df[in_df['date'].dt.normalize() == cur_day].copy()
        if day_df.empty:
            logger.info(f"📭 No houses to harvest on {current_date.date()} — skipping.")
            continue

        # -------- dynamic penultimate-all-houses check (no global pre-scan) --------
        elig_no_thu = in_df[in_df['date'].dt.dayofweek != 3].copy()
        last_date = elig_no_thu['date'].dt.normalize().max()
        use_all_houses = False
        if pd.notna(last_date):
            last_rows = elig_no_thu[elig_no_thu['date'].dt.normalize() == last_date]
            # if the *last* eligible day has exactly one farm-house
            if not last_rows.empty and last_rows[['farm', 'house']].drop_duplicates().shape[0] == 1:
                fh = last_rows[['farm', 'house']].drop_duplicates().iloc[0]
                # find the closest earlier day where this same FH also appears
                earlier = elig_no_thu[
                    (elig_no_thu['farm'] == fh['farm']) &
                    (elig_no_thu['house'] == fh['house']) &
                    (elig_no_thu['date'].dt.normalize() < last_date)
                ]
                if not earlier.empty:
                    penultimate = earlier['date'].dt.normalize().max()
                    # if today is that penultimate day, switch to all-houses optimizer
                    if cur_day == penultimate:
                        use_all_houses = True
                        logger.info(f"▶ Using SH_all_houses_uniform_extra on {current_date.date()} (penultimate day rule).")
        # --------------------------------------------------------------------------

        # Run optimizer
        if use_all_houses:
            result = SH_all_houses_uniform_extra(
                day_df,
                max_pct_per_house=max_pct_per_house,
                min_total_stock=min_stock,
                min_per_house=min_per_house
            )
        else:
            # Check for NF flag to determine optimizer type
            if "NF" in day_df.columns and (day_df["NF"] == 1).any():
                result = SH_min_houses_uniform_extra_pct(
                    day_df,
                    max_pct_per_house=max_pct_per_house,
                    min_total_stock=min_stock,
                    min_per_house=min_per_house
                )
            else:
                result = optimizer_fn(
                    day_df,
                    max_pct_per_house=max_pct_per_house,
                    min_total_stock=min_stock,
                    min_per_house=min_per_house
                )

        if result.empty:
            logger.warning(f"⚠️ Optimizer returned no harvest for {current_date.date()} — skipping.")
            continue

        # Keep only today's rows
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result = result[result['date'].dt.normalize() == cur_day]
        if result.empty:
            logger.warning(f"❌ Optimizer returned rows, but none match {current_date.date()}. Skipping.")
            continue

        # Apply updates
        full_results.append(result)
        df_all = apply_harvest_updates(df_all, result)
        df = apply_harvest_updates_only_once(df, result)

        # Early stop if no future rows remain
        if df.empty:
            logger.info("✅ No future rows after updates. Ending loop.")
            break
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if not df['date'].dt.normalize().gt(cur_day).any():
            logger.info("✅ No future rows after updates. Ending loop.")
            break

    harvest_df = pd.concat(full_results, ignore_index=True) if full_results else pd.DataFrame()
    return harvest_df, df_all


def SH_run_multiple_harvest_starts(df_input,
                                   optimizer_fn,
                                   min_weight, max_weight,
                                   min_stock, max_stock,
                                   max_pct_per_house=0.3, min_per_house=2000):
    """Run multiple harvest starts to find the best scenario."""
    df_base = df_input.copy()
    df_base['date'] = pd.to_datetime(df_base['date'], errors='coerce')

    results_by_start_day, all_updated_dfs = {}, {}

    # Eligible non-Thursday start days
    scan = df_base.copy()
    scan = flag_ready_avg_weight(scan, min_weight, max_weight, 'SH')
    scan = flag_ready_daily_stock(scan, min_stock, max_stock, 'SH', min_per_house)
    scan['date'] = pd.to_datetime(scan['date'], errors='coerce')

    elig_mask = (scan['flag_day_sh'] == 1) & (scan['ready_SH'] == 1) & (scan['date'].dt.dayofweek != 3)
    elig_days = sorted(scan.loc[elig_mask, 'date'].dt.normalize().dropna().unique().tolist())
    if not elig_days:
        return {}, {}

    last_eligible_day = pd.to_datetime(elig_days[-1])

    for start_day in elig_days:
        start_day = pd.to_datetime(start_day)
        num_days = int((last_eligible_day - start_day).days) + 1
        if num_days <= 1:
            continue

        logger.info(f"🔁 Running harvest loop starting at {start_day.date()}")
        harvest_df, updated_df = SH_run_daily_harvest_loop(
            df_input=df_base.copy(),
            optimizer_fn=optimizer_fn,
            start_date=start_day,
            min_weight=min_weight,
            max_weight=max_weight,
            min_stock=min_stock,
            max_stock=max_stock,
            num_days=num_days,
            max_pct_per_house=max_pct_per_house,
            min_per_house=min_per_house
        )

        if harvest_df.empty:
            logger.warning(f"⛔ No harvest generated for start date {start_day.date()}")
            if start_day == last_eligible_day:
                logger.info("🛑 Reached last eligible start day. Stopping.")
                break
            continue

        results_by_start_day[start_day.date()] = {
            'harvest': harvest_df,
            'updated_df': updated_df
        }
        all_updated_dfs[start_day.date()] = updated_df.copy()

    return results_by_start_day, all_updated_dfs


def get_best_harvest_stock_plan(plan_dict):
    """
    Finds the best plan (by total harvested stock) from a dict with date keys and {'harvest', 'updated_df'} values.

    Parameters:
        plan_dict (dict): {
            date_obj: {
                'harvest': DataFrame,
                'updated_df': DataFrame
            }
        }

    Returns:
        best_date (datetime.date): Date of the best harvest plan.
        best_harvest_df (pd.DataFrame): Harvest dataframe with highest total harvest stock.
        best_updated_df (pd.DataFrame): Corresponding updated dataframe.
        sorted_summary (dict): {date: total_harvest_stock} sorted descending.
    """
    summary = {}

    for date_key, data in plan_dict.items():
        harvest_df = data.get('harvest')
        if isinstance(harvest_df, pd.DataFrame) and not harvest_df.empty and 'harvest_stock' in harvest_df.columns:
            total_stock = harvest_df['harvest_stock'].sum()
            summary[date_key] = total_stock

    if not summary:
        logger.warning("⚠️ No valid harvest plans with harvest_stock found.")
        return None, None, None, {}

    # Sort descending by total harvested stock
    sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))
    best_date = next(iter(sorted_summary))
    best_harvest_df = plan_dict[best_date]['harvest']
    best_updated_df = plan_dict[best_date]['updated_df']

    return best_date, best_harvest_df, best_updated_df, sorted_summary


def summarize_harvest_plans(plan_dict):
    """Summarize harvest plans for analysis."""
    summary_rows = []

    for date_key, plan in plan_dict.items():
        df = plan.get('harvest')
        if df is None or df.empty or 'net_meat' not in df.columns:
            continue

        plan_name = f"harvest_{date_key.strftime('%Y%m%d')}"
        start_date = df['date'].min().date()
        end_date = df['date'].max().date()
        total_days = (end_date - start_date).days + 1
        total_net_meat = df['net_meat'].sum()
        total_stock = df['harvest_stock'].sum()
        unique_houses = df[['farm', 'house']].drop_duplicates().shape[0]

        summary_rows.append({
            'plan_name': plan_name,
            'start_date': start_date,
            'end_date': end_date,
            'number_of_days': total_days,
            'number_of_unique_houses': unique_houses,
            'total_net_meat': total_net_meat,
            'total_stock': total_stock
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='total_stock', ascending=False)
    return summary_df


def build_daily_summary(harvest_plans_dict):
    """Build daily summary from harvest plans."""
    combined = []

    for plan in harvest_plans_dict.values():
        df = plan.get('harvest')
        if df is not None and not df.empty:
            combined.append(df[['date', 'harvest_stock', 'net_meat']])

    if not combined:
        return pd.DataFrame(columns=['date', 'total_harvest_stock', 'total_net_meat'])

    all_data = pd.concat(combined)
    summary = all_data.groupby('date', as_index=False).agg({
        'harvest_stock': 'sum',
        'net_meat': 'sum'
    })

    summary.rename(columns={
        'harvest_stock': 'total_harvest_stock',
        'net_meat': 'total_net_meat'
    }, inplace=True)

    return summary


def SH_min_houses_uniform_extra(
    df_day,
    min_total_stock=70000,
    max_pct_per_house=0.30,   # starting cap
    min_per_house=2000,
    pct_step=0.05,            # increment
    pct_max=1.00              # ceiling
):
    """
    Optimizer for all dates except last eligible date that aims to minimize the number of harvested houses
    as long as they can be harvested in another day.
    """
    df0 = df_day.reset_index(drop=True).copy()
    need = {'farm', 'date', 'house', 'avg_weight', 'expected_stock', 'opportunity'}
    miss = need - set(df0.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # validity
    df0 = df0[(df0['avg_weight'] > 0) & (df0['expected_stock'] > 0)].copy()
    if df0.empty:
        return df0.iloc[0:0]

    def build_selection(df_src, pct):
        df = df_src.copy()
        df['cap'] = np.minimum(
            np.floor(df['expected_stock'] * pct).astype(int),
            df['expected_stock'].astype(int)
        )
        # opp==1 must be selectable at this pct
        opp1 = df[df['opportunity'] == 1]
        if not opp1.empty and (opp1['cap'] < min_per_house).any():
            return None, 0

        df = df[df['cap'] >= min_per_house].copy()
        if df.empty:
            return None, 0

        # mandatory: all opp==1
        sel = df[df['opportunity'] == 1].copy()
        cap_sum = int(sel['cap'].sum()) if not sel.empty else 0

        # add minimal opp>1 needed (opportunity asc, weight asc, cap desc)
        pool = df[df['opportunity'] > 1].sort_values(
            ['opportunity', 'avg_weight', 'cap'], ascending=[True, True, False]
        )
        for _, row in pool.iterrows():
            if cap_sum >= min_total_stock:
                break
            sel = pd.concat([sel, row.to_frame().T], ignore_index=True)
            cap_sum += int(row['cap'])

        return sel, cap_sum

    # escalate pct until target feasible or ceiling reached
    pct = float(max_pct_per_house)
    best_sel, best_cap = None, 0
    while pct <= pct_max + 1e-12:
        sel, cap_sum = build_selection(df0, pct)
        if sel is not None:
            best_sel, best_cap = sel, cap_sum
            if cap_sum >= min_total_stock:
                break
        pct = round(min(pct_max, pct + pct_step), 6)

    if best_sel is None or best_sel.empty:
        return df0.iloc[0:0]

    # final cap pct used for selection phase (for reference)
    sel = best_sel.copy()
    sel['cap'] = np.minimum(
        np.floor(sel['expected_stock'] * min(pct, pct_max)).astype(int),
        sel['expected_stock'].astype(int)
    )

    achievable = int(sel['cap'].sum())
    T = min_total_stock if achievable >= min_total_stock else achievable

    # per-house minimum
    n = len(sel)
    base_total = n * int(min_per_house)
    if base_total > T:
        T = base_total

    # allocate: min + uniform % extra within residual caps, then top-up by lowest weight
    sel['harvest_stock'] = int(min_per_house)
    sel['residual_cap'] = (sel['cap'] - sel['harvest_stock']).clip(lower=0).astype(int)

    remaining = int(T - base_total)
    if remaining > 0:
        low, high = 0.0, float(min(pct, pct_max))
        best_extra = np.zeros(n, dtype=int)
        exp = sel['expected_stock'].to_numpy()
        res = sel['residual_cap'].to_numpy()

        for _ in range(60):
            mid = (low + high) / 2.0
            extra = np.floor(exp * mid).astype(int)
            extra = np.minimum(extra, res)
            if int(extra.sum()) <= remaining:
                best_extra = extra
                low = mid
            else:
                high = mid

        sel['harvest_stock'] += best_extra
        spill = remaining - int(best_extra.sum())
        if spill > 0:
            for i in sel.sort_values('avg_weight').index:
                if spill <= 0:
                    break
                used = int(sel.at[i, 'harvest_stock'] - min_per_house)
                can_add = int(sel.at[i, 'residual_cap'] - used)
                if can_add > 0:
                    add = min(can_add, spill)
                    sel.at[i, 'harvest_stock'] += add
                    spill -= add

    # finalize
    sel['harvest_stock'] = sel['harvest_stock'].astype(int)
    sel = sel[sel['harvest_stock'] > 0].copy()
    sel['net_meat'] = sel['harvest_stock'] * sel['avg_weight']
    sel['selected'] = 1
    sel['harvest_type'] = 'SH'
    # actual per-house pct harvested and cap pct
    sel['final_pct_per_house'] = sel['harvest_stock'].astype(float) / sel['expected_stock'].astype(float)
    sel['cap_pct_per_house'] = sel['cap'].astype(float) / sel['expected_stock'].astype(float)

    cols = ['farm', 'date', 'house', 'age', 'expected_mortality', 'expected_stock',
            'expected_mortality_rate', 'avg_weight', 'opportunity', 'selected',
            'harvest_stock', 'net_meat', 'harvest_type', 'final_pct_per_house', 'cap_pct_per_house',"priority","profit_per_bird"]
    cols = [c for c in cols if c in sel.columns]
    return sel[cols]


def SH_all_houses_uniform_extra(
    df_day,
    min_total_stock=70000,
    max_pct_per_house=0.30,   # starting cap
    min_per_house=2000,
    pct_step=0.05,            # increment
    pct_max=1.00              # ceiling
):
    """
    Optimizer for the last eligible day to include all unharvested houses.
    """
    df0 = df_day.reset_index(drop=True).copy()
    need = {'farm', 'date', 'house', 'avg_weight', 'expected_stock'}
    miss = need - set(df0.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # validity
    df0 = df0[(df0['avg_weight'] > 0) & (df0['expected_stock'] > 0)].copy()
    if df0.empty:
        return df0.iloc[0:0]

    def selectable_at_pct(df_src, pct):
        df = df_src.copy()
        df['cap'] = np.minimum(
            np.floor(df['expected_stock'] * pct).astype(int),
            df['expected_stock'].astype(int)
        )
        df = df[df['cap'] >= min_per_house].copy()
        return df

    # escalate pct until target feasible or ceiling reached
    pct = float(max_pct_per_house)
    sel = None
    while pct <= pct_max + 1e-12:
        cand = selectable_at_pct(df0, pct)
        if not cand.empty:
            sel = cand
            if int(cand['cap'].sum()) >= int(min_total_stock):
                break
        pct = round(min(pct_max, pct + pct_step), 6)

    if sel is None or sel.empty:
        return df0.iloc[0:0]

    final_cap_pct = float(min(pct, pct_max))
    sel = sel.copy()  # all available houses at this pct

    achievable = int(sel['cap'].sum())
    T = min(int(min_total_stock), achievable)

    # per-house minimum
    n = len(sel)
    base_total = n * int(min_per_house)
    if base_total > T:
        T = base_total

    # allocate: min + uniform % extra within residual caps, then top-up by lowest weight
    sel['harvest_stock'] = int(min_per_house)
    sel['residual_cap'] = (sel['cap'] - sel['harvest_stock']).clip(lower=0).astype(int)

    remaining = int(T - base_total)
    if remaining > 0:
        low, high = 0.0, final_cap_pct
        best_extra = np.zeros(n, dtype=int)
        exp = sel['expected_stock'].to_numpy()
        res = sel['residual_cap'].to_numpy()

        for _ in range(60):
            mid = (low + high) / 2.0
            extra = np.floor(exp * mid).astype(int)
            extra = np.minimum(extra, res)
            if int(extra.sum()) <= remaining:
                best_extra = extra
                low = mid
            else:
                high = mid

        sel['harvest_stock'] += best_extra
        spill = remaining - int(best_extra.sum())
        if spill > 0:
            for i in sel.sort_values('avg_weight').index:
                if spill <= 0:
                    break
                used = int(sel.at[i, 'harvest_stock'] - min_per_house)
                can_add = int(sel.at[i, 'residual_cap'] - used)
                if can_add > 0:
                    add = min(can_add, spill)
                    sel.at[i, 'harvest_stock'] += add
                    spill -= add

    # finalize
    sel['harvest_stock'] = sel['harvest_stock'].astype(int)
    sel = sel[sel['harvest_stock'] > 0].copy()
    sel['net_meat'] = sel['harvest_stock'] * sel['avg_weight']
    sel['selected'] = 1
    sel['harvest_type'] = 'SH'
    # actual per-house pct harvested and cap pct
    sel['final_pct_per_house'] = sel['harvest_stock'].astype(float) / sel['expected_stock'].astype(float)
    sel['cap_pct_per_house'] = sel['cap'].astype(float) / sel['expected_stock'].astype(float)

    cols = ['farm', 'date', 'house', 'age', 'expected_mortality', 'expected_stock',
            'expected_mortality_rate', 'avg_weight', 'selected',
            'harvest_stock', 'net_meat', 'harvest_type', 'final_pct_per_house', 'cap_pct_per_house',"priority","profit_per_bird"]
    cols = [c for c in cols if c in sel.columns]
    return sel[cols]


def SH_min_houses_uniform_extra_base(
    df_day,
    min_total_stock=70000,
    max_pct_per_house=0.30,   # fixed cap (no escalation)
    min_per_house=2000,
    pct_step=0.05,            # unused (kept for signature compatibility)
    pct_max=1.00              # ceiling
):
    """
    Base optimizer with fixed percentage cap (no escalation).
    """
    df0 = df_day.reset_index(drop=True).copy()
    need = {'farm', 'date', 'house', 'avg_weight', 'expected_stock', 'opportunity'}
    miss = need - set(df0.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # validity
    df0 = df0[(df0['avg_weight'] > 0) & (df0['expected_stock'] > 0)].copy()
    if df0.empty:
        return df0.iloc[0:0]

    final_cap_pct = float(min(max_pct_per_house, pct_max))

    def build_selection_fixed(df_src, pct):
        df = df_src.copy()
        df['cap'] = np.minimum(
            np.floor(df['expected_stock'] * pct).astype(int),
            df['expected_stock'].astype(int)
        )

        # opp==1 must be feasible at this pct
        opp1 = df[df['opportunity'] == 1]
        if not opp1.empty and (opp1['cap'] < min_per_house).any():
            return None, 0

        df = df[df['cap'] >= min_per_house].copy()
        if df.empty:
            return None, 0

        # mandatory: all opp==1
        sel = df[df['opportunity'] == 1].copy()
        cap_sum = int(sel['cap'].sum()) if not sel.empty else 0

        # add minimal opp>1 until reaching min_total_stock if possible
        pool = df[df['opportunity'] > 1].sort_values(
            ['opportunity', 'avg_weight', 'cap'], ascending=[True, True, False]
        )
        for _, row in pool.iterrows():
            if cap_sum >= min_total_stock:
                break
            sel = pd.concat([sel, row.to_frame().T], ignore_index=True)
            cap_sum += int(row['cap'])

        return sel, cap_sum

    sel, cap_sum = build_selection_fixed(df0, final_cap_pct)
    if sel is None or sel.empty:
        return df0.iloc[0:0]

    # use fixed-cap selection
    sel = sel.copy()
    sel['cap'] = np.minimum(
        np.floor(sel['expected_stock'] * final_cap_pct).astype(int),
        sel['expected_stock'].astype(int)
    )

    achievable = int(sel['cap'].sum())
    T = min(int(min_total_stock), achievable)

    # per-house minimum
    n = len(sel)
    base_total = n * int(min_per_house)
    if base_total > T:
        T = base_total  # still ≤ achievable because cap ≥ min_per_house per house

    # allocate: min + uniform % extra within residual caps, then top-up by lowest weight
    sel['harvest_stock'] = int(min_per_house)
    sel['residual_cap'] = (sel['cap'] - sel['harvest_stock']).clip(lower=0).astype(int)

    remaining = int(T - base_total)
    if remaining > 0:
        low, high = 0.0, final_cap_pct
        best_extra = np.zeros(n, dtype=int)
        exp = sel['expected_stock'].to_numpy()
        res = sel['residual_cap'].to_numpy()

        for _ in range(60):
            mid = (low + high) / 2.0
            extra = np.floor(exp * mid).astype(int)
            extra = np.minimum(extra, res)
            if int(extra.sum()) <= remaining:
                best_extra = extra
                low = mid
            else:
                high = mid

        sel['harvest_stock'] += best_extra
        spill = remaining - int(best_extra.sum())
        if spill > 0:
            for i in sel.sort_values('avg_weight').index:
                if spill <= 0:
                    break
                used = int(sel.at[i, 'harvest_stock'] - min_per_house)
                can_add = int(sel.at[i, 'residual_cap'] - used)
                if can_add > 0:
                    add = min(can_add, spill)
                    sel.at[i, 'harvest_stock'] += add
                    spill -= add

    # finalize
    sel['harvest_stock'] = sel['harvest_stock'].astype(int)
    sel = sel[sel['harvest_stock'] > 0].copy()
    sel['net_meat'] = sel['harvest_stock'] * sel['avg_weight']
    sel['selected'] = 1
    sel['harvest_type'] = 'SH'
    sel['final_pct_per_house'] = sel['harvest_stock'].astype(float) / sel['expected_stock'].astype(float)
    sel['cap_pct_per_house'] = sel['cap'].astype(float) / sel['expected_stock'].astype(float)

    cols = ['farm', 'date', 'house', 'age', 'expected_mortality', 'expected_stock',
            'expected_mortality_rate', 'avg_weight', 'opportunity', 'selected',
            'harvest_stock', 'net_meat', 'harvest_type', 'final_pct_per_house', 'cap_pct_per_house']
    cols = [c for c in cols if c in sel.columns]
    return sel[cols]


def SH_min_houses_uniform_extra_pct(
    df_day,
    min_total_stock=70000,
    max_pct_per_house=0.30,   # starting cap
    min_per_house=2000,
    pct_step=0.05,            # increment
    pct_max=1.00              # ceiling
):
    """
    Percentage optimizer with escalation logic.
    """
    df0 = df_day.reset_index(drop=True).copy()
    need = {'farm', 'date', 'house', 'avg_weight', 'expected_stock', 'opportunity'}
    miss = need - set(df0.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # validity
    df0 = df0[(df0['avg_weight'] > 0) & (df0['expected_stock'] > 0)].copy()
    if df0.empty:
        return df0.iloc[0:0]

    def build_selection(df_src, pct):
        df = df_src.copy()
        df['cap'] = np.minimum(
            np.floor(df['expected_stock'] * pct).astype(int),
            df['expected_stock'].astype(int)
        )
        # opp==1 must be selectable at this pct
        opp1 = df[df['opportunity'] == 1]
        if not opp1.empty and (opp1['cap'] < min_per_house).any():
            return None, 0

        df = df[df['cap'] >= min_per_house].copy()
        if df.empty:
            return None, 0

        # mandatory: all opp==1
        sel = df[df['opportunity'] == 1].copy()
        cap_sum = int(sel['cap'].sum()) if not sel.empty else 0

        # add minimal opp>1 needed (opportunity asc, weight asc, cap desc)
        pool = df[df['opportunity'] > 1].sort_values(
            ['opportunity', 'avg_weight', 'cap'], ascending=[True, True, False]
        )
        for _, row in pool.iterrows():
            if cap_sum >= min_total_stock:
                break
            sel = pd.concat([sel, row.to_frame().T], ignore_index=True)
            cap_sum += int(row['cap'])

        return sel, cap_sum

    # escalate pct until target feasible or ceiling reached
    pct = float(max_pct_per_house)
    best_sel, best_cap = None, 0
    while pct <= pct_max + 1e-12:
        sel, cap_sum = build_selection(df0, pct)
        if sel is not None:
            best_sel, best_cap = sel, cap_sum
            if cap_sum >= min_total_stock:
                break
        pct = round(min(pct_max, pct + pct_step), 6)

    # If 100% still cannot reach target -> take maximum available weight (take everything eligible)
    if best_cap < min_total_stock:
        df_full = df0.copy()
        df_full['cap'] = np.floor(df_full['expected_stock']).astype(int)

        # opp==1 houses must be feasible
        opp1 = df_full[df_full['opportunity'] == 1]
        if not opp1.empty and (opp1['cap'] < min_per_house).any():
            return df0.iloc[0:0]

        df_full = df_full[df_full['cap'] >= min_per_house].copy()
        if df_full.empty:
            return df0.iloc[0:0]

        df_full['harvest_stock'] = df_full['cap'].astype(int)
        df_full['net_meat'] = df_full['harvest_stock'] * df_full['avg_weight']
        df_full['selected'] = 1
        df_full['harvest_type'] = 'SH'
        df_full['final_pct_per_house'] = df_full['harvest_stock'] / df_full['expected_stock']
        df_full['cap_pct_per_house'] = 1.0

        cols = ['farm', 'date', 'house', 'age', 'expected_mortality', 'expected_stock',
                'expected_mortality_rate', 'avg_weight', 'opportunity', 'selected',
                'harvest_stock', 'net_meat', 'harvest_type', 'final_pct_per_house', 'cap_pct_per_house']
        cols = [c for c in cols if c in df_full.columns]
        return df_full[cols]

    # Otherwise proceed with uniform-extra allocation using the best selection
    sel = best_sel.copy()
    sel['cap'] = np.minimum(
        np.floor(sel['expected_stock'] * min(pct, pct_max)).astype(int),
        sel['expected_stock'].astype(int)
    )

    achievable = int(sel['cap'].sum())
    T = min_total_stock if achievable >= min_total_stock else achievable

    # per-house minimum
    n = len(sel)
    base_total = n * int(min_per_house)
    if base_total > T:
        T = base_total

    # allocate: min + uniform % extra within residual caps, then top-up by lowest weight
    sel['harvest_stock'] = int(min_per_house)
    sel['residual_cap'] = (sel['cap'] - sel['harvest_stock']).clip(lower=0).astype(int)

    remaining = int(T - base_total)
    if remaining > 0:
        low, high = 0.0, float(min(pct, pct_max))
        best_extra = np.zeros(n, dtype=int)
        exp = sel['expected_stock'].to_numpy()
        res = sel['residual_cap'].to_numpy()

        for _ in range(60):
            mid = (low + high) / 2.0
            extra = np.floor(exp * mid).astype(int)
            extra = np.minimum(extra, res)
            if int(extra.sum()) <= remaining:
                best_extra = extra
                low = mid
            else:
                high = mid

        sel['harvest_stock'] += best_extra
        spill = remaining - int(best_extra.sum())
        if spill > 0:
            for i in sel.sort_values('avg_weight').index:
                if spill <= 0:
                    break
                used = int(sel.at[i, 'harvest_stock'] - min_per_house)
                can_add = int(sel.at[i, 'residual_cap'] - used)
                if can_add > 0:
                    add = min(can_add, spill)
                    sel.at[i, 'harvest_stock'] += add
                    spill -= add

    # finalize
    sel['harvest_stock'] = sel['harvest_stock'].astype(int)
    sel = sel[sel['harvest_stock'] > 0].copy()
    sel['net_meat'] = sel['harvest_stock'] * sel['avg_weight']
    sel['selected'] = 1
    sel['harvest_type'] = 'SH'
    sel['final_pct_per_house'] = sel['harvest_stock'].astype(float) / sel['expected_stock'].astype(float)
    sel['cap_pct_per_house'] = sel['cap'].astype(float) / sel['expected_stock'].astype(float)

    cols = ['farm', 'date', 'house', 'age', 'expected_mortality', 'expected_stock',
            'expected_mortality_rate', 'avg_weight', 'opportunity', 'selected',
            'harvest_stock', 'net_meat', 'harvest_type', 'final_pct_per_house', 'cap_pct_per_house']
    cols = [c for c in cols if c in sel.columns]
    return sel[cols]
