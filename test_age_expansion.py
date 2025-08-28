#!/usr/bin/env python3
"""
Test script for the new age expansion functionality.
"""

import pandas as pd
import sys
import os

# Add the harvest package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'harvest'))

from harvest.models import SlaughterHouseConfig, MarketConfig, ServiceConfig
from harvest.main import PoultryHarvestOptimizationService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data():
    """Create test growth data for testing."""
    # Create sample data with a few farms and houses
    data = []
    
    # Farm 1, House 1
    for age in range(1, 41):
        data.append({
            'farm': 'Farm1',
            'house': 'House1',
            'date': pd.date_range('2025-01-01', periods=40)[age-1].strftime('%Y-%m-%d'),
            'age': age,
            'expected_stock': 1000,
            'avg_weight': 1.5 + (age * 0.05),  # Weight increases with age
            'expected_mortality': int(1000 * (0.02 + age * 0.001)),  # Mortality increases with age
            'expected_mortality_rate': 0.02 + age * 0.001
        })
    
    # Farm 1, House 2
    for age in range(1, 41):
        data.append({
            'farm': 'Farm1',
            'house': 'House2',
            'date': pd.date_range('2025-01-01', periods=40)[age-1].strftime('%Y-%m-%d'),
            'age': age,
            'expected_stock': 1000,
            'avg_weight': 1.5 + (age * 0.05),
            'expected_mortality': int(1000 * (0.02 + age * 0.001)),
            'expected_mortality_rate': 0.02 + age * 0.001
        })
    
    # Farm 2, House 1
    for age in range(1, 41):
        data.append({
            'farm': 'Farm2',
            'house': 'House1',
            'date': pd.date_range('2025-01-01', periods=40)[age-1].strftime('%Y-%m-%d'),
            'age': age,
            'expected_stock': 1000,
            'avg_weight': 1.5 + (age * 0.05),
            'expected_mortality': int(1000 * (0.02 + age * 0.001)),
            'expected_mortality_rate': 0.02 + age * 0.001
        })
    
    df = pd.DataFrame(data)
    return df


def test_age_expansion():
    """Test the age expansion functionality."""
    logger.info("Creating test data...")
    test_df = create_test_data()
    test_df.to_csv('test_growth_data.csv', index=False)
    logger.info(f"Created test data with {len(test_df)} rows")
    
    # Create configuration objects
    sh_config = SlaughterHouseConfig(
        min_weight=1.55,
        max_weight=1.7,
        min_stock=500,
        max_stock=1000,
        max_pct_per_house=0.3
    )
    
    market_config = MarketConfig(
        min_weight=1.90,
        max_weight=2.20,
        min_stock=500,
        max_stock=5000,
        max_pct_per_house=1.0,
        tolerance_step=100,
        max_tolerance=1000
    )
    
    service_config = ServiceConfig(
        duration_days=40,
        max_harvest_date=None,
        cull_adjustment=0.03,
        cleaning_days=15,
        safety_days=3,
        culls_avg_weight=1.2
    )
    
    # Initialize the optimization service
    optimization_service = PoultryHarvestOptimizationService(service_config)
    optimization_service.sh_config = sh_config
    optimization_service.market_config = market_config
    
    # Run the full optimization
    try:
        logger.info("Running full optimization with age expansion...")
        result = optimization_service.run_full_optimization(
            input_file_path="test_growth_data.csv",
            output_dir="test_output"
        )
        
        if result.get("status") == "success":
            logger.info("Optimization completed successfully!")
            logger.info(f"Summary: {result['summary']}")
            logger.info(f"Exported files: {list(result['exported_files'].keys())}")
            
            # Check if age expansion results exist
            if 'age_expansion' in result['exported_files']:
                logger.info("✓ Age expansion scenario completed successfully!")
                logger.info(f"Age expansion market harvest: {len(result['age_expansion_market_harvest'])} rows")
                logger.info(f"Age expansion total harvest stock: {result['summary']['age_expansion_total_harvest_stock']}")
                logger.info(f"Age expansion total harvest meat: {result['summary']['age_expansion_total_harvest_meat']}")
            else:
                logger.warning("✗ Age expansion scenario not found in results")
                
        else:
            logger.error(f"Optimization failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise
    
    finally:
        # Clean up test files
        if os.path.exists('test_growth_data.csv'):
            os.remove('test_growth_data.csv')
            logger.info("Cleaned up test data file")


if __name__ == "__main__":
    test_age_expansion()


