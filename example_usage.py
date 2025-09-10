"""
Example usage of the harvest optimization system.
"""

from harvest.models import SlaughterHouseConfig, MarketConfig, ServiceConfig
from harvest.main import PoultryHarvestOptimizationService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    
    # Create configuration objects
    sh_config = SlaughterHouseConfig(
        min_weight=1.55,
        max_weight=1.7,
        min_stock=30000,
        max_stock=30000,
        max_pct_per_house=0.3,
        optimizer_type="base"  # Will run all three scenarios: base, weight, pct
    )
    
    market_config = MarketConfig(
        min_weight=1.90,
        max_weight=2.20,
        min_stock=10000,
        max_stock=100000,
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
        result = optimization_service.run_full_optimization(
            input_file_path="growth_data.csv",
            input_file_path_price="market_price.csv",
            feed_price=18,
            output_dir="output"
        )
        
        if result.get("status") == "success":
            logger.info("Optimization completed successfully!")
        else:
            logger.error(f"Optimization failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise


if __name__ == "__main__":
    main()