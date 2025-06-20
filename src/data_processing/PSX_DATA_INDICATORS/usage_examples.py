#!/usr/bin/env python3
"""
Example usage of the Enhanced PSX Indicator Processor

This script demonstrates various ways to use the enhanced processor
with different configurations and options.
"""

import asyncio
import logging
from pathlib import Path

from enhanced_psx_indicator_processor import (
    EnhancedPSXIndicatorProcessor, 
    ProcessorConfig
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    logger.info("Example 1: Basic usage")
    
    # Use default configuration
    async with EnhancedPSXIndicatorProcessor() as processor:
        # Process all symbols
        results = await processor.process_all_symbols()
        
        logger.info(f"Processed {results['successful']} symbols successfully")
        logger.info(f"Failed to process {results['failed']} symbols")


async def example_custom_config():
    """Example 2: Using custom configuration."""
    logger.info("Example 2: Custom configuration")
    
    # Create custom configuration
    config = ProcessorConfig(
        max_workers=4,
        calculate_advanced_indicators=True,
        include_ml_features=True,
        export_formats=["sqlite", "csv"],
        rsi_periods=[14, 21, 26],
        ma_periods=[20, 50, 100, 200]
    )
    
    async with EnhancedPSXIndicatorProcessor(config) as processor:
        # Process specific symbols
        symbols = ["KSE100", "OGDC", "PPL", "LUCK"]
        results = await processor.process_symbols_async(symbols)
        
        logger.info(f"Processing completed in {results['processing_time']:.2f} seconds")


async def example_config_from_file():
    """Example 3: Loading configuration from file."""
    logger.info("Example 3: Configuration from file")
    
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        config = ProcessorConfig.from_file(str(config_path))
        
        async with EnhancedPSXIndicatorProcessor(config) as processor:
            # Get processing statistics
            stats = processor.get_processing_stats()
            logger.info(f"System configuration: {stats}")
            
            # Process all symbols
            results = await processor.process_all_symbols()
            
            # Cleanup unused tables
            processor.cleanup_unused_tables()
    else:
        logger.warning(f"Config file not found: {config_path}")


async def example_performance_monitoring():
    """Example 4: Performance monitoring and detailed reporting."""
    logger.info("Example 4: Performance monitoring")
    
    config = ProcessorConfig(
        max_workers=8,
        enable_data_validation=True,
        export_formats=["sqlite", "csv"]
    )
    
    async with EnhancedPSXIndicatorProcessor(config) as processor:
        # Process symbols with detailed monitoring
        results = await processor.process_symbols_async()
        
        # Analyze results
        successful_results = [r for r in results['details'] if r['success']]
        failed_results = [r for r in results['details'] if not r['success']]
        
        if successful_results:
            avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            avg_rows = sum(r['row_count'] for r in successful_results) / len(successful_results)
            
            logger.info(f"Average processing time: {avg_time:.2f} seconds")
            logger.info(f"Average rows processed: {avg_rows:.0f}")
        
        if failed_results:
            logger.warning("Failed symbols:")
            for failed in failed_results:
                logger.warning(f"  - {failed['table_name']}: {failed['error']}")


async def example_data_validation():
    """Example 5: Data validation and quality assessment."""
    logger.info("Example 5: Data validation")
    
    config = ProcessorConfig(
        enable_data_validation=True,
        max_workers=2  # Use fewer workers for detailed analysis
    )
    
    async with EnhancedPSXIndicatorProcessor(config) as processor:
        # Process a few symbols with validation
        symbols = ["KSE100", "OGDC"]
        results = await processor.process_symbols_async(symbols)
        
        # Analyze validation results
        for detail in results['details']:
            if detail['validation_results']:
                validation = detail['validation_results']
                logger.info(f"Symbol {detail['table_name']}:")
                logger.info(f"  Quality Score: {validation['quality_score']:.1f}")
                logger.info(f"  Issues: {validation['issues']}")
                logger.info(f"  Metrics: {validation['metrics']}")


def main():
    """Run all examples."""
    examples = [
        example_basic_usage,
        example_custom_config,
        example_config_from_file,
        example_performance_monitoring,
        example_data_validation
    ]
    
    for example in examples:
        try:
            asyncio.run(example())
            print("-" * 50)
        except Exception as e:
            logger.error(f"Error in {example.__name__}: {e}")


if __name__ == "__main__":
    main()
