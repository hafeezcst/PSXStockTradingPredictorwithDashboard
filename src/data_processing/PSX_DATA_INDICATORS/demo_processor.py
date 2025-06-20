#!/usr/bin/env python3
"""
PSX Indicator Processor - Demo Version

This is a standalone demo that shows the indicator calculation capabilities
without requiring a database connection.
"""

import pandas as pd
import numpy as np
from enhanced_psx_processor_simple import PSXIndicatorProcessor, ProcessorConfig, IndicatorCalculator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample PSX stock data for demonstration."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Generate realistic stock price data
    np.random.seed(42)  # For reproducible results
    
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns with slight upward bias
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1))  # Ensure price doesn't go below 1
    
    # Create volume data
    volume = np.random.randint(1000, 10000, len(dates))
    
    # Create OHLC data
    high = [p * np.random.uniform(1.00, 1.05) for p in prices]
    low = [p * np.random.uniform(0.95, 1.00) for p in prices]
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': high,
        'Low': low,
        'Close': prices,
        'Volume': volume,
        'Symbol': 'DEMO'
    })
    
    return data

def demo_indicator_calculation():
    """Demonstrate the indicator calculation capabilities."""
    print("=" * 60)
    print("🚀 PSX INDICATOR PROCESSOR DEMO")
    print("=" * 60)
    
    try:
        # Create sample data
        print("📊 Creating sample stock data...")
        df = create_sample_data()
        print(f"✅ Generated {len(df)} days of sample data")
        print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")        # Initialize processor
        print("\n⚙️ Initializing processor...")
        config = ProcessorConfig()
        calculator = IndicatorCalculator(config)
        print("✅ Processor initialized successfully")
        
        # Calculate indicators
        print("\n🔍 Calculating technical indicators...")
        result_df = calculator.calculate_comprehensive_indicators(df)
        
        # Show results
        indicator_columns = [col for col in result_df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
        
        print(f"✅ Calculated {len(indicator_columns)} indicators:")
        for i, col in enumerate(indicator_columns[:10], 1):  # Show first 10
            latest_value = result_df[col].iloc[-1]
            if pd.notna(latest_value):
                print(f"   {i:2d}. {col}: {latest_value:.4f}")
            else:
                print(f"   {i:2d}. {col}: N/A (calculating...)")
        
        if len(indicator_columns) > 10:
            print(f"   ... and {len(indicator_columns) - 10} more indicators")
        
        # Show latest data
        print("\n📈 Latest data (last 5 days):")
        display_cols = ['Date', 'Close', 'RSI_14', 'SMA_20', 'Volume']
        recent_data = result_df[display_cols].tail(5)
        print(recent_data.to_string(index=False, float_format='%.2f'))
        
        # Performance stats
        print("\n📊 Data Quality:")
        total_values = len(result_df) * len(indicator_columns)
        valid_values = result_df[indicator_columns].notna().sum().sum()
        quality_pct = (valid_values / total_values) * 100
        print(f"   Total data points: {total_values:,}")
        print(f"   Valid data points: {valid_values:,}")
        print(f"   Data quality: {quality_pct:.1f}%")
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Configure your database in config.yaml")
        print("2. Run: python enhanced_psx_indicator_processor.py")
        print("3. Or use: python enhanced_psx_processor_simple.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    demo_indicator_calculation()
