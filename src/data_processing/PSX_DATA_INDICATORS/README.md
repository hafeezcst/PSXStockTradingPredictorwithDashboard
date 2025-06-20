# Enhanced PSX Indicator Processor

A high-performance, feature-rich technical indicator processor for Pakistan Stock Exchange (PSX) data with modern Python capabilities and advanced optimizations.

## ✨ Enhanced Features (2024/2025)

### 🚀 Performance Optimizations
- **Async/Await Processing**: Non-blocking I/O operations for better concurrency
- **Multiprocessing Support**: Parallel processing with configurable worker pools
- **GPU Acceleration**: Optional CUDA support for high-speed calculations
- **Memory Optimization**: Efficient memory usage for large datasets
- **Caching System**: LRU cache with TTL for frequently accessed data

### 📊 Advanced Technical Indicators
- **50+ Technical Indicators**: Comprehensive suite including RSI, MACD, Bollinger Bands, etc.
- **Multi-timeframe Analysis**: Daily, weekly, monthly, quarterly, and annual timeframes
- **Volume Analysis**: Advanced volume-based indicators (OBV, CMF, VWAP)
- **Volatility Measures**: ATR, Bollinger Bands, Keltner Channels
- **Momentum Indicators**: Stochastic, Williams %R, CCI, ROC

### 🤖 Machine Learning Features
- **Feature Engineering**: Automated ML feature generation
- **Price Patterns**: Gap analysis, momentum features, volatility ratios
- **Statistical Features**: Rolling statistics, percentile rankings
- **Data Quality Scoring**: Automated data validation and quality assessment

### 🔧 Modern Development Features
- **Type Hints**: Full Python 3.12+ type annotations
- **Pydantic Models**: Data validation and serialization
- **Rich Console Output**: Beautiful progress bars and tables
- **Configuration Management**: YAML/JSON configuration files
- **Comprehensive Logging**: Structured logging with multiple handlers

### 📤 Export Capabilities
- **Multiple Formats**: SQLite, CSV, Parquet support
- **Batch Processing**: Efficient batch operations
- **Data Validation**: Quality checks before export
- **Compression**: Optimized file sizes

## 🛠️ Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Optional GPU Support
For GPU acceleration (requires NVIDIA GPU with CUDA):
```bash
pip install cudf-cu11 cupy-cuda11x
```

## 🚀 Quick Start

### Basic Usage
```python
import asyncio
from enhanced_psx_indicator_processor import EnhancedPSXIndicatorProcessor

async def main():
    async with EnhancedPSXIndicatorProcessor() as processor:
        results = await processor.process_all_symbols()
        print(f"Processed {results['successful']} symbols")

asyncio.run(main())
```

### Configuration File
```python
from enhanced_psx_indicator_processor import ProcessorConfig

# Load from YAML
config = ProcessorConfig.from_file("config.yaml")
processor = EnhancedPSXIndicatorProcessor(config)
```

### Command Line Interface
```bash
# Process all symbols with default settings
python enhanced_psx_indicator_processor.py

# Process specific symbols with custom workers
python enhanced_psx_indicator_processor.py --symbols KSE100 OGDC PPL --workers 4

# Use configuration file with CSV export
python enhanced_psx_indicator_processor.py --config config.yaml --export csv

# Enable GPU acceleration
python enhanced_psx_indicator_processor.py --gpu
```

## ⚙️ Configuration

### YAML Configuration Example
```yaml
# Performance settings
max_workers: 8
use_gpu: false
cache_size: 128

# Processing options
calculate_advanced_indicators: true
include_ml_features: true
enable_data_validation: true

# Export formats
export_formats:
  - sqlite
  - csv

# Indicator parameters
rsi_periods: [9, 14, 21, 26]
ma_periods: [20, 30, 50, 100, 200]
```

### Programmatic Configuration
```python
config = ProcessorConfig(
    max_workers=4,
    use_gpu=True,
    calculate_advanced_indicators=True,
    export_formats=["sqlite", "parquet"],
    rsi_periods=[14, 21],
    ma_periods=[20, 50, 200]
)
```

## 📈 Calculated Indicators

### Basic Indicators
- **RSI**: Multiple periods (9, 14, 21, 26) with SMA smoothing
- **Moving Averages**: SMA, EMA, WMA for various periods
- **Price Metrics**: Percentage change, daily fluctuation

### Advanced Indicators
- **MACD**: Signal line, histogram, divergence
- **Bollinger Bands**: Upper, lower, squeeze detection
- **Stochastic**: %K, %D, slow stochastic
- **Awesome Oscillator**: Multiple timeframes
- **Ichimoku Cloud**: Complete cloud system
- **Parabolic SAR**: Trend reversal points
- **SuperTrend**: Dynamic support/resistance

### Volume Indicators
- **OBV**: On Balance Volume
- **CMF**: Chaikin Money Flow
- **VPT**: Volume Price Trend
- **VWAP**: Volume Weighted Average Price
- **A/D Line**: Accumulation/Distribution

### Multi-timeframe Analysis
- **Daily**: Standard indicators
- **Weekly**: 5-day aggregated indicators
- **Monthly**: 21-day aggregated indicators
- **Quarterly**: 63-day aggregated indicators
- **Annual**: 252-day aggregated indicators

## 🔍 Data Validation

### Quality Checks
- **Missing Data**: Percentage and impact analysis
- **Price Validation**: Logical price relationships
- **Volume Validation**: Negative value detection
- **Date Continuity**: Gap analysis
- **Statistical Outliers**: Anomaly detection

### Quality Scoring
- **0-100 Scale**: Automated quality scoring
- **Issue Reporting**: Detailed problem identification
- **Threshold Management**: Configurable quality thresholds

## 📊 Performance Features

### Async Processing
```python
# Process multiple symbols concurrently
symbols = ["KSE100", "OGDC", "PPL", "LUCK"]
results = await processor.process_symbols_async(symbols)
```

### Progress Monitoring
- **Rich Progress Bars**: Real-time processing status
- **Performance Metrics**: Time per symbol, throughput
- **Error Tracking**: Failed symbol identification
- **Memory Usage**: Resource consumption monitoring

### Caching
- **Result Caching**: Avoid redundant calculations
- **Table Name Caching**: Database metadata caching
- **Configuration Caching**: Settings optimization

## 🔧 Advanced Usage

### Custom Indicator Calculation
```python
from enhanced_psx_indicator_processor import IndicatorCalculator

calculator = IndicatorCalculator(config)
enhanced_data = calculator.calculate_comprehensive_indicators(raw_data)
```

### Data Validation
```python
from enhanced_psx_indicator_processor import DataValidator

validator = DataValidator()
validation_results = validator.validate_ohlcv_data(data)
print(f"Quality Score: {validation_results['quality_score']}")
```

### Export Options
```python
# Configure multiple export formats
config = ProcessorConfig(
    export_formats=["sqlite", "csv", "parquet"]
)

# Data will be automatically exported to all specified formats
```

## 🐛 Error Handling

### Comprehensive Error Recovery
- **Database Connection**: Automatic retry with exponential backoff
- **Data Processing**: Graceful handling of corrupted data
- **Memory Management**: Automatic garbage collection
- **Resource Cleanup**: Context manager-based resource management

### Logging
- **Structured Logging**: JSON-formatted log entries
- **Multiple Handlers**: Console and file logging
- **Rich Formatting**: Colored console output
- **Error Tracking**: Detailed error information

## 📋 System Requirements

### Minimum Requirements
- Python 3.9+
- 4GB RAM
- 1GB disk space

### Recommended Requirements
- Python 3.12+
- 16GB RAM
- 10GB disk space
- NVIDIA GPU (for GPU acceleration)

### Dependencies
- pandas >= 2.0.0
- numpy >= 1.24.0
- sqlalchemy >= 2.0.0
- pandas-ta >= 0.3.14b
- rich >= 13.0.0
- pydantic >= 2.0.0

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd PSXStockTradingPredictorwithDashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/data_processing/PSX_DATA_INDICATORS/requirements.txt

# Run tests
pytest tests/
```

### Code Quality
- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Unit testing

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

## 🚀 Roadmap

### Planned Features
- [ ] Real-time data streaming
- [ ] WebSocket API integration
- [ ] Interactive dashboard
- [ ] Backtesting framework
- [ ] Portfolio optimization
- [ ] Risk management indicators
- [ ] Sentiment analysis integration
- [ ] Cloud deployment support
