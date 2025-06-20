# Enhanced PSX Data Processing Module

This module provides a comprehensive, enterprise-grade solution for downloading, processing, and managing Pakistan Stock Exchange (PSX) data. It represents a significant enhancement over the original implementation with improved architecture, monitoring, error handling, and data quality management.

## 🚀 Key Improvements

### Architecture Enhancements
- **Modular Design**: Separated concerns into specialized classes
- **Configuration Management**: Centralized YAML-based configuration
- **Dependency Injection**: Loose coupling between components
- **Context Management**: Proper resource cleanup with context managers

### Data Quality & Validation
- **Comprehensive Validation**: OHLCV relationship checks, price range validation
- **Data Cleaning**: Automatic removal of duplicates and invalid entries
- **Anomaly Detection**: Identification of unusual price movements and volume spikes
- **Quality Metrics**: Detailed data quality reporting

### Monitoring & Observability
- **Performance Metrics**: Response times, success rates, throughput
- **Health Checks**: Database, memory, and disk space monitoring
- **Structured Logging**: Enhanced logging with context and performance data
- **Progress Tracking**: Real-time progress reporting with detailed statistics

### Error Handling & Reliability
- **Custom Exceptions**: Specific exception types for different error scenarios
- **Retry Logic**: Configurable retry mechanisms with exponential backoff
- **Graceful Degradation**: Fallback mechanisms for failed operations
- **Resource Cleanup**: Proper cleanup of connections and resources

### Database Management
- **Connection Pooling**: Efficient database connection management
- **Backup & Recovery**: Automated database backup functionality
- **Integrity Checks**: Regular database integrity validation
- **Performance Optimization**: Query optimization and maintenance

### API & Networking
- **Rate Limiting**: Respectful server interaction with configurable delays
- **Connection Management**: Persistent connections with retry strategies
- **Performance Tuning**: Dynamic thread count adjustment based on response times
- **Error Recovery**: Robust handling of network issues

## 📁 File Structure

```
src/data_processing/
├── 01-PSX_Database_data_download_to_SQL_db_PSX.py  # Enhanced main module
├── __init__.py                                      # Package initialization
├── config.yaml                                     # Configuration file
├── config_manager.py                               # Configuration management
├── exceptions.py                                   # Custom exception classes
├── monitoring.py                                   # Metrics and health monitoring
├── data_validator.py                              # Data validation utilities
├── enhanced_db_manager.py                         # Database management
├── api_client.py                                  # PSX API client
├── enhanced_data_processor.py                     # Data processing utilities
└── README.md                                      # This documentation
```

## 🛠️ Configuration

The module uses a YAML configuration file (`config.yaml`) for centralized settings:

```yaml
database:
  pool_size: 10
  max_overflow: 20
  main_db_path: "data/databases/production/PSX_consolidated_data_PSX.db"
  alt_db_path: "data/databases/production/PSX_consolidated_data_PSX_Alternative.db"

threading:
  max_workers: 4
  max_threads: 8
  min_threads: 2
  request_timeout: 60

api:
  history_url: "https://dps.psx.com.pk/historical"
  symbols_url: "https://dps.psx.com.pk/symbols"
  max_retries: 3
  delay_between_requests: 0.5

data_validation:
  max_price_change_percent: 50.0
  min_volume: 0
  check_ohlc_relationships: true

monitoring:
  enable_metrics: true
  metrics_file: "download_metrics.json"
  health_check_interval: 300
```

## 💻 Usage Examples

### Basic Usage

```python
from src.data_processing import EnhancedDataReader
from datetime import date

# Using context manager (recommended)
with EnhancedDataReader() as reader:
    # Download data for a specific symbol
    data = reader.stocks('HBL', date(2024, 1, 1), date(2024, 6, 1))
    print(f"Downloaded {len(data)} records for HBL")
    
    # Get performance metrics
    metrics = reader.get_performance_metrics()
    print(f"Success rate: {metrics['download_metrics'].success_rate:.1f}%")
```

### Custom Configuration

```python
from src.data_processing import EnhancedDataReader

# Use custom configuration file
reader = EnhancedDataReader(config_path='/path/to/custom/config.yaml')

# Process multiple symbols
symbols = ['HBL', 'ENGRO', 'NESTLE']
for symbol in symbols:
    try:
        data = reader.stocks(symbol, start_date, end_date)
        if not data.empty:
            print(f"Successfully processed {symbol}: {len(data)} records")
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
```

### Advanced Data Validation

```python
from src.data_processing import DataValidator, DataValidationConfig

# Configure validation
config = DataValidationConfig(
    max_price_change_percent=30.0,
    check_ohlc_relationships=True,
    validate_date_continuity=True
)

validator = DataValidator(config)

# Validate data
is_valid, issues = validator.validate_stock_data(data, 'HBL')
if not is_valid:
    print(f"Data quality issues: {issues}")
    
    # Clean the data
    cleaned_data = validator.clean_data(data, 'HBL')
```

### Monitoring and Metrics

```python
from src.data_processing import MetricsCollector

# Initialize metrics collection
metrics = MetricsCollector("custom_metrics.json")
metrics.start_session(total_symbols=100)

# Record individual operations
metrics.record_download_attempt('HBL', success=True, response_time=1.2)

# Generate summary report
report = metrics.get_summary_report()
print(report)
```

## 🔧 Key Classes and Components

### EnhancedDataReader
Main orchestrator class that coordinates all components:
- **Configuration management**
- **Component initialization**
- **Data download orchestration**
- **Performance monitoring**
- **Resource cleanup**

### DataValidator
Comprehensive data validation and cleaning:
- **OHLCV relationship validation**
- **Price range checks**
- **Volume validation**
- **Date continuity verification**
- **Anomaly detection**

### MetricsCollector
Performance monitoring and metrics collection:
- **Download success rates**
- **Response time tracking**
- **Error classification**
- **Performance reporting**

### EnhancedDatabaseManager
Advanced database operations:
- **Connection pooling**
- **Backup management**
- **Integrity checks**
- **Performance optimization**

### PSXAPIClient
Robust API communication:
- **Retry mechanisms**
- **Rate limiting**
- **Response parsing**
- **Performance tracking**

## 📊 Monitoring and Metrics

The enhanced module provides comprehensive monitoring capabilities:

### Performance Metrics
- **Download success rates** by symbol and overall
- **Average response times** with trend analysis
- **Database performance** statistics
- **Threading efficiency** metrics

### Health Monitoring
- **Database connectivity** checks
- **Memory usage** monitoring
- **Disk space** availability
- **API endpoint** health

### Data Quality Metrics
- **Validation success rates**
- **Data completeness** percentages
- **Anomaly detection** counts
- **Cleaning operation** statistics

## 🚨 Error Handling

The module implements comprehensive error handling:

### Custom Exception Types
```python
PSXDataDownloadError    # Data download failures
DatabaseConnectionError # Database connectivity issues
APIConnectionError     # API communication failures
DataValidationError    # Data quality problems
ConfigurationError     # Configuration issues
```

### Retry Mechanisms
- **Exponential backoff** for API calls
- **Configurable retry counts**
- **Graceful degradation** on persistent failures
- **Alternative database** failover

## 🔒 Security Considerations

- **Input validation** for all user inputs
- **SQL injection prevention** through parameterized queries
- **Rate limiting** to respect server resources
- **Resource cleanup** to prevent memory leaks

## 📈 Performance Optimizations

### Threading
- **Dynamic thread adjustment** based on response times
- **Resource-aware** threading limits
- **Health-based** thread scaling

### Database
- **Connection pooling** for efficient resource usage
- **Query optimization** with proper indexing
- **Regular maintenance** operations

### Caching
- **Response caching** for frequently accessed data
- **Configuration caching** to reduce file I/O
- **Metadata caching** for improved performance

## 🧪 Testing and Validation

### Unit Testing
```python
# Example test structure
def test_data_validation():
    validator = DataValidator(config)
    sample_data = create_sample_data()
    is_valid, issues = validator.validate_stock_data(sample_data, 'TEST')
    assert is_valid == True
```

### Integration Testing
```python
# Example integration test
def test_complete_download_flow():
    with EnhancedDataReader() as reader:
        data = reader.stocks('HBL', start_date, end_date)
        assert len(data) > 0
        assert 'Close' in data.columns
```

## 📋 Migration Guide

### From Legacy DataReader

The enhanced module maintains backward compatibility:

```python
# Old usage still works
data_reader = DataReader()
data = data_reader.stocks('HBL', start_date, end_date)

# Enhanced usage (recommended)
with EnhancedDataReader() as reader:
    data = reader.stocks('HBL', start_date, end_date)
```

### Configuration Migration

Convert hard-coded values to configuration:

```python
# Old: Hard-coded values
max_workers = 4
timeout = 60

# New: Configuration-based
config = AppConfig.from_yaml('config.yaml')
max_workers = config.threading.max_workers
timeout = config.threading.request_timeout
```

## 🔮 Future Enhancements

### Planned Features
- **Real-time data streaming** support
- **Machine learning** data quality prediction
- **Advanced caching** with Redis support
- **Distributed processing** capabilities
- **Enhanced visualization** tools

### Scalability Improvements
- **Horizontal scaling** support
- **Load balancing** for multiple API endpoints
- **Database sharding** strategies
- **Cloud deployment** configurations

## 🤝 Contributing

To contribute to this module:

1. **Follow the established architecture** patterns
2. **Add comprehensive tests** for new features
3. **Update documentation** for any changes
4. **Follow logging conventions** for observability
5. **Maintain backward compatibility** where possible

## 📞 Support

For issues or questions:
- **Check the logs** in `data_reader.log`
- **Review configuration** in `config.yaml`
- **Examine metrics** in `download_metrics.json`
- **Validate data quality** using built-in validators

---

*This enhanced module represents a production-ready solution for PSX data management with enterprise-grade features for reliability, monitoring, and data quality.*
