# PSX Data to SQL Database - Enhanced Module

**Version 2.0.0** | **Production Ready** | **Enterprise Grade**

This standalone module provides a comprehensive, enterprise-grade solution for downloading, processing, and managing Pakistan Stock Exchange (PSX) data. It represents a complete rewrite and enhancement of the original data download system with improved architecture, monitoring, error handling, and data quality management.

## 🎯 Quick Start

### 1. Setup
```bash
# Navigate to the module directory
cd PSX_DATA_TO_SQL_DATABASE

# Run the setup script (installs dependencies and creates directories)
python setup.py

# Or manually install requirements
pip install -r requirements.txt
```

### 2. Configuration
Review and update `config.yaml` for your environment:
```yaml
database:
  main_db_path: "data/databases/production/PSX_consolidated_data_PSX.db"
  
symbols:
  file_path: "data/databases/production/psxsymbols.xlsx"
  sheet_name: "KSEALL"
```

### 3. Run Data Download
```bash
# Simple execution
python run_psx_download.py

# Or run the main module directly
python enhanced_psx_data_reader.py

# Run tests to validate setup
python test_enhanced_reader.py
```

## 📁 Module Structure

```
PSX_DATA_TO_SQL_DATABASE/
├── enhanced_psx_data_reader.py      # Main enhanced data reader
├── config_manager.py                # Configuration management
├── exceptions.py                    # Custom exception classes
├── monitoring.py                    # Metrics and health monitoring
├── data_validator.py                # Data validation utilities
├── enhanced_db_manager.py           # Database management
├── api_client.py                    # PSX API client
├── enhanced_data_processor.py       # Data processing utilities
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
├── setup.py                         # Setup and installation script
├── run_psx_download.py              # Simple launcher script
├── test_enhanced_reader.py          # Test suite
├── __init__.py                      # Package initialization
└── README.md                        # This documentation
```

## 🚀 Key Features & Improvements

### ✨ Architecture Enhancements
- **Modular Design**: Separated concerns into specialized classes
- **Configuration Management**: Centralized YAML-based configuration
- **Context Management**: Proper resource cleanup with context managers
- **Dependency Injection**: Loose coupling between components

### 📊 Data Quality & Validation
- **Comprehensive Validation**: OHLCV relationship checks, price range validation
- **Automatic Data Cleaning**: Duplicate removal and invalid entry cleanup
- **Anomaly Detection**: Identification of unusual price movements
- **Quality Metrics**: Detailed data quality reporting

### 📈 Monitoring & Observability
- **Performance Metrics**: Response times, success rates, throughput
- **Health Checks**: Database, memory, and disk space monitoring
- **Structured Logging**: Enhanced logging with context and performance data
- **Progress Tracking**: Real-time progress reporting with statistics

### 🛡️ Error Handling & Reliability
- **Custom Exceptions**: Specific exception types for different scenarios
- **Retry Logic**: Configurable retry mechanisms with exponential backoff
- **Graceful Degradation**: Fallback mechanisms for failed operations
- **Resource Cleanup**: Proper cleanup of connections and resources

### 🗄️ Database Management
- **Connection Pooling**: Efficient database connection management
- **Backup & Recovery**: Automated database backup functionality
- **Integrity Checks**: Regular database integrity validation
- **Performance Optimization**: Query optimization and maintenance

## 💻 Usage Examples

### Basic Usage
```python
from enhanced_psx_data_reader import EnhancedDataReader
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
from enhanced_psx_data_reader import EnhancedDataReader

# Use custom configuration file
reader = EnhancedDataReader(config_path='custom_config.yaml')

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
from data_validator import DataValidator
from config_manager import DataValidationConfig

# Configure validation
config = DataValidationConfig(
    max_price_change_percent=30.0,
    check_ohlc_relationships=True
)

validator = DataValidator(config)

# Validate data
is_valid, issues = validator.validate_stock_data(data, 'HBL')
if not is_valid:
    print(f"Data quality issues: {issues}")
```

## 🔧 Configuration Options

The module uses `config.yaml` for all configuration. Key sections include:

### Database Configuration
```yaml
database:
  pool_size: 10                    # Connection pool size
  main_db_path: "path/to/main.db"  # Primary database path
  backup_enabled: true             # Enable automatic backups
```

### Threading Configuration
```yaml
threading:
  max_workers: 4                   # Maximum concurrent workers
  max_threads: 8                   # Maximum thread count
  request_timeout: 60              # Request timeout in seconds
```

### API Configuration
```yaml
api:
  history_url: "https://dps.psx.com.pk/historical"
  max_retries: 3                   # Maximum retry attempts
  delay_between_requests: 0.5      # Delay between requests
```

### Data Validation Configuration
```yaml
data_validation:
  max_price_change_percent: 50.0   # Maximum allowed price change
  check_ohlc_relationships: true   # Validate OHLC relationships
```

## 📊 Monitoring & Metrics

### Performance Metrics
- **Download Success Rates**: By symbol and overall
- **Response Time Analysis**: Average, min, max response times
- **Database Performance**: Query times, connection health
- **Threading Efficiency**: Worker utilization, thread performance

### Health Monitoring
- **Database Connectivity**: Connection health checks
- **System Resources**: Memory usage, disk space
- **API Health**: Endpoint availability and performance

### Data Quality Metrics
- **Validation Success Rates**: Data quality percentages
- **Anomaly Detection**: Unusual pattern identification
- **Completeness**: Data coverage analysis

## 🧪 Testing

### Run All Tests
```bash
python test_enhanced_reader.py
```

### Individual Test Categories
```python
# Test configuration management
python -c "from test_enhanced_reader import test_configuration; test_configuration()"

# Test database functionality
python -c "from test_enhanced_reader import test_database_manager; test_database_manager()"

# Test data validation
python -c "from test_enhanced_reader import test_data_validation; test_data_validation()"
```

## 🔒 Security & Best Practices

### Security Features
- **Input Validation**: SQL injection prevention
- **Rate Limiting**: Respectful server interaction
- **Resource Cleanup**: Memory leak prevention
- **Error Handling**: Secure error reporting

### Best Practices
- **Use Context Managers**: Always use `with` statements
- **Monitor Resources**: Check health metrics regularly
- **Validate Data**: Use built-in validation tools
- **Regular Backups**: Enable automatic backup features

## 📈 Performance Optimizations

### Threading Optimizations
- **Dynamic Adjustment**: Automatic thread count optimization
- **Resource-Aware**: Health-based scaling
- **Load Balancing**: Efficient work distribution

### Database Optimizations
- **Connection Pooling**: Efficient resource usage
- **Query Optimization**: Indexed queries and batch operations
- **Maintenance**: Regular integrity checks and optimization

### Caching Strategies
- **Response Caching**: Reduce redundant API calls
- **Configuration Caching**: Minimize file I/O
- **Metadata Caching**: Improve performance

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Database Connection Issues
```python
# Test database connectivity
from enhanced_db_manager import EnhancedDatabaseManager
from config_manager import DatabaseConfig

db_manager = EnhancedDatabaseManager(DatabaseConfig())
print(db_manager.check_connection())
```

#### Configuration Problems
```python
# Validate configuration
from config_manager import AppConfig
config = AppConfig.from_yaml()
config.validate()
```

### Log Analysis
Check the following log files:
- `data_reader.log` - Main application logs
- `download_metrics.json` - Performance metrics
- Console output for real-time status

## 🔮 Future Enhancements

### Planned Features
- **Real-time Data Streaming**: Live market data support
- **Advanced Analytics**: Built-in technical indicators
- **Cloud Integration**: AWS/Azure deployment support
- **API Gateway**: RESTful API for data access

### Scalability Improvements
- **Distributed Processing**: Multi-node support
- **Load Balancing**: Multiple API endpoint support
- **Horizontal Scaling**: Container-based deployment
- **Advanced Caching**: Redis/Memcached integration

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests to ensure compatibility
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for changes
- Maintain backward compatibility

## 📞 Support & Documentation

### Getting Help
1. **Check Logs**: Review `data_reader.log` for errors
2. **Validate Config**: Ensure `config.yaml` is correct
3. **Run Tests**: Use `test_enhanced_reader.py` for diagnostics
4. **Check Metrics**: Review `download_metrics.json`

### Documentation
- **Code Documentation**: Comprehensive docstrings in all modules
- **Configuration Guide**: Detailed config.yaml documentation
- **API Reference**: Type hints and method documentation
- **Examples**: Practical usage examples in README

---

**This enhanced module provides a production-ready, enterprise-grade solution for PSX data management with comprehensive monitoring, validation, and reliability features.**

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
