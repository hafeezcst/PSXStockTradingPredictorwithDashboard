# Portfolio Management System

## Overview

A robust, automated portfolio management system that integrates with your existing signal database to manage a 35 million PKR portfolio. The system automatically trades based on the top 50 buy signals from your proven signaling system, with comprehensive risk management, real-time monitoring, and detailed analytics.

## Key Features

### 🎯 **Signal Integration**
- Seamlessly integrates with existing `PSX_investing_Stocks_KMI100.db` signal database
- Uses `buy_stocks`, `sell_stocks`, and `neutral_stocks` tables as signal sources
- Automatically trades based on top 50 performing buy signals
- Real-time signal monitoring and portfolio updates

### 💰 **Portfolio Management**
- Manages 35 Million PKR portfolio with dynamic position sizing
- Entry: Only invests in stocks among top 50 buy signals
- Exit: Automatically sells when stocks fall out of top 50 or receive sell signals
- Optimal capital allocation with risk-based position sizing
- Kelly Criterion position sizing for optimal risk-adjusted returns

### 🛡️ **Risk Management**
- Maximum position limits (8% per stock, 30% per sector)
- Daily loss limits and portfolio concentration monitoring
- Real-time Value at Risk (VaR) calculations
- Stop-loss and take-profit mechanisms
- Cash reserve management (minimum 5% cash)

### 📊 **Analytics & Reporting**
- Real-time performance tracking and portfolio analytics
- Sharpe ratio, max drawdown, win rate calculations
- Signal effectiveness analysis
- Risk metrics and concentration analysis
- Automated report generation (JSON, Excel formats)

### 🔄 **Automation Features**
- Continuous monitoring mode with configurable intervals
- Automatic rebalancing based on signal changes
- Error-resistant operation with comprehensive logging
- Paper trading mode for testing strategies

## System Architecture

```
Portfolio Management System
├── portfolio_manager.py      # Main portfolio management engine
├── risk_manager.py          # Risk management and validation
├── portfolio_analytics.py   # Performance analytics and reporting
├── portfolio_config.py      # System configuration
├── launch_portfolio_system.py # Main launcher
└── test_portfolio_system.py  # System validation
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Existing signal database at `data/databases/production/PSX_investing_Stocks_KMI100.db`
- Required Python packages (see requirements below)

### Required Packages
```bash
pip install pandas numpy sqlite3 matplotlib seaborn
```

### Verification
Run the system validation test:
```bash
python test_portfolio_system.py
```

This will verify:
- Database connectivity and structure
- Required dependencies
- Directory permissions
- Signal processing capabilities
- Risk calculation functions

## Quick Start

### 1. Initial Setup
```bash
python launch_portfolio_system.py --mode setup
```

This will:
- Validate system components
- Initialize portfolio data
- Generate initial analytics report
- Display current portfolio status

### 2. Single Rebalancing
```bash
python launch_portfolio_system.py --mode single
```
Performs one-time portfolio rebalancing based on current signals.

### 3. Continuous Monitoring
```bash
python launch_portfolio_system.py --mode continuous
```
Starts continuous monitoring with automatic rebalancing every 5 minutes.

### 4. Analytics Only
```bash
python launch_portfolio_system.py --mode analytics
```
Generates comprehensive analytics without executing trades.

## Configuration

### Portfolio Settings
```python
PORTFOLIO_CONFIG = {
    'initial_capital': 35_000_000,  # 35 Million PKR
    'max_positions': 50,            # Maximum positions
    'min_position_size': 100_000,   # 100K PKR minimum
    'max_position_size': 2_000_000, # 2M PKR maximum
    'transaction_cost': 0.002,      # 0.2% transaction cost
}
```

### Risk Management
```python
RISK_CONFIG = {
    'max_single_position': 0.08,    # 8% max per stock
    'max_sector_exposure': 0.30,    # 30% max per sector
    'cash_reserve_minimum': 0.05,   # 5% minimum cash
    'daily_loss_limit': -0.03,      # 3% daily loss limit
}
```

### Trading Rules
```python
TRADING_RULES = {
    'market_hours_only': True,      # Trade only during market hours
    'market_open_time': '09:30',    # PSX market open
    'market_close_time': '15:30',   # PSX market close
    'exit_on_sell_signal': True,    # Exit on sell signals
    'exit_on_top50_removal': True,  # Exit if removed from top 50
}
```

## Trading Strategy

### Entry Criteria
1. Stock must be in top 50 buy signals (ranked by P&L percentage)
2. RSI Weekly Average > 40
3. Signal marked as "Success = Yes"
4. Sufficient liquidity (volume > 10K daily)
5. Passes risk management validation

### Exit Criteria
1. Stock falls out of top 50 buy signals
2. Signal changes from "Buy" to "Sell"
3. Risk management triggers (stop-loss, concentration limits)
4. Rebalancing requirements

### Position Sizing
- Kelly Criterion-based optimal position sizing
- Constrained by risk limits (min 100K, max 2M PKR)
- Signal strength weighting based on:
  - Historical P&L percentage (40% weight)
  - RSI levels (30% weight)
  - Volume confirmation (20% weight)
  - Signal age (10% weight)

## Risk Management Features

### Position Limits
- **Single Position**: Maximum 8% of portfolio value
- **Sector Concentration**: Maximum 30% per sector
- **Position Count**: Maximum 50 simultaneous positions
- **Minimum Size**: 100,000 PKR per position
- **Maximum Size**: 2,000,000 PKR per position

### Risk Monitoring
- **Daily Loss Limit**: 3% maximum daily portfolio loss
- **Value at Risk**: 95% and 99% VaR calculations
- **Portfolio Concentration**: Herfindahl-Hirschman Index monitoring
- **Cash Reserve**: Minimum 5% cash reserve maintained
- **Correlation Risk**: Basic correlation analysis for similar stocks

### Validation Checks
- Signal quality validation (age, success rate, RSI levels)
- Liquidity validation (volume requirements)
- Market hours validation
- Daily trade limit enforcement

## Analytics & Reporting

### Performance Metrics
- **Total Return**: Portfolio performance vs. initial capital
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Volatility**: Portfolio return volatility (daily/annualized)

### Position Analysis
- Individual stock performance tracking
- Trade frequency and holding periods
- P&L distribution and statistics
- Win rate per position

### Signal Effectiveness
- Historical signal performance analysis
- Signal-to-outcome mapping
- Average returns by signal type
- Signal age and success correlation

### Risk Analytics
- Portfolio concentration analysis
- Sector exposure breakdown
- Value at Risk calculations
- Expected shortfall analysis
- Correlation matrix (when data available)

## File Structure

```
Portfolio Management System/
├── portfolio_manager.py           # Main portfolio engine
├── risk_manager.py               # Risk management
├── portfolio_analytics.py        # Analytics engine
├── portfolio_config.py           # Configuration
├── launch_portfolio_system.py    # System launcher
├── test_portfolio_system.py      # Validation tests
├── examine_signal_db.py          # Database inspection tool
├── data/
│   ├── portfolio_data.json       # Portfolio state
│   ├── logs/                     # System logs
│   ├── reports/                  # Generated reports
│   ├── backups/                  # Portfolio backups
│   └── exports/                  # Exported data
└── data/databases/production/
    └── PSX_investing_Stocks_KMI100.db  # Signal database
```

## Logging & Monitoring

### Log Files
- `portfolio_manager.log`: Portfolio operations and trades
- `risk_manager.log`: Risk management decisions
- `portfolio_analytics.log`: Analytics calculations
- `portfolio_system_YYYYMMDD.log`: Daily system log

### Log Levels
- **INFO**: Normal operations, trades, rebalancing
- **WARNING**: Risk limit warnings, validation failures
- **ERROR**: System errors, database issues
- **DEBUG**: Detailed calculation steps (when enabled)

## Sample Usage

### Basic Operations
```python
from portfolio_manager import PortfolioManager
from portfolio_analytics import PortfolioAnalytics

# Initialize portfolio manager
pm = PortfolioManager()

# Run single rebalancing
pm.rebalance_portfolio()

# Generate performance report
report = pm.generate_report()

# Run analytics
analytics = PortfolioAnalytics()
dashboard = analytics.create_performance_dashboard()
```

### Custom Configuration
```python
from portfolio_config import ALL_CONFIGS

# Modify configuration
custom_config = ALL_CONFIGS.copy()
custom_config['portfolio']['max_positions'] = 30
custom_config['risk']['daily_loss_limit'] = -0.02

# Initialize with custom config
pm = PortfolioManager(config=custom_config)
```

## Error Handling & Recovery

### Automatic Recovery
- Database connection retry mechanisms
- Trade validation and rollback capabilities
- Portfolio state backup and restoration
- Signal refresh error handling

### Manual Recovery
- Portfolio data backup files with timestamps
- Trade history preservation
- Error log analysis tools
- System state validation commands

## Performance Optimization

### Database Optimization
- Efficient SQL queries for signal retrieval
- Connection pooling for database operations
- Indexed queries on frequently accessed columns

### Memory Management
- Streaming data processing for large datasets
- Garbage collection optimization
- Memory-efficient data structures

### Processing Speed
- Vectorized calculations using NumPy/Pandas
- Parallel processing for analytics calculations
- Optimized position sizing algorithms

## Security Considerations

### Data Protection
- Portfolio data encryption (recommended)
- Secure database connections
- Access control for configuration files

### Trade Security
- Transaction validation and confirmation
- Position limit enforcement
- Audit trail for all operations

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database path and permissions
   python examine_signal_db.py
   ```

2. **Signal Data Empty**
   ```bash
   # Verify signal generation system is running
   # Check database table contents
   ```

3. **Risk Validation Failures**
   ```bash
   # Review risk configuration in portfolio_config.py
   # Check portfolio concentration and limits
   ```

4. **Performance Issues**
   ```bash
   # Enable debug logging
   # Check system resources
   # Optimize signal refresh interval
   ```

### Debug Mode
```bash
# Enable debug logging
python launch_portfolio_system.py --mode setup
# Edit portfolio_config.py: 'log_level': 'DEBUG'
```

## Integration with Existing Systems

### Signal System Integration
- Direct database integration with existing signal tables
- Real-time signal monitoring
- Backward compatibility with existing data structure

### Reporting Integration
- JSON/Excel export compatibility
- API endpoints for external systems (future enhancement)
- Custom report formatting options

## Future Enhancements

### Planned Features
- Web-based dashboard interface
- Mobile app notifications
- Advanced ML-based position sizing
- Sector classification and analysis
- Options and derivatives support
- Multi-timeframe signal analysis

### API Development
- RESTful API for external integration
- Real-time WebSocket feeds
- Third-party broker integration
- Cloud deployment options

## Support & Maintenance

### Regular Maintenance
- Daily portfolio backup verification
- Weekly performance analysis
- Monthly risk assessment
- Quarterly strategy review

### Updates & Patches
- Signal system compatibility updates
- Risk management enhancements
- Performance optimizations
- Bug fixes and security patches

## License & Disclaimer

This software is provided as-is for portfolio management purposes. Users are responsible for:
- Validating signal quality and accuracy
- Compliance with local regulations
- Risk management and loss prevention
- System monitoring and maintenance

**Trading Disclaimer**: This system is for automated portfolio management based on provided signals. Past performance does not guarantee future results. All investments carry risk of loss.

---

**Contact Information**
- System Developer: Portfolio Management Team
- Last Updated: June 2025
- Version: 1.0.0
