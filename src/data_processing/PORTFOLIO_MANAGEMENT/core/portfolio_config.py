"""
Portfolio Manager Configuration
Customize portfolio parameters, risk management, and trading rules
"""

# Portfolio Configuration
PORTFOLIO_CONFIG = {
    'initial_capital': 35_000_000,  # 35 Million PKR
    'max_positions': 50,            # Maximum number of stock positions
    'min_position_size': 100_000,   # Minimum 100K PKR per position
    'max_position_size': 2_000_000, # Maximum 2M PKR per position
    'rebalance_threshold': 0.20,    # 20% - trigger rebalancing if position deviates by this much
    'transaction_cost': 0.002,      # 0.2% transaction cost (brokerage + taxes)
}

# Signal Configuration
SIGNAL_CONFIG = {
    'signal_db_path': r"data\databases\production\PSX_investing_Stocks_KMI100.db",
    'top_signals_count': 50,        # Number of top buy signals to consider
    'signal_refresh_interval': 300, # 5 minutes - how often to check for new signals
    'min_signal_age_hours': 1,      # Minimum age of signal before acting (avoid noise)
    'signal_strength_factors': {
        'pnl_weight': 0.4,          # Weight for P&L percentage in signal strength
        'rsi_weight': 0.3,          # Weight for RSI in signal strength
        'volume_weight': 0.2,       # Weight for volume in signal strength
        'holding_days_weight': 0.1  # Weight for holding days in signal strength
    }
}

# Risk Management
RISK_CONFIG = {
    'max_sector_exposure': 0.30,    # Maximum 30% exposure to any single sector
    'max_single_position': 0.08,    # Maximum 8% of portfolio in single stock
    'cash_reserve_minimum': 0.05,   # Keep minimum 5% cash reserve
    'stop_loss_percentage': -0.15,  # 15% stop loss (if implemented)
    'take_profit_percentage': 0.50, # 50% take profit (if implemented)
    'daily_loss_limit': -0.03,      # 3% daily portfolio loss limit
}

# Trading Rules
TRADING_RULES = {
    'market_hours_only': True,      # Only trade during market hours
    'market_open_time': '09:30',    # PSX market open time
    'market_close_time': '15:30',   # PSX market close time
    'no_trade_days': ['saturday', 'sunday'],  # Days when market is closed
    'position_entry_rules': {
        'require_volume_confirmation': True,    # Require volume > average
        'require_rsi_confirmation': True,       # Require RSI > 40 for buy signals
        'require_positive_momentum': True,      # Require positive price momentum
        'max_daily_trades': 10,                 # Maximum trades per day
    },
    'position_exit_rules': {
        'exit_on_sell_signal': True,           # Exit immediately on sell signal
        'exit_on_neutral_conversion': False,    # Keep position if signal becomes neutral
        'exit_on_top50_removal': True,         # Exit if stock falls out of top 50
        'partial_profit_taking': True,         # Take partial profits on large gains
    }
}

# Reporting Configuration
REPORTING_CONFIG = {
    'daily_report': True,           # Generate daily performance report
    'weekly_summary': True,         # Generate weekly summary
    'monthly_analysis': True,       # Generate monthly analysis
    'report_directory': 'data/reports',
    'backup_directory': 'data/backups',
    'log_directory': 'data/logs',
    'export_formats': ['json', 'csv', 'excel'],  # Report export formats
    'email_reports': False,         # Email reports (requires email config)
    'telegram_alerts': False,      # Telegram alerts (requires bot config)
}

# Performance Tracking
PERFORMANCE_CONFIG = {
    'benchmark_symbol': 'KSE100',   # Benchmark for comparison
    'track_metrics': [
        'total_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'average_holding_period',
        'profit_factor',
        'calmar_ratio'
    ],
    'rolling_windows': [7, 30, 90, 365],  # Days for rolling calculations
}

# Alert Configuration
ALERT_CONFIG = {
    'portfolio_loss_alert': -0.05,     # Alert if portfolio drops 5%
    'single_position_loss_alert': -0.10, # Alert if single position drops 10%
    'large_gain_alert': 0.25,          # Alert if single position gains 25%
    'cash_depletion_alert': 0.02,      # Alert if cash drops below 2%
    'signal_age_alert': 24,             # Alert if signals are older than 24 hours
}

# Database Configuration
DATABASE_CONFIG = {
    'connection_timeout': 30,       # Database connection timeout (seconds)
    'retry_attempts': 3,            # Number of retry attempts for database operations
    'backup_frequency': 'daily',    # Database backup frequency
    'data_retention_days': 365,     # How long to keep trade history
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',            # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_rotation': 'daily',        # daily, weekly, monthly
    'max_log_files': 30,           # Maximum number of log files to keep
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'console_output': True,         # Show logs in console
    'file_output': True,           # Save logs to file
}

# Development/Testing Configuration
DEV_CONFIG = {
    'paper_trading': False,         # Set to True for paper trading mode
    'simulation_mode': False,       # Set to True for backtesting
    'debug_mode': False,           # Enable debug features
    'test_data_path': 'data/test',  # Path for test data
    'mock_signals': False,         # Use mock signals for testing
}

# Export all configurations
ALL_CONFIGS = {
    'portfolio': PORTFOLIO_CONFIG,
    'signals': SIGNAL_CONFIG,
    'risk': RISK_CONFIG,
    'trading': TRADING_RULES,
    'reporting': REPORTING_CONFIG,
    'performance': PERFORMANCE_CONFIG,
    'alerts': ALERT_CONFIG,
    'database': DATABASE_CONFIG,
    'logging': LOGGING_CONFIG,
    'development': DEV_CONFIG,
}
