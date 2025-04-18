# Dashboard Components

This directory contains the various components used in the PSX Stock Trading Dashboard.

## Signal Tracker Component

The `signal_tracker.py` component provides advanced analytics and visualizations for tracking stock signals. It integrates with the existing signal tracking database to provide real-time insights and alerts.

### Features

1. **Signal Overview**
   - Current signal distribution (Buy/Sell/Neutral)
   - Performance metrics
   - Filterable data tables

2. **Performance Metrics**
   - Profit/Loss analysis by signal type
   - Days in signal distribution
   - Profit vs. Days scatter plot

3. **Signal Transitions**
   - Transition history and analysis
   - Sankey diagram for signal flow visualization
   - Recent transitions table

4. **Alerts & Notifications**
   - Configurable alert thresholds
   - High profit/loss signal detection
   - Long duration signal detection
   - Manual Telegram alert sending

5. **Run Tracker**
   - Manual execution of signal tracking script
   - Options for backup, reporting, and alert sending

### Integration

The Signal Tracker component integrates with:
- The existing `run_stock_signal_tracker.py` script
- The signal tracking database in `/data/databases/production/PSX_investing_Stocks_KMI30_tracking.db`
- The Telegram messaging system

### Configuration

The component uses a configuration file at `/scripts/config/alert_config.json` with the following settings:

```json
{
    "telegram": {
        "enabled": true
    },
    "alerts": {
        "signal_transitions": true,
        "profit_threshold": 5.0,
        "loss_threshold": -5.0,
        "days_in_signal_threshold": 14
    },
    "analysis": {
        "trend_detection": true,
        "volume_analysis": true,
        "performance_metrics": true
    }
}
```

### Usage

To use this component in the dashboard, import it and call the `display_signal_tracker(config)` function with the appropriate configuration.

Example:
```python
from scripts.data_processing.dashboard.components.signal_tracker import display_signal_tracker

# In your Streamlit app
display_signal_tracker(config)
```

## Other Components

The dashboard includes other components for various analysis functions:
- `trading_signals.py`: Basic trading signal visualization
- `charts.py`: Technical chart visualization
- `financial_reports.py`: Financial report analysis
- `portfolio.py`: Portfolio management and analysis
- And more... 