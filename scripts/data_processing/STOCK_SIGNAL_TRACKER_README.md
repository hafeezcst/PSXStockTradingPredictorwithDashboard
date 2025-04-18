# Stock Signal Tracker

This tool tracks stock signals from the PSX Stock Trading database, focusing on buy, sell, and neutral signals.

## Features

- Tracks stock transitions between buy, sell, and neutral signals
- Maintains historical record of signal changes
- Calculates profit/loss metrics for active signals
- Generates comprehensive reports of current positions and signal transitions
- Creates visualizations for performance analysis
- Integrates with the existing PSX analysis workflow

## Setup

1. Ensure you have the required dependencies installed:

```bash
pip install -r stock_signal_tracker_requirements.txt
```

2. The tracker uses a copy of the main database by default. The original database remains unchanged.

## Usage

### Running Directly

```bash
python stock_signal_tracker.py --db /path/to/database.db --backup --report=/path/to/report.txt --visualize --output-dir=./reports
```

### Command-line Arguments

- `--db`: Path to the SQLite database (default: PSX_investing_Stocks_KMI30_tracking.db)
- `--backup`: Create a backup of the database before making changes
- `--report`: Path to save the generated report
- `--visualize`: Generate visualizations of signal performance
- `--output-dir`: Directory to save visualizations (default: ./reports)

### Using the Runner Script

For convenience, you can use the runner script:

```bash
python ../run_stock_signal_tracker.py
```

Command-line options for the runner:
- `--db`: Path to the SQLite database
- `--no-backup`: Skip creating a backup
- `--no-report`: Skip generating a report
- `--no-visualize`: Skip generating visualizations
- `--output-dir`: Directory to save outputs

## Integration with PSX Analysis

The Stock Signal Tracker is automatically run as part of the daily PSX analysis job. After all data processing scripts are executed, the tracker will:

1. Update signal tracking information
2. Generate a daily report
3. Create visualizations of signal performance

## Understanding the Reports

### Active Buy Signals

This section shows all stocks currently in a "Buy" signal state, along with:
- Initial signal date
- Entry price
- Current price
- Days in signal
- Profit/Loss percentage

### Signal Transitions

This section tracks stocks that have changed signals, showing:
- Original signal type
- Current signal type
- Entry and current prices
- Days in signal
- Profit/Loss percentage
- Signal change history

### Performance Analysis

This section provides statistical analysis of signal performance:
- Average profit/loss by signal type
- Median holding days
- Success rate metrics
- Signal transition patterns

## Visualizations

The tracker generates several visualizations:
- Profit/Loss distribution by signal type
- Signal holding period analysis
- Signal transition patterns

These are saved in the specified output directory.

## Database Structure

The tracker creates a new table called `signal_tracking` in the database with the following structure:

- `id`: Unique identifier
- `Stock`: Stock symbol
- `Initial_Signal`: Original signal type (Buy/Sell/Neutral)
- `Initial_Signal_Date`: Date of the initial signal
- `Initial_Close`: Price at initial signal
- `Current_Signal`: Current signal type
- `Current_Signal_Date`: Date of the current signal
- `Current_Close`: Current price
- `Days_In_Signal`: Days since initial signal
- `Profit_Loss_Pct`: Calculated profit/loss percentage
- `Signal_Changes`: Number of signal changes
- `Last_Updated`: Last update timestamp
- `Notes`: Additional information about signal changes 