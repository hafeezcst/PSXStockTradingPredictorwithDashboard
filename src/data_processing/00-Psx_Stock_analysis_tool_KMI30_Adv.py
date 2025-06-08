import subprocess
import os
import sys
import logging
import schedule
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pandas_ta patch
from src.data_processing.pandas_ta_patch import pandas_ta

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('psx_analysis.log'),
        logging.StreamHandler()
    ]
)

def get_available_symbols(cursor):
    """Get list of available stock symbols from database"""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'PSX_%_stock_data'")
    tables = cursor.fetchall()
    return [table[0].replace('PSX_', '').replace('_stock_data', '') for table in tables]

def fetch_column_names(engine, table_name):
    """Fetch column names from a specific table"""
    try:
        query = f"SELECT * FROM {table_name} LIMIT 1"
        df = pd.read_sql(query, engine)
        return df.columns.tolist()
    except Exception as e:
        logging.error(f"Error fetching columns for {table_name}: {e}")
        return []

def get_buy_sell_signals(symbol):
    """Get buy and sell signals for a specific symbol"""
    try:
        # Implementation depends on your signal tracking system
        # This is a placeholder - implement based on your actual signal tracking
        return [], []
    except Exception as e:
        logging.error(f"Error getting signals for {symbol}: {e}")
        return [], []

def calculate_market_phase(df, symbol):
    """Calculate market phase and probability for a symbol"""
    try:
        # Implementation depends on your market phase calculation logic
        # This is a placeholder - implement based on your actual market phase calculation
        return "NEUTRAL", 50.0, None
    except Exception as e:
        logging.error(f"Error calculating market phase for {symbol}: {e}")
        return "NEUTRAL", 50.0, None

def get_latest_buy_stocks():
    """Get latest buy signals from tracking database"""
    try:
        # Implementation depends on your signal tracking system
        # This is a placeholder - implement based on your actual signal tracking
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error getting latest buy stocks: {e}")
        return pd.DataFrame()

def create_category_tables(df, output_dir, current_date):
    """Create tabular dashboards for different stock categories"""
    try:
        # Create tables for different categories
        categories = {
            'BUY_HOLD': df[df['Status'] == 'BUY/HOLD'],
            'OPPORTUNITY': df[df['Status'] == 'OPPORTUNITY'],
            'SELL': df[df['Status'] == 'SELL']
        }
        
        for category, category_df in categories.items():
            if not category_df.empty:
                # Sort by phase probability
                category_df = category_df.sort_values('Phase_Probability', ascending=False)
                
                # Save to CSV
                output_file = os.path.join(output_dir, f'{category.lower()}_stocks_{current_date}.csv')
                category_df.to_csv(output_file, index=False)
                
                # Create HTML table
                html_file = os.path.join(output_dir, f'{category.lower()}_stocks_{current_date}.html')
                with open(html_file, 'w') as f:
                    f.write(f"<h2>{category} Stocks - {current_date}</h2>")
                    f.write(category_df.to_html(index=False))
    except Exception as e:
        logging.error(f"Error creating category tables: {e}")

def send_telegram_message_with_image(image_path, message):
    """Send message with image to Telegram"""
    try:
        from src.data_processing.telegram_message import send_telegram_message_with_image
        send_telegram_message_with_image(image_path, message)
    except Exception as e:
        logging.error(f"Error sending telegram message with image: {e}")

def generate_stock_dashboard():
    """Generate a dashboard showing buy, sell and neutral stocks with key metrics"""
    try:
        database_path = 'data/databases/production/psx_consolidated_data_indicators_PSX.db'
        
        # Create a connection to the database
        engine = create_engine(f'sqlite:///{database_path}')
        connection = engine.connect()
        cursor = connection.connection.cursor()
        
        # Get all available symbols
        available_symbols = get_available_symbols(cursor)
        
        # Get all buy signals
        buy_df = get_latest_buy_stocks()
        buy_symbols = set(buy_df['Stock'].tolist()) if not buy_df.empty else set()
        
        # Prepare containers for results
        all_results = []
        
        # Process each symbol individually
        print("Analyzing all available stocks for dashboard...")
        for symbol in tqdm(available_symbols, desc="Processing stocks"):
            try:
                # Get stock data for this specific symbol only
                table_name = f"PSX_{symbol}_stock_data"
                
                # First, check which columns exist in this table
                available_columns = fetch_column_names(engine, table_name)
                if not available_columns:
                    logging.warning(f"No columns found for {table_name}")
                    continue
                
                # Define required and optional columns
                required_cols = ["Date", "Close"]
                optional_cols = {
                    "RSI_weekly_Avg": None,
                    "AO_weekly_AVG": None,
                    "MA_30": None,
                    "RSI_weekly": None,
                    "Volume": None
                }
                
                # Check if required columns exist
                if not all(col in available_columns for col in required_cols):
                    logging.warning(f"Missing required columns in {table_name}")
                    continue
                
                # Build SELECT clause with only available columns
                select_cols = required_cols.copy()
                for col in optional_cols:
                    if col in available_columns:
                        select_cols.append(col)
                
                # Build and execute query
                query = f"""SELECT {', '.join(select_cols)} 
                           FROM {table_name}
                           ORDER BY Date DESC
                           LIMIT 60"""
                
                df = pd.read_sql(query, connection)
                
                if df.empty:
                    continue
                
                # Add missing columns as NaN
                for col, default_val in optional_cols.items():
                    if col not in df.columns:
                        df[col] = default_val
                
                # Work with a proper copy to avoid SettingWithCopyWarning
                df = df.copy()
                
                # Convert the date column to datetime
                df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
                
                # Calculate percentage change
                df.loc[:, 'pct_change'] = df['Close'].pct_change() * 100
                
                # Get latest data
                latest = df.iloc[0] if not df.empty else None
                if latest is None:
                    continue
                
                # Get buy and sell signals
                buy_signals, sell_signals = get_buy_sell_signals(symbol)
                
                # Determine stock status
                if buy_signals and sell_signals:
                    latest_buy = max(buy_signals, key=lambda x: x[0])
                    latest_sell = max(sell_signals, key=lambda x: x[0])
                    
                    if latest_buy[0] > latest_sell[0]:
                        status = "BUY/HOLD"
                    else:
                        status = "SELL"
                elif buy_signals:
                    status = "BUY/HOLD"
                elif sell_signals:
                    status = "SELL"
                else:
                    status = "OPPORTUNITY"
                
                # Calculate market phase
                market_phase, phase_probability, _ = calculate_market_phase(df, symbol)
                
                # Get holding days and profit/loss for buy stocks
                holding_days = None
                profit_loss = None
                
                if status == "BUY/HOLD" and symbol in buy_symbols:
                    stock_info = buy_df[buy_df['Stock'] == symbol].iloc[0]
                    holding_days = int(stock_info['holding_days']) if 'holding_days' in stock_info else None
                    
                    # Calculate profit/loss if we have signal price
                    if 'Signal_Close' in stock_info:
                        signal_price = float(stock_info['Signal_Close'])
                        current_price = latest['Close']
                        profit_loss = ((current_price - signal_price) / signal_price) * 100
                
                # Collect metrics
                result = {
                    'Symbol': symbol,
                    'Status': status,
                    'Close': latest['Close'],
                    'RSI': latest['RSI_weekly'],
                    'AO': latest['AO_weekly_AVG'],
                    'Market_Phase': market_phase,
                    'Phase_Probability': phase_probability,
                    'Holding_Days': holding_days,
                    'Profit_Loss': profit_loss,
                    'Above_MA30': latest['Close'] > latest['MA_30'] if 'MA_30' in latest and pd.notna(latest['MA_30']) else False
                }
                
                all_results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing {symbol} for dashboard: {e}")
                continue
        
        # Convert results to DataFrame
        dashboard_df = pd.DataFrame(all_results)
        
        # Create the dashboard
        create_dashboard_visualization(dashboard_df)
        
        # Close connection
        connection.close()
        
        return dashboard_df
        
    except Exception as e:
        logging.error(f"Error generating dashboard: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def create_dashboard_visualization(df):
    """Create visual dashboard using Matplotlib"""
    if df.empty:
        logging.error("No data available for dashboard visualization")
        return False
    
    # Create a figure with multiple subplots - 4x3 grid
    fig = plt.figure(figsize=(22, 18))
    plt.subplots_adjust(hspace=0.8, wspace=0.4)
    
    # Add a title
    fig.suptitle('PSX Market Dashboard', fontsize=24, y=0.98)
    
    # 1. Status Distribution Pie Chart
    plt.subplot(4, 3, 1)
    status_counts = df['Status'].value_counts()
    colors = {'BUY/HOLD': 'green', 'SELL': 'red', 'OPPORTUNITY': 'blue'}
    status_colors = [colors.get(s, 'gray') for s in status_counts.index]
    plt.pie(status_counts, labels=status_counts.index, autopct='%.2f%%', colors=status_colors)
    plt.title('Stock Signal Distribution')
    
    # 2. Market Phase Distribution Pie Chart
    plt.subplot(4, 3, 2)
    phase_counts = df['Market_Phase'].value_counts()
    phase_colors = {'ACCUMULATION': 'green', 'DISTRIBUTION': 'red', 'NEUTRAL': 'gray'}
    plt.pie(phase_counts, labels=phase_counts.index, autopct='%.2f%%',
            colors=[phase_colors.get(p, 'blue') for p in phase_counts.index])
    plt.title('Market Phase Distribution')
    
    # 3. Market Breadth Indicator
    plt.subplot(4, 3, 3)
    # Calculate key market breadth metrics
    above_ma = df['Above_MA30'].sum() / len(df) * 100
    acc_stocks = len(df[df['Market_Phase'] == 'ACCUMULATION']) / len(df) * 100
    high_rsi = len(df[df['RSI'] > 50]) / len(df) * 100
    pos_ao = len(df[df['AO'] > 0]) / len(df) * 100

    metrics = ['Above MA30', 'Accumulation', 'RSI > 50', 'AO > 0']
    values = [above_ma, acc_stocks, high_rsi, pos_ao]
    colors = ['navy', 'green', 'purple', 'orange']

    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Calculate overall market breadth score
    market_score = sum(values) / len(values)
    plt.axhline(y=market_score, color='black', linestyle='-', linewidth=2, alpha=0.5)
    plt.text(len(metrics) - 0.5, market_score + 2, f'Avg: {market_score:.1f}%', 
             ha='center', va='bottom', fontweight='bold')

    # Determine market condition
    market_condition = "Neutral Market"
    if market_score > 60:
        market_condition = "Strong Bullish Market"
    elif market_score > 50:
        market_condition = "Moderately Bullish Market" 
    elif market_score < 40:
        market_condition = "Strong Bearish Market"
    elif market_score < 50:
        market_condition = "Moderately Bearish Market"

    plt.title(f'Market Breadth Indicator\n{market_condition}')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel('Percentage of Stocks (%)')
    
    # 4. Exit Timing Indicator
    plt.subplot(4, 3, 4)
    buy_df = df[df['Status'] == 'BUY/HOLD'].copy()

    if not buy_df.empty and 'Holding_Days' in buy_df.columns and len(buy_df.dropna(subset=['Holding_Days', 'Profit_Loss'])) > 0:
        # Create holding period buckets
        buy_df['Holding_Bucket'] = pd.cut(
            buy_df['Holding_Days'].fillna(0), 
            bins=[0, 5, 20, 60, 120, float('inf')],
            labels=['0-5d', '6-20d', '21-60d', '61-120d', '>120d']
        )
        
        # Calculate average profit/loss by holding period
        profitability = buy_df.groupby('Holding_Bucket', observed=False)['Profit_Loss'].agg(
            ['mean', 'count']).reset_index()
        
        if not profitability.empty:
            # Create plot
            bars = plt.bar(profitability['Holding_Bucket'], 
                          profitability['mean'], 
                          alpha=0.7,
                          color=['green' if x >= 0 else 'red' for x in profitability['mean']])
            
            # Add count labels
            for i, bar in enumerate(bars):
                count = profitability.iloc[i]['count']
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (1 if bar.get_height() >= 0 else -3),
                        f"n={count}", 
                        ha='center', va='bottom', fontsize=8)
            
            plt.title('Profit/Loss by Holding Period')
            plt.xlabel('Holding Period')
            plt.ylabel('Average Profit/Loss %')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add optimal exit guidance if we have valid data
            if not profitability['mean'].isna().all():
                best_period_idx = profitability['mean'].idxmax()
                best_period = profitability.iloc[best_period_idx]
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.text(0.5, 0.9, 
                        f"Best exit window: {best_period['Holding_Bucket']}", 
                        transform=plt.gca().transAxes, ha='center',
                        bbox=dict(facecolor='yellow', alpha=0.5))
    else:
        plt.text(0.5, 0.5, 'No buy/hold stocks data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 5. Market Momentum Heat Map
    plt.subplot(4, 3, 5)
    # Create a filtered DataFrame with valid data for both axes
    valid_data = df.dropna(subset=['RSI', 'AO'])
    
    if len(valid_data) >= 5:  # Make sure we have enough data points
        # Create a 2D histogram (heatmap)
        h = plt.hist2d(valid_data['RSI'], valid_data['AO'], 
                      bins=[10, 10], cmap='RdYlGn', alpha=0.7,
                      range=[[0, 100], [-5, 5]])
        
        # Add quadrant lines
        plt.axvline(x=50, color='white', linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='white', linestyle='--', alpha=0.7)
        
        # Add quadrant labels
        plt.text(25, 2.5, "Strong Buy\nZone", fontsize=9, ha='center', color='white', weight='bold')
        plt.text(75, 2.5, "Overbought", fontsize=9, ha='center', color='white', weight='bold')
        plt.text(25, -2.5, "Weak", fontsize=9, ha='center', color='white', weight='bold')
        plt.text(75, -2.5, "Distribution\nZone", fontsize=9, ha='center', color='white', weight='bold')
        
        # Calculate and show market center of gravity
        avg_rsi = valid_data['RSI'].mean()
        avg_ao = valid_data['AO'].mean()
        plt.scatter(avg_rsi, avg_ao, color='white', edgecolor='black', s=100, marker='*')
        
        # Determine market status based on center of gravity
        market_status = ""
        if avg_rsi < 50 and avg_ao > 0:
            market_status = "Accumulation Phase"
        elif avg_rsi > 50 and avg_ao > 0:
            market_status = "Bullish Phase"
        elif avg_rsi > 50 and avg_ao < 0:
            market_status = "Distribution Phase"
        else:
            market_status = "Bearish Phase"
        
        plt.colorbar(label='Stock Concentration')
        plt.title(f'Market Momentum Heat Map\n{market_status}')
        plt.xlabel('RSI Value')
        plt.ylabel('AO Value')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for heat map', 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # 6. Top 10 BUY/HOLD stocks with highest phase probability
    plt.subplot(4, 3, 6)
    if not buy_df.empty:
        top_buys = buy_df.sort_values('Phase_Probability', ascending=False).head(10)
        y_pos = range(len(top_buys))
        plt.barh(y_pos, top_buys['Phase_Probability'], color='green', alpha=0.7)
        plt.yticks(y_pos, top_buys['Symbol'])
        plt.title('Top BUY/HOLD Stocks by Accumulation Probability')
        plt.xlabel('Accumulation Probability %')
    
    # 7. Top 10 OPPORTUNITY stocks with highest accumulation probability
    plt.subplot(4, 3, 7)
    opportunity_df = df[(df['Status'] == 'OPPORTUNITY') & (df['Market_Phase'] == 'ACCUMULATION')]
    if not opportunity_df.empty:
        top_opps = opportunity_df.sort_values('Phase_Probability', ascending=False).head(10)
        y_pos = range(len(top_opps))
        plt.barh(y_pos, top_opps['Phase_Probability'], color='blue', alpha=0.7)
        plt.yticks(y_pos, top_opps['Symbol'])
        plt.title('Top OPPORTUNITY Stocks (Accumulation Phase)')
        plt.xlabel('Accumulation Probability %')
    
    # 8. Latest Buy Signals by Holding Days
    plt.subplot(4, 3, 8)
    if not buy_df.empty and 'Holding_Days' in buy_df.columns:
        # Sort by holding days (newest first) and take top 10
        recent_buys = buy_df.sort_values('Holding_Days').head(10)
        
        # Prepare data for the plot
        y_pos = range(len(recent_buys))
        symbols = recent_buys['Symbol']
        days = recent_buys['Holding_Days']
        
        # Define colors based on Market Phase
        colors = {'ACCUMULATION': 'green', 'DISTRIBUTION': 'red', 'NEUTRAL': 'gray'}
        bar_colors = [colors.get(phase, 'blue') for phase in recent_buys['Market_Phase']]
        
        # Create horizontal bar chart
        bars = plt.barh(y_pos, days, color=bar_colors, alpha=0.7)
        plt.yticks(y_pos, symbols)
        
        # Add holding days labels at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            phase_prob = recent_buys.iloc[i]['Phase_Probability']
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f"{int(width)} days ({phase_prob:.1f}%)", ha='left', va='center', fontsize=8)
        
        plt.title('Latest Buy Signals by Holding Days')
        plt.xlabel('Days Since Buy Signal')
        
        # Add a legend for market phases
        markers = [plt.Rectangle((0,0),1,1,color=color) for color in [colors['ACCUMULATION'], colors['DISTRIBUTION'], colors['NEUTRAL']]]
        plt.legend(markers, ['Accumulation', 'Distribution', 'Neutral'], loc='upper right')
    else:
        plt.text(0.5, 0.5, 'No buy/hold stocks data available', 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    # 9. Market Rotation Analysis
    ax9 = plt.subplot(4, 3, 9)
    ax9.set_title('Market Rotation Analysis', fontsize=12)

    # Compute average metrics for different stock price ranges
    def get_price_category(price):
        if price < 50:
            return "Small Cap (<50)"
        elif price < 200:
            return "Mid Cap (50-200)"
        else:
            return "Large Cap (>200)"

    # Add price category to dataframe
    df['Price_Category'] = df['Close'].apply(get_price_category)

    # Calculate metrics by price category
    price_cat_metrics = df.groupby('Price_Category').agg({
        'Profit_Loss': lambda x: x.dropna().mean(),
        'RSI': 'mean',
        'AO': 'mean',
        'Symbol': 'count'
    }).reset_index()

    price_cat_metrics = price_cat_metrics.rename(columns={'Symbol': 'Count'})

    if not price_cat_metrics.empty and len(price_cat_metrics) > 1:
        bars = ax9.bar(price_cat_metrics['Price_Category'], price_cat_metrics['RSI'],
                      color=['lightblue', 'royalblue', 'darkblue'])
        
        # Add count labels on top of each bar
        for bar, count in zip(bars, price_cat_metrics['Count']):
            height = bar.get_height()
            ax9.annotate(f'n={count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        ax9.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax9.set_ylim(0, 100)
        
        # Add a short summary text for rotation indicators
        if (price_cat_metrics['RSI'].iloc[0] > price_cat_metrics['RSI'].iloc[-1] and
            len(price_cat_metrics) > 1):
            ax9.set_xlabel("⬆️ Rotation to Smaller Caps", fontweight='bold')
        elif (price_cat_metrics['RSI'].iloc[0] < price_cat_metrics['RSI'].iloc[-1] and
                len(price_cat_metrics) > 1):
            ax9.set_xlabel("⬇️ Rotation to Larger Caps", fontweight='bold')
                
    else:
        plt.text(0.5, 0.5, 'Insufficient data for rotation analysis',
                 ha='center', va='center', transform=ax9.transAxes)
        ax9.axis('off')
    
    # 10. Risk-Adjusted Performance Matrix
    ax10 = plt.subplot(4, 3, 10)
    ax10.set_title('Risk-Adjusted Performance Matrix', fontsize=12)
    
    # Only use BUY/HOLD stocks with valid data
    perf_df = df[(df['Status'] == 'BUY/HOLD') & df['Profit_Loss'].notna() & df['Holding_Days'].notna()].copy()
    
    if not perf_df.empty and len(perf_df) >= 3:
        # Calculate daily return and volatility
        perf_df['Daily_Return'] = perf_df['Profit_Loss'] / perf_df['Holding_Days']
        perf_df['Risk_Category'] = pd.qcut(perf_df['Profit_Loss'].abs(), 3, labels=['Low', 'Medium', 'High'])
        perf_df['Return_Category'] = pd.qcut(perf_df['Daily_Return'], 3, labels=['Low', 'Medium', 'High'])
        
        # Create scatter plot
        risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        for risk, group in perf_df.groupby('Risk_Category'):
            ax10.scatter(group['Holding_Days'], group['Profit_Loss'], 
                        label=f'{risk} Risk', color=risk_colors[risk], 
                        alpha=0.7, s=100)
            
            # Label top performers in each category
            top = group.nlargest(3, 'Profit_Loss')
            for _, row in top.iterrows():
                ax10.annotate(row['Symbol'], 
                             (row['Holding_Days'], row['Profit_Loss']),
                             xytext=(5, 5), textcoords='offset points')
                             
        # Add optimal hold period range
        if len(perf_df) > 5:
            # Find optimal holding period range (highest avg daily returns)
            perf_df['Hold_Bucket'] = pd.cut(perf_df['Holding_Days'], 
                                          bins=[0, 10, 30, 60, 120, float('inf')],
                                          labels=['0-10d', '11-30d', '31-60d', '61-120d', '>120d'])
            best_bucket = perf_df.groupby('Hold_Bucket')['Daily_Return'].mean().idxmax()
            
            # Shade the optimal region
            bucket_ranges = {'0-10d': (0, 10), '11-30d': (11, 30), 
                           '31-60d': (31, 60), '61-120d': (61, 120), '>120d': (121, 200)}
            if best_bucket in bucket_ranges:
                min_x, max_x = bucket_ranges[best_bucket]
                ax10.axvspan(min_x, max_x, alpha=0.2, color='green')
                ax10.text((min_x + max_x)/2, ax10.get_ylim()[1]*0.9, 
                         f"Optimal Hold: {best_bucket}", ha='center',
                         bbox=dict(facecolor='white', alpha=0.8))
            
        ax10.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax10.set_xlabel('Holding Period (Days)')
        ax10.set_ylabel('Profit/Loss (%)')
        ax10.legend(title='Risk Level')
        ax10.grid(True, alpha=0.3)
    else:
        ax10.text(0.5, 0.5, 'Insufficient data for performance matrix',
                 ha='center', va='center', transform=ax10.transAxes)

    # 11. Decision Support - Action Recommendations
    ax11 = plt.subplot(4, 3, 11)
    ax11.set_title('Action Recommendations', fontsize=12)
    
    # Create decision support categories
    action_counts = {
        'Strong Buy': len(df[(df['Market_Phase'] == 'ACCUMULATION') & 
                             (df['Phase_Probability'] > 70) & 
                             (df['RSI'] < 50) & 
                             (df['AO'] > 0)]),
        'Buy': len(df[(df['Market_Phase'] == 'ACCUMULATION') & 
                      (df['Phase_Probability'] > 55)]),
        'Hold': len(df[(df['Status'] == 'BUY/HOLD') & 
                       (df['Market_Phase'] != 'DISTRIBUTION')]),
        'Take Profit': len(df[(df['Status'] == 'BUY/HOLD') & 
                             (df['Market_Phase'] == 'DISTRIBUTION') & 
                             (df['Profit_Loss'] > 0 if 'Profit_Loss' in df.columns else False)]),
        'Cut Loss': len(df[(df['Status'] == 'BUY/HOLD') & 
                          (df['Market_Phase'] == 'DISTRIBUTION') & 
                          (df['Profit_Loss'] < 0 if 'Profit_Loss' in df.columns else False)]),
        'Avoid': len(df[(df['Market_Phase'] == 'DISTRIBUTION') & 
                       (df['Phase_Probability'] > 70)])
    }
    
    # Create action guidance visualization
    actions = list(action_counts.keys())
    values = list(action_counts.values())
    colors = ['darkgreen', 'green', 'blue', 'orange', 'red', 'darkred']
    
    bars = ax11.bar(actions, values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax11.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                     str(int(height)), ha='center', va='bottom')
    
    ax11.set_ylabel('Number of Stocks')
    ax11.set_xticklabels(actions, rotation=45, ha='right')
    ax11.grid(True, axis='y', alpha=0.3)
    
    # Add a note for how to use this chart
    ax11.text(0.5, -0.3, 'Focus on Strong Buy for new entries, Take Profit for overbought positions',
             transform=ax11.transAxes, ha='center', fontsize=9)

    # 12. Sector Rotation Heat Map
    ax12 = plt.subplot(4, 3, 12)
    ax12.set_title('Market Sector Performance', fontsize=12)
    
    # Create sector categories (could be based on industry or market cap)
    if 'Price_Category' in df.columns:
        # We already have price categories from previous analysis
        # Get average RSI and AO values by category
        sector_metrics = df.groupby('Price_Category').agg({
            'RSI': 'mean',
            'AO': 'mean',
            'Phase_Probability': 'mean',
            'Symbol': 'count'
        }).reset_index()
        
        sector_metrics = sector_metrics.rename(columns={'Symbol': 'Count'})
        
        if not sector_metrics.empty:
            # Create array for heatmap
            sectors = sector_metrics['Price_Category'].tolist()
            metrics = ['RSI', 'AO', 'Phase_Probability']
            data = sector_metrics[metrics].values
            
            # Create heatmap
            im = ax12.imshow(data.T, cmap='RdYlGn', aspect='auto')
            
            # Add labels
            ax12.set_xticks(np.arange(len(sectors)))
            ax12.set_yticks(np.arange(len(metrics)))
            ax12.set_xticklabels(sectors)
            ax12.set_yticklabels(metrics)
            plt.setp(ax12.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations in each cell
            for i in range(len(sectors)):
                for j in range(len(metrics)):
                    value = data[i, j]
                    text_color = 'black' if 30 < value < 70 else 'white'
                    ax12.text(i, j, f"{value:.1f}", ha="center", va="center", 
                             color=text_color, fontweight="bold")
            
            # Add count below each column
            for i, count in enumerate(sector_metrics['Count']):
                ax12.text(i, len(metrics), f"n={count}", ha="center", va="center")
                
            # Add a title explaining what we're seeing
            lead_sector = sector_metrics.loc[sector_metrics['Phase_Probability'].idxmax(), 'Price_Category']
            ax12.set_title(f'Market Sector Performance (Leader: {lead_sector})', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax12, orientation='horizontal', pad=0.2)
            cbar.set_label('Score (Higher is Better)')
        else:
            ax12.text(0.5, 0.5, 'Insufficient data for sector analysis',
                     ha='center', va='center', transform=ax12.transAxes)
            ax12.axis('off')
    else:
        ax12.text(0.5, 0.5, 'Sector data not available',
                 ha='center', va='center', transform=ax12.transAxes)
        ax12.axis('off')
    
    # Save dashboard
    dashboards_folder = 'outputs/dashboards/PSX_DASHBOARDS'
    os.makedirs(dashboards_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y-%m-%d')
    dashboard_path = os.path.join(dashboards_folder, f'psx_dashboard_{current_date}.png')
    plt.savefig(dashboard_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Create tabular dashboards
    create_category_tables(df, dashboards_folder, current_date)
    
    # Send dashboard to Telegram
    message = f"PSX Market Dashboard - Generated on {current_date}"
    send_telegram_message_with_image(dashboard_path, message)
    
    print(f"Dashboard saved to {dashboard_path} and sent to Telegram")
    return True

def run_scripts(scripts: list):
    """Execute a list of scripts in sequence"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for script in scripts:
        # Split script and arguments, but keep the script name intact
        script_parts = script.split()
        script_name = script_parts[0]
        script_args = script_parts[1:] if len(script_parts) > 1 else []
        
        script_path = os.path.join(script_dir, script_name)
        logging.info(f"Executing {script}")
        
        try:
            # Build command with script path and arguments
            cmd = [sys.executable, script_path] + script_args
            result = subprocess.run(
                cmd,
                check=True
            )
            if result.returncode != 0:
                logging.error(f"{script} failed with return code {result.returncode}")
                break
        except Exception as e:
            logging.error(f"Error executing {script}: {str(e)}")
            break

def run_signal_tracker():
    """Run the stock signal tracker with advanced analysis and Telegram alerts"""
    try:
        logging.info("Running stock signal tracker with Telegram alerts")
        
        # Get the path to the run_stock_signal_tracker.py script
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tracker_script = os.path.join(script_dir, "scripts", "run_stock_signal_tracker.py")
        
        # Create reports directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        reports_dir = os.path.join(script_dir, "reports", f"signal_tracking_{timestamp}")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Set path to database
        db_path = os.path.join(script_dir, "data", "databases", "production", "PSX_investing_Stocks_KMI30_tracking.db")
        
        # Run the tracker script with enhanced parameters
        cmd = [
            sys.executable, 
            tracker_script,
            f"--db={db_path}",
            f"--output-dir={reports_dir}",
            "--create-backup",
            "--generate-report",
            "--send-alerts"
        ]
        
        logging.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log output
        if result.stdout:
            logging.info(f"Stock signal tracker output: {result.stdout[:500]}...")
        
        if result.stderr:
            logging.warning(f"Stock signal tracker errors: {result.stderr}")
        
        if result.returncode != 0:
            logging.error(f"Stock signal tracker failed with return code {result.returncode}")
        else:
            logging.info("Stock signal tracker completed successfully")
        
        # Add to summary notification
        if result.returncode == 0:
            try:
                from src.data_processing.telegram_message import send_telegram_message
                
                summary = f"🔄 PSX Analysis Batch Job - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                summary += f"✅ Stock signal tracker executed successfully\n"
                summary += f"📊 Reports saved to: {os.path.basename(reports_dir)}\n"
                send_telegram_message(summary)
            except Exception as e:
                logging.error(f"Error sending telegram summary: {str(e)}")
            
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Error running stock signal tracker: {str(e)}")
        return False

def job():
    """Main job function to run all scripts"""
    logging.info("Starting scheduled PSX analysis job")
    try:
        # Define script execution order
        scripts = [
            'manual_kmi_shariah_processor.py',
            'PSXAnnouncement.py --fresh',
            '01-PSX_Database_data_download_to_SQL_db_PSX.py',
            '02-sql_duplicate_remover_ALL.py',
            '01-PSX_SQL_Indicator_PSX.py',
            '06_PSX_Dividend_Schedule.py',
            'stable_versions\04-List_Weekly_RSI_GT_40_BUY_SELL_KMI30_100_Weekly_v1.0_stable.py',
            #'10-draw_indicator_trend_lines_with_signals_Stable_V_1.0.py',
            'stable_versions\draw_indicator_trend_lines_v1.0_KMI30_stable.py',

        ]
        
        start_time = time.time()
        
        # Run the analysis scripts
        run_scripts(scripts)
        
        # Generate the dashboard
        generate_stock_dashboard()
        
        # Run the stock signal tracker after all other scripts
        run_signal_tracker()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logging.info(f"PSX analysis job completed successfully in {execution_time:.2f} seconds")
        
        # Send completion notification
        try:
            from src.data_processing.telegram_message import send_telegram_message
            completion_msg = f"✅ PSX Stock Analysis Batch Job Completed\n"
            completion_msg += f"⏱️ Total execution time: {execution_time:.2f} seconds\n"
            completion_msg += f"🕒 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            send_telegram_message(completion_msg)
        except Exception as e:
            logging.error(f"Error sending completion notification: {str(e)}")
            
    except Exception as e:
        logging.error(f"Error in scheduled job: {str(e)}")

def main():
    """Main function to set up and run the scheduler"""
    logging.info("Starting PSX analysis scheduler")
    
    # Schedule the job to run every day at 17:30
    schedule.every().day.at("17:30").do(job)
    
    # Run the job immediately on startup
    job()
    
    # Keep the script running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logging.error(f"Scheduler error: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    main()

    