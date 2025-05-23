"""
Main dashboard application for PSX Stock Trading Predictor
"""

# Import necessary modules for path setup first
import os
import sys
from pathlib import Path

# Add project root to Python path (MUST be done before any other imports)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import streamlit next
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state for API keys
if 'huggingface_api_key' not in st.session_state:
    st.session_state.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', '')
if 'anthropic_api_key' not in st.session_state:
    st.session_state.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY', '')
if 'deepseek_api_key' not in st.session_state:
    st.session_state.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', '')
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = os.getenv('GOOGLE_API_KEY', '')

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="PSX Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/PSXStockTradingPredictorwithDashboard/issues',
        'Report a bug': 'https://github.com/yourusername/PSXStockTradingPredictorwithDashboard/issues/new',
        'About': "PSX Stock Trading Predictor Dashboard v2.0.0"
    }
)

# Now import all other modules
from datetime import datetime, time, timedelta
import pytz

# Import components after setting page config
# Use absolute imports to be explicit and avoid any potential issues
from src.data_processing.dashboard.config.settings import initialize_config
from src.data_processing.dashboard.components.database_manager import DatabaseManager
from src.data_processing.dashboard.components.portfolio import display_portfolio_analysis
from src.data_processing.dashboard.components.market import display_market
from src.data_processing.dashboard.components.charts import display_charts
from src.data_processing.dashboard.components.mutual_funds import display_mutual_funds
from src.data_processing.dashboard.components.financial_reports import display_financial_reports
from src.data_processing.dashboard.components.trading_signals import display_trading_signals
from src.data_processing.dashboard.components.indicator_analysis import display_indicator_analysis
from src.data_processing.dashboard.components.dividend_analysis import display_dividend_analysis
from src.data_processing.dashboard.components.signal_tracker import display_signal_tracker
from src.data_processing.dashboard.utils.styles import apply_custom_css
from src.data_processing.dashboard.components.analysis_tool import create_analysis_tool_dashboard
from src.data_processing.dashboard.components.shared_styles import apply_shared_styles
from src.data_processing.dashboard.components.signal_analysis import display_signal_analysis

def get_remaining_time(current_time, end_time):
    """
    Calculate remaining time until a specific time.
    
    Args:
        current_time (datetime): Current time (timezone-aware)
        end_time (time): Target end time
        
    Returns:
        str: Formatted remaining time string
    """
    # Get the timezone from current_time
    tz = current_time.tzinfo
    
    # Convert end_time to datetime for today with the same timezone
    end_datetime = datetime.combine(current_time.date(), end_time)
    end_datetime = tz.localize(end_datetime)
    
    # If end time is earlier than current time, it's for tomorrow
    if end_datetime < current_time:
        end_datetime = datetime.combine(current_time.date() + timedelta(days=1), end_time)
        end_datetime = tz.localize(end_datetime)
    
    # Calculate time difference
    time_diff = end_datetime - current_time
    
    # Convert to hours, minutes, seconds
    hours = time_diff.seconds // 3600
    minutes = (time_diff.seconds % 3600) // 60
    seconds = time_diff.seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_market_status():
    """
    Check if PSX market is open based on Pakistan trading hours.
    
    Regular Trading Days (Monday-Thursday):
    - Pre-Open: 09:15 AM to 09:30 AM
    - Break (Order Matching): 09:30 AM to 09:32 AM
    - Open: 09:32 AM to 03:30 PM
    - Break/Close: 03:30 PM
    - Post-Close: 03:35 PM to 03:50 PM
    - NDM: 09:30 AM to 04:30 PM
    - NDM T+0: 09:30 AM to 01:00 PM
    
    Friday Trading (Two Sessions):
    1st Session:
    - Pre-Open: 09:00 AM to 09:15 AM
    - Break (Order Matching): 09:15 AM to 09:17 AM
    - Open: 09:17 AM to 12:00 PM
    - Break/Close: 12:00 PM
    - NDM: 09:15 AM to 12:00 PM
    - NDM T+0: 09:15 AM to 12:00 PM
    - Break: 12:00 PM to 02:30 PM
    
    2nd Session:
    - Pre-Open: 02:15 PM to 02:30 PM
    - Break (Order Matching): 02:30 PM to 02:32 PM
    - Open: 02:32 PM to 04:30 PM
    - Break/Close: 04:30 PM
    - Post-Close: 04:35 PM to 04:50 PM
    - NDM: 02:30 PM to 05:00 PM
    
    Returns:
        tuple: (is_open, status_text, status_emoji, next_session, remaining_time)
    """
    # Get current time in Pakistan
    pakistan_tz = pytz.timezone('Asia/Karachi')
    current_time = datetime.now(pakistan_tz)
    
    # Check if it's a weekday (Monday = 0, Sunday = 6)
    is_weekday = current_time.weekday() < 5
    is_friday = current_time.weekday() == 4  # Friday is 4
    
    # Get current time components
    current_time_only = current_time.time()
    
    # Handle weekend
    if not is_weekday:
        return False, "Weekend", "⚫", "Next Market: Monday 09:15 AM", None
    
    # Handle Friday trading sessions
    if is_friday:
        # Define Friday session times
        # 1st Session
        friday_pre_open_start = time(9, 0)    # 09:00 AM
        friday_pre_open_end = time(9, 15)     # 09:15 AM
        friday_break_start = time(9, 15)      # 09:15 AM
        friday_break_end = time(9, 17)        # 09:17 AM
        friday_market_open = time(9, 17)      # 09:17 AM
        friday_market_close = time(12, 0)     # 12:00 PM
        friday_break_period_start = time(12, 0)  # 12:00 PM
        friday_break_period_end = time(14, 30)   # 02:30 PM
        
        # 2nd Session
        friday_session2_pre_open_start = time(14, 15)  # 02:15 PM
        friday_session2_pre_open_end = time(14, 30)    # 02:30 PM
        friday_session2_break_start = time(14, 30)     # 02:30 PM
        friday_session2_break_end = time(14, 32)       # 02:32 PM
        friday_session2_market_open = time(14, 32)     # 02:32 PM
        friday_session2_market_close = time(16, 30)    # 04:30 PM
        friday_session2_post_close_start = time(16, 35)  # 04:35 PM
        friday_session2_post_close_end = time(16, 50)    # 04:50 PM
        
        # Check Friday 1st Session
        if friday_pre_open_start <= current_time_only < friday_pre_open_end:
            remaining = get_remaining_time(current_time, friday_pre_open_end)
            return False, "Friday Pre-Open (1st)", "🟡", f"Market Opens: {friday_pre_open_end.strftime('%H:%M')} AM", remaining
        
        if friday_break_start <= current_time_only < friday_break_end:
            remaining = get_remaining_time(current_time, friday_break_end)
            return False, "Friday Break (1st)", "🟠", f"Market Opens: {friday_break_end.strftime('%H:%M')} AM", remaining
        
        if friday_market_open <= current_time_only < friday_market_close:
            remaining = get_remaining_time(current_time, friday_market_close)
            return True, "Friday Open (1st)", "🟢", f"Market Closes: {friday_market_close.strftime('%H:%M')} PM", remaining
        
        if current_time_only == friday_market_close:
            return False, "Friday Closing (1st)", "🔴", "Break Period: 12:00 PM - 02:30 PM", None
        
        # Check Friday Break Period
        if friday_break_period_start < current_time_only < friday_break_period_end:
            remaining = get_remaining_time(current_time, friday_break_period_end)
            return False, "Friday Break Period", "⚫", f"2nd Session: {friday_session2_pre_open_start.strftime('%H:%M')} PM", remaining
        
        # Check Friday 2nd Session
        if friday_session2_pre_open_start <= current_time_only < friday_session2_pre_open_end:
            remaining = get_remaining_time(current_time, friday_session2_pre_open_end)
            return False, "Friday Pre-Open (2nd)", "🟡", f"Market Opens: {friday_session2_pre_open_end.strftime('%H:%M')} PM", remaining
        
        if friday_session2_break_start <= current_time_only < friday_session2_break_end:
            remaining = get_remaining_time(current_time, friday_session2_break_end)
            return False, "Friday Break (2nd)", "🟠", f"Market Opens: {friday_session2_break_end.strftime('%H:%M')} PM", remaining
        
        if friday_session2_market_open <= current_time_only < friday_session2_market_close:
            remaining = get_remaining_time(current_time, friday_session2_market_close)
            return True, "Friday Open (2nd)", "🟢", f"Market Closes: {friday_session2_market_close.strftime('%H:%M')} PM", remaining
        
        if current_time_only == friday_session2_market_close:
            return False, "Friday Closing (2nd)", "🔴", "Post-Close Session: 04:35 PM", None
        
        if friday_session2_post_close_start <= current_time_only < friday_session2_post_close_end:
            remaining = get_remaining_time(current_time, friday_session2_post_close_end)
            return False, "Friday Post-Close", "🔴", f"Session Ends: {friday_session2_post_close_end.strftime('%H:%M')} PM", remaining
        
        # Before first session
        if current_time_only < friday_pre_open_start:
            remaining = get_remaining_time(current_time, friday_pre_open_start)
            return False, "Friday Closed", "⚫", f"Pre-Open: {friday_pre_open_start.strftime('%H:%M')} AM", remaining
        
        # After second session
        if current_time_only > friday_session2_post_close_end:
            return False, "Friday Closed", "⚫", "Next Market: Monday 09:15 AM", None
    
    # Regular trading days (Monday-Thursday)
    else:
        # Define regular market session times
        pre_open_start = time(9, 15)  # 09:15 AM
        pre_open_end = time(9, 30)    # 09:30 AM
        break_start = time(9, 30)     # 09:30 AM
        break_end = time(9, 32)       # 09:32 AM
        market_open = time(9, 32)     # 09:32 AM
        market_close = time(15, 30)   # 03:30 PM
        post_close_start = time(15, 35)  # 03:35 PM
        post_close_end = time(15, 50)    # 03:50 PM
        
        # Check regular market status
        if pre_open_start <= current_time_only < pre_open_end:
            remaining = get_remaining_time(current_time, pre_open_end)
            return False, "Pre-Open", "🟡", f"Market Opens: {pre_open_end.strftime('%H:%M')} AM", remaining
        
        if break_start <= current_time_only < break_end:
            remaining = get_remaining_time(current_time, break_end)
            return False, "Break (Matching)", "🟠", f"Market Opens: {break_end.strftime('%H:%M')} AM", remaining
        
        if market_open <= current_time_only < market_close:
            remaining = get_remaining_time(current_time, market_close)
            return True, "Open", "🟢", f"Market Closes: {market_close.strftime('%H:%M')} PM", remaining
        
        if current_time_only == market_close:
            return False, "Closing", "🔴", "Post-Close Session: 03:35 PM", None
        
        if post_close_start <= current_time_only < post_close_end:
            remaining = get_remaining_time(current_time, post_close_end)
            return False, "Post-Close", "🔴", f"Session Ends: {post_close_end.strftime('%H:%M')} PM", remaining
        
        # Before market opens
        if current_time_only < pre_open_start:
            remaining = get_remaining_time(current_time, pre_open_start)
            return False, "Closed", "⚫", f"Pre-Open: {pre_open_start.strftime('%H:%M')} AM", remaining
        
        # After market closes
        if current_time_only > post_close_end:
            return False, "Closed", "⚫", "Next Market: Tomorrow 09:15 AM", None
    
    # Default case (should not reach here)
    return False, "Unknown", "⚫", "Check market hours", None

def display_header():
    """Display the dashboard header with additional information."""
    # Create a container for the entire header
    header_container = st.container()
    
    with header_container:
        # Create a gradient background effect with custom CSS
        st.markdown("""
        <style>
        .header-container {
            background: linear-gradient(90deg, #1a237e, #0d47a1, #01579b);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .header-subtitle {
            color: #e3f2fd;
            font-size: 1.2rem;
            margin-bottom: 15px;
        }
        .clock-container {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 10px;
            text-align: right;
        }
        .clock-date {
            color: #bbdefb;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        .clock-time {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }
        .header-divider {
            height: 3px;
            background: linear-gradient(90deg, #ff9800, #ff5722);
            margin: 15px 0;
            border-radius: 3px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a three-column layout for the header
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown("""
            <div class="header-container">
                <div class="header-title">📈 PSX Stock Analyzer</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="header-container">
                <div class="header-subtitle">Pakistan Stock Exchange Analysis Platform</div>
                <div class="header-divider"></div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 5px 10px; border-radius: 5px; color: #bbdefb;">
                        <span style="font-weight: bold;">Data Source:</span> PSX Database
                    </div>
                    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 5px 10px; border-radius: 5px; color: #bbdefb;">
                        <span style="font-weight: bold;">Version:</span> 2.0.0
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Get current time in Pakistan
            pakistan_tz = pytz.timezone('Asia/Karachi')
            current_time = datetime.now(pakistan_tz)
            
            # Create a container for the live clock
            clock_container = st.empty()
            
            # Update the clock every second
            def update_clock():
                current_time = datetime.now(pakistan_tz)
                clock_container.markdown(f"""
                <div class="header-container">
                    <div class="clock-container">
                        <div class="clock-date">Last Updated:</div>
                        <div style="color: #bbdefb; font-size: 1.1rem; font-weight: bold;">{current_time.strftime("%Y-%m-%d")}</div>
                        <div class="clock-time">{current_time.strftime("%H:%M:%S")}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Initial clock update
            update_clock()
            
            # Set up auto-refresh for the clock
            if 'clock_placeholder' not in st.session_state:
                st.session_state.clock_placeholder = st.empty()
            
            # Use JavaScript to update the clock every second
            st.markdown("""
            <script>
                function updateClock() {
                    const now = new Date();
                    const timeString = now.toLocaleTimeString('en-US', { 
                        hour12: false,
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit'
                    });
                    const dateString = now.toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit'
                    });
                    document.getElementById('clock').innerHTML = `
                        <div class="header-container">
                            <div class="clock-container">
                                <div class="clock-date">Last Updated:</div>
                                <div style="color: #bbdefb; font-size: 1.1rem; font-weight: bold;">${dateString}</div>
                                <div class="clock-time">${timeString}</div>
                            </div>
                        </div>
                    `;
                }
                setInterval(updateClock, 1000);
                updateClock();
            </script>
            <div id="clock"></div>
            """, unsafe_allow_html=True)
    
    # Add a subtle divider after the header
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

def display_sidebar():
    """Display enhanced sidebar with categorized navigation."""
    st.sidebar.title("Navigation")
    
    # Initialize session state for navigation if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Define all navigation options with their categories
    navigation_options = {
        "📊 Market Overview": [
            "Dashboard",
            "Market Analysis",
            "Charts"
        ],
        "🔍 Analysis Tools": [
            "Technical Indicators",
            "Trading Signals",
            "Signal Analysis",
            "Signal Tracker",
            "Financial Reports",
            "Analysis Tool"
        ],
        "💰 Investment Tools": [
            "Portfolio",
            "Mutual Funds",
            "Dividend Analysis"
        ]
    }
    
    # Create a flat list of all options for the radio button
    all_options = []
    for category, options in navigation_options.items():
        all_options.extend(options)
    
    # Display category headers and radio buttons
    for category, options in navigation_options.items():
        st.sidebar.markdown(f"### {category}")
        # Add a small space between categories
        st.sidebar.markdown("")
    
    # Single radio button for all options
    selected_page = st.sidebar.radio(
        "Select Page",
        all_options,
        key="page_selection",
        index=all_options.index(st.session_state.current_page) if st.session_state.current_page in all_options else 0,
        label_visibility="collapsed"
    )
    
    # Update current page
    st.session_state.current_page = selected_page
    
    # Quick Stats Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    
    # Get market status
    is_open, status_text, status_emoji, next_session, remaining_time = get_market_status()
    
    # Display market status with appropriate color
    st.sidebar.markdown(f"- Market Status: **{status_text}** {status_emoji}")
    st.sidebar.markdown(f"- Next Session: **{next_session}**")
    if remaining_time:
        st.sidebar.markdown(f"- Time Remaining: **{remaining_time}**")
    
    # Get current time in Pakistan
    pakistan_tz = pytz.timezone('Asia/Karachi')
    current_time_pk = datetime.now(pakistan_tz)
    
    st.sidebar.markdown(f"- Pakistan Time: **{current_time_pk.strftime('%H:%M')}**")
    
    # Display appropriate trading hours based on day
    if current_time_pk.weekday() == 4:  # Friday
        st.sidebar.markdown("- Friday 1st Session: **09:17 AM - 12:00 PM**")
        st.sidebar.markdown("- Friday 2nd Session: **02:32 PM - 04:30 PM**")
    else:
        st.sidebar.markdown("- Regular Market: **09:32 AM - 03:30 PM**")
    
    st.sidebar.markdown("- Data Source: **PSX Database**")
    
    # Help Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Help & Support")
    st.sidebar.markdown("""
    - [Documentation](https://github.com/yourusername/PSXStockTradingPredictorwithDashboard/wiki)
    - [Report Issues](https://github.com/yourusername/PSXStockTradingPredictorwithDashboard/issues)
    - [Contact Support](mailto:support@example.com)
    """)
    
    return selected_page

def main():
    """
    Main function to run the dashboard.
    """
    # Initialize configuration
    config = initialize_config()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Display header
    display_header()
    
    # Display sidebar and get selected page
    selected_page = display_sidebar()
    
    # Display main content based on selected page
    if selected_page == "Dashboard":
        st.markdown("## Portfolio Analysis")
        display_portfolio_analysis(config)
        
        st.markdown("## Market Overview")
        display_market(config)
        
        st.markdown("## Technical Analysis")
        display_charts(config)
        
        st.markdown("## Signal Analysis")
        display_signal_analysis(config)
        
        st.markdown("## Mutual Funds")
        display_mutual_funds(config)
        
        st.markdown("## Financial Reports")
        display_financial_reports(config)
        
        st.markdown("## Trading Signals")
        display_trading_signals(config)
        
        st.markdown("## Signal Tracker")
        display_signal_tracker(config)
        
        st.markdown("## Indicator Analysis")
        display_indicator_analysis(config)
        
        st.markdown("## Dividend Analysis")
        display_dividend_analysis(config)
    
    elif selected_page == "Market Analysis":
        st.markdown("## Market Overview")
        display_market(config)
    
    elif selected_page == "Charts":
        st.markdown("## Technical Analysis")
        display_charts(config)
    
    elif selected_page == "Technical Indicators":
        st.markdown("## Indicator Analysis")
        display_indicator_analysis(config)
    
    elif selected_page == "Trading Signals":
        st.markdown("## Trading Signals")
        display_trading_signals(config)
    
    elif selected_page == "Signal Analysis":
        st.markdown("## Signal Analysis")
        display_signal_analysis(config)
    
    elif selected_page == "Signal Tracker":
        st.markdown("## Signal Tracker")
        display_signal_tracker(config)
    
    elif selected_page == "Financial Reports":
        st.markdown("## Financial Reports")
        display_financial_reports(config)
    
    elif selected_page == "Portfolio":
        st.markdown("## Portfolio Analysis")
        display_portfolio_analysis(config)
    
    elif selected_page == "Mutual Funds":
        st.markdown("## Mutual Funds")
        display_mutual_funds(config)
    
    elif selected_page == "Dividend Analysis":
        st.markdown("## Dividend Analysis")
        display_dividend_analysis(config)
    
    elif selected_page == "Analysis Tool":
        create_analysis_tool_dashboard()

def initialize_config():
    """Initialize dashboard configuration"""
    config = {
        "theme": "light",
        "layout": "wide",
        "menu_items": {
            "Get Help": "https://github.com/yourusername/PSXStockTradingPredictorwithDashboard/issues",
            "Report a bug": "https://github.com/yourusername/PSXStockTradingPredictorwithDashboard/issues/new",
            "About": "PSX Stock Trading Predictor Dashboard v2.0.0"
        },
        "databases": {
            "main_db_path": str(PROJECT_ROOT / "psx_consolidated_data_indicators_PSX.db"),
            "signals_db_path": str(PROJECT_ROOT / "data" / "databases" / "production" / "fairvalue.db"),
        },
        "database_path": str(PROJECT_ROOT / "data" / "databases" / "production" / "fairvalue.db"),
    }
    return config

if __name__ == "__main__":
    main() 