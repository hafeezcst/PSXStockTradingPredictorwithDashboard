import streamlit as st
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the component using relative import
from components.signal_tracker import display_signal_tracker

# Set page config
st.set_page_config(
    page_title="PSX Signal Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a simple config for testing
test_config = {
    'database_path': os.path.join(project_root, 'data/databases/production/PSX_investing_Stocks_KMI30_tracking.db')
}

# Display the component
display_signal_tracker(test_config) 