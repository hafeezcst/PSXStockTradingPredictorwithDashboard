"""
Main dashboard application for PSX stock analysis.
"""

import streamlit as st
from components.financial_reports import display_financial_reports

# Set page config
st.set_page_config(
    page_title="PSX Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Display the financial reports component
display_financial_reports({}) 