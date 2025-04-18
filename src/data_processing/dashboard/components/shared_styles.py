"""
Shared styling components for the PSX dashboard.
"""

import streamlit as st

def apply_shared_styles():
    """Apply shared styles across all dashboard components."""
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
            padding: 1rem;
        }
        
        /* Header styling */
        .stApp > header {
            background-color: #2c3e50;
            color: white;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Metric card styling */
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            gap: 1px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #f0f2f6;
            font-weight: bold;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            font-weight: 600;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        
        /* Chart container styling */
        .chart-container {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Alert styling */
        .alert-success {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .alert-error {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        /* Custom divider */
        .custom-divider {
            height: 2px;
            background: linear-gradient(to right, #3498db, #2ecc71);
            margin: 1rem 0;
            border-radius: 2px;
        }
        
        /* Custom header */
        .custom-header {
            color: #2c3e50;
            font-weight: 700;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3498db;
        }
        
        /* Custom subheader */
        .custom-subheader {
            color: #34495e;
            font-weight: 600;
            margin-top: 1.2rem;
            margin-bottom: 1rem;
        }
        
        /* Custom metric value */
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
        }
        
        /* Custom metric label */
        .metric-label {
            font-size: 0.9rem;
            color: #7f8c8d;
        }
        
        /* Custom metric delta */
        .metric-delta {
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        /* Custom expander */
        .stExpander {
            border-radius: 6px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Custom selectbox */
        .stSelectbox {
            border-radius: 6px;
        }
        
        /* Custom slider */
        .stSlider {
            border-radius: 6px;
        }
        
        /* Custom multiselect */
        .stMultiSelect {
            border-radius: 6px;
        }
        
        /* Custom date input */
        .stDateInput {
            border-radius: 6px;
        }
        
        /* Custom time input */
        .stTimeInput {
            border-radius: 6px;
        }
        
        /* Custom number input */
        .stNumberInput {
            border-radius: 6px;
        }
        
        /* Custom text input */
        .stTextInput {
            border-radius: 6px;
        }
        
        /* Custom text area */
        .stTextArea {
            border-radius: 6px;
        }
        
        /* Custom checkbox */
        .stCheckbox {
            border-radius: 6px;
        }
        
        /* Custom radio */
        .stRadio {
            border-radius: 6px;
        }
        
        /* Custom toggle */
        .stToggle {
            border-radius: 6px;
        }
        
        /* Custom file uploader */
        .stFileUploader {
            border-radius: 6px;
        }
        
        /* Custom download button */
        .stDownloadButton {
            border-radius: 6px;
        }
        
        /* Custom progress bar */
        .stProgress {
            border-radius: 6px;
        }
        
        /* Custom spinner */
        .stSpinner {
            border-radius: 6px;
        }
        
        /* Custom error message */
        .stError {
            border-radius: 6px;
        }
        
        /* Custom success message */
        .stSuccess {
            border-radius: 6px;
        }
        
        /* Custom info message */
        .stInfo {
            border-radius: 6px;
        }
        
        /* Custom warning message */
        .stWarning {
            border-radius: 6px;
        }
        
        /* Custom exception message */
        .stException {
            border-radius: 6px;
        }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-delta" style="color: {"#2ecc71" if delta_color == "normal" else "#e74c3c"};">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def create_custom_header(text: str):
    """Create a styled header."""
    st.markdown(f'<h1 class="custom-header">{text}</h1>', unsafe_allow_html=True)

def create_custom_subheader(text: str):
    """Create a styled subheader."""
    st.markdown(f'<h2 class="custom-subheader">{text}</h2>', unsafe_allow_html=True)

def create_custom_divider():
    """Create a styled divider."""
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

def create_alert(message: str, alert_type: str = "info"):
    """Create a styled alert message."""
    st.markdown(f"""
    <div class="alert-{alert_type}">
        {message}
    </div>
    """, unsafe_allow_html=True)

def create_chart_container(fig, title: str = None):
    """Create a styled chart container."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    if title:
        create_custom_subheader(title)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True) 