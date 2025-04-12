"""
Styles module for the PSX dashboard.
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS styles to the dashboard."""
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 104, 201, 0.1);
        }
        .stButton>button {
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: rgba(0, 104, 201, 0.2);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: bold;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.8rem;
        }
        div.row-widget.stRadio > div {
            flex-direction: row;
            align-items: center;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            margin: 0px 10px;
            padding: 10px;
            background-color: rgba(0, 104, 201, 0.1);
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True) 