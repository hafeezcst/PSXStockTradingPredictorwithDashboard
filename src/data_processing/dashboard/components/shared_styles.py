"""
Shared styles and UI components for the PSX dashboard.
"""

import streamlit as st
import os
import sys
from typing import Optional, Any

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def apply_shared_styles():
    """Apply shared styles to the dashboard."""
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .custom-header {
            color: #1E88E5;
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .custom-subheader {
            color: #424242;
            font-size: 1.5rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .custom-divider {
            margin: 1rem 0;
            border-top: 2px solid #E0E0E0;
        }
        
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chart-container {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .alert {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .alert-info {
            background-color: #E3F2FD;
            border: 1px solid #90CAF9;
            color: #1565C0;
        }
        
        .alert-success {
            background-color: #E8F5E9;
            border: 1px solid #A5D6A7;
            color: #2E7D32;
        }
        
        .alert-warning {
            background-color: #FFF3E0;
            border: 1px solid #FFCC80;
            color: #EF6C00;
        }
        
        .alert-error {
            background-color: #FFEBEE;
            border: 1px solid #EF9A9A;
            color: #C62828;
        }
        </style>
    """, unsafe_allow_html=True)

def create_custom_header(text: str):
    """Create a custom header with consistent styling."""
    st.markdown(f'<h1 class="custom-header">{text}</h1>', unsafe_allow_html=True)

def create_custom_subheader(text: str):
    """Create a custom subheader with consistent styling."""
    st.markdown(f'<h2 class="custom-subheader">{text}</h2>', unsafe_allow_html=True)

def create_custom_divider():
    """Create a custom divider with consistent styling."""
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

def create_chart_container(chart: Any, title: Optional[str] = None):
    """Create a container for charts with consistent styling."""
    with st.container():
        if title:
            st.markdown(f'<h3 class="custom-subheader">{title}</h3>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_metric_card(label: str, value: Any, delta: Optional[Any] = None):
    """Create a metric card with consistent styling."""
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label=label, value=value, delta=delta)
        st.markdown('</div>', unsafe_allow_html=True)

def create_alert(message: str, alert_type: str = "info"):
    """Create an alert box with consistent styling.
    
    Args:
        message: The message to display
        alert_type: One of "info", "success", "warning", "error"
    """
    alert_types = {
        "info": "alert-info",
        "success": "alert-success",
        "warning": "alert-warning",
        "error": "alert-error"
    }
    alert_class = alert_types.get(alert_type.lower(), "alert-info")
    st.markdown(f'<div class="alert {alert_class}">{message}</div>', unsafe_allow_html=True) 