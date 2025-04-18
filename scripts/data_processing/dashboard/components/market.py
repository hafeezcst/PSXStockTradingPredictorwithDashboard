"""
Market analysis component for the PSX dashboard.
"""

import streamlit as st
import pandas as pd
import os
import glob
from typing import Dict, Any
from datetime import datetime, timedelta
from PIL import Image

from scripts.data_processing.dashboard.components.shared_styles import (
    apply_shared_styles,
    create_custom_header,
    create_custom_subheader,
    create_custom_divider
)

def get_latest_dashboard_images(dashboard_dir: str, n_days: int = 5):
    """
    Get the latest dashboard PNG images from the specified directory.
    
    Args:
        dashboard_dir: Directory containing dashboard images
        n_days: Number of most recent days to retrieve
        
    Returns:
        List of tuples containing (date, image_path)
    """
    # Get all dashboard PNG files
    dashboard_files = glob.glob(os.path.join(dashboard_dir, "psx_dashboard_*.png"))
    
    # Extract dates and create (date, file_path) tuples
    dashboard_data = []
    for file_path in dashboard_files:
        filename = os.path.basename(file_path)
        date_str = filename.replace("psx_dashboard_", "").replace(".png", "")
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            dashboard_data.append((date_obj, file_path))
        except ValueError:
            continue
    
    # Sort by date (newest first) and take the most recent n_days
    dashboard_data.sort(reverse=True, key=lambda x: x[0])
    return dashboard_data[:n_days]

def display_market(config: Dict[str, Any]):
    """Display market analysis dashboard using pre-generated dashboard images."""
    apply_shared_styles()
    create_custom_header("Market Analysis")
    create_custom_divider()
    
    # Get dashboard directory from config or use default
    dashboard_dir = config.get("dashboard_dir", 
                              "/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/outputs/dashboards/PSX_DASHBOARDS")
    
    # Get the latest dashboard images
    dashboard_data = get_latest_dashboard_images(dashboard_dir)
    
    if not dashboard_data:
        st.error("No dashboard images found in the specified directory.")
        return
    
    # Create date selector
    dates = [d[0].strftime("%Y-%m-%d") for d in dashboard_data]
    selected_date = st.selectbox("Select date:", dates)
    
    # Display the selected dashboard image
    selected_image_path = None
    for date_obj, image_path in dashboard_data:
        if date_obj.strftime("%Y-%m-%d") == selected_date:
            selected_image_path = image_path
            break
    
    if selected_image_path:
        try:
            image = Image.open(selected_image_path)
            st.image(image, caption=f"PSX Dashboard - {selected_date}", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        st.error(f"Dashboard image for {selected_date} not found.") 