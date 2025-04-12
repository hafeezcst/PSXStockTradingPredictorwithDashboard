"""
Charts component for the PSX dashboard.
"""

import streamlit as st
import os
from pathlib import Path
from typing import Dict, Any, List
import glob
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime

# Update the charts directory to use the RSI_AO_CHARTS directory
CHARTS_DIR = Path("/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/outputs/charts/RSI_AO_CHARTS")

def get_chart_files(directory: Path, extensions: List[str] = ['.png', '.jpg', '.jpeg']) -> List[Path]:
    """
    Get all chart image files from a directory.
    
    Args:
        directory: Directory to scan for chart images
        extensions: List of file extensions to include
        
    Returns:
        List of Path objects for chart files
    """
    chart_files = []
    for ext in extensions:
        chart_files.extend(glob.glob(str(directory / f"*{ext}")))
    
    # Sort by creation time (newest first)
    return sorted([Path(f) for f in chart_files], key=lambda x: os.path.getctime(x), reverse=True)

def display_chart_gallery(chart_files: List[Path]):
    """Display charts in a gallery format."""
    if not chart_files:
        st.warning("No chart files found in the specified directory.")
        return
    
    # Create tabs for different view options
    tab1, tab2 = st.tabs(["Gallery View", "List View"])
    
    with tab1:
        # Gallery view with multiple charts per row
        cols = st.columns(2)
        for i, chart_file in enumerate(chart_files):
            col_idx = i % 2
            with cols[col_idx]:
                try:
                    img = Image.open(chart_file)
                    # Add creation time to caption
                    creation_time = datetime.fromtimestamp(os.path.getctime(chart_file))
                    formatted_time = creation_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Highlight the most recent charts (first 4)
                    if i < 4:
                        st.markdown(f"""
                        <div style="border: 2px solid #ff5722; border-radius: 5px; padding: 5px; margin-bottom: 10px;">
                            <span style="color: #ff5722; font-weight: bold;">Latest Signal</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.image(img, caption=f"{chart_file.stem.replace('_', ' ').title()} ({formatted_time})", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image {chart_file.name}: {str(e)}")
    
    with tab2:
        # List view with larger charts and more details
        for i, chart_file in enumerate(chart_files):
            try:
                # Add creation time to expander title
                creation_time = datetime.fromtimestamp(os.path.getctime(chart_file))
                formatted_time = creation_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Highlight the most recent charts (first 4)
                if i < 4:
                    expander_title = f"🆕 {chart_file.stem.replace('_', ' ').title()} ({formatted_time})"
                else:
                    expander_title = f"{chart_file.stem.replace('_', ' ').title()} ({formatted_time})"
                
                expander = st.expander(expander_title)
                with expander:
                    st.image(str(chart_file), use_container_width=True)
                    st.caption(f"Created: {formatted_time}")
            except Exception as e:
                st.error(f"Error loading image {chart_file.name}: {str(e)}")

def display_charts(config: Dict[str, Any]):
    """Display technical analysis charts."""
    st.subheader("Technical Analysis Charts")
    
    # Get custom charts directory from config or use default
    charts_dir = Path(config.get("charts_dir", str(CHARTS_DIR)))
    
    # Check if the charts directory exists
    if not charts_dir.exists():
        st.error(f"Charts directory not found: {charts_dir}")
        st.info("Please ensure the charts directory exists and contains chart files.")
        return
    
    # Add filter options
    st.sidebar.subheader("Chart Filters")
    
    # Get all chart files
    chart_files = get_chart_files(charts_dir)
    
    if not chart_files:
        st.warning(f"No chart files found in {charts_dir}")
        return
    
    # Get unique chart types from filenames
    chart_types = set()
    for f in chart_files:
        parts = f.stem.split('_')
        if len(parts) > 1:
            chart_types.add(parts[0].lower())
    
    # Add chart type filter
    selected_type = st.sidebar.selectbox(
        "Chart Type",
        ["All"] + sorted(list(chart_types)),
        index=0
    )
    
    # Filter by selected type
    if selected_type != "All":
        filtered_charts = [f for f in chart_files if f.stem.lower().startswith(selected_type.lower())]
    else:
        filtered_charts = chart_files
    
    # Add stock filter based on filename
    stock_identifiers = set()
    for f in chart_files:
        # Assume stock identifier is in the filename
        parts = f.stem.split('_')
        if len(parts) > 1:
            # Try to find stock symbol in the file name
            for part in parts:
                if part.isupper() and len(part) >= 2 and len(part) <= 6:
                    stock_identifiers.add(part)
    
    if stock_identifiers:
        selected_stock = st.sidebar.selectbox(
            "Stock Symbol",
            ["All"] + sorted(list(stock_identifiers)),
            index=0
        )
        
        if selected_stock != "All":
            filtered_charts = [f for f in filtered_charts if selected_stock in f.stem]
    
    # Display number of charts found
    st.info(f"Found {len(filtered_charts)} charts matching the selected filters.")
    
    # Display the charts
    display_chart_gallery(filtered_charts) 