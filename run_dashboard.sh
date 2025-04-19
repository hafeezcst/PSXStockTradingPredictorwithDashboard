#!/bin/bash

# Change to the project directory
cd /Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard

# Activate the Python environment
source /opt/anaconda3/bin/activate python_trading

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run the Streamlit app using the full path
/opt/anaconda3/envs/python_trading/bin/streamlit run src/data_processing/dashboard/main.py 