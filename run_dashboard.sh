#!/bin/bash

# Change to the project directory
cd /Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard

# Activate the Python environment
source /opt/anaconda3/bin/activate python_trading

# Set PYTHONPATH to include the project root and src directory
export PYTHONPATH=$PYTHONPATH:/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard
export PYTHONPATH=$PYTHONPATH:/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/src

# Run the Streamlit app using the full path
/opt/anaconda3/envs/python_trading/bin/streamlit run src/data_processing/dashboard/main.py 