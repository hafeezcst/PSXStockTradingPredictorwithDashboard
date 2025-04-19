#!/bin/bash

# Change to the project directory
cd "$(dirname "$0")"

# Run the Streamlit app
streamlit run src/data_processing/dashboard/app.py 