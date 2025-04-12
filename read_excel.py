#!/usr/bin/env python
"""
Script to read and display the structure of the PSX_Announcements.xlsx file.
"""

import pandas as pd
import sys

# Path to the Excel file
excel_file = "data/excel/PSX_Announcements.xlsx"

try:
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Print basic information
    print(f"Excel file has {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    # Print first few rows
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    
    # Check if there's a 'Category' column
    if 'Category' in df.columns:
        print("\nUnique categories:")
        for category in df['Category'].unique():
            count = df[df['Category'] == category].shape[0]
            print(f"- {category}: {count} entries")
            
    # If there's no Category column, look for similar columns
    else:
        print("\nNo 'Category' column found. Looking for similar columns...")
        for col in df.columns:
            if 'type' in col.lower() or 'category' in col.lower() or 'class' in col.lower():
                print(f"\nUnique values in '{col}':")
                for val in df[col].unique():
                    count = df[df[col] == val].shape[0]
                    print(f"- {val}: {count} entries")
                    
except Exception as e:
    print(f"Error reading Excel file: {e}", file=sys.stderr) 