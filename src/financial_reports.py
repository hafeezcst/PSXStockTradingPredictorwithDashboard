"""
PSX Financial Reports Viewer App - Standalone Version
"""

import streamlit as st
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="PSX Financial Reports",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("PSX Financial Reports Viewer")
st.markdown("---")

def display_financial_reports():
    """Display financial reports with the ability to view them inline."""
    
    # Path to the actual Excel file
    excel_file = Path("data/excel/PSX_Announcements.xlsx")
    
    if not excel_file.exists():
        st.error(f"Excel file not found at {excel_file}")
        # Fall back to sample data if file not found
        use_sample_data = True
    else:
        use_sample_data = False
    
    if use_sample_data:
        # Sample financial reports data with links (fallback)
        financial_reports_data = {
            'Symbol': ['OGDC', 'PPL', 'LUCK', 'MARI', 'ENGRO', 'UBL', 'HBL', 'MCB', 'PSO', 'EFERT'],
            'Company': ['Oil & Gas Development Company Ltd', 'Pakistan Petroleum Limited', 'Lucky Cement Limited', 'Mari Petroleum Company Limited', 'Engro Corporation Limited', 'United Bank Limited', 'Habib Bank Limited', 'MCB Bank Limited', 'Pakistan State Oil', 'Engro Fertilizers Limited'],
            'Report Type': ['Annual Report', 'Annual Report', 'Quarterly Report', 'Annual Report', 'Quarterly Report', 'Annual Report', 'Quarterly Report', 'Annual Report', 'Quarterly Report', 'Annual Report'],
            'Date': ['2024-03-15', '2024-02-28', '2024-01-31', '2024-03-10', '2024-02-15', '2024-01-20', '2024-03-05', '2024-02-10', '2024-01-25', '2024-03-01'],
            'Subject': [
                'Annual Report 2023', 
                'Annual Report 2023', 
                'Quarterly Report Q1 2024', 
                'Annual Report 2023', 
                'Quarterly Report Q1 2024', 
                'Annual Report 2023', 
                'Quarterly Report Q1 2024', 
                'Annual Report 2023', 
                'Quarterly Report Q1 2024', 
                'Annual Report 2023'
            ],
            'Report Link': [
                'https://dps.psx.com.pk/download/document/17042.pdf',
                'https://dps.psx.com.pk/download/document/16541.pdf',
                'https://dps.psx.com.pk/download/document/17024.pdf',
                'https://dps.psx.com.pk/download/document/16662.pdf',
                'https://dps.psx.com.pk/download/document/17040.pdf',
                'https://dps.psx.com.pk/download/document/17035.pdf',
                'https://dps.psx.com.pk/download/document/17038.pdf',
                'https://dps.psx.com.pk/download/document/17044.pdf',
                'https://dps.psx.com.pk/download/document/17012.pdf',
                'https://dps.psx.com.pk/download/document/17051.pdf'
            ]
        }
        df = pd.DataFrame(financial_reports_data)
        df['Date'] = pd.to_datetime(df['Date'])
        st.warning("Using sample data as Excel file was not found.")
    else:
        try:
            # Load the actual financial reports data from Excel
            st.info(f"Loading financial reports from {excel_file}...")
            all_data = pd.read_excel(excel_file)
            
            # Show basic info about the data
            st.write(f"Total records: {len(all_data)}")
            st.write(f"Columns: {', '.join(all_data.columns)}")
            
            # Filter for financial results only
            df = all_data[all_data['Category'] == 'Financial Results'].copy()
            
            # Add a Report Type column based on the Subject
            df['Report Type'] = df['Subject'].apply(lambda x: 
                'Annual Report' if 'annual' in x.lower() or 'year ended' in x.lower()
                else ('Quarterly Report' if 'quarter' in x.lower() 
                     else ('Half-Yearly Report' if 'half' in x.lower() or 'half-year' in x.lower() 
                          else 'Financial Report')))
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Rename URL column to Report Link if it exists
            if 'URL' in df.columns:
                df = df.rename(columns={'URL': 'Report Link'})
            
            # Sort by date (newest first)
            df = df.sort_values('Date', ascending=False)
            
            st.success(f"Loaded {len(df)} financial reports from {excel_file}")
        except Exception as e:
            st.error(f"Error loading financial reports data: {str(e)}")
            st.error("Details:", exc_info=True)
            st.error("Using sample data due to error.")
            # Fall back to sample data if there's an error
            use_sample_data = True
            financial_reports_data = {
                'Symbol': ['OGDC', 'PPL', 'LUCK', 'MARI', 'ENGRO'],
                'Company': ['Oil & Gas Development Company Ltd', 'Pakistan Petroleum Limited', 'Lucky Cement Limited', 'Mari Petroleum Company Limited', 'Engro Corporation Limited'],
                'Report Type': ['Annual Report', 'Annual Report', 'Quarterly Report', 'Annual Report', 'Quarterly Report'],
                'Date': ['2024-03-15', '2024-02-28', '2024-01-31', '2024-03-10', '2024-02-15'],
                'Subject': [
                    'Annual Report 2023', 
                    'Annual Report 2023', 
                    'Quarterly Report Q1 2024', 
                    'Annual Report 2023', 
                    'Quarterly Report Q1 2024'
                ],
                'Report Link': [
                    'https://dps.psx.com.pk/download/document/17042.pdf',
                    'https://dps.psx.com.pk/download/document/16541.pdf',
                    'https://dps.psx.com.pk/download/document/17024.pdf',
                    'https://dps.psx.com.pk/download/document/16662.pdf',
                    'https://dps.psx.com.pk/download/document/17040.pdf'
                ]
            }
            df = pd.DataFrame(financial_reports_data)
            df['Date'] = pd.to_datetime(df['Date'])
    
    # Custom Report URL Input Section
    st.subheader("Custom Financial Report")
    st.write("Enter a direct PDF URL to view any financial report")
    
    custom_url = st.text_input("PDF URL:", 
                              help="Enter the full URL to a PDF report from PSX or other source")
    
    if custom_url and custom_url.strip():
        if custom_url.lower().endswith('.pdf'):
            try:
                st.write("### Viewing Custom Report")
                
                # Display the PDF using HTML iframe
                pdf_display = f'<iframe src="{custom_url}" width="100%" height="600" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                
                # Provide direct link as fallback
                st.markdown(f"If the PDF doesn't display correctly above, [open it directly in a new tab]({custom_url})")
            except Exception as e:
                st.error(f"Error displaying PDF: {str(e)}")
        else:
            st.warning("URL does not appear to be a PDF. Please ensure the URL ends with .pdf")
    
    st.markdown("---")
    
    # Add filters
    st.subheader("Financial Reports")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol_filter = st.multiselect("Filter by Symbol", options=sorted(df['Symbol'].unique()), 
                                      help="Select one or more symbols to filter the reports")
    
    with col2:
        report_type_filter = st.multiselect("Filter by Report Type", options=sorted(df['Report Type'].unique()),
                                           help="Select one or more report types to filter the reports")
    
    with col3:
        # Add a date range filter
        date_range = st.date_input(
            "Date Range",
            value=(df['Date'].min().date(), df['Date'].max().date()),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date(),
            help="Select a date range to filter the reports"
        )
    
    # Apply filters
    filtered_df = df.copy()
    if symbol_filter:
        filtered_df = filtered_df[filtered_df['Symbol'].isin(symbol_filter)]
    if report_type_filter:
        filtered_df = filtered_df[filtered_df['Report Type'].isin(report_type_filter)]
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'].dt.date >= start_date) & 
                                 (filtered_df['Date'].dt.date <= end_date)]
    
    # Display metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Total Reports", len(filtered_df))
    with metrics_col2:
        st.metric("Unique Companies", filtered_df['Company'].nunique())
    with metrics_col3:
        if not filtered_df.empty:
            st.metric("Date Range", f"{filtered_df['Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Date'].max().strftime('%Y-%m-%d')}")
        else:
            st.metric("Date Range", "N/A")
    
    # Create display dataframe with links
    display_df = filtered_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Display table
    if not display_df.empty:
        st.dataframe(display_df[['Symbol', 'Company', 'Report Type', 'Date', 'Subject']], use_container_width=True)
    else:
        st.warning("No reports match your filter criteria.")
    
    # Create report selection and viewer
    st.subheader("Report Viewer")
    
    if not display_df.empty:
        # Create a dropdown to select a report to view
        report_options = [f"{row['Symbol']} - {row['Date']} - {row['Subject']}" 
                        for _, row in display_df.iterrows()]
        
        report_options.insert(0, "Select a report to view")
        
        selected_report = st.selectbox("Select Report", options=report_options)
        
        if selected_report != "Select a report to view":
            # Get the index of the selected report in the filtered dataframe
            selected_index = report_options.index(selected_report) - 1  # -1 to account for the "Select a report" option
            
            # Get the URL of the selected report
            selected_url = display_df.iloc[selected_index]['Report Link']
            
            st.markdown(f"### Viewing: {selected_report}")
            
            # Display the PDF content
            pdf_display = f'<iframe src="{selected_url}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Add direct links for different access methods
            st.markdown(f"If the report doesn't display correctly above, [open it directly in a new tab]({selected_url})")
            
            # Download button
            try:
                response = requests.get(selected_url, timeout=10)
                if response.status_code == 200:
                    filename = f"{display_df.iloc[selected_index]['Symbol']}_{display_df.iloc[selected_index]['Date']}_{display_df.iloc[selected_index]['Subject']}.pdf"
                    
                    st.download_button(
                        label="Download PDF",
                        data=response.content,
                        file_name=filename,
                        mime="application/pdf"
                    )
                else:
                    st.warning(f"Could not download the file: HTTP {response.status_code}")
            except Exception as e:
                st.warning(f"Could not prepare download: {str(e)}")
                st.markdown(f"You can download the file directly from: [{selected_url}]({selected_url})")
    else:
        st.warning("No reports available to view. Please adjust your filters.")

if __name__ == "__main__":
    display_financial_reports() 