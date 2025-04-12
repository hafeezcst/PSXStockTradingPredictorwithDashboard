import re

file_path = '/Users/muhammadhafeez/Documents/GitHub/PSXStockTradingPredictorwithDashboard/scripts/data_processing/PSX_dashboard.py'

with open(file_path, 'r') as file:
    content = file.read()

# Add debugging output after the database connection
debug_pattern = r"conn = engine\.connect\(\)"
debug_replacement = """conn = engine.connect()
        
        # Debug: Check if neutral_stocks table exists
        try:
            check_table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neutral_stocks'").fetchall()
            st.write(f"Debug - neutral_stocks table check: {check_table}")
            if not check_table:
                st.error("neutral_stocks table does not exist in the database")
                st.info(f"Current database path: {signals_db_path}")
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                st.write(f"Available tables: {[t[0] for t in tables]}")
        except Exception as e:
            st.error(f"Debug - Table check error: {str(e)}")"""

# Add the debugging code
modified_content = re.sub(debug_pattern, debug_replacement, content)

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.write(modified_content)

print("Added debugging code to check for the neutral_stocks table")
