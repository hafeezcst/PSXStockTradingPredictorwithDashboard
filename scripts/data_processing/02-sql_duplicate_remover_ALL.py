import sqlite3
import pandas as pd
import time

def remove_duplicates_from_table(conn, table_name):
    """
    Remove duplicate records from a table based on the first 6 columns.
    
    Args:
        conn: SQLite database connection
        table_name: Name of the table to process
        
    Returns:
        tuple: (number of duplicates found, number of records deleted)
    """
    try:
        # Fetch all records from the table
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        if df.empty:
            print(f"Table {table_name} is empty. Skipping.")
            return 0, 0
        
        # Define the columns to check for duplicates (first 6 columns)
        columns_to_check = df.columns[:6].tolist()
        print(f"Checking for duplicates in table {table_name} based on columns: {columns_to_check}")

        # Find duplicates based on the specified columns
        duplicates = df[df.duplicated(subset=columns_to_check, keep=False)]

        # Number of duplicates found
        num_duplicates = len(duplicates)

        # Remove duplicates and keep the first occurrence
        df_cleaned = df.drop_duplicates(subset=columns_to_check, keep='first')

        # Number of records to be deleted
        num_deleted = len(df) - len(df_cleaned)

        # Write the cleaned DataFrame back to the database
        df_cleaned.to_sql(table_name, conn, if_exists='replace', index=False)

        return num_duplicates, num_deleted
    except Exception as e:
        print(f"Error processing table {table_name}: {str(e)}")
        return 0, 0

def process_database(conn, db_name):
    """
    Process all tables in a database to remove duplicates.
    
    Args:
        conn: SQLite database connection
        db_name: Name of the database (for reporting)
    """
    print(f"\n--- Processing database: {db_name} ---")
    
    try:
        # Get the list of all tables in the database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Initialize counters
        total_duplicates = 0
        total_deleted = 0

        # Iterate over each table and remove duplicates
        for table in tables:
            table_name = table[0]
            num_duplicates, num_deleted = remove_duplicates_from_table(conn, table_name)
            total_duplicates += num_duplicates
            total_deleted += num_deleted
            print(f"Table: {table_name}")
            print(f"Number of duplicate records found: {num_duplicates}")
            print(f"Number of records deleted: {num_deleted}\n")

        # Commit the changes
        conn.commit()

        # Final reporting
        print(f"Total number of duplicate records found in {db_name}: {total_duplicates}")
        print(f"Total number of records deleted in {db_name}: {total_deleted}")
        
    except Exception as e:
        print(f"Error processing database {db_name}: {str(e)}")
    finally:
        # Close the connection
        conn.close()

def main():
    """Main function to execute the duplicate removal process."""
    start_time = time.time()
    
    # Database connections and processing
    databases = [

        ("data/databases/production/psx_consolidated_data_psx.db", "PSX"),
    ]
    
    for db_file, db_name in databases:
        try:
            conn = sqlite3.connect(db_file)
            process_database(conn, db_name)
        except Exception as e:
            print(f"Failed to connect to {db_file}: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

