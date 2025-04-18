import sqlite3
import os
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parents[2]
DB_DIR = BASE_DIR / "data" / "databases" / "production"
DB_PATH = DB_DIR / "PSX_Dividend_Schedule.db"

def create_dividend_database():
    """Create the dividend schedule database with required tables"""
    try:
        # Ensure directory exists
        os.makedirs(DB_DIR, exist_ok=True)
        
        # Connect to database (will create if doesn't exist)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create dividend schedule table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dividend_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            company_name TEXT,
            face_value REAL,
            last_close REAL,
            bc_from TEXT,
            bc_to TEXT,
            dividend_amount REAL,
            right_amount REAL,
            payout_text TEXT,
            data_type TEXT DEFAULT 'current',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create index on symbol and dates
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON dividend_schedule(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bc_dates ON dividend_schedule(bc_from, bc_to)")
        
        # Create dividend history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dividend_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            company_name TEXT,
            face_value REAL,
            last_close REAL,
            bc_from TEXT,
            bc_to TEXT,
            dividend_amount REAL,
            right_amount REAL,
            payout_text TEXT,
            data_type TEXT DEFAULT 'historical',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create index on symbol
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_symbol ON dividend_history(symbol)")
        
        # Create dividend statistics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dividend_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            total_dividends INTEGER,
            total_rights INTEGER,
            avg_dividend_amount REAL,
            avg_right_amount REAL,
            last_dividend_date TEXT,
            last_right_date TEXT,
            dividend_frequency TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create index on statistics
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_symbol ON dividend_statistics(symbol)")
        
        # Create triggers for updated_at
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS update_dividend_schedule_timestamp 
        AFTER UPDATE ON dividend_schedule
        BEGIN
            UPDATE dividend_schedule SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        """)
        
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS update_dividend_statistics_timestamp 
        AFTER UPDATE ON dividend_statistics
        BEGIN
            UPDATE dividend_statistics SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        """)
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Successfully created dividend database at {DB_PATH}")
        return True
        
    except Exception as e:
        print(f"Error creating dividend database: {str(e)}")
        return False

if __name__ == "__main__":
    create_dividend_database() 