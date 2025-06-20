import sqlite3
import logging
from contextlib import contextmanager
from typing import Iterator
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

@contextmanager
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def db_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    """Context manager for SQLite database connections with retry"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error for {db_path}: {e}")
        raise
    finally:
        if conn:
            conn.close()
