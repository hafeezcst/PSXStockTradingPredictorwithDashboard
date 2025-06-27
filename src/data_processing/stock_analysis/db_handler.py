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

# --- SQLAlchemy Async ORM Setup ---
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, mapped_column, Mapped
from sqlalchemy import String, Integer, Float, Text, DateTime, select
from typing import Optional, List

Base = declarative_base()

class Signal(Base):
    __tablename__ = 'signal'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    buy_stocks: Mapped[Optional[str]] = mapped_column(String)
    sell_stocks: Mapped[Optional[str]] = mapped_column(String)
    neutral_stocks: Mapped[Optional[str]] = mapped_column(String)
    # Add other columns as needed

class StockSignal(Base):
    __tablename__ = 'stock_signals'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String)
    date: Mapped[str] = mapped_column(String)
    signal: Mapped[str] = mapped_column(String)
    close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[float]] = mapped_column(Float)
    rsi_weekly: Mapped[Optional[float]] = mapped_column(Float)
    rsi_3months: Mapped[Optional[float]] = mapped_column(Float)
    ao_weekly: Mapped[Optional[float]] = mapped_column(Float)
    ma_30: Mapped[Optional[float]] = mapped_column(Float)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    technical_score: Mapped[Optional[float]] = mapped_column(Float)
    multibagger: Mapped[Optional[str]] = mapped_column(String)
    free_float_ratio: Mapped[Optional[str]] = mapped_column(String)
    success: Mapped[Optional[str]] = mapped_column(String)
    profit_loss: Mapped[Optional[float]] = mapped_column(Float)
    signal_date: Mapped[Optional[str]] = mapped_column(String)
    signal_close: Mapped[Optional[float]] = mapped_column(Float)
    holding_days: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[Optional[str]] = mapped_column(String)
    analysis_summary: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[str]] = mapped_column(String)

# Async engine and session setup
ASYNC_DB_URL = "sqlite+aiosqlite:///data/databases/production/PSX_investing_Stocks_KMI100.db"
engine = create_async_engine(ASYNC_DB_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_async_session() -> AsyncSession:
    """Yield an async SQLAlchemy session."""
    async with AsyncSessionLocal() as session:
        yield session

# Example async query function
async def fetch_signals(limit: int = 100) -> List[Signal]:
    """Fetch signals from the signal table asynchronously."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Signal).limit(limit))
        return result.scalars().all()

async def fetch_stock_signals(limit: int = 100) -> List[StockSignal]:
    """Fetch stock signals from the stock_signals table asynchronously."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(StockSignal).limit(limit))
        return result.scalars().all()
