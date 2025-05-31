from sqlalchemy import Column, String, Numeric, Boolean, Date, ForeignKey, Integer, BigInteger
from sqlalchemy.orm import relationship
from app.models.base import Base

class KMI30Stock(Base):
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    sector = Column(String(50))
    market_cap = Column(Numeric(20, 2))

    # Relationships
    buy_signals = relationship("BuySignal", back_populates="stock", cascade="all, delete-orphan")
    sell_signals = relationship("SellSignal", back_populates="stock", cascade="all, delete-orphan")
    neutral_signals = relationship("NeutralSignal", back_populates="stock", cascade="all, delete-orphan")
    signal_transitions = relationship("SignalTransition", back_populates="stock", cascade="all, delete-orphan")
    dividends = relationship("Dividend", back_populates="stock", cascade="all, delete-orphan")
    fund_investments = relationship("FundInvestment", back_populates="stock", cascade="all, delete-orphan")
    etf_memberships = relationship("ETFMembership", back_populates="stock", cascade="all, delete-orphan")
    portfolio_holdings = relationship("PortfolioHolding", back_populates="stock", cascade="all, delete-orphan")

class Portfolio(Base):
    user_id = Column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String)

    # Relationships
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("PortfolioHolding", back_populates="portfolio", cascade="all, delete-orphan")

class PortfolioHolding(Base):
    portfolio_id = Column(ForeignKey("portfolio.id", ondelete="CASCADE"), nullable=False)
    stock_id = Column(ForeignKey("kmi30stock.id", ondelete="CASCADE"), nullable=False)
    shares = Column(Integer, nullable=False)
    purchase_price = Column(Numeric(10, 2), nullable=False)
    entry_date = Column(Date, nullable=False)
    take_profit = Column(Numeric(10, 2))
    stop_loss = Column(Numeric(10, 2))

    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")
    stock = relationship("KMI30Stock", back_populates="portfolio_holdings")

class BuySignal(Base):
    stock_id = Column(ForeignKey("kmi30stock.id", ondelete="CASCADE"), nullable=False)
    signal_date = Column(Date, nullable=False)
    technical_score = Column(Numeric(5, 2))
    fundamental_score = Column(Numeric(5, 2))
    tradingview_signal = Column(String(20))
    volume = Column(BigInteger)
    rsi_weekly = Column(Numeric(5, 2))
    rsi_monthly = Column(Numeric(5, 2))
    rsi_quarterly = Column(Numeric(5, 2))
    ao_cross_positive = Column(Boolean)

    # Relationships
    stock = relationship("KMI30Stock", back_populates="buy_signals")

class SellSignal(Base):
    stock_id = Column(ForeignKey("kmi30stock.id", ondelete="CASCADE"), nullable=False)
    signal_date = Column(Date, nullable=False)
    technical_score = Column(Numeric(5, 2))
    fundamental_score = Column(Numeric(5, 2))
    tradingview_signal = Column(String(20))
    stop_loss_triggered = Column(Boolean, default=False)
    take_profit_triggered = Column(Boolean, default=False)

    # Relationships
    stock = relationship("KMI30Stock", back_populates="sell_signals")

class NeutralSignal(Base):
    stock_id = Column(ForeignKey("kmi30stock.id", ondelete="CASCADE"), nullable=False)
    signal_date = Column(Date, nullable=False)
    technical_score = Column(Numeric(5, 2))
    fundamental_score = Column(Numeric(5, 2))
    tradingview_signal = Column(String(20))

    # Relationships
    stock = relationship("KMI30Stock", back_populates="neutral_signals")

class SignalTransition(Base):
    stock_id = Column(ForeignKey("kmi30stock.id", ondelete="CASCADE"), nullable=False)
    from_signal_type = Column(String(20), nullable=False)
    to_signal_type = Column(String(20), nullable=False)
    transition_date = Column(Date, nullable=False)

    # Relationships
    stock = relationship("KMI30Stock", back_populates="signal_transitions")

class Dividend(Base):
    stock_id = Column(ForeignKey("kmi30stock.id", ondelete="CASCADE"), nullable=False)
    declaration_date = Column(Date, nullable=False)
    payment_date = Column(Date)
    amount = Column(Numeric(10, 4), nullable=False)
    dividend_yield = Column(Numeric(5, 2))
    source = Column(String(50))

    # Relationships
    stock = relationship("KMI30Stock", back_populates="dividends")

class FundInvestment(Base):
    stock_id = Column(ForeignKey("kmi30stock.id", ondelete="CASCADE"), nullable=False)
    fund_name = Column(String(100), nullable=False)
    ownership_percentage = Column(Numeric(5, 2))
    source = Column(String(50))
    last_updated = Column(Date)

    # Relationships
    stock = relationship("KMI30Stock", back_populates="fund_investments")

class ETFMembership(Base):
    stock_id = Column(ForeignKey("kmi30stock.id", ondelete="CASCADE"), nullable=False)
    etf_name = Column(String(100), nullable=False)
    weight = Column(Numeric(5, 2))
    source = Column(String(50))
    last_updated = Column(Date)

    # Relationships
    stock = relationship("KMI30Stock", back_populates="etf_memberships") 