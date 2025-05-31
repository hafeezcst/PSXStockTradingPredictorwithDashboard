from datetime import date, datetime
from typing import Optional, List
from decimal import Decimal
from pydantic import BaseModel, UUID4
from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema

# Stock Schemas
class KMI30StockBase(BaseModel):
    symbol: str
    name: str
    sector: Optional[str] = None
    market_cap: Optional[Decimal] = None

class KMI30StockCreate(KMI30StockBase):
    pass

class KMI30StockUpdate(BaseUpdateSchema):
    name: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[Decimal] = None

class KMI30Stock(KMI30StockBase, BaseSchema):
    pass

# Portfolio Schemas
class PortfolioBase(BaseModel):
    name: str
    description: Optional[str] = None

class PortfolioCreate(PortfolioBase):
    pass

class PortfolioUpdate(BaseUpdateSchema):
    name: Optional[str] = None
    description: Optional[str] = None

class Portfolio(PortfolioBase, BaseSchema):
    user_id: UUID4

# Portfolio Holding Schemas
class PortfolioHoldingBase(BaseModel):
    shares: int
    purchase_price: Decimal
    entry_date: date
    take_profit: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None

class PortfolioHoldingCreate(PortfolioHoldingBase):
    stock_id: UUID4

class PortfolioHoldingUpdate(BaseUpdateSchema):
    shares: Optional[int] = None
    take_profit: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None

class PortfolioHolding(PortfolioHoldingBase, BaseSchema):
    portfolio_id: UUID4
    stock_id: UUID4

# Signal Schemas
class SignalBase(BaseModel):
    signal_date: date
    technical_score: Optional[Decimal] = None
    fundamental_score: Optional[Decimal] = None
    tradingview_signal: Optional[str] = None

class BuySignalBase(SignalBase):
    volume: Optional[int] = None
    rsi_weekly: Optional[Decimal] = None
    rsi_monthly: Optional[Decimal] = None
    rsi_quarterly: Optional[Decimal] = None
    ao_cross_positive: Optional[bool] = None

class SellSignalBase(SignalBase):
    stop_loss_triggered: bool = False
    take_profit_triggered: bool = False

class NeutralSignalBase(SignalBase):
    pass

class BuySignal(BuySignalBase, BaseSchema):
    stock_id: UUID4

class SellSignal(SellSignalBase, BaseSchema):
    stock_id: UUID4

class NeutralSignal(NeutralSignalBase, BaseSchema):
    stock_id: UUID4

# Dividend Schemas
class DividendBase(BaseModel):
    declaration_date: date
    payment_date: Optional[date] = None
    amount: Decimal
    dividend_yield: Optional[Decimal] = None
    source: Optional[str] = None

class DividendCreate(DividendBase):
    stock_id: UUID4

class DividendUpdate(BaseUpdateSchema):
    payment_date: Optional[date] = None
    amount: Optional[Decimal] = None
    dividend_yield: Optional[Decimal] = None
    source: Optional[str] = None

class Dividend(DividendBase, BaseSchema):
    stock_id: UUID4

# Fund Investment Schemas
class FundInvestmentBase(BaseModel):
    fund_name: str
    ownership_percentage: Optional[Decimal] = None
    source: Optional[str] = None
    last_updated: Optional[date] = None

class FundInvestmentCreate(FundInvestmentBase):
    stock_id: UUID4

class FundInvestmentUpdate(BaseUpdateSchema):
    ownership_percentage: Optional[Decimal] = None
    source: Optional[str] = None
    last_updated: Optional[date] = None

class FundInvestment(FundInvestmentBase, BaseSchema):
    stock_id: UUID4

# ETF Membership Schemas
class ETFMembershipBase(BaseModel):
    etf_name: str
    weight: Optional[Decimal] = None
    source: Optional[str] = None
    last_updated: Optional[date] = None

class ETFMembershipCreate(ETFMembershipBase):
    stock_id: UUID4

class ETFMembershipUpdate(BaseUpdateSchema):
    weight: Optional[Decimal] = None
    source: Optional[str] = None
    last_updated: Optional[date] = None

class ETFMembership(ETFMembershipBase, BaseSchema):
    stock_id: UUID4 