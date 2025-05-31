from sqlalchemy import Boolean, Column, String
from sqlalchemy.orm import relationship
from app.models.base import Base

class User(Base):
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
    subscription_tier = Column(String(20), default="free")

    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    activity_logs = relationship("UserActivityLog", back_populates="user", cascade="all, delete-orphan")
    api_usage = relationship("ApiUsage", back_populates="user", cascade="all, delete-orphan")
    analytics_reports = relationship("AnalyticsReport", back_populates="user", cascade="all, delete-orphan")
    blockchain_transactions = relationship("BlockchainTransaction", back_populates="user", cascade="all, delete-orphan") 