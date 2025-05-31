from sqlalchemy import Column, String, ForeignKey, JSON, DateTime, Integer
from sqlalchemy.orm import relationship
from app.models.base import Base

class UserActivityLog(Base):
    user_id = Column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    activity_type = Column(String(50), nullable=False)
    description = Column(String)
    ip_address = Column(String(45))

    # Relationships
    user = relationship("User", back_populates="activity_logs")

class ApiUsage(Base):
    user_id = Column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    api_name = Column(String(50), nullable=False)
    request_count = Column(Integer, default=0)
    last_used = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="api_usage")

class AnalyticsReport(Base):
    user_id = Column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    report_type = Column(String(50), nullable=False)
    report_data = Column(JSON, nullable=False)

    # Relationships
    user = relationship("User", back_populates="analytics_reports")

class ContentQueue(Base):
    content_type = Column(String(50), nullable=False)
    content_data = Column(JSON, nullable=False)
    status = Column(String(20), default="pending")
    platform = Column(String(50))
    scheduled_time = Column(DateTime(timezone=True))

class BlockchainTransaction(Base):
    user_id = Column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    transaction_hash = Column(String(66), nullable=False)
    transaction_type = Column(String(50), nullable=False)
    status = Column(String(20), default="pending")

    # Relationships
    user = relationship("User", back_populates="blockchain_transactions") 