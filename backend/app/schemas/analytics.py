from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, UUID4
from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema

# User Activity Log Schemas
class UserActivityLogBase(BaseModel):
    activity_type: str
    description: Optional[str] = None
    ip_address: Optional[str] = None

class UserActivityLogCreate(UserActivityLogBase):
    user_id: UUID4

class UserActivityLog(UserActivityLogBase, BaseSchema):
    user_id: UUID4

# API Usage Schemas
class ApiUsageBase(BaseModel):
    api_name: str
    request_count: int = 0
    last_used: Optional[datetime] = None

class ApiUsageCreate(ApiUsageBase):
    user_id: UUID4

class ApiUsageUpdate(BaseUpdateSchema):
    request_count: Optional[int] = None
    last_used: Optional[datetime] = None

class ApiUsage(ApiUsageBase, BaseSchema):
    user_id: UUID4

# Analytics Report Schemas
class AnalyticsReportBase(BaseModel):
    report_type: str
    report_data: Dict[str, Any]

class AnalyticsReportCreate(AnalyticsReportBase):
    user_id: UUID4

class AnalyticsReport(AnalyticsReportBase, BaseSchema):
    user_id: UUID4

# Content Queue Schemas
class ContentQueueBase(BaseModel):
    content_type: str
    content_data: Dict[str, Any]
    status: str = "pending"
    platform: Optional[str] = None
    scheduled_time: Optional[datetime] = None

class ContentQueueCreate(ContentQueueBase):
    pass

class ContentQueueUpdate(BaseUpdateSchema):
    status: Optional[str] = None
    platform: Optional[str] = None
    scheduled_time: Optional[datetime] = None

class ContentQueue(ContentQueueBase, BaseSchema):
    pass

# Blockchain Transaction Schemas
class BlockchainTransactionBase(BaseModel):
    transaction_hash: str
    transaction_type: str
    status: str = "pending"

class BlockchainTransactionCreate(BlockchainTransactionBase):
    user_id: UUID4

class BlockchainTransactionUpdate(BaseUpdateSchema):
    status: Optional[str] = None

class BlockchainTransaction(BlockchainTransactionBase, BaseSchema):
    user_id: UUID4 