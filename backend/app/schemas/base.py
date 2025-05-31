from datetime import datetime
from typing import Optional
from pydantic import BaseModel, UUID4

class BaseSchema(BaseModel):
    id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class BaseCreateSchema(BaseModel):
    pass

class BaseUpdateSchema(BaseModel):
    pass 