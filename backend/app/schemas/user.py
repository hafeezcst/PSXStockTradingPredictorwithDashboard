from typing import Optional
from pydantic import BaseModel, EmailStr, UUID4
from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False
    subscription_tier: str = "free"

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseUpdateSchema):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    subscription_tier: Optional[str] = None

class UserInDBBase(UserBase, BaseSchema):
    pass

class User(UserInDBBase):
    pass

class UserInDB(UserInDBBase):
    hashed_password: str 