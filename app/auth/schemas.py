"""
Pydantic Schemas for Authentication

These schemas define the structure of request and response data.
Pydantic provides automatic validation, serialization, and documentation.
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime


# ============================================================================
# User Schemas
# ============================================================================

class UserBase(BaseModel):
    """Base user schema with common fields"""
    email: EmailStr = Field(..., description="User email address")


class UserCreate(UserBase):
    """Schema for user registration"""
    password: str = Field(
        ...,
        min_length=8,
        description="Password (min 8 characters)"
    )
    
    @validator("password")
    def validate_password(cls, v):
        """
        Validate password strength.
        
        Requirements:
        - At least 8 characters
        - Contains uppercase and lowercase
        - Contains at least one digit
        """
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str


class UserResponse(UserBase):
    """Schema for user response (excludes password)"""
    id: int
    is_verified: bool
    is_active: bool
    role: str
    created_at: datetime
    
    class Config:
        from_attributes = True  # Allows creation from ORM models


# ============================================================================
# Token Schemas
# ============================================================================

class Token(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """JWT token payload"""
    sub: int  # User ID
    exp: datetime
    type: str  # "access" or "refresh"


class RefreshTokenRequest(BaseModel):
    """Request to refresh access token"""
    refresh_token: str


# ============================================================================
# Email Verification Schemas
# ============================================================================

class EmailVerificationRequest(BaseModel):
    """Request to verify email"""
    token: str = Field(..., description="Verification token from email")


class ResendVerificationRequest(BaseModel):
    """Request to resend verification email"""
    email: EmailStr


# ============================================================================
# Password Reset Schemas
# ============================================================================

class ForgotPasswordRequest(BaseModel):
    """Request to initiate password reset"""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Request to reset password with token"""
    token: str = Field(..., description="Reset token from email")
    new_password: str = Field(..., min_length=8)
    
    @validator("new_password")
    def validate_password(cls, v):
        """Validate new password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# ============================================================================
# Educational Note: Pydantic Schemas
# ============================================================================
"""
Why use Pydantic schemas?

1. **Automatic Validation**: Validates input data automatically
2. **Type Safety**: Ensures correct data types
3. **Documentation**: Auto-generates OpenAPI/Swagger docs
4. **Serialization**: Converts between JSON and Python objects
5. **Custom Validators**: Add complex validation logic

Key patterns:
- Separate schemas for input (Create) and output (Response)
- Use EmailStr for email validation
- Use Field() for additional constraints and documentation
- Use validators for complex validation logic
- Exclude sensitive fields (password) from response schemas

Example usage in FastAPI:
    @router.post("/register", response_model=UserResponse)
    async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
        # user is automatically validated
        # response is automatically serialized to UserResponse
        ...
"""
