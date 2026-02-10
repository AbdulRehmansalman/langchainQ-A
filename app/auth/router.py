"""
Authentication Router

FastAPI router for authentication endpoints:
- User registration
- Login
- Token refresh
- Email verification
- Password reset
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from typing import Optional

from app.database import get_db
from app.database.models import User, VerificationToken, PasswordResetToken
from app.auth.schemas import (
    UserCreate, UserLogin, UserResponse, Token,
    EmailVerificationRequest, ForgotPasswordRequest, ResetPasswordRequest,
    RefreshTokenRequest
)
from app.auth.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, verify_token,
    generate_verification_token, generate_password_reset_token
)
from app.config import settings


router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================================
# Helper Functions
# ============================================================================

async def send_verification_email(email: str, token: str):
    """
    Send verification email (mock implementation).
    
    In production, integrate with SMTP or email service (SendGrid, AWS SES, etc.)
    """
    # Mock implementation - in production, send actual email
    print(f"📧 Verification email to {email}")
    print(f"   Token: {token}")
    print(f"   Link: http://localhost:3000/verify-email?token={token}")


async def send_password_reset_email(email: str, token: str):
    """Send password reset email (mock implementation)"""
    print(f"📧 Password reset email to {email}")
    print(f"   Token: {token}")
    print(f"   Link: http://localhost:3000/reset-password?token={token}")


# ============================================================================
# Registration Endpoint
# ============================================================================

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user.
    
    Steps:
    1. Check if email already exists
    2. Hash password
    3. Create user
    4. Generate verification token
    5. Send verification email (background task)
    
    Returns:
        User object (without password)
    """
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_pwd = hash_password(user_data.password)
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_pwd,
        is_verified=False  # Requires email verification
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Generate verification token
    token = generate_verification_token()
    verification_token = VerificationToken(
        user_id=new_user.id,
        token=token,
        expires_at=datetime.utcnow() + timedelta(hours=24)
    )
    db.add(verification_token)
    await db.commit()
    
    # Send verification email in background
    background_tasks.add_task(send_verification_email, new_user.email, token)
    
    return new_user


# ============================================================================
# Login Endpoint
# ============================================================================

@router.post("/login", response_model=Token)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Login user and return JWT tokens.
    
    Steps:
    1. Find user by email
    2. Verify password
    3. Check if email is verified
    4. Generate access and refresh tokens
    
    Returns:
        Access token and refresh token
    """
    # Find user
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your email."
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


# ============================================================================
# Token Refresh Endpoint
# ============================================================================

@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    
    This allows users to stay logged in without re-entering credentials.
    """
    user_id = verify_token(token_data.refresh_token, token_type="refresh")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Verify user still exists and is active
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Generate new tokens
    access_token = create_access_token(user.id)
    new_refresh_token = create_refresh_token(user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


# ============================================================================
# Email Verification Endpoint
# ============================================================================

@router.post("/verify-email")
async def verify_email(
    verification: EmailVerificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Verify user email with token.
    
    Tokens are single-use and expire after 24 hours.
    """
    # Find verification token
    result = await db.execute(
        select(VerificationToken).where(
            VerificationToken.token == verification.token,
            VerificationToken.is_used == False
        )
    )
    token_record = result.scalar_one_or_none()
    
    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    # Check expiration
    if token_record.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired"
        )
    
    # Mark user as verified
    result = await db.execute(select(User).where(User.id == token_record.user_id))
    user = result.scalar_one_or_none()
    
    if user:
        user.is_verified = True
        token_record.is_used = True
        await db.commit()
    
    return {"message": "Email verified successfully"}


# ============================================================================
# Forgot Password Endpoint
# ============================================================================

@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Initiate password reset flow.
    
    Sends password reset email if user exists.
    Always returns success to prevent email enumeration.
    """
    # Find user
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()
    
    if user:
        # Generate reset token
        token = generate_password_reset_token()
        reset_token = PasswordResetToken(
            user_id=user.id,
            token=token,
            expires_at=datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
        )
        db.add(reset_token)
        await db.commit()
        
        # Send reset email in background
        background_tasks.add_task(send_password_reset_email, user.email, token)
    
    # Always return success (prevent email enumeration)
    return {"message": "If the email exists, a password reset link has been sent"}


# ============================================================================
# Reset Password Endpoint
# ============================================================================

@router.post("/reset-password")
async def reset_password(
    reset_data: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Reset password using reset token.
    
    Tokens are single-use and expire after 1 hour.
    """
    # Find reset token
    result = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token == reset_data.token,
            PasswordResetToken.is_used == False
        )
    )
    token_record = result.scalar_one_or_none()
    
    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Check expiration
    if token_record.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired"
        )
    
    # Update password
    result = await db.execute(select(User).where(User.id == token_record.user_id))
    user = result.scalar_one_or_none()
    
    if user:
        user.hashed_password = hash_password(reset_data.new_password)
        token_record.is_used = True
        await db.commit()
    
    return {"message": "Password reset successfully"}


# ============================================================================
# Educational Note: Authentication Flow
# ============================================================================
"""
Complete authentication flow:

1. **Registration**:
   - User submits email + password
   - Password is hashed with bcrypt
   - Verification email is sent
   - User is created but not verified

2. **Email Verification**:
   - User clicks link in email
   - Token is validated and marked as used
   - User is marked as verified

3. **Login**:
   - User submits email + password
   - Password is verified against hash
   - JWT access + refresh tokens are generated
   - Tokens are returned to client

4. **API Requests**:
   - Client includes access token in Authorization header
   - Middleware validates token
   - Request is processed

5. **Token Refresh**:
   - When access token expires, client uses refresh token
   - New access + refresh tokens are generated
   - User stays logged in without re-entering password

6. **Password Reset**:
   - User requests reset
   - Reset email is sent
   - User clicks link and enters new password
   - Password is updated

Security features:
- Passwords are never stored in plaintext
- Tokens are single-use and expire
- Email enumeration is prevented (always return success)
- Background tasks don't block API responses
"""
