"""
Security Utilities for Authentication

This module provides security functions for:
- Password hashing and verification
- JWT token creation and validation
- Token generation for email verification and password reset
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
import secrets
from passlib.context import CryptContext
from jose import JWTError, jwt

from app.config import settings


# ============================================================================
# Password Hashing
# ============================================================================
# Use bcrypt for secure password hashing
# bcrypt is designed to be slow, making brute-force attacks difficult
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
        
    Example:
        hashed = hash_password("MySecurePass123")
        # Returns: $2b$12$...
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
        
    Example:
        is_valid = verify_password("MySecurePass123", user.hashed_password)
    """
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================================
# JWT Token Management
# ============================================================================

def create_access_token(user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Access tokens are short-lived (default: 30 minutes) and used for API authentication.
    
    Args:
        user_id: User ID to encode in token
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
        
    Example:
        token = create_access_token(user_id=123)
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    to_encode = {
        "sub": str(user_id),  # Subject (user ID)
        "exp": expire,  # Expiration time
        "type": "access"  # Token type
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    return encoded_jwt


def create_refresh_token(user_id: int) -> str:
    """
    Create a JWT refresh token.
    
    Refresh tokens are long-lived (default: 7 days) and used to obtain new access tokens.
    
    Args:
        user_id: User ID to encode in token
        
    Returns:
        Encoded JWT token string
    """
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    
    to_encode = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[int]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        token_type: Expected token type ("access" or "refresh")
        
    Returns:
        User ID if token is valid, None otherwise
        
    Example:
        user_id = verify_token(token, "access")
        if user_id:
            # Token is valid
            ...
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        
        # Verify token type
        if payload.get("type") != token_type:
            return None
        
        # Extract user ID
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
            
        return int(user_id)
        
    except JWTError:
        return None


# ============================================================================
# Verification Token Generation
# ============================================================================

def generate_verification_token() -> str:
    """
    Generate a secure random token for email verification.
    
    Uses secrets module for cryptographically strong random generation.
    
    Returns:
        Random token string (32 bytes, hex encoded = 64 characters)
        
    Example:
        token = generate_verification_token()
        # Returns: "a3f5b2c1d4e6f7a8b9c0d1e2f3a4b5c6..."
    """
    return secrets.token_hex(32)


def generate_password_reset_token() -> str:
    """
    Generate a secure random token for password reset.
    
    Returns:
        Random token string (32 bytes, hex encoded = 64 characters)
    """
    return secrets.token_hex(32)


# ============================================================================
# Educational Note: Security Best Practices
# ============================================================================
"""
Security concepts demonstrated:

1. **Password Hashing with bcrypt**:
   - Never store plaintext passwords
   - bcrypt is intentionally slow (prevents brute-force)
   - Automatic salt generation
   - Configurable work factor

2. **JWT Tokens**:
   - Stateless authentication (no server-side session storage)
   - Access tokens: Short-lived, used for API calls
   - Refresh tokens: Long-lived, used to get new access tokens
   - Signed with secret key (prevents tampering)
   - Contains expiration time (prevents replay attacks)

3. **Token Types**:
   - Access token: For API authentication (30 min)
   - Refresh token: For getting new access tokens (7 days)
   - Verification token: For email verification (one-time use)
   - Reset token: For password reset (one-time use)

4. **Cryptographic Randomness**:
   - Use secrets module (not random module)
   - Cryptographically strong random generation
   - Prevents token prediction attacks

Best practices:
- Always hash passwords before storing
- Use different tokens for different purposes
- Set appropriate expiration times
- Validate token type before use
- Use HTTPS in production (prevents token interception)
"""
