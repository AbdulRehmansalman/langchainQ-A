"""
Database Setup and Session Management

This module configures SQLAlchemy for async database operations.
We use async throughout for better performance and scalability.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import StaticPool
from typing import AsyncGenerator

from app.config import settings


# ============================================================================
# SQLAlchemy Base
# ============================================================================
# All database models will inherit from this Base class
Base = declarative_base()


# ============================================================================
# Async Engine Configuration
# ============================================================================
# Create async engine with appropriate settings
# For SQLite: use StaticPool to avoid threading issues
# For PostgreSQL: would use NullPool or QueuePool
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,  # Log SQL queries in debug mode
    poolclass=StaticPool if "sqlite" in settings.database_url else None,
    future=True,
)


# ============================================================================
# Session Factory
# ============================================================================
# async_sessionmaker creates a factory for async sessions
# expire_on_commit=False prevents lazy-loading issues after commit
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ============================================================================
# Dependency Injection for FastAPI
# ============================================================================
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    
    Usage in routes:
        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            ...
    
    The session is automatically closed after the request completes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# Database Initialization
# ============================================================================
async def init_db():
    """
    Initialize database tables.
    
    This creates all tables defined in models.
    Called during application startup.
    """
    async with engine.begin() as conn:
        # Import all models here to ensure they're registered
        from app.database.models import User, Conversation, Message, Document
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """
    Close database connections.
    
    Called during application shutdown for clean cleanup.
    """
    await engine.dispose()


# ============================================================================
# Educational Note: Async SQLAlchemy
# ============================================================================
"""
Why async database operations?

1. **Non-blocking**: Database queries don't block the event loop
2. **Scalability**: Handle more concurrent requests with fewer resources
3. **Performance**: Better throughput for I/O-bound operations
4. **Consistency**: Matches FastAPI's async nature

Key differences from sync SQLAlchemy:
- Use `async with` for sessions
- Use `await` for queries
- Use `AsyncSession` instead of `Session`
- Use `create_async_engine` instead of `create_engine`

Example query:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
"""
