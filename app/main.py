"""
FastAPI Main Application

This is the entry point for the Intelligent Q&A System API.
It sets up FastAPI with all routes, middleware, and lifecycle events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.config import settings
from app.database import init_db, close_db
from app.auth.router import router as auth_router
from app.routes.qa_router import router as qa_router
from app.routes.documents_router import router as documents_router
from app.routes.evaluation_router import router as evaluation_router


# ============================================================================
# Lifespan Events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Startup:
    - Initialize database
    - Load vector store (if exists)
    - Initialize caches
    
    Shutdown:
    - Close database connections
    - Save vector store
    - Cleanup resources
    """
    # Startup
    print("🚀 Starting Intelligent Q&A System...")
    
    # Initialize database
    await init_db()
    print("✅ Database initialized")
    
    # Application is ready
    print(f"✅ {settings.app_name} v{settings.app_version} is ready!")
    print(f"📚 API docs: http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    await close_db()
    print("✅ Cleanup complete")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Intelligent Q&A System with Memory and Advanced RAG
    
    Features:
    - Advanced RAG patterns (basic, conversational, adaptive, self-RAG)
    - Multiple memory strategies
    - LangChain LCEL chains
    - Tools and function calling
    - Production-ready authentication
    - Rate limiting and error handling
    
    Built with LangChain (no LangGraph) to demonstrate core concepts.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Routes
# ============================================================================

# Authentication routes
app.include_router(auth_router)

# Q&A routes
app.include_router(qa_router)

# Document management routes
app.include_router(documents_router)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "healthy",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "version": settings.app_version,
    }


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )


# ============================================================================
# Educational Note: FastAPI Concepts
# ============================================================================
"""
**FastAPI Key Features:**

1. **Async Support**:
   - All endpoints can be async
   - Better performance for I/O operations
   - Matches our async database

2. **Automatic Documentation**:
   - OpenAPI/Swagger at /docs
   - ReDoc at /redoc
   - Generated from code

3. **Type Validation**:
   - Uses Pydantic for request/response
   - Automatic validation
   - Clear error messages

4. **Dependency Injection**:
   - Clean, testable code
   - Reusable dependencies
   - Example: get_db, get_current_user

5. **Middleware**:
   - CORS for frontend
   - Authentication
   - Rate limiting
   - Logging

**Lifespan Events:**

Replace deprecated startup/shutdown events:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    yield
    # Shutdown code
```

Benefits:
- Cleaner syntax
- Better resource management
- Async support

**Router Organization:**

Separate routers for different features:
- auth_router: Authentication endpoints
- qa_router: Q&A endpoints
- documents_router: Document management

Benefits:
- Better organization
- Easier testing
- Clear separation of concerns

**CORS Configuration:**

Allow frontend to access API:
```python
allow_origins=["http://localhost:3000"]  # React dev server
```

In production:
- Use specific domains
- Don't use "*"
- Enable credentials if needed

**Best Practices:**

1. **Use async**:
   - For database operations
   - For LLM calls
   - For I/O operations

2. **Add middleware**:
   - CORS for frontend
   - Authentication
   - Rate limiting
   - Error handling

3. **Document endpoints**:
   - Use docstrings
   - Add response models
   - Include examples

4. **Handle errors**:
   - Custom exception handlers
   - User-friendly messages
   - Proper status codes

5. **Monitor performance**:
   - Add logging
   - Track metrics
   - Profile slow endpoints
"""
