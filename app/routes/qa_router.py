"""
Q&A Router

FastAPI router for question-answering endpoints.
Implements the main Q&A functionality using RAG chains.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.database import get_db
from app.database.models import User, Conversation, Message
from app.auth.security import verify_token
from app.rag.vector_store import load_vector_store, vector_store_exists
from app.rag.adaptive_rag import create_adaptive_conversational_rag
from app.models.llm_config import get_chat_model_with_fallback


router = APIRouter(prefix="/qa", tags=["Q&A"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class QuestionRequest(BaseModel):
    """Request schema for asking a question"""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    conversation_id: Optional[int] = Field(None, description="Conversation ID for context")
    use_streaming: bool = Field(default=False, description="Enable streaming response")


class SourceInfo(BaseModel):
    """Source document information"""
    source: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None


class QuestionResponse(BaseModel):
    """Response schema for Q&A"""
    answer: str
    sources: List[SourceInfo] = []
    conversation_id: int
    message_id: int
    tokens_used: Optional[int] = None


class ConversationListResponse(BaseModel):
    """Response schema for conversation list"""
    id: int
    title: str
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: int


class MessageResponse(BaseModel):
    """Response schema for a message"""
    id: int
    role: str
    content: str
    sources: Optional[List[Dict]] = None
    created_at: datetime


# ============================================================================
# Dependency: Get Current User
# ============================================================================

async def get_current_user(
    token: str = Depends(lambda: "mock_token"),  # In production, extract from header
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user.
    
    In production, this would:
    1. Extract token from Authorization header
    2. Verify JWT token
    3. Get user from database
    
    For now, we'll use a mock user for demonstration.
    """
    # Mock user for demonstration
    # In production, verify token and get real user
    result = await db.execute(select(User).limit(1))
    user = result.scalar_one_or_none()
    
    if not user:
        # Create mock user if none exists
        user = User(
            email="demo@example.com",
            hashed_password="mock",
            is_verified=True
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    
    return user


# ============================================================================
# Get or Create Conversation
# ============================================================================

async def get_or_create_conversation(
    user_id: int,
    conversation_id: Optional[int],
    db: AsyncSession
) -> Conversation:
    """Get existing conversation or create new one"""
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=user_id,
            title="New Conversation"
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
    
    return conversation


# ============================================================================
# Ask Question Endpoint
# ============================================================================

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Ask a question and get an answer.
    
    This endpoint:
    1. Gets or creates a conversation
    2. Loads conversation history
    3. Uses adaptive RAG to answer
    4. Saves question and answer to database
    5. Returns response with sources
    
    Example:
        POST /qa/ask
        {
            "question": "How do I reset my password?",
            "conversation_id": null
        }
        
        Response:
        {
            "answer": "To reset your password...",
            "sources": [...],
            "conversation_id": 1,
            "message_id": 2
        }
    """
    # Check if vector store exists
    if not vector_store_exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base not initialized. Please upload documents first."
        )
    
    # Load vector store
    vector_store = load_vector_store()
    
    # Get or create conversation
    conversation = await get_or_create_conversation(
        user.id,
        request.conversation_id,
        db
    )
    
    # Load conversation history
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
        .limit(10)  # Last 10 messages
    )
    messages = result.scalars().all()
    
    # Convert to LangChain message format
    from langchain_core.messages import HumanMessage, AIMessage
    chat_history = []
    for msg in messages:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))
    
    # Create adaptive RAG chain
    llm = get_chat_model_with_fallback()
    chain = create_adaptive_conversational_rag(vector_store, llm)
    
    # Get answer
    try:
        answer = chain.invoke({
            "question": request.question,
            "chat_history": chat_history
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}"
        )
    
    # Save question to database
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.question
    )
    db.add(user_message)
    
    # Save answer to database
    ai_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources=[]  # Would extract from chain in production
    )
    db.add(ai_message)
    
    await db.commit()
    await db.refresh(ai_message)
    
    # Update conversation title if it's the first message
    if len(messages) == 0:
        conversation.title = request.question[:50] + "..." if len(request.question) > 50 else request.question
        await db.commit()
    
    return QuestionResponse(
        answer=answer,
        sources=[],  # Would extract from chain in production
        conversation_id=conversation.id,
        message_id=ai_message.id
    )


# ============================================================================
# List Conversations Endpoint
# ============================================================================

@router.get("/conversations", response_model=List[ConversationListResponse])
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List user's conversations.
    
    Returns all conversations for the current user,
    ordered by most recent first.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()
    
    # Count messages for each conversation
    response = []
    for conv in conversations:
        msg_result = await db.execute(
            select(Message).where(Message.conversation_id == conv.id)
        )
        message_count = len(msg_result.scalars().all())
        
        response.append(ConversationListResponse(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=message_count
        ))
    
    return response


# ============================================================================
# Get Conversation History Endpoint
# ============================================================================

@router.get("/conversations/{conversation_id}", response_model=List[MessageResponse])
async def get_conversation_history(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get conversation history.
    
    Returns all messages in a conversation.
    """
    # Verify conversation belongs to user
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    
    return [
        MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            sources=msg.sources,
            created_at=msg.created_at
        )
        for msg in messages
    ]


# ============================================================================
# Delete Conversation Endpoint
# ============================================================================

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a conversation and all its messages"""
    # Verify conversation belongs to user
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Delete conversation (cascade deletes messages)
    await db.delete(conversation)
    await db.commit()
    
    return {"message": "Conversation deleted successfully"}


# ============================================================================
# Educational Note: FastAPI Endpoint Patterns
# ============================================================================
"""
**Key Patterns Demonstrated:**

1. **Dependency Injection**:
   - get_current_user: Authentication
   - get_db: Database session
   - Reusable, testable

2. **Pydantic Validation**:
   - Request models validate input
   - Response models validate output
   - Automatic documentation

3. **Error Handling**:
   - HTTPException for user errors
   - Proper status codes
   - Clear error messages

4. **Async Operations**:
   - All endpoints are async
   - Better performance
   - Matches async database

5. **Database Patterns**:
   - Get or create
   - Cascade operations
   - Transaction management

**Best Practices:**

1. **Use response models**:
   - Validates output
   - Documents API
   - Type safety

2. **Handle errors gracefully**:
   - User-friendly messages
   - Proper status codes
   - Log errors

3. **Validate input**:
   - Use Pydantic
   - Set constraints
   - Clear error messages

4. **Use dependencies**:
   - Reusable logic
   - Easy testing
   - Clean code

5. **Document endpoints**:
   - Docstrings
   - Examples
   - Response models
"""
