"""
Database Models

This module defines all SQLAlchemy models for the application.
Models represent database tables and relationships.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from app.database.database import Base


# ============================================================================
# User Model (Authentication)
# ============================================================================
class User(Base):
    """
    User model for authentication and authorization.
    
    Stores user credentials, verification status, and role.
    Passwords are hashed using bcrypt (never stored in plaintext).
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    role = Column(String(50), default="user")  # user, admin, premium
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


# ============================================================================
# Conversation Model (Memory)
# ============================================================================
class Conversation(Base):
    """
    Conversation model to track chat sessions.
    
    Each conversation has multiple messages and belongs to a user.
    Stores conversation-level metadata like title and summary.
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), default="New Conversation")
    summary = Column(Text, nullable=True)  # Conversation summary for memory
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"


# ============================================================================
# Message Model (Conversation History)
# ============================================================================
class Message(Base):
    """
    Message model for individual chat messages.
    
    Stores both user queries and AI responses.
    Includes metadata like sources, tool calls, and tokens used.
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # Metadata
    sources = Column(JSON, nullable=True)  # List of source documents used
    tool_calls = Column(JSON, nullable=True)  # Tools called during generation
    tokens_used = Column(Integer, default=0)
    cost = Column(String(20), nullable=True)  # Estimated cost
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role})>"


# ============================================================================
# Document Model (RAG Knowledge Base)
# ============================================================================
class Document(Base):
    """
    Document model for uploaded files.
    
    Tracks documents added to the knowledge base.
    Stores metadata about the document and its processing status.
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, txt, html
    file_size = Column(Integer, nullable=False)  # in bytes
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    chunk_count = Column(Integer, default=0)  # Number of chunks created
    
    # Metadata
    custom_metadata = Column(JSON, nullable=True)  # Custom metadata
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="documents")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename})>"


# ============================================================================
# Verification Token Model (Email Verification)
# ============================================================================
class VerificationToken(Base):
    """
    Email verification token model.
    
    Stores tokens for email verification.
    Tokens expire after a set time for security.
    """
    __tablename__ = "verification_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============================================================================
# Password Reset Token Model
# ============================================================================
class PasswordResetToken(Base):
    """
    Password reset token model.
    
    Stores tokens for password reset flow.
    Tokens are single-use and expire after a set time.
    """
    __tablename__ = "password_reset_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============================================================================
# Educational Note: SQLAlchemy Models
# ============================================================================
"""
Key SQLAlchemy concepts demonstrated:

1. **Declarative Base**: All models inherit from Base
2. **Columns**: Define table structure with types and constraints
3. **Relationships**: Define foreign keys and relationships between tables
4. **Indexes**: Speed up queries on frequently searched columns
5. **Timestamps**: Auto-managed created_at/updated_at fields
6. **Cascade**: Automatic deletion of related records
7. **JSON Columns**: Store structured data (sources, metadata)

Relationship patterns:
- One-to-Many: User → Conversations, Conversation → Messages
- Cascade delete: Deleting a user deletes their conversations and messages

Best practices:
- Use indexes on foreign keys and frequently queried fields
- Store timestamps for audit trails
- Use JSON for flexible metadata
- Never store plaintext passwords (use hashed_password)
"""
