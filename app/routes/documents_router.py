"""
Documents Router

FastAPI router for document upload and management.
Handles PDF/text uploads and vector store indexing.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from pathlib import Path

from app.database import get_db
from app.database.models import User, Document
from app.auth.security import verify_token
from app.rag.loaders import load_pdf, load_text_file
from app.rag.splitters import split_documents_with_metadata
from app.rag.vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
    vector_store_exists,
    add_documents_to_vector_store
)
from app.routes.qa_router import get_current_user


router = APIRouter(prefix="/documents", tags=["Documents"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class DocumentResponse(BaseModel):
    """Response schema for document"""
    id: int
    filename: str
    file_type: str
    file_size: int
    status: str
    uploaded_at: datetime
    chunk_count: Optional[int] = None


class UploadResponse(BaseModel):
    """Response schema for upload"""
    message: str
    document_id: int
    chunks_created: int


# ============================================================================
# Upload Directory
# ============================================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================================
# Upload Document Endpoint
# ============================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document and index it in the vector store.
    
    Supported formats:
    - PDF (.pdf)
    - Text (.txt, .md)
    
    Process:
    1. Save file to disk
    2. Load and parse document
    3. Split into chunks
    4. Create/update vector store
    5. Save metadata to database
    
    Example:
        POST /documents/upload
        Content-Type: multipart/form-data
        
        file: manual.pdf
    """
    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )
    
    # Create document record
    document = Document(
        user_id=user.id,
        filename=file.filename,
        file_path=str(file_path),
        file_type=file_ext[1:],  # Remove dot
        file_size=len(content),
        status="processing"
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    
    try:
        # Load document
        if file_ext == ".pdf":
            docs = load_pdf(str(file_path))
        else:
            docs = load_text_file(str(file_path))
        
        # Split into chunks
        chunks = split_documents_with_metadata(docs)
        
        # Create or update vector store
        if vector_store_exists():
            # Load existing and add new documents
            vector_store = load_vector_store()
            vector_store = add_documents_to_vector_store(vector_store, chunks)
        else:
            # Create new vector store
            vector_store = create_vector_store(chunks)
        
        # Save vector store
        save_vector_store(vector_store)
        
        # Update document status
        document.status = "indexed"
        document.chunk_count = len(chunks)
        await db.commit()
        
        return UploadResponse(
            message="Document uploaded and indexed successfully",
            document_id=document.id,
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        # Update document status to failed
        document.status = "failed"
        document.error_message = str(e)
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


# ============================================================================
# List Documents Endpoint
# ============================================================================

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List user's uploaded documents.
    
    Returns all documents uploaded by the current user.
    """
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user.id)
        .order_by(Document.uploaded_at.desc())
    )
    documents = result.scalars().all()
    
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            status=doc.status,
            uploaded_at=doc.uploaded_at,
            chunk_count=doc.chunk_count
        )
        for doc in documents
    ]


# ============================================================================
# Delete Document Endpoint
# ============================================================================

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a document.
    
    Note: This deletes the database record and file,
    but does NOT remove from vector store (requires rebuild).
    """
    # Get document
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id
        )
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete file
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        # Log error but continue
        print(f"Error deleting file: {e}")
    
    # Delete database record
    await db.delete(document)
    await db.commit()
    
    return {
        "message": "Document deleted successfully",
        "note": "Vector store rebuild recommended"
    }


# ============================================================================
# Educational Note: File Upload Patterns
# ============================================================================
"""
**File Upload Best Practices:**

1. **Validate file type**:
   - Check extension
   - Verify content type
   - Scan for malware (production)

2. **Limit file size**:
   - Set max size (e.g., 10MB)
   - Prevent abuse
   - Protect server resources

3. **Secure storage**:
   - Store outside web root
   - Use unique filenames
   - Set proper permissions

4. **Async processing**:
   - Don't block request
   - Use background tasks
   - Return immediately

5. **Error handling**:
   - Graceful failures
   - Clean up on error
   - User-friendly messages

**Vector Store Management:**

1. **Incremental updates**:
   - Add to existing store
   - Don't rebuild from scratch
   - Save after each update

2. **Metadata tracking**:
   - Track source documents
   - Enable filtering
   - Support deletion

3. **Rebuild strategy**:
   - Periodic full rebuilds
   - Remove deleted documents
   - Optimize index

4. **Backup**:
   - Save vector store regularly
   - Version control
   - Disaster recovery

**Production Improvements:**

1. **Background processing**:
   ```python
   from fastapi import BackgroundTasks
   
   @router.post("/upload")
   async def upload(file: UploadFile, bg: BackgroundTasks):
       bg.add_task(process_document, file)
       return {"message": "Processing started"}
   ```

2. **Progress tracking**:
   - WebSocket updates
   - Status endpoint
   - Notifications

3. **Chunking strategy**:
   - Optimize chunk size
   - Test with your data
   - Monitor quality

4. **Deduplication**:
   - Check for duplicates
   - Hash-based detection
   - Prevent redundancy
"""
