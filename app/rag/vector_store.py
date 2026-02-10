"""
Vector Store Management

This module demonstrates LangChain's vector store capabilities.
Vector stores enable semantic search over document embeddings.

Key concepts:
- FAISS vector store (fast, local)
- Document indexing
- Similarity search
- MMR (Maximum Marginal Relevance) search
- Metadata filtering
- Persistence
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import os

from app.rag.embeddings import get_cached_embeddings
from app.config import settings


# ============================================================================
# What is a Vector Store?
# ============================================================================
"""
A vector store is a database optimized for vector similarity search.

How it works:
1. Documents are embedded into vectors
2. Vectors are indexed for fast search
3. Query is embedded into a vector
4. Similar vectors are retrieved (cosine similarity)

Example:
    Query: "How do I reset my password?"
    Query vector: [0.2, 0.5, -0.1, ...]
    
    Document 1: "Password recovery steps" → [0.21, 0.48, -0.09, ...]
    Document 2: "Product pricing" → [-0.5, 0.1, 0.8, ...]
    
    Similarity(query, doc1) = 0.95 (high! ✓)
    Similarity(query, doc2) = 0.12 (low ✗)
    
    Returns: Document 1

Why FAISS:
- Fast: Optimized for similarity search
- Local: No external database needed
- Scalable: Handles millions of vectors
- Free: Open source
"""


# ============================================================================
# Initialize Vector Store
# ============================================================================

def create_vector_store(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
) -> FAISS:
    """
    Create a new FAISS vector store from documents.
    
    Args:
        documents: List of documents to index
        embeddings: Embeddings instance (default: cached OpenAI)
        
    Returns:
        FAISS vector store
        
    Example:
        from app.rag.loaders import load_pdf
        from app.rag.splitters import split_documents_with_metadata
        
        # Load and split documents
        docs = load_pdf("manual.pdf")
        chunks = split_documents_with_metadata(docs)
        
        # Create vector store
        vector_store = create_vector_store(chunks)
        
        # Search
        results = vector_store.similarity_search("How do I reset?", k=3)
    """
    if embeddings is None:
        embeddings = get_cached_embeddings()
    
    if not documents:
        raise ValueError("Cannot create vector store from empty document list")
    
    # Create FAISS index from documents
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    
    return vector_store


# ============================================================================
# Add Documents to Existing Vector Store
# ============================================================================

def add_documents_to_vector_store(
    vector_store: FAISS,
    documents: List[Document],
) -> FAISS:
    """
    Add new documents to existing vector store.
    
    Args:
        vector_store: Existing FAISS vector store
        documents: New documents to add
        
    Returns:
        Updated vector store
        
    Example:
        # Create initial vector store
        vector_store = create_vector_store(initial_docs)
        
        # Add more documents later
        vector_store = add_documents_to_vector_store(vector_store, new_docs)
    """
    if not documents:
        return vector_store
    
    vector_store.add_documents(documents)
    return vector_store


# ============================================================================
# Similarity Search
# ============================================================================

def similarity_search(
    vector_store: FAISS,
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Search for similar documents.
    
    Args:
        vector_store: FAISS vector store
        query: Search query
        k: Number of results to return
        filter: Metadata filter (e.g., {"source": "manual.pdf"})
        
    Returns:
        List of most similar documents
        
    Example:
        results = similarity_search(
            vector_store,
            "How do I reset my password?",
            k=3
        )
        
        for doc in results:
            print(doc.page_content)
            print(doc.metadata)
    """
    if filter:
        return vector_store.similarity_search(query, k=k, filter=filter)
    else:
        return vector_store.similarity_search(query, k=k)


def similarity_search_with_score(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Search with similarity scores.
    
    Returns documents with their similarity scores.
    Scores are distances (lower = more similar for FAISS).
    
    Args:
        vector_store: FAISS vector store
        query: Search query
        k: Number of results
        
    Returns:
        List of (document, score) tuples
        
    Example:
        results = similarity_search_with_score(vector_store, "reset password", k=3)
        
        for doc, score in results:
            print(f"Score: {score:.4f}")
            print(f"Content: {doc.page_content[:100]}")
    """
    return vector_store.similarity_search_with_score(query, k=k)


# ============================================================================
# MMR Search (Maximum Marginal Relevance)
# ============================================================================

def mmr_search(
    vector_store: FAISS,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> List[Document]:
    """
    Search using Maximum Marginal Relevance.
    
    MMR balances relevance and diversity:
    - Retrieves relevant documents
    - Ensures diversity (not all similar to each other)
    - Reduces redundancy
    
    Args:
        vector_store: FAISS vector store
        query: Search query
        k: Number of results to return
        fetch_k: Number of candidates to consider
        lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
        
    Returns:
        List of diverse, relevant documents
        
    Example:
        # Standard search might return 3 very similar docs
        standard = similarity_search(vector_store, "Python", k=3)
        
        # MMR returns diverse docs about Python
        diverse = mmr_search(vector_store, "Python", k=3, lambda_mult=0.5)
    
    How it works:
        1. Fetch top fetch_k candidates by similarity
        2. Select most relevant document
        3. For remaining selections, balance:
           - Relevance to query (lambda_mult)
           - Diversity from already selected (1 - lambda_mult)
    """
    return vector_store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )


# ============================================================================
# Metadata Filtering
# ============================================================================

def search_with_metadata_filter(
    vector_store: FAISS,
    query: str,
    metadata_filter: Dict[str, Any],
    k: int = 4,
) -> List[Document]:
    """
    Search with metadata filtering.
    
    Only returns documents matching the metadata filter.
    
    Args:
        vector_store: FAISS vector store
        query: Search query
        metadata_filter: Metadata conditions
        k: Number of results
        
    Returns:
        Filtered search results
        
    Example:
        # Only search in specific document
        results = search_with_metadata_filter(
            vector_store,
            "reset password",
            {"source": "user_manual.pdf"},
            k=3
        )
        
        # Only search specific page
        results = search_with_metadata_filter(
            vector_store,
            "pricing",
            {"source": "pricing.pdf", "page": 5},
            k=2
        )
    """
    return vector_store.similarity_search(
        query,
        k=k,
        filter=metadata_filter,
    )


# ============================================================================
# Vector Store Persistence
# ============================================================================

def save_vector_store(
    vector_store: FAISS,
    path: Optional[str] = None,
) -> str:
    """
    Save vector store to disk.
    
    Args:
        vector_store: FAISS vector store to save
        path: Save path (default: from settings)
        
    Returns:
        Path where vector store was saved
        
    Example:
        # Create and save
        vector_store = create_vector_store(documents)
        save_vector_store(vector_store)
        
        # Load later
        loaded = load_vector_store()
    """
    save_path = path or settings.vector_store_path
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save vector store
    vector_store.save_local(save_path)
    
    return save_path


def load_vector_store(
    path: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
) -> FAISS:
    """
    Load vector store from disk.
    
    Args:
        path: Load path (default: from settings)
        embeddings: Embeddings instance (must match saved store)
        
    Returns:
        Loaded FAISS vector store
        
    Example:
        vector_store = load_vector_store()
        results = vector_store.similarity_search("query")
    """
    load_path = path or settings.vector_store_path
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Vector store not found at {load_path}")
    
    if embeddings is None:
        embeddings = get_cached_embeddings()
    
    # Load vector store
    vector_store = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required for FAISS
    )
    
    return vector_store


def vector_store_exists(path: Optional[str] = None) -> bool:
    """
    Check if vector store exists.
    
    Args:
        path: Path to check (default: from settings)
        
    Returns:
        True if vector store exists
    """
    check_path = path or settings.vector_store_path
    return os.path.exists(check_path)


# ============================================================================
# Educational Note: Vector Store Best Practices
# ============================================================================
"""
**FAISS vs Other Vector Stores:**

1. **FAISS** (used here):
   - Pros: Fast, local, free, no setup
   - Cons: In-memory, no built-in persistence
   - Use for: Development, small-medium datasets, local deployment

2. **Chroma**:
   - Pros: Persistent, easy to use, good for development
   - Cons: Slower than FAISS
   - Use for: Development, prototyping

3. **Pinecone**:
   - Pros: Managed, scalable, production-ready
   - Cons: Costs money, requires account
   - Use for: Production, large scale

4. **Weaviate**:
   - Pros: Feature-rich, self-hosted or cloud
   - Cons: More complex setup
   - Use for: Production, advanced features needed

**Similarity Search Strategies:**

1. **Standard Similarity Search**:
   - Returns top-k most similar documents
   - Fast and simple
   - May return redundant results

2. **MMR (Maximum Marginal Relevance)**:
   - Balances relevance and diversity
   - Reduces redundancy
   - Better for varied information needs

3. **With Score**:
   - Returns similarity scores
   - Allows thresholding (only accept score > X)
   - Useful for quality control

4. **With Metadata Filter**:
   - Narrows search to specific documents
   - Faster (fewer vectors to search)
   - Useful for multi-tenant systems

**Performance Optimization:**

1. **Index Size**:
   - FAISS is fast up to ~1M vectors
   - Beyond that, consider sharding or Pinecone

2. **Batch Operations**:
   - Add documents in batches
   - More efficient than one-by-one

3. **Persistence**:
   - Save vector store to avoid re-indexing
   - Load on startup
   - Update incrementally

4. **Memory Management**:
   - FAISS loads entire index into memory
   - Monitor memory usage
   - Consider disk-based stores for large datasets

**Best Practices:**

1. **Always save vector stores**:
   - Embedding is expensive
   - Save after creation/updates
   - Version your vector stores

2. **Use metadata effectively**:
   - Add source, page, timestamp
   - Enables filtering and citation
   - Helps with debugging

3. **Tune k parameter**:
   - Too small: Miss relevant docs
   - Too large: Include irrelevant docs
   - Typical: 3-5 for Q&A

4. **Consider MMR**:
   - Use when diversity matters
   - Especially for multi-document queries
   - Adjust lambda_mult based on needs

5. **Monitor quality**:
   - Check retrieved documents
   - Measure relevance
   - Adjust chunk size if needed

**Common Issues:**

1. **Poor retrieval quality**:
   - Check chunk size (too large/small?)
   - Check embedding quality
   - Try different k values
   - Consider metadata filtering

2. **Slow search**:
   - Index too large?
   - Consider approximate search
   - Use metadata filtering
   - Shard the index

3. **Memory issues**:
   - FAISS loads full index
   - Reduce index size
   - Use disk-based store
   - Shard the data

4. **Stale results**:
   - Vector store not updated
   - Re-index when documents change
   - Implement incremental updates
"""
