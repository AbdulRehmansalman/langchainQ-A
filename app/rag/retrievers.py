"""
Advanced Retrievers

This module demonstrates LangChain's retriever capabilities.
Retrievers are interfaces for fetching relevant documents.

Key concepts:
- VectorStoreRetriever: Basic vector search
- MultiQueryRetriever: Generate multiple query variations
- EnsembleRetriever: Combine multiple retrievers
- Hybrid search (BM25 + vector)
"""

from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import (
    MultiQueryRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from typing import List, Optional

from app.rag.vector_store import FAISS
from app.models.llm_config import get_openai_chat_model


# ============================================================================
# Basic Vector Store Retriever
# ============================================================================

def get_vector_store_retriever(
    vector_store: FAISS,
    search_type: str = "similarity",
    k: int = 4,
    **kwargs,
) -> BaseRetriever:
    """
    Get basic vector store retriever.
    
    Args:
        vector_store: FAISS vector store
        search_type: "similarity" or "mmr"
        k: Number of documents to retrieve
        **kwargs: Additional search parameters
        
    Returns:
        Vector store retriever
        
    Example:
        retriever = get_vector_store_retriever(vector_store, k=4)
        docs = retriever.get_relevant_documents("How do I reset?")
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, **kwargs}
    )


# ============================================================================
# Multi-Query Retriever
# ============================================================================

def get_multi_query_retriever(
    vector_store: FAISS,
    llm: Optional[BaseChatModel] = None,
    k: int = 4,
) -> MultiQueryRetriever:
    """
    Get multi-query retriever.
    
    Multi-query retriever:
    1. Takes user's question
    2. Generates multiple variations of the question using LLM
    3. Retrieves documents for each variation
    4. Returns unique documents from all queries
    
    This improves recall by capturing different phrasings.
    
    Args:
        vector_store: FAISS vector store
        llm: LLM for query generation (default: OpenAI)
        k: Documents per query
        
    Returns:
        Multi-query retriever
        
    Example:
        retriever = get_multi_query_retriever(vector_store)
        
        # User asks: "How do I reset my password?"
        # LLM generates variations:
        # - "What are the steps to reset my password?"
        # - "How can I recover my password?"
        # - "Password reset instructions"
        # Retrieves docs for all variations
        
        docs = retriever.get_relevant_documents("How do I reset my password?")
    
    Benefits:
        - Better recall (more relevant docs)
        - Handles different phrasings
        - Reduces impact of poor query formulation
    
    Tradeoffs:
        - More LLM calls (costs more)
        - Slower (multiple retrievals)
        - May retrieve more irrelevant docs
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0)
    
    base_retriever = get_vector_store_retriever(vector_store, k=k)
    
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )


# ============================================================================
# BM25 Retriever (Keyword Search)
# ============================================================================

def get_bm25_retriever(
    documents: List[Document],
    k: int = 4,
) -> BM25Retriever:
    """
    Get BM25 retriever for keyword-based search.
    
    BM25 is a traditional keyword search algorithm (like search engines).
    It's complementary to vector search:
    - Vector search: Semantic similarity
    - BM25: Keyword matching
    
    Args:
        documents: All documents to search
        k: Number of documents to retrieve
        
    Returns:
        BM25 retriever
        
    Example:
        retriever = get_bm25_retriever(all_documents, k=4)
        docs = retriever.get_relevant_documents("password reset")
    
    When BM25 is better:
        - Exact keyword matches needed
        - Technical terms, IDs, codes
        - Proper nouns
    
    When vector search is better:
        - Semantic understanding needed
        - Synonyms, paraphrasing
        - Conceptual queries
    """
    return BM25Retriever.from_documents(
        documents,
        k=k,
    )


# ============================================================================
# Ensemble Retriever (Hybrid Search)
# ============================================================================

def get_ensemble_retriever(
    vector_store: FAISS,
    documents: List[Document],
    weights: Optional[List[float]] = None,
    k: int = 4,
) -> EnsembleRetriever:
    """
    Get ensemble retriever combining vector and keyword search.
    
    Ensemble retriever combines multiple retrievers:
    - Vector search (semantic)
    - BM25 (keyword)
    
    Results are weighted and merged for best of both worlds.
    
    Args:
        vector_store: FAISS vector store
        documents: All documents (for BM25)
        weights: Weights for each retriever (default: [0.5, 0.5])
        k: Number of documents to retrieve
        
    Returns:
        Ensemble retriever
        
    Example:
        retriever = get_ensemble_retriever(
            vector_store,
            all_documents,
            weights=[0.7, 0.3],  # 70% vector, 30% BM25
            k=4
        )
        
        docs = retriever.get_relevant_documents("reset password")
    
    Weight tuning:
        - [0.7, 0.3]: Prefer semantic (general Q&A)
        - [0.5, 0.5]: Balanced (default)
        - [0.3, 0.7]: Prefer keywords (technical docs)
    
    Benefits:
        - Best of both worlds
        - Better recall and precision
        - Robust to different query types
    
    Tradeoffs:
        - Slower (two retrievals)
        - More complex
        - Needs tuning
    """
    if weights is None:
        weights = [0.5, 0.5]  # Equal weight
    
    # Create individual retrievers
    vector_retriever = get_vector_store_retriever(vector_store, k=k)
    bm25_retriever = get_bm25_retriever(documents, k=k)
    
    # Combine into ensemble
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights,
    )


# ============================================================================
# Contextual Compression Retriever (Advanced)
# ============================================================================
"""
Contextual Compression Retriever (not implemented here, but explained):

This retriever:
1. Retrieves documents (over-retrieve, e.g., k=10)
2. Uses LLM to extract only relevant parts
3. Returns compressed, relevant excerpts

Benefits:
- More relevant content
- Less noise in context
- Better LLM performance

Tradeoffs:
- Extra LLM call (slower, costs more)
- More complex

Example usage:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
"""


# ============================================================================
# Educational Note: Retriever Selection Guide
# ============================================================================
"""
**When to use each retriever:**

1. **VectorStoreRetriever** (Basic):
   - Use for: Most cases, semantic search
   - Pros: Fast, simple, good quality
   - Cons: May miss exact keyword matches
   - Example: General Q&A, documentation search

2. **MultiQueryRetriever**:
   - Use for: When queries are poorly formulated
   - Pros: Better recall, handles variations
   - Cons: Slower, more expensive
   - Example: User-generated questions, chatbots

3. **BM25Retriever**:
   - Use for: Keyword-heavy domains
   - Pros: Fast, good for exact matches
   - Cons: No semantic understanding
   - Example: Code search, technical docs, IDs

4. **EnsembleRetriever** (Hybrid):
   - Use for: Best quality, production systems
   - Pros: Best of both worlds
   - Cons: Slower, more complex
   - Example: Production Q&A, critical applications

5. **ContextualCompressionRetriever**:
   - Use for: When context quality is critical
   - Pros: Most relevant content
   - Cons: Slowest, most expensive
   - Example: Complex queries, long documents

**Retrieval Quality Metrics:**

1. **Recall**: Did we retrieve all relevant docs?
   - Improve: Increase k, use MultiQuery, use Ensemble

2. **Precision**: Are retrieved docs relevant?
   - Improve: Decrease k, use compression, better chunks

3. **Latency**: How fast is retrieval?
   - Improve: Use simple retriever, optimize index

4. **Cost**: How much does it cost?
   - Improve: Avoid MultiQuery, avoid compression

**Best Practices:**

1. **Start simple**:
   - Begin with VectorStoreRetriever
   - Add complexity only if needed
   - Measure before optimizing

2. **Tune k parameter**:
   - Too small: Miss relevant docs
   - Too large: Include irrelevant docs
   - Typical: 3-5 for Q&A

3. **Consider hybrid search**:
   - Especially for production
   - Tune weights for your domain
   - Test with real queries

4. **Monitor quality**:
   - Check retrieved documents
   - Measure user satisfaction
   - Iterate based on feedback

5. **Optimize for your use case**:
   - Technical docs: More BM25 weight
   - General Q&A: More vector weight
   - User questions: Consider MultiQuery

**Common Patterns:**

1. **Development**: VectorStoreRetriever
2. **Production (general)**: EnsembleRetriever
3. **Production (high quality)**: MultiQuery + Ensemble
4. **Production (critical)**: MultiQuery + Ensemble + Compression

**Performance Tips:**

1. **Cache retrievals**:
   - Same query = same results
   - Significant speedup

2. **Parallel retrieval**:
   - For ensemble, retrieve in parallel
   - Use async when possible

3. **Index optimization**:
   - Keep index in memory
   - Use appropriate index type
   - Shard for large datasets

4. **Batch queries**:
   - Process multiple queries together
   - More efficient
"""
