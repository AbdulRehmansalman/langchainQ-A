"""
Embeddings Configuration

This module demonstrates LangChain's embedding capabilities.
Embeddings convert text into vector representations for semantic search.

Key concepts:
- OpenAI embeddings
- Document vs query embeddings
- Embedding caching
- Dimension awareness
"""

from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List
import hashlib
import json
import os
from pathlib import Path

from app.config import settings


# ============================================================================
# What are Embeddings?
# ============================================================================
"""
Embeddings are vector representations of text that capture semantic meaning.

Example:
    "dog" → [0.2, 0.5, -0.1, ..., 0.3]  (1536 dimensions for OpenAI)
    "puppy" → [0.21, 0.48, -0.09, ..., 0.29]  (similar vector!)
    "car" → [-0.5, 0.1, 0.8, ..., -0.2]  (different vector)

Similar meanings = similar vectors (measured by cosine similarity).

Why embeddings for RAG:
1. **Semantic Search**: Find documents by meaning, not just keywords
2. **Multilingual**: Works across languages
3. **Typo Tolerant**: Handles misspellings
4. **Context Aware**: Understands context and nuance

Example:
    Query: "How do I reset my password?"
    Matches: "Password recovery steps" (even without exact words!)
"""


# ============================================================================
# OpenAI Embeddings Configuration
# ============================================================================

def get_openai_embeddings(
    model: str = "text-embedding-ada-002",
    chunk_size: int = 1000,
) -> OpenAIEmbeddings:
    """
    Get configured OpenAI embeddings.
    
    OpenAI's text-embedding-ada-002:
    - Dimensions: 1536
    - Cost: $0.0001 per 1K tokens
    - Max input: 8191 tokens
    - Best for: General purpose, multilingual
    
    Args:
        model: Embedding model name
        chunk_size: Batch size for embedding (for efficiency)
        
    Returns:
        Configured OpenAI embeddings
        
    Example:
        embeddings = get_openai_embeddings()
        
        # Embed a single text
        vector = embeddings.embed_query("What is Python?")
        print(len(vector))  # 1536
        
        # Embed multiple documents
        docs = ["doc1", "doc2", "doc3"]
        vectors = embeddings.embed_documents(docs)
        print(len(vectors))  # 3
        print(len(vectors[0]))  # 1536
    """
    return OpenAIEmbeddings(
        model=model,
        chunk_size=chunk_size,
        # API key is automatically loaded from environment
    )


# ============================================================================
# Document vs Query Embeddings
# ============================================================================
"""
**Important Distinction:**

1. **embed_documents()**: For indexing documents
   - Used when building vector store
   - Batch processing for efficiency
   - Example: Embedding all chunks of a PDF

2. **embed_query()**: For search queries
   - Used at query time
   - Single embedding
   - Example: Embedding user's question

Some embedding models use different prompts for documents vs queries
to optimize retrieval performance.

Example usage:
    embeddings = get_openai_embeddings()
    
    # Index documents (batch)
    doc_vectors = embeddings.embed_documents([
        "Python is a programming language",
        "JavaScript is used for web development"
    ])
    
    # Search query (single)
    query_vector = embeddings.embed_query("What is Python?")
    
    # Find similarity
    from numpy import dot
    from numpy.linalg import norm
    
    similarity = dot(query_vector, doc_vectors[0]) / (
        norm(query_vector) * norm(doc_vectors[0])
    )
    print(f"Similarity: {similarity}")  # High similarity!
"""


# ============================================================================
# Embedding Caching
# ============================================================================

class CachedEmbeddings:
    """
    Wrapper for embeddings with file-based caching.
    
    Caching saves costs by avoiding re-embedding the same text.
    Especially useful during development and testing.
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        cache_dir: str = None,
    ):
        """
        Initialize cached embeddings.
        
        Args:
            embeddings: Base embeddings instance
            cache_dir: Directory to store cache (default: from settings)
        """
        self.embeddings = embeddings
        self.cache_dir = Path(cache_dir or settings.embeddings_cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for current session
        self._memory_cache = {}
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, text: str) -> List[float]:
        """Load embedding from cache"""
        # Check memory cache first
        if text in self._memory_cache:
            return self._memory_cache[text]
        
        # Check file cache
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                embedding = json.load(f)
                self._memory_cache[text] = embedding
                return embedding
        
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]):
        """Save embedding to cache"""
        # Save to memory cache
        self._memory_cache[text] = embedding
        
        # Save to file cache
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        with open(cache_path, 'w') as f:
            json.dump(embedding, f)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with caching"""
        cached = self._load_from_cache(text)
        if cached is not None:
            return cached
        
        embedding = self.embeddings.embed_query(text)
        self._save_to_cache(text, embedding)
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching"""
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.embeddings.embed_documents(texts_to_embed)
            
            # Save to cache and update results
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                self._save_to_cache(texts[idx], embedding)
                results[idx] = embedding
        
        return results


def get_cached_embeddings() -> CachedEmbeddings:
    """
    Get embeddings with caching enabled.
    
    Returns:
        Cached embeddings instance
        
    Example:
        embeddings = get_cached_embeddings()
        
        # First call: hits API
        vec1 = embeddings.embed_query("What is Python?")
        
        # Second call: uses cache (free!)
        vec2 = embeddings.embed_query("What is Python?")
        
        assert vec1 == vec2  # Same result, no API call
    """
    base_embeddings = get_openai_embeddings()
    return CachedEmbeddings(base_embeddings)


# ============================================================================
# Embedding Dimensions
# ============================================================================

def get_embedding_dimension(embeddings: Embeddings = None) -> int:
    """
    Get embedding dimension for a model.
    
    Useful for:
    - Vector store configuration
    - Dimension validation
    - Storage planning
    
    Args:
        embeddings: Embeddings instance (default: OpenAI)
        
    Returns:
        Embedding dimension
        
    Example:
        dim = get_embedding_dimension()
        print(dim)  # 1536 for OpenAI ada-002
    """
    if embeddings is None:
        embeddings = get_openai_embeddings()
    
    # Embed a test string to get dimension
    test_embedding = embeddings.embed_query("test")
    return len(test_embedding)


# ============================================================================
# Educational Note: Embedding Best Practices
# ============================================================================
"""
**Embedding Model Selection:**

1. **OpenAI text-embedding-ada-002** (used here):
   - Pros: High quality, multilingual, well-supported
   - Cons: Costs money, requires API
   - Use for: Production, high quality needed

2. **OpenAI text-embedding-3-small**:
   - Pros: Cheaper, faster, good quality
   - Cons: Newer, less tested
   - Use for: Cost optimization

3. **OpenAI text-embedding-3-large**:
   - Pros: Best quality
   - Cons: More expensive, slower
   - Use for: When quality is critical

4. **Local models** (HuggingFace):
   - Pros: Free, private, offline
   - Cons: Lower quality, slower
   - Use for: Development, privacy requirements

**Caching Strategy:**

1. **When to cache:**
   - Development and testing
   - Frequently queried documents
   - Static knowledge base

2. **When NOT to cache:**
   - Frequently changing documents
   - User-generated content
   - Real-time data

3. **Cache invalidation:**
   - Clear cache when documents update
   - Periodic cache cleanup
   - Version cache by document version

**Cost Optimization:**

1. **Batch embedding:**
   - Embed multiple documents at once
   - More efficient than one-by-one

2. **Cache aggressively:**
   - Cache both documents and queries
   - Saves significant costs

3. **Choose right model:**
   - ada-002 for most use cases
   - 3-small for cost optimization
   - 3-large only when necessary

**Performance Tips:**

1. **Parallel embedding:**
   - Embed documents in parallel
   - Use async for better throughput

2. **Batch size:**
   - Larger batches = more efficient
   - But don't exceed API limits

3. **Dimension awareness:**
   - Know your embedding dimensions
   - Configure vector store correctly
   - Plan storage requirements

**Common Issues:**

1. **Rate limits:**
   - OpenAI has rate limits
   - Implement retry logic
   - Use exponential backoff

2. **Token limits:**
   - Max 8191 tokens per input
   - Split long documents first
   - Use text splitters

3. **Cost management:**
   - Monitor usage
   - Set budgets
   - Use caching

4. **Quality issues:**
   - Test with your data
   - Compare different models
   - Measure retrieval quality
"""
