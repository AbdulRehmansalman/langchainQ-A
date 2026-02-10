"""
Text Splitters

This module demonstrates LangChain's text splitting capabilities.
Text splitters break documents into chunks for embedding and retrieval.

Key concepts:
- RecursiveCharacterTextSplitter: Smart splitting
- TokenTextSplitter: Token-aware splitting
- Chunk size and overlap strategies
- Metadata preservation
"""

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document
from typing import List, Optional
import tiktoken


# ============================================================================
# Why Text Splitting?
# ============================================================================
"""
Text splitting is crucial for RAG because:

1. **Embedding Models have limits**:
   - Most embedding models have max input length (e.g., 8191 tokens for OpenAI)
   - Documents often exceed this limit
   - Need to split into smaller chunks

2. **Retrieval Precision**:
   - Smaller chunks = more precise retrieval
   - Large chunks may contain irrelevant information
   - Optimal chunk size balances precision and context

3. **Context Window**:
   - LLMs have context limits (e.g., 4K, 8K, 16K tokens)
   - Retrieved chunks must fit in context
   - Smaller chunks allow retrieving more documents

4. **Semantic Coherence**:
   - Chunks should be semantically meaningful
   - Don't split mid-sentence or mid-paragraph
   - Preserve context within chunks
"""


# ============================================================================
# Recursive Character Text Splitter (Recommended)
# ============================================================================

def get_recursive_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Get recursive character text splitter.
    
    This is the RECOMMENDED splitter for most use cases.
    It tries to split on paragraphs, then sentences, then words.
    
    Args:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks (preserves context)
        separators: Custom separators (default: ["\n\n", "\n", " ", ""])
        
    Returns:
        Configured text splitter
        
    Example:
        splitter = get_recursive_text_splitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
    
    How it works:
        1. Try to split on "\n\n" (paragraphs)
        2. If chunks still too large, split on "\n" (lines)
        3. If still too large, split on " " (words)
        4. If still too large, split on "" (characters)
    
    This preserves semantic structure as much as possible.
    """
    if separators is None:
        # Default separators in order of preference
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            " ",     # Words
            "",      # Characters
        ]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


# ============================================================================
# Token-Based Text Splitter
# ============================================================================

def get_token_text_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> TokenTextSplitter:
    """
    Get token-based text splitter.
    
    Splits based on token count (not characters).
    More accurate for LLM context limits.
    
    Args:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap in tokens
        encoding_name: Tokenizer encoding (cl100k_base for GPT-3.5/4)
        
    Returns:
        Configured token splitter
        
    Example:
        splitter = get_token_text_splitter(chunk_size=500)
        chunks = splitter.split_documents(documents)
    
    When to use:
        - Need precise token counts
        - Working with token limits
        - Cost optimization (tokens = cost)
    
    Encodings:
        - cl100k_base: GPT-3.5-turbo, GPT-4
        - p50k_base: GPT-3 (davinci, curie)
        - r50k_base: GPT-3 (ada, babbage)
    """
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )


# ============================================================================
# Character Text Splitter (Simple)
# ============================================================================

def get_character_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n",
) -> CharacterTextSplitter:
    """
    Get simple character text splitter.
    
    Splits on a single separator.
    Simpler but less flexible than RecursiveCharacterTextSplitter.
    
    Args:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap in characters
        separator: Separator to split on
        
    Returns:
        Configured character splitter
        
    Example:
        # Split on paragraphs
        splitter = get_character_text_splitter(separator="\n\n")
        chunks = splitter.split_documents(documents)
    """
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
        length_function=len,
    )


# ============================================================================
# Chunk Size Optimization
# ============================================================================

def calculate_optimal_chunk_size(
    avg_query_length: int = 50,
    max_chunks_to_retrieve: int = 4,
    llm_context_window: int = 4096,
    buffer_tokens: int = 1000,
) -> int:
    """
    Calculate optimal chunk size based on constraints.
    
    Formula:
        chunk_size = (context_window - query - buffer) / num_chunks
    
    Args:
        avg_query_length: Average query length in tokens
        max_chunks_to_retrieve: Max chunks to retrieve
        llm_context_window: LLM's context window size
        buffer_tokens: Buffer for prompt template, etc.
        
    Returns:
        Recommended chunk size in tokens
        
    Example:
        chunk_size = calculate_optimal_chunk_size(
            avg_query_length=50,
            max_chunks_to_retrieve=4,
            llm_context_window=4096,
            buffer_tokens=1000
        )
        # Returns: ~750 tokens per chunk
    
    Reasoning:
        - Context window: 4096 tokens
        - Query: 50 tokens
        - Buffer (prompt, etc.): 1000 tokens
        - Available: 4096 - 50 - 1000 = 3046 tokens
        - Per chunk: 3046 / 4 = 761 tokens
    """
    available_tokens = llm_context_window - avg_query_length - buffer_tokens
    chunk_size = available_tokens // max_chunks_to_retrieve
    
    # Ensure reasonable bounds
    chunk_size = max(100, min(chunk_size, 2000))
    
    return chunk_size


# ============================================================================
# Split Documents with Metadata Preservation
# ============================================================================

def split_documents_with_metadata(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents while preserving metadata.
    
    Each chunk inherits the original document's metadata.
    Adds chunk-specific metadata (chunk_index).
    
    Args:
        documents: List of documents to split
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with metadata
        
    Example:
        docs = load_pdf("manual.pdf")
        chunks = split_documents_with_metadata(docs, chunk_size=1000)
        
        for chunk in chunks:
            print(f"Source: {chunk.metadata['source']}")
            print(f"Page: {chunk.metadata['page']}")
            print(f"Chunk: {chunk.metadata['chunk_index']}")
    """
    splitter = get_recursive_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    return chunks


# ============================================================================
# Educational Note: Chunking Strategies
# ============================================================================
"""
**Chunk Size Guidelines:**

1. **Small chunks (200-500 tokens)**:
   - Pros: Precise retrieval, more documents fit in context
   - Cons: May lose context, more chunks to manage
   - Use for: Q&A, fact lookup, specific information

2. **Medium chunks (500-1000 tokens)**:
   - Pros: Good balance of precision and context
   - Cons: None major
   - Use for: General RAG, most applications (RECOMMENDED)

3. **Large chunks (1000-2000 tokens)**:
   - Pros: More context, fewer chunks
   - Cons: Less precise retrieval, fewer chunks fit in context
   - Use for: Summarization, when context is critical

**Chunk Overlap Guidelines:**

Overlap preserves context across chunk boundaries.

- **No overlap (0)**:
  - May lose context at boundaries
  - Faster processing
  - Use when chunks are independent

- **Small overlap (50-100 tokens)**:
  - Minimal context preservation
  - Good for well-structured documents

- **Medium overlap (100-200 tokens)**:
  - Good context preservation (RECOMMENDED)
  - Balances redundancy and context

- **Large overlap (200-500 tokens)**:
  - Maximum context preservation
  - More redundancy
  - Use for critical applications

**Choosing a Splitter:**

1. **RecursiveCharacterTextSplitter** (RECOMMENDED):
   - Best for most use cases
   - Preserves semantic structure
   - Handles various document types

2. **TokenTextSplitter**:
   - When token count matters
   - Cost optimization
   - Precise context management

3. **CharacterTextSplitter**:
   - Simple use cases
   - Well-structured documents
   - Single separator works well

**Best Practices:**

1. **Test different chunk sizes**:
   - Optimal size depends on your data and queries
   - Measure retrieval quality
   - Adjust based on results

2. **Consider document structure**:
   - Code: Split on functions/classes
   - Articles: Split on sections
   - Conversations: Split on exchanges

3. **Preserve metadata**:
   - Source file
   - Page numbers
   - Chunk index
   - Any custom metadata

4. **Monitor chunk quality**:
   - Check if chunks are semantically coherent
   - Ensure overlap captures context
   - Verify metadata is preserved

**Common Issues:**

1. **Chunks too small**:
   - Lose context
   - Poor retrieval quality
   - Solution: Increase chunk_size

2. **Chunks too large**:
   - Can't fit enough in context
   - Less precise retrieval
   - Solution: Decrease chunk_size

3. **Lost context at boundaries**:
   - Information split across chunks
   - Solution: Increase chunk_overlap

4. **Too much redundancy**:
   - Same info in multiple chunks
   - Wastes storage/compute
   - Solution: Decrease chunk_overlap
"""
