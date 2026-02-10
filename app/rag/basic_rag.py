"""
Basic RAG Chain

This module implements the fundamental RAG (Retrieval-Augmented Generation) pattern.
RAG combines retrieval with generation for knowledge-grounded responses.

Key concepts:
- LCEL (LangChain Expression Language)
- Pipe operator (|) for chain composition
- RunnablePassthrough for data flow
- Document formatting
- Source tracking
"""

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List, Dict, Any

from app.prompts.templates import RAG_PROMPT_TEMPLATE
from app.models.llm_config import get_openai_chat_model
from app.rag.vector_store import FAISS


# ============================================================================
# What is RAG?
# ============================================================================
"""
RAG (Retrieval-Augmented Generation) is a pattern that combines:
1. **Retrieval**: Find relevant documents from a knowledge base
2. **Augmentation**: Add retrieved docs to the prompt
3. **Generation**: LLM generates answer based on retrieved context

Traditional LLM:
    User: "What is our refund policy?"
    LLM: "I don't have information about your specific refund policy."
    
RAG:
    User: "What is our refund policy?"
    System: [Retrieves policy document]
    LLM: "According to the policy, you can get a full refund within 30 days..."

Benefits:
- Grounds responses in factual data
- Reduces hallucinations
- Enables domain-specific knowledge
- Can cite sources
- Updatable knowledge (just update documents)
"""


# ============================================================================
# Document Formatting Helper
# ============================================================================

def format_docs(docs: List[Document]) -> str:
    """
    Format documents for inclusion in prompt.
    
    Combines multiple documents into a single string with clear separation.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        Formatted string with all document content
        
    Example:
        docs = retriever.get_relevant_documents("query")
        formatted = format_docs(docs)
        # Returns:
        # Document 1:
        # [content]
        # 
        # Document 2:
        # [content]
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (Page {page})" if page else ""
        
        formatted.append(
            f"Document {i} [Source: {source}{page_info}]:\n{doc.page_content}"
        )
    
    return "\n\n".join(formatted)


# ============================================================================
# Basic RAG Chain
# ============================================================================

def create_basic_rag_chain(
    vector_store: FAISS,
    llm=None,
    k: int = 4,
):
    """
    Create a basic RAG chain using LCEL.
    
    This demonstrates the fundamental RAG pattern with LCEL syntax.
    
    Chain flow:
        1. User question comes in
        2. Retrieve relevant documents
        3. Format documents
        4. Create prompt with question + context
        5. LLM generates answer
        6. Parse output to string
    
    Args:
        vector_store: FAISS vector store for retrieval
        llm: Language model (default: OpenAI)
        k: Number of documents to retrieve
        
    Returns:
        Runnable chain
        
    Example:
        chain = create_basic_rag_chain(vector_store)
        
        response = chain.invoke({
            "question": "How do I reset my password?"
        })
        
        print(response)
        # "To reset your password, follow these steps: ..."
    
    LCEL Explanation:
        The pipe operator (|) chains runnables together.
        Data flows left to right through the chain.
        
        RunnableParallel executes multiple runnables concurrently:
        - "context": retriever | format_docs
        - "question": RunnablePassthrough()
        
        This creates a dict: {"context": "...", "question": "..."}
        which is passed to the prompt.
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0.3)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # ========================================================================
    # LCEL Chain Construction
    # ========================================================================
    # This is the core RAG chain using LCEL syntax
    
    chain = (
        # Step 1: Prepare inputs in parallel
        RunnableParallel({
            # Retrieve and format documents
            "context": retriever | format_docs,
            # Pass through the question unchanged
            "question": RunnablePassthrough(),
        })
        # Step 2: Create prompt from inputs
        | RAG_PROMPT_TEMPLATE
        # Step 3: Generate response with LLM
        | llm
        # Step 4: Parse output to string
        | StrOutputParser()
    )
    
    return chain


# ============================================================================
# RAG Chain with Metadata
# ============================================================================

def create_rag_chain_with_metadata(
    vector_store: FAISS,
    llm=None,
    k: int = 4,
):
    """
    Create RAG chain that returns answer with source metadata.
    
    This version returns both the answer and the source documents.
    
    Args:
        vector_store: FAISS vector store
        llm: Language model
        k: Number of documents to retrieve
        
    Returns:
        Runnable chain that returns dict with answer and sources
        
    Example:
        chain = create_rag_chain_with_metadata(vector_store)
        
        result = chain.invoke({
            "question": "How do I reset my password?"
        })
        
        print(result["answer"])
        print(result["sources"])
        # [
        #   {"source": "manual.pdf", "page": 5, "content": "..."},
        #   ...
        # ]
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0.3)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    def process_with_sources(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question and return answer with sources.
        
        This function:
        1. Retrieves documents
        2. Generates answer
        3. Returns both answer and source metadata
        """
        question = inputs["question"]
        
        # Retrieve documents
        docs = retriever.get_relevant_documents(question)
        
        # Format context
        context = format_docs(docs)
        
        # Generate answer
        prompt_value = RAG_PROMPT_TEMPLATE.invoke({
            "context": context,
            "question": question,
        })
        
        answer = llm.invoke(prompt_value)
        
        # Extract source metadata
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:200] + "...",  # Preview
            }
            for doc in docs
        ]
        
        return {
            "answer": answer.content,
            "sources": sources,
            "num_sources": len(sources),
        }
    
    # Chain using RunnableLambda for custom logic
    from langchain_core.runnables import RunnableLambda
    
    chain = RunnableLambda(process_with_sources)
    
    return chain


# ============================================================================
# Educational Note: LCEL Concepts
# ============================================================================
"""
**LCEL (LangChain Expression Language) Key Concepts:**

1. **Pipe Operator (|)**:
   - Chains runnables together
   - Data flows left to right
   - Example: retriever | format_docs | prompt | llm
   
2. **RunnablePassthrough**:
   - Passes input through unchanged
   - Useful for preserving data in parallel chains
   - Example: {"question": RunnablePassthrough()}

3. **RunnableParallel**:
   - Executes multiple runnables concurrently
   - Returns dict with results
   - Example: {"context": retriever, "question": passthrough}

4. **RunnableLambda**:
   - Wraps custom functions as runnables
   - Enables custom logic in chains
   - Example: RunnableLambda(custom_function)

**Why LCEL?**

1. **Composability**: Easy to build complex chains from simple parts
2. **Readability**: Clear data flow
3. **Streaming**: Built-in streaming support
4. **Async**: Automatic async support
5. **Debugging**: Easy to inspect intermediate steps

**LCEL vs Traditional Chains:**

Traditional (deprecated):
    from langchain.chains import RetrievalQA
    chain = RetrievalQA.from_chain_type(...)

LCEL (modern):
    chain = retriever | format_docs | prompt | llm | parser

LCEL is:
- More flexible
- Easier to customize
- Better performance
- Actively maintained

**Best Practices:**

1. **Use LCEL for new code**: It's the modern approach
2. **Break chains into steps**: Easier to debug
3. **Use RunnableParallel**: For concurrent operations
4. **Add type hints**: Helps with debugging
5. **Test each component**: Before chaining together

**Common Patterns:**

1. **Simple chain**:
   prompt | llm | parser

2. **RAG chain**:
   retriever | format | prompt | llm | parser

3. **Parallel retrieval**:
   RunnableParallel({
       "docs": retriever,
       "summary": summarizer
   }) | combine | llm

4. **Conditional routing**:
   RunnableBranch(
       (condition1, chain1),
       (condition2, chain2),
       default_chain
   )
"""
