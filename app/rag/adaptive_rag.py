"""
Adaptive RAG

This module implements adaptive RAG with intelligent routing.
It decides whether to use RAG or answer directly based on the question.

Key concepts:
- RunnableBranch for conditional routing
- Routing decision logic
- Fallback handling
- Multiple RAG strategies
"""

from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, Literal

from app.prompts.templates import ROUTING_PROMPT, RAG_PROMPT_TEMPLATE, DIRECT_ANSWER_PROMPT
from app.models.llm_config import get_openai_chat_model
from app.rag.vector_store import FAISS
from app.rag.basic_rag import format_docs, create_basic_rag_chain
from app.rag.conversational_rag import create_conversational_rag_chain
from app.parsers.output_parsers import SimpleRoutingParser


# ============================================================================
# What is Adaptive RAG?
# ============================================================================
"""
Adaptive RAG intelligently routes queries to the best answering strategy.

Problem with always using RAG:
- General knowledge questions don't need retrieval
- "What is Python?" → Wastes time retrieving docs
- "2 + 2 = ?" → No need for documents

Adaptive RAG solution:
1. Analyze the question
2. Decide: RAG or Direct answer?
3. Route to appropriate chain

Example routing decisions:
- "What is Python?" → DIRECT (general knowledge)
- "What is our refund policy?" → RAG (company-specific)
- "How do I reset my password?" → RAG (documentation)
- "What's the capital of France?" → DIRECT (general knowledge)

Benefits:
- Faster for general questions (no retrieval)
- Cheaper (fewer embeddings)
- Better answers (use right strategy)
- More efficient system
"""


# ============================================================================
# Routing Decision Function
# ============================================================================

def make_routing_decision(
    question: str,
    llm=None,
) -> Literal["RAG", "DIRECT"]:
    """
    Decide whether to use RAG or direct answer.
    
    Uses LLM to analyze question and decide routing.
    
    Args:
        question: User's question
        llm: Language model for decision
        
    Returns:
        "RAG" or "DIRECT"
        
    Example:
        decision = make_routing_decision("What is Python?")
        # Returns: "DIRECT"
        
        decision = make_routing_decision("What is our refund policy?")
        # Returns: "RAG"
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0)
    
    # Use routing prompt
    chain = ROUTING_PROMPT | llm | SimpleRoutingParser()
    
    decision = chain.invoke({"question": question})
    
    return decision


# ============================================================================
# Adaptive RAG Chain
# ============================================================================

def create_adaptive_rag_chain(
    vector_store: FAISS,
    llm=None,
    k: int = 4,
):
    """
    Create adaptive RAG chain with intelligent routing.
    
    This chain:
    1. Analyzes the question
    2. Decides RAG vs Direct
    3. Routes to appropriate chain
    4. Returns answer
    
    Args:
        vector_store: FAISS vector store
        llm: Language model
        k: Number of documents for RAG
        
    Returns:
        Adaptive RAG chain
        
    Example:
        chain = create_adaptive_rag_chain(vector_store)
        
        # General knowledge → Direct answer
        response1 = chain.invoke({"question": "What is Python?"})
        
        # Company-specific → RAG
        response2 = chain.invoke({"question": "What is our refund policy?"})
    
    LCEL Explanation:
        RunnableBranch enables conditional routing:
        - First argument: (condition, chain) pairs
        - Last argument: default chain
        
        It evaluates conditions in order and runs the first matching chain.
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0.3)
    
    # Create RAG chain
    rag_chain = create_basic_rag_chain(vector_store, llm, k)
    
    # Create direct answer chain
    direct_chain = DIRECT_ANSWER_PROMPT | llm | StrOutputParser()
    
    # Routing function
    def route_question(inputs: Dict[str, Any]) -> str:
        """Determine routing decision"""
        question = inputs["question"]
        decision = make_routing_decision(question, llm)
        return decision
    
    # Adaptive chain with RunnableBranch
    adaptive_chain = RunnableBranch(
        # If RAG decision, use RAG chain
        (
            lambda x: route_question(x) == "RAG",
            rag_chain
        ),
        # Otherwise (DIRECT), use direct answer chain
        direct_chain
    )
    
    return adaptive_chain


# ============================================================================
# Adaptive RAG with Fallback
# ============================================================================

def create_adaptive_rag_with_fallback(
    vector_store: FAISS,
    llm=None,
    k: int = 4,
):
    """
    Create adaptive RAG with fallback handling.
    
    This version adds fallback logic:
    - If RAG fails (no relevant docs), fall back to direct answer
    - If direct answer fails, try RAG
    - Ensures we always return something
    
    Args:
        vector_store: FAISS vector store
        llm: Language model
        k: Number of documents for RAG
        
    Returns:
        Adaptive RAG chain with fallbacks
        
    Example:
        chain = create_adaptive_rag_with_fallback(vector_store)
        
        # Even if routing is wrong, we get an answer
        response = chain.invoke({"question": "..."})
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0.3)
    
    # Create chains
    rag_chain = create_basic_rag_chain(vector_store, llm, k)
    direct_chain = DIRECT_ANSWER_PROMPT | llm | StrOutputParser()
    
    # Add fallbacks
    rag_with_fallback = rag_chain.with_fallbacks([direct_chain])
    direct_with_fallback = direct_chain.with_fallbacks([rag_chain])
    
    # Routing function
    def route_question(inputs: Dict[str, Any]) -> str:
        question = inputs["question"]
        return make_routing_decision(question, llm)
    
    # Adaptive chain
    adaptive_chain = RunnableBranch(
        (lambda x: route_question(x) == "RAG", rag_with_fallback),
        direct_with_fallback
    )
    
    return adaptive_chain


# ============================================================================
# Adaptive Conversational RAG
# ============================================================================

def create_adaptive_conversational_rag(
    vector_store: FAISS,
    llm=None,
    k: int = 4,
):
    """
    Create adaptive RAG with conversation support.
    
    Combines:
    - Adaptive routing (RAG vs Direct)
    - Conversational memory
    - Fallback handling
    
    This is the most sophisticated RAG pattern in this system.
    
    Args:
        vector_store: FAISS vector store
        llm: Language model
        k: Number of documents for RAG
        
    Returns:
        Adaptive conversational RAG chain
        
    Example:
        chain = create_adaptive_conversational_rag(vector_store)
        
        # First question (general knowledge)
        response1 = chain.invoke({
            "question": "What is Python?",
            "chat_history": []
        })
        
        # Follow-up (company-specific, uses RAG)
        response2 = chain.invoke({
            "question": "Do we use it in our products?",
            "chat_history": [
                HumanMessage("What is Python?"),
                AIMessage(response1)
            ]
        })
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0.3)
    
    # Create conversational RAG
    conv_rag_chain = create_conversational_rag_chain(vector_store, llm, k)
    
    # Create conversational direct answer
    conv_direct_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on your general knowledge."),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])
    conv_direct_chain = conv_direct_prompt | llm | StrOutputParser()
    
    # Routing function
    def route_with_history(inputs: Dict[str, Any]) -> str:
        question = inputs["question"]
        # Consider history in routing decision
        # For now, simple routing based on question
        return make_routing_decision(question, llm)
    
    # Adaptive conversational chain
    adaptive_chain = RunnableBranch(
        (lambda x: route_with_history(x) == "RAG", conv_rag_chain),
        conv_direct_chain
    )
    
    return adaptive_chain


# ============================================================================
# Educational Note: Adaptive RAG Concepts
# ============================================================================
"""
**RunnableBranch Explained:**

RunnableBranch is LCEL's way to do conditional routing.

Syntax:
    RunnableBranch(
        (condition1, chain1),
        (condition2, chain2),
        default_chain
    )

Example:
    RunnableBranch(
        (lambda x: x["score"] > 0.8, high_confidence_chain),
        (lambda x: x["score"] > 0.5, medium_confidence_chain),
        low_confidence_chain
    )

How it works:
1. Evaluates conditions in order
2. Runs first matching chain
3. If no match, runs default chain

**Routing Strategies:**

1. **LLM-based** (used here):
   - Pros: Flexible, understands nuance
   - Cons: Slower, costs tokens
   - Use for: Complex routing logic

2. **Rule-based**:
   - Pros: Fast, free, deterministic
   - Cons: Brittle, needs maintenance
   - Use for: Simple, clear rules

3. **Embedding-based**:
   - Pros: Fast, semantic understanding
   - Cons: Needs training data
   - Use for: Category classification

4. **Hybrid**:
   - Combine multiple strategies
   - Best accuracy
   - More complex

**Fallback Strategies:**

1. **Chain-level fallback**:
   ```python
   chain.with_fallbacks([fallback_chain])
   ```

2. **Model-level fallback**:
   ```python
   llm.with_fallbacks([backup_llm])
   ```

3. **Custom fallback**:
   ```python
   try:
       result = chain.invoke(input)
   except Exception:
       result = fallback_chain.invoke(input)
   ```

**Best Practices:**

1. **Test routing decisions**:
   - Collect real queries
   - Measure routing accuracy
   - Adjust prompts/logic

2. **Monitor routing**:
   - Track RAG vs Direct ratio
   - Identify misrouted queries
   - Improve over time

3. **Add fallbacks**:
   - Always have a backup
   - Graceful degradation
   - Never fail completely

4. **Optimize for speed**:
   - Cache routing decisions
   - Use fast routing when possible
   - Parallel operations

**Common Patterns:**

1. **Simple adaptive**:
   - RAG vs Direct
   - Based on question type

2. **Multi-strategy**:
   - RAG vs Direct vs Search vs Calculator
   - Based on intent

3. **Confidence-based**:
   - High confidence → Direct
   - Low confidence → RAG
   - Very low → Human escalation

4. **Hybrid**:
   - Try Direct first
   - If low confidence, try RAG
   - Best of both

**Performance Optimization:**

1. **Cache routing decisions**:
   - Same question = same routing
   - Significant speedup

2. **Parallel routing**:
   - Make decision while preparing chains
   - Reduce latency

3. **Fast routing**:
   - Use simple rules when possible
   - LLM only for complex cases

4. **Batch routing**:
   - Route multiple questions together
   - More efficient
"""
