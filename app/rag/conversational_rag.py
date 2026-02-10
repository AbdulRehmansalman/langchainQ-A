"""
Conversational RAG

This module implements conversational RAG with memory.
It maintains conversation history for context-aware responses.

Key concepts:
- History-aware retrieval
- Context reformulation
- Memory integration
- RunnablePassthrough for history
"""

from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any

from app.prompts.templates import RAG_PROMPT_TEMPLATE
from app.models.llm_config import get_openai_chat_model
from app.rag.vector_store import FAISS
from app.rag.basic_rag import format_docs


# ============================================================================
# What is Conversational RAG?
# ============================================================================
"""
Conversational RAG extends basic RAG with conversation history.

Basic RAG:
    User: "What is the refund policy?"
    System: [Retrieves policy] "You can get a refund within 30 days..."
    
    User: "What about international orders?"
    System: [Doesn't know "it" refers to refunds] "What about them?"

Conversational RAG:
    User: "What is the refund policy?"
    System: [Retrieves policy] "You can get a refund within 30 days..."
    
    User: "What about international orders?"
    System: [Knows context, reformulates: "refund policy for international orders"]
            [Retrieves relevant docs] "For international orders, refunds..."

Key improvements:
1. **Context awareness**: Understands pronouns and references
2. **Query reformulation**: Converts follow-ups to standalone questions
3. **Better retrieval**: More accurate document retrieval
4. **Natural conversation**: Feels like talking to a human
"""


# ============================================================================
# History-Aware Retriever
# ============================================================================

def create_history_aware_retriever(
    vector_store: FAISS,
    llm=None,
    k: int = 4,
):
    """
    Create a retriever that considers conversation history.
    
    This retriever:
    1. Takes question + chat history
    2. Reformulates question to be standalone
    3. Retrieves documents based on reformulated question
    
    Args:
        vector_store: FAISS vector store
        llm: Language model for reformulation
        k: Number of documents to retrieve
        
    Returns:
        History-aware retriever chain
        
    Example:
        retriever = create_history_aware_retriever(vector_store)
        
        docs = retriever.invoke({
            "question": "What about international orders?",
            "chat_history": [
                HumanMessage("What is the refund policy?"),
                AIMessage("You can get a refund within 30 days...")
            ]
        })
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0)
    
    # Prompt for query reformulation
    reformulation_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question.
        
Do NOT answer the question, just reformulate it to include necessary context from the chat history.

Examples:
Chat history: "What is Python?" / "Python is a programming language"
Follow-up: "What is it used for?"
Standalone: "What is Python used for?"

Chat history: "Tell me about the refund policy" / "You can get a refund within 30 days"
Follow-up: "What about international orders?"
Standalone: "What is the refund policy for international orders?"
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    # Base retriever
    base_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # Chain: reformulate question → retrieve documents
    def retrieve_with_history(inputs: Dict[str, Any]) -> List:
        """Reformulate question with history and retrieve"""
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        
        if not chat_history:
            # No history, use question as-is
            standalone_question = question
        else:
            # Reformulate with history
            reformulation_result = (reformulation_prompt | llm | StrOutputParser()).invoke({
                "question": question,
                "chat_history": chat_history,
            })
            standalone_question = reformulation_result
        
        # Retrieve with standalone question
        docs = base_retriever.get_relevant_documents(standalone_question)
        return docs
    
    return RunnableLambda(retrieve_with_history)


# ============================================================================
# Conversational RAG Chain
# ============================================================================

def create_conversational_rag_chain(
    vector_store: FAISS,
    llm=None,
    k: int = 4,
):
    """
    Create a conversational RAG chain with memory.
    
    This chain:
    1. Reformulates question with history
    2. Retrieves relevant documents
    3. Generates answer with full context
    
    Args:
        vector_store: FAISS vector store
        llm: Language model
        k: Number of documents to retrieve
        
    Returns:
        Conversational RAG chain
        
    Example:
        chain = create_conversational_rag_chain(vector_store)
        
        # First question
        response1 = chain.invoke({
            "question": "What is the refund policy?",
            "chat_history": []
        })
        
        # Follow-up (with history)
        response2 = chain.invoke({
            "question": "What about international orders?",
            "chat_history": [
                HumanMessage("What is the refund policy?"),
                AIMessage(response1)
            ]
        })
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0.3)
    
    # Create history-aware retriever
    history_retriever = create_history_aware_retriever(vector_store, llm, k)
    
    # Conversational prompt (includes history)
    conversational_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question based on the provided context and conversation history.

Context:
{context}

Be conversational and refer to previous messages when relevant."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    # Build chain
    chain = (
        RunnableParallel({
            "context": history_retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        })
        | conversational_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# ============================================================================
# Conversational RAG with Memory Management
# ============================================================================

class ConversationalRAGWithMemory:
    """
    Conversational RAG with built-in memory management.
    
    This class maintains conversation history automatically.
    """
    
    def __init__(
        self,
        vector_store: FAISS,
        llm=None,
        k: int = 4,
        max_history: int = 10,
    ):
        """
        Initialize conversational RAG with memory.
        
        Args:
            vector_store: FAISS vector store
            llm: Language model
            k: Number of documents to retrieve
            max_history: Max messages to keep in history
        """
        self.chain = create_conversational_rag_chain(vector_store, llm, k)
        self.chat_history: List = []
        self.max_history = max_history
    
    def invoke(self, question: str) -> str:
        """
        Ask a question with automatic history management.
        
        Args:
            question: User's question
            
        Returns:
            AI's response
            
        Example:
            rag = ConversationalRAGWithMemory(vector_store)
            
            response1 = rag.invoke("What is the refund policy?")
            print(response1)
            
            # History is maintained automatically
            response2 = rag.invoke("What about international orders?")
            print(response2)  # Knows we're talking about refunds
        """
        # Invoke chain with current history
        response = self.chain.invoke({
            "question": question,
            "chat_history": self.chat_history,
        })
        
        # Update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))
        
        # Trim history if too long
        if len(self.chat_history) > self.max_history * 2:  # *2 for human+ai pairs
            self.chat_history = self.chat_history[-(self.max_history * 2):]
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []
    
    def get_history(self) -> List:
        """Get current conversation history"""
        return self.chat_history.copy()


# ============================================================================
# Educational Note: Conversational RAG Patterns
# ============================================================================
"""
**Key Concepts:**

1. **Query Reformulation**:
   - Converts follow-up questions to standalone questions
   - Essential for accurate retrieval
   - Uses LLM to understand context
   
   Example:
   History: "What is Python?" → "Python is a language"
   Follow-up: "What is it used for?"
   Reformulated: "What is Python used for?"

2. **History Management**:
   - Keep recent messages (e.g., last 10)
   - Trim old messages to save tokens
   - Balance context vs token usage

3. **MessagesPlaceholder**:
   - Dynamically inject chat history
   - Supports variable-length history
   - Clean prompt templates

**Best Practices:**

1. **Limit history length**:
   - Too much history = too many tokens
   - Keep last 5-10 exchanges
   - Summarize older history if needed

2. **Reformulation is critical**:
   - Don't skip this step
   - Poor reformulation = poor retrieval
   - Test with real conversations

3. **Handle empty history**:
   - First question has no history
   - Don't reformulate if history is empty
   - Graceful fallback

4. **Memory persistence**:
   - Store history in database
   - Associate with user/session
   - Load on conversation resume

**Common Patterns:**

1. **Stateless** (pass history each time):
   ```python
   response = chain.invoke({
       "question": question,
       "chat_history": history
   })
   ```

2. **Stateful** (built-in memory):
   ```python
   rag = ConversationalRAGWithMemory(vector_store)
   response = rag.invoke(question)  # History automatic
   ```

3. **Database-backed**:
   ```python
   # Load history from database
   history = load_history_from_db(session_id)
   response = chain.invoke({"question": q, "chat_history": history})
   # Save to database
   save_history_to_db(session_id, history + [question, response])
   ```

**Performance Tips:**

1. **Cache reformulations**:
   - Same history + question = same reformulation
   - Significant speedup

2. **Parallel operations**:
   - Reformulate and retrieve in parallel when possible
   - Use RunnableParallel

3. **Optimize history size**:
   - Fewer messages = fewer tokens = faster
   - Find minimum needed for quality

**Common Issues:**

1. **Lost context**:
   - History too short
   - Reformulation failed
   - Solution: Increase history, improve reformulation

2. **Too slow**:
   - History too long
   - Too many tokens
   - Solution: Trim history, summarize

3. **Poor reformulation**:
   - LLM not understanding context
   - Solution: Better prompt, examples, higher quality LLM
"""
