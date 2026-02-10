"""
Custom Tools

This module demonstrates LangChain's tool system.
Tools extend LLM capabilities with external functions.

Key concepts:
- @tool decorator for simple tools
- StructuredTool with Pydantic schemas
- Tool binding and execution
- Error handling
"""

from langchain_core.tools import tool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List
import requests
from datetime import datetime


# ============================================================================
# What are Tools?
# ============================================================================
"""
Tools are functions that LLMs can call to perform actions or get information.

Without tools:
    User: "What's the weather in New York?"
    LLM: "I don't have access to real-time weather data."

With tools:
    User: "What's the weather in New York?"
    LLM: [Calls weather_tool("New York")]
    Tool: Returns weather data
    LLM: "The weather in New York is 72°F and sunny."

Common tool types:
- Search: Web search, database lookup
- Calculation: Math, data processing
- API calls: External services
- Retrieval: Document search (retriever as tool)
- Actions: Send email, create ticket
"""


# ============================================================================
# Simple Tools with @tool Decorator
# ============================================================================

@tool
def get_current_time() -> str:
    """
    Get the current time.
    
    Returns:
        Current time as a string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate (e.g., "2 + 2", "10 * 5")
        
    Returns:
        Result of the calculation
        
    Example:
        calculate("2 + 2")  # Returns: "4"
        calculate("10 * 5")  # Returns: "50"
    """
    try:
        # Safe evaluation (only allow numbers and basic operators)
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query
        
    Returns:
        Search results summary
        
    Note:
        This is a mock implementation. In production, integrate with:
        - DuckDuckGo API
        - Google Custom Search
        - Tavily Search
        - SerpAPI
    """
    # Mock implementation
    return f"Mock search results for: {query}\n\nThis would contain real search results in production."


# ============================================================================
# Structured Tools with Pydantic
# ============================================================================

class DocumentSearchInput(BaseModel):
    """Input schema for document search tool"""
    query: str = Field(description="Search query for documents")
    max_results: int = Field(default=3, description="Maximum number of results")


def document_search_function(query: str, max_results: int = 3) -> str:
    """
    Search documents in the knowledge base.
    
    This would integrate with the vector store in production.
    """
    return f"Found {max_results} documents matching '{query}'"


document_search_tool = StructuredTool.from_function(
    func=document_search_function,
    name="document_search",
    description="Search the knowledge base for relevant documents",
    args_schema=DocumentSearchInput,
)


# ============================================================================
# Retriever as Tool
# ============================================================================

def create_retriever_tool(vector_store, name: str = "knowledge_base"):
    """
    Convert a retriever to a tool.
    
    This allows the LLM to search the knowledge base as a tool.
    
    Args:
        vector_store: FAISS vector store
        name: Tool name
        
    Returns:
        Retriever tool
        
    Example:
        from app.rag.vector_store import load_vector_store
        
        vector_store = load_vector_store()
        kb_tool = create_retriever_tool(vector_store)
        
        # LLM can now call this tool
        tools = [kb_tool, calculator, get_current_time]
    """
    from langchain.tools.retriever import create_retriever_tool as lc_create_retriever_tool
    
    retriever = vector_store.as_retriever()
    
    tool = lc_create_retriever_tool(
        retriever,
        name=name,
        description="Search the knowledge base for information. Use this when you need to find specific information from documents.",
    )
    
    return tool


# ============================================================================
# Human-in-the-Loop Tool
# ============================================================================

@tool
def ask_human(question: str) -> str:
    """
    Ask a human for help.
    
    Use this tool when:
    - You don't know the answer
    - The question requires human judgment
    - You need clarification
    
    Args:
        question: Question to ask the human
        
    Returns:
        Human's response
        
    Note:
        In production, this would:
        - Create a ticket
        - Send notification
        - Wait for human response
        - Return the response
    """
    # Mock implementation
    return f"[Human escalation] Question forwarded to support team: {question}"


# ============================================================================
# Tool Execution with Error Handling
# ============================================================================

def execute_tool_safely(tool, tool_input: str) -> str:
    """
    Execute a tool with error handling.
    
    Args:
        tool: Tool to execute
        tool_input: Input for the tool
        
    Returns:
        Tool output or error message
        
    Example:
        result = execute_tool_safely(calculator, "2 + 2")
    """
    try:
        result = tool.run(tool_input)
        return result
    except Exception as e:
        return f"Tool execution error: {str(e)}"


# ============================================================================
# Tool Collection
# ============================================================================

def get_default_tools(vector_store=None) -> List:
    """
    Get default tool collection.
    
    Args:
        vector_store: Optional vector store for retriever tool
        
    Returns:
        List of tools
        
    Example:
        tools = get_default_tools(vector_store)
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
    """
    tools = [
        get_current_time,
        calculate,
        search_web,
        document_search_tool,
        ask_human,
    ]
    
    if vector_store is not None:
        retriever_tool = create_retriever_tool(vector_store)
        tools.append(retriever_tool)
    
    return tools


# ============================================================================
# Educational Note: Tool Concepts
# ============================================================================
"""
**Tool Creation Methods:**

1. **@tool decorator** (simplest):
   ```python
   @tool
   def my_tool(input: str) -> str:
       '''Tool description'''
       return result
   ```
   - Pros: Simple, quick
   - Cons: Limited validation
   - Use for: Simple tools

2. **StructuredTool** (recommended):
   ```python
   class InputSchema(BaseModel):
       param: str = Field(description="...")
   
   tool = StructuredTool.from_function(
       func=my_function,
       args_schema=InputSchema
   )
   ```
   - Pros: Type validation, better docs
   - Cons: More code
   - Use for: Production tools

3. **Custom Tool class**:
   ```python
   class MyTool(BaseTool):
       name = "my_tool"
       description = "..."
       
       def _run(self, input: str) -> str:
           return result
   ```
   - Pros: Full control
   - Cons: Most code
   - Use for: Complex tools

**Tool Binding:**

Bind tools to LLM for function calling:
```python
llm_with_tools = llm.bind_tools(tools)

# LLM can now call tools
response = llm_with_tools.invoke("What's 2 + 2?")
# LLM returns tool call: calculator("2 + 2")
```

**Tool Execution:**

1. **Manual execution**:
   ```python
   tool_calls = extract_tool_calls(llm_response)
   results = [tool.run(call.input) for call in tool_calls]
   ```

2. **Automatic execution** (with agents):
   ```python
   # Agent automatically executes tools
   agent.invoke("What's the weather?")
   # Agent calls weather_tool, gets result, responds
   ```

**Best Practices:**

1. **Clear descriptions**:
   - LLM uses description to decide when to call
   - Be specific about what the tool does
   - Include examples in docstring

2. **Input validation**:
   - Use Pydantic schemas
   - Validate before execution
   - Return clear error messages

3. **Error handling**:
   - Always wrap in try-except
   - Return user-friendly errors
   - Log errors for debugging

4. **Idempotency**:
   - Same input = same output
   - No side effects when possible
   - Safe to retry

5. **Performance**:
   - Keep tools fast
   - Cache when possible
   - Timeout long operations

**Common Tool Patterns:**

1. **Search tools**:
   - Web search
   - Database search
   - Document retrieval

2. **Calculation tools**:
   - Math operations
   - Data processing
   - Statistics

3. **API tools**:
   - Weather API
   - Stock prices
   - External services

4. **Action tools**:
   - Send email
   - Create ticket
   - Update database

5. **Human-in-the-loop**:
   - Escalate to human
   - Request approval
   - Get clarification

**Tool Selection:**

LLM decides which tool to call based on:
- Tool description
- User's question
- Context

Improve tool selection:
- Clear, specific descriptions
- Good examples
- Distinct purposes
- Test with real queries
"""
