"""
Output Parsers

This module demonstrates LangChain's output parsing capabilities.
Output parsers transform LLM string outputs into structured data.

Key concepts:
- StrOutputParser: Basic string extraction
- PydanticOutputParser: Parse into Pydantic models
- with_structured_output(): Modern structured output
- OutputFixingParser: Auto-fix malformed outputs
- RetryOutputParser: Retry on parse failures
"""

from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from enum import Enum


# ============================================================================
# Basic String Output Parser
# ============================================================================
"""
StrOutputParser is the simplest parser - extracts string content from LLM response.

Usage:
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": "What is Python?"})
    # result is a string: "Python is a programming language..."
"""

# Create a reusable instance
string_parser = StrOutputParser()


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class RoutingDecision(BaseModel):
    """
    Routing decision for adaptive RAG.
    
    Determines whether to use RAG or direct answer.
    """
    decision: Literal["RAG", "DIRECT"] = Field(
        description="Whether to use RAG (knowledge base) or DIRECT (general knowledge)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the decision (0-1)"
    )
    reasoning: str = Field(
        description="Brief explanation of why this decision was made"
    )
    
    @validator("decision")
    def validate_decision(cls, v):
        """Ensure decision is uppercase"""
        return v.upper()


class SourceCitation(BaseModel):
    """Source document citation"""
    filename: str = Field(description="Source document filename")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant this source is (0-1)"
    )


class QAResponse(BaseModel):
    """
    Structured Q&A response with sources and metadata.
    
    This is the main output format for the Q&A system.
    """
    answer: str = Field(description="The answer to the question")
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="List of source documents used"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the answer (0-1)"
    )
    requires_followup: bool = Field(
        default=False,
        description="Whether the question requires follow-up"
    )
    suggested_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )


class DocumentRelevance(BaseModel):
    """
    Document relevance assessment for Self-RAG.
    
    Used to evaluate if a retrieved document is relevant.
    """
    is_relevant: bool = Field(description="Whether the document is relevant")
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1)"
    )
    reasoning: str = Field(description="Why this document is/isn't relevant")


# ============================================================================
# Pydantic Output Parsers
# ============================================================================

# Create parsers for each model
routing_parser = PydanticOutputParser(pydantic_object=RoutingDecision)
qa_response_parser = PydanticOutputParser(pydantic_object=QAResponse)
document_relevance_parser = PydanticOutputParser(pydantic_object=DocumentRelevance)


def get_routing_parser() -> PydanticOutputParser:
    """
    Get parser for routing decisions.
    
    Example:
        from app.parsers.output_parsers import get_routing_parser
        
        parser = get_routing_parser()
        chain = prompt | llm | parser
        
        result = chain.invoke({"question": "What is Python?"})
        # result is a RoutingDecision object
        print(result.decision)  # "DIRECT"
        print(result.confidence)  # 0.95
    """
    return routing_parser


def get_qa_response_parser() -> PydanticOutputParser:
    """
    Get parser for Q&A responses.
    
    Example:
        parser = get_qa_response_parser()
        chain = prompt | llm | parser
        
        result = chain.invoke({"question": "...", "context": "..."})
        # result is a QAResponse object
        print(result.answer)
        print(result.sources)
        print(result.confidence)
    """
    return qa_response_parser


# ============================================================================
# with_structured_output() - Modern Approach
# ============================================================================
"""
with_structured_output() is the modern way to get structured output.
It uses function calling under the hood (more reliable than parsing).

Advantages over PydanticOutputParser:
- More reliable (uses function calling, not text parsing)
- Better error handling
- Cleaner syntax
- Supported by GPT-3.5-turbo, GPT-4, and other modern models

Example usage:
    from app.models.llm_config import get_openai_chat_model
    from app.parsers.output_parsers import RoutingDecision
    
    llm = get_openai_chat_model()
    structured_llm = llm.with_structured_output(RoutingDecision)
    
    result = structured_llm.invoke("Should I use RAG for: What is Python?")
    # result is a RoutingDecision object (no parsing needed!)
    print(result.decision)  # "DIRECT"
"""


def get_structured_llm(llm, schema: type[BaseModel]):
    """
    Wrap LLM with structured output.
    
    This is a helper function that uses with_structured_output().
    
    Args:
        llm: Base LLM (ChatOpenAI, etc.)
        schema: Pydantic model class
        
    Returns:
        LLM that outputs structured data
        
    Example:
        from app.models.llm_config import get_openai_chat_model
        
        llm = get_openai_chat_model()
        routing_llm = get_structured_llm(llm, RoutingDecision)
        
        result = routing_llm.invoke("Question: What is Python?")
        # result is RoutingDecision object
    """
    return llm.with_structured_output(schema)


# ============================================================================
# Output Fixing Parser
# ============================================================================
"""
OutputFixingParser automatically fixes malformed outputs.

If the LLM returns invalid JSON or doesn't match the schema,
this parser uses another LLM call to fix it.

Example:
    from langchain.output_parsers import OutputFixingParser
    
    # Base parser
    base_parser = PydanticOutputParser(pydantic_object=RoutingDecision)
    
    # Fixing parser (uses LLM to fix errors)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=get_openai_chat_model()
    )
    
    # Even if LLM returns malformed output, fixing_parser will correct it
    result = fixing_parser.parse(malformed_output)

Note: This makes an extra LLM call, so it's slower and more expensive.
Use only when necessary.
"""


# ============================================================================
# Retry Output Parser
# ============================================================================
"""
RetryOutputParser retries parsing with additional context.

If parsing fails, it shows the LLM the error and asks it to fix the output.

Example:
    from langchain.output_parsers import RetryOutputParser
    
    base_parser = PydanticOutputParser(pydantic_object=QAResponse)
    
    retry_parser = RetryOutputParser.from_llm(
        parser=base_parser,
        llm=get_openai_chat_model()
    )
    
    # If parsing fails, retry_parser will:
    # 1. Show the LLM the error
    # 2. Ask it to fix the output
    # 3. Parse again
    result = retry_parser.parse_with_prompt(output, original_prompt)

Note: Requires the original prompt, so use with chains.
"""


# ============================================================================
# Custom Output Parser Example
# ============================================================================

class SimpleRoutingParser(StrOutputParser):
    """
    Custom parser for simple routing decisions.
    
    Extracts "RAG" or "DIRECT" from LLM output.
    More robust than expecting exact format.
    """
    
    def parse(self, text: str) -> str:
        """
        Parse routing decision from text.
        
        Looks for "RAG" or "DIRECT" anywhere in the response.
        """
        text_upper = text.upper()
        
        if "RAG" in text_upper and "DIRECT" not in text_upper:
            return "RAG"
        elif "DIRECT" in text_upper and "RAG" not in text_upper:
            return "DIRECT"
        elif "RAG" in text_upper:
            # If both present, prefer RAG (safer to retrieve)
            return "RAG"
        else:
            # Default to DIRECT if unclear
            return "DIRECT"


# Create instance
simple_routing_parser = SimpleRoutingParser()


# ============================================================================
# Educational Note: Output Parsing Strategies
# ============================================================================
"""
**When to use each parser:**

1. **StrOutputParser**:
   - Simple string responses
   - No structure needed
   - Fastest, no overhead
   - Example: General Q&A, summaries

2. **PydanticOutputParser**:
   - Need structured data
   - LLM doesn't support function calling
   - Legacy models
   - Example: Extracting specific fields from text

3. **with_structured_output()**:
   - Modern models (GPT-3.5-turbo+)
   - Most reliable structured output
   - Uses function calling
   - Example: Any structured data with modern models

4. **OutputFixingParser**:
   - Unreliable LLM outputs
   - Complex schemas
   - Can afford extra LLM call
   - Example: Parsing complex JSON from older models

5. **RetryOutputParser**:
   - Need high reliability
   - Can afford extra LLM call
   - Have access to original prompt
   - Example: Critical parsing that must succeed

6. **Custom Parsers**:
   - Specific format requirements
   - Need custom logic
   - Performance optimization
   - Example: Simple keyword extraction

**Best practices:**
- Use with_structured_output() for modern models
- Use PydanticOutputParser for legacy models
- Add validation in Pydantic models
- Handle parsing errors gracefully
- Test parsers with various inputs
- Consider cost of fixing/retry parsers

**Error handling:**
Always wrap parsing in try-except:
    try:
        result = parser.parse(output)
    except OutputParserException as e:
        # Handle parsing error
        # Maybe use fixing parser or return default
        ...
"""
