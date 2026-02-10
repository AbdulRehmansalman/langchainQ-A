"""
LLM Configuration and Management

This module demonstrates LangChain's model abstraction layer.
It covers ChatModels, configuration, fallbacks, caching, and streaming.

Key concepts:
- ChatModels vs LLMs
- Model configuration (temperature, max_tokens, etc.)
- Model fallback strategy
- Response caching
- Streaming responses
- Token counting
"""

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from typing import Optional, Dict, Any
import tiktoken

from app.config import settings


# ============================================================================
# Response Caching
# ============================================================================
# Cache LLM responses to save costs and improve performance
# Same input = same output (for deterministic temperatures)
set_llm_cache(InMemoryCache())


# ============================================================================
# ChatModels vs LLMs - Educational Note
# ============================================================================
"""
**ChatModels vs LLMs** - Understanding the difference:

1. **LLMs (Language Models)**:
   - Input: String
   - Output: String
   - Example: GPT-3 (davinci, curie)
   - Use case: Text completion
   - Prompt: "The capital of France is"
   - Response: "Paris"

2. **ChatModels**:
   - Input: List of messages (with roles: system, human, ai)
   - Output: Message
   - Example: GPT-3.5-turbo, GPT-4, Claude
   - Use case: Conversational AI
   - Prompt: [SystemMessage("You are helpful"), HumanMessage("What is the capital of France?")]
   - Response: AIMessage("The capital of France is Paris.")

**Why ChatModels are preferred:**
- Better instruction following
- Proper conversation context
- Role-based prompting (system, human, ai)
- More cost-effective (GPT-3.5-turbo vs GPT-3)
- Better performance on most tasks

**When to use LLMs:**
- Simple text completion
- Legacy code
- Specific use cases requiring completion models

In this project, we use ChatModels exclusively for better quality and cost.
"""


# ============================================================================
# OpenAI ChatModel Configuration
# ============================================================================

def get_openai_chat_model(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """
    Get configured OpenAI ChatModel.
    
    Args:
        model: Model name (default: from settings)
        temperature: Temperature 0-2 (default: from settings)
            - 0: Deterministic, focused
            - 1: Balanced
            - 2: Creative, random
        max_tokens: Max tokens in response
        streaming: Enable streaming responses
        
    Returns:
        Configured ChatOpenAI instance
        
    Example:
        # Deterministic model for factual Q&A
        llm = get_openai_chat_model(temperature=0)
        
        # Creative model for brainstorming
        llm = get_openai_chat_model(temperature=1.5)
        
        # Streaming for real-time UI
        llm = get_openai_chat_model(streaming=True)
    """
    return ChatOpenAI(
        model=model or settings.default_model,
        temperature=temperature if temperature is not None else settings.default_temperature,
        max_tokens=max_tokens or settings.default_max_tokens,
        streaming=streaming,
        request_timeout=settings.response_timeout_seconds,
        # Callbacks can be added here for logging
    )


# ============================================================================
# Ollama ChatModel Configuration (Local LLM)
# ============================================================================

def get_ollama_chat_model(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOllama:
    """
    Get configured Ollama ChatModel for local inference.
    
    Ollama allows running models locally without API costs:
    - llama2: General purpose
    - mistral: Fast and capable
    - codellama: Code-focused
    - phi: Lightweight
    
    Args:
        model: Ollama model name (default: from settings)
        temperature: Temperature 0-2
        
    Returns:
        Configured ChatOllama instance
        
    Example:
        # Use local Llama2
        llm = get_ollama_chat_model(model="llama2")
        
        # Use Mistral for faster responses
        llm = get_ollama_chat_model(model="mistral")
    
    Note:
        Requires Ollama running locally: `ollama serve`
        Install models: `ollama pull llama2`
    """
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model or settings.ollama_model,
        temperature=temperature if temperature is not None else settings.default_temperature,
    )


# ============================================================================
# Model Fallback Strategy
# ============================================================================

def get_chat_model_with_fallback(
    prefer_openai: bool = True,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Get ChatModel with automatic fallback.
    
    This implements a fallback strategy:
    1. Try OpenAI (if preferred and API key available)
    2. Fall back to Ollama (local)
    
    This ensures the system works even if:
    - OpenAI API is down
    - API key is invalid
    - Rate limits are hit
    
    Args:
        prefer_openai: Try OpenAI first (default: True)
        streaming: Enable streaming
        
    Returns:
        ChatModel with fallback configured
        
    Example:
        llm = get_chat_model_with_fallback()
        # Will use OpenAI, fall back to Ollama if it fails
    """
    if prefer_openai and settings.openai_api_key:
        # Primary: OpenAI with Ollama fallback
        primary = get_openai_chat_model(streaming=streaming)
        fallback = get_ollama_chat_model()
        return primary.with_fallbacks([fallback])
    else:
        # Use Ollama directly
        return get_ollama_chat_model()


# ============================================================================
# Token Counting Utilities
# ============================================================================

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text for a given model.
    
    Token counting is important for:
    - Cost estimation
    - Context window management
    - Chunking decisions
    
    Args:
        text: Text to count tokens for
        model: Model name (affects tokenization)
        
    Returns:
        Number of tokens
        
    Example:
        tokens = count_tokens("Hello, world!")
        # Returns: 4
        
        # Estimate cost
        cost_per_1k = 0.002  # GPT-3.5-turbo
        cost = (tokens / 1000) * cost_per_1k
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-3.5-turbo"
) -> float:
    """
    Estimate cost for LLM call.
    
    Pricing (as of 2024):
    - GPT-3.5-turbo: $0.0015/1K input, $0.002/1K output
    - GPT-4: $0.03/1K input, $0.06/1K output
    - GPT-4-turbo: $0.01/1K input, $0.03/1K output
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        
    Returns:
        Estimated cost in USD
        
    Example:
        cost = estimate_cost(1000, 500, "gpt-3.5-turbo")
        # Returns: 0.0025 ($0.0025)
    """
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    }
    
    # Default to GPT-3.5-turbo pricing
    prices = pricing.get(model, pricing["gpt-3.5-turbo"])
    
    input_cost = (input_tokens / 1000) * prices["input"]
    output_cost = (output_tokens / 1000) * prices["output"]
    
    return input_cost + output_cost


# ============================================================================
# Model Configuration Presets
# ============================================================================

class ModelPresets:
    """
    Pre-configured model settings for common use cases.
    
    This demonstrates best practices for different scenarios.
    """
    
    @staticmethod
    def factual_qa() -> ChatOpenAI:
        """
        Model for factual Q&A.
        
        - Low temperature (0.3) for consistency
        - Moderate max_tokens
        - No streaming (batch processing)
        """
        return get_openai_chat_model(
            temperature=0.3,
            max_tokens=500,
            streaming=False
        )
    
    @staticmethod
    def creative_writing() -> ChatOpenAI:
        """
        Model for creative tasks.
        
        - High temperature (1.2) for variety
        - High max_tokens for longer responses
        """
        return get_openai_chat_model(
            temperature=1.2,
            max_tokens=2000,
            streaming=False
        )
    
    @staticmethod
    def chat_interface() -> ChatOpenAI:
        """
        Model for chat interface.
        
        - Balanced temperature (0.7)
        - Streaming enabled for real-time UX
        """
        return get_openai_chat_model(
            temperature=0.7,
            max_tokens=1000,
            streaming=True
        )
    
    @staticmethod
    def code_generation() -> ChatOpenAI:
        """
        Model for code generation.
        
        - Low temperature (0.2) for correctness
        - High max_tokens for complete code
        """
        return get_openai_chat_model(
            model="gpt-4",  # GPT-4 is better for code
            temperature=0.2,
            max_tokens=2000,
            streaming=False
        )


# ============================================================================
# Educational Note: Model Configuration Best Practices
# ============================================================================
"""
**Temperature Guide:**
- 0.0-0.3: Factual, deterministic (Q&A, classification)
- 0.4-0.7: Balanced (general chat, summarization)
- 0.8-1.2: Creative (brainstorming, writing)
- 1.3-2.0: Very creative (experimental, artistic)

**Max Tokens Guide:**
- 100-300: Short answers, classifications
- 500-1000: Standard Q&A, explanations
- 1000-2000: Detailed responses, code
- 2000+: Long-form content, articles

**Streaming:**
- Enable: Chat interfaces, real-time feedback
- Disable: Batch processing, when full response needed

**Model Selection:**
- GPT-3.5-turbo: Fast, cheap, good for most tasks
- GPT-4: Better reasoning, complex tasks, code
- GPT-4-turbo: Faster GPT-4, larger context
- Ollama (local): No cost, privacy, offline

**Caching:**
- Enabled by default in this module
- Saves costs for repeated queries
- Use deterministic temperature (0-0.3) for best results
- Clear cache periodically to prevent stale responses

**Fallbacks:**
- Always have a fallback strategy
- OpenAI → Ollama is a good pattern
- Prevents complete system failure
- Graceful degradation
"""
