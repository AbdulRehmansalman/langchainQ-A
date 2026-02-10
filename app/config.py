"""
Configuration Management using Pydantic Settings

This module centralizes all application configuration, loading from environment variables.
Pydantic Settings provides:
- Type validation
- Default values
- Environment variable parsing
- Easy testing with overrides
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via .env file or environment variables.
    This pattern makes configuration explicit and type-safe.
    """
    
    # ============================================================================
    # OpenAI Configuration
    # ============================================================================
    openai_api_key: str = Field(..., description="OpenAI API key for GPT models")
    
    # ============================================================================
    # Ollama Configuration (Local LLM)
    # ============================================================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL for local models"
    )
    ollama_model: str = Field(
        default="llama2",
        description="Ollama model name (e.g., llama2, mistral, codellama)"
    )
    
    # ============================================================================
    # Database Configuration
    # ============================================================================
    database_url: str = Field(
        default="sqlite+aiosqlite:///./qa_system.db",
        description="Async database URL (SQLite for simplicity, can use PostgreSQL)"
    )
    
    # ============================================================================
    # JWT Authentication
    # ============================================================================
    secret_key: str = Field(..., description="Secret key for JWT signing")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration in days"
    )
    
    # ============================================================================
    # Email Configuration (for verification)
    # ============================================================================
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    from_email: str = Field(
        default="noreply@qa-system.com",
        description="From email address"
    )
    
    # ============================================================================
    # Vector Store Configuration
    # ============================================================================
    vector_store_path: str = Field(
        default="./data/vector_store",
        description="Path to persist FAISS vector store"
    )
    embeddings_cache_path: str = Field(
        default="./data/embeddings_cache",
        description="Path to cache embeddings for cost optimization"
    )
    
    # ============================================================================
    # Rate Limiting
    # ============================================================================
    rate_limit_per_minute: int = Field(
        default=60,
        description="Max requests per minute per user"
    )
    rate_limit_per_hour: int = Field(
        default=1000,
        description="Max requests per hour per user"
    )
    
    # ============================================================================
    # LLM Configuration
    # ============================================================================
    default_model: str = Field(
        default="gpt-3.5-turbo",
        description="Default OpenAI model (gpt-3.5-turbo, gpt-4, etc.)"
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM (0=deterministic, 2=creative)"
    )
    default_max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Default max tokens for LLM responses"
    )
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming responses for better UX"
    )
    
    # ============================================================================
    # Performance Configuration
    # ============================================================================
    max_concurrent_requests: int = Field(
        default=10,
        description="Max concurrent LLM requests"
    )
    response_timeout_seconds: int = Field(
        default=30,
        description="Timeout for LLM responses"
    )
    
    # ============================================================================
    # Application Metadata
    # ============================================================================
    app_name: str = Field(default="Intelligent Q&A System")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False, description="Debug mode")
    
    # ============================================================================
    # Pydantic Settings Configuration
    # ============================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables
    )
    
    @validator("default_temperature")
    def validate_temperature(cls, v):
        """Ensure temperature is in valid range"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    def get_openai_config(self) -> dict:
        """
        Get OpenAI model configuration.
        
        Returns a dict suitable for ChatOpenAI initialization.
        This centralizes model config for consistency.
        """
        return {
            "model": self.default_model,
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens,
            "streaming": self.enable_streaming,
            "request_timeout": self.response_timeout_seconds,
        }
    
    def get_ollama_config(self) -> dict:
        """
        Get Ollama model configuration.
        
        Returns a dict suitable for ChatOllama initialization.
        """
        return {
            "base_url": self.ollama_base_url,
            "model": self.ollama_model,
            "temperature": self.default_temperature,
        }


# ============================================================================
# Global Settings Instance
# ============================================================================
# This singleton pattern ensures settings are loaded once and reused.
# It's efficient and makes testing easier (can override in tests).
settings = Settings()


# ============================================================================
# Educational Note: Why Pydantic Settings?
# ============================================================================
"""
Pydantic Settings provides several benefits over manual environment variable handling:

1. **Type Safety**: Automatic type conversion and validation
2. **Documentation**: Field descriptions serve as inline documentation
3. **Defaults**: Clear default values in one place
4. **Validation**: Custom validators for complex rules
5. **Testing**: Easy to override settings in tests
6. **IDE Support**: Autocomplete and type hints

Example usage in other modules:
    from app.config import settings
    
    llm = ChatOpenAI(**settings.get_openai_config())
"""
