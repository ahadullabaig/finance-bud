"""
FinPilot Configuration Management

Centralized configuration using Pydantic Settings.
Loads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    For example, set API_HOST=0.0.0.0 to override the default host.
    """

    # ============================================================================
    # API Server Configuration
    # ============================================================================

    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )

    api_port: int = Field(
        default=8000,
        description="API server port"
    )

    api_title: str = Field(
        default="FinPilot VP-MAS API",
        description="API title for documentation"
    )

    api_version: str = Field(
        default="1.0.0",
        description="API version"
    )

    api_description: str = Field(
        default="Verifiable Planning Multi-Agent System for Financial Planning",
        description="API description for documentation"
    )

    reload: bool = Field(
        default=True,
        description="Enable auto-reload in development"
    )

    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error)"
    )

    # ============================================================================
    # CORS Configuration
    # ============================================================================

    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )

    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )

    cors_allow_methods: List[str] = Field(
        default=["*"],
        description="Allowed HTTP methods for CORS"
    )

    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed headers for CORS"
    )

    # ============================================================================
    # Agent Configuration
    # ============================================================================

    use_mock_orchestration: bool = Field(
        default=True,
        description="Use mock orchestration agent (for development)"
    )

    use_mock_planning: bool = Field(
        default=True,
        description="Use mock planning agent (for development)"
    )

    use_mock_retrieval: bool = Field(
        default=True,
        description="Use mock information retrieval agent (for development)"
    )

    use_real_verification: bool = Field(
        default=True,
        description="Use real verification agent"
    )

    conversational_agent_enabled: bool = Field(
        default=True,
        description="Enable conversational agent"
    )

    # ============================================================================
    # External API Configuration
    # ============================================================================

    alpha_vantage_api_key: Optional[str] = Field(
        default=None,
        description="Alpha Vantage API key for market data"
    )

    yahoo_finance_enabled: bool = Field(
        default=True,
        description="Enable Yahoo Finance integration"
    )

    # ============================================================================
    # Ollama Configuration (for Conversational Agent)
    # ============================================================================

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )

    ollama_model: str = Field(
        default="llama2",
        description="Ollama model to use"
    )

    ollama_timeout: int = Field(
        default=30,
        description="Ollama request timeout in seconds"
    )

    # ============================================================================
    # Database Configuration (for future use)
    # ============================================================================

    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL"
    )

    supabase_url: Optional[str] = Field(
        default=None,
        description="Supabase project URL"
    )

    supabase_key: Optional[str] = Field(
        default=None,
        description="Supabase API key"
    )

    # ============================================================================
    # Performance & Caching
    # ============================================================================

    cache_enabled: bool = Field(
        default=True,
        description="Enable caching for market data and API responses"
    )

    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache time-to-live in seconds"
    )

    max_request_timeout: int = Field(
        default=120,
        description="Maximum request timeout in seconds"
    )

    # ============================================================================
    # Security
    # ============================================================================

    enable_api_key_auth: bool = Field(
        default=False,
        description="Enable API key authentication"
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (if enabled)"
    )

    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting"
    )

    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per client"
    )

    # ============================================================================
    # Development & Debug
    # ============================================================================

    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )

    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )

    class Config:
        """Pydantic settings configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are only loaded once.
    Call this function to access application settings.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Convenience access
settings = get_settings()
