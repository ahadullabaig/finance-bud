"""
FinPilot Configuration Management

Centralized configuration using Pydantic Settings.
Loads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional, Dict
from functools import lru_cache
import secrets


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
        default=False,
        description="Use mock orchestration agent (for development only - NOT for production)"
    )

    use_mock_planning: bool = Field(
        default=False,
        description="Use mock planning agent (for development only - NOT for production)"
    )

    use_mock_retrieval: bool = Field(
        default=False,
        description="Use mock information retrieval agent (for development only - NOT for production)"
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
    # Database Configuration
    # ============================================================================

    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL (PostgreSQL)"
    )

    database_pool_size: int = Field(
        default=5,
        description="Database connection pool size"
    )

    database_max_overflow: int = Field(
        default=10,
        description="Maximum number of connections that can be created beyond pool_size"
    )

    database_pool_pre_ping: bool = Field(
        default=True,
        description="Enable connection health checks before using"
    )

    database_pool_recycle: int = Field(
        default=3600,
        description="Recycle connections after N seconds (prevents stale connections)"
    )

    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries (for debugging only)"
    )

    # Supabase (alternative to self-hosted PostgreSQL)
    supabase_url: Optional[str] = Field(
        default=None,
        description="Supabase project URL"
    )

    supabase_key: Optional[str] = Field(
        default=None,
        description="Supabase API key"
    )

    # ============================================================================
    # Redis Configuration
    # ============================================================================

    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL (for caching and rate limiting)"
    )

    redis_max_connections: int = Field(
        default=50,
        description="Maximum Redis connections in pool"
    )

    redis_decode_responses: bool = Field(
        default=True,
        description="Decode Redis responses to strings"
    )

    redis_socket_timeout: int = Field(
        default=5,
        description="Redis socket timeout in seconds"
    )

    redis_socket_connect_timeout: int = Field(
        default=5,
        description="Redis socket connect timeout in seconds"
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
    # Environment Detection
    # ============================================================================

    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production)"
    )

    # ============================================================================
    # Security - Authentication & Authorization
    # ============================================================================

    enable_api_key_auth: bool = Field(
        default=False,
        description="Enable API key authentication (legacy, use JWT instead)"
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (if enabled)"
    )

    # JWT Authentication Configuration
    jwt_secret_key: Optional[str] = Field(
        default=None,
        description="JWT secret key for token signing (REQUIRED in production)"
    )

    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm (HS256, RS256)"
    )

    jwt_access_token_expire_minutes: int = Field(
        default=15,
        description="JWT access token expiration in minutes"
    )

    jwt_refresh_token_expire_days: int = Field(
        default=7,
        description="JWT refresh token expiration in days"
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting"
    )

    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per client"
    )

    rate_limit_burst_multiplier: float = Field(
        default=1.5,
        description="Burst multiplier for rate limiting"
    )

    # Security Headers & Protection
    enable_https_redirect: bool = Field(
        default=False,
        description="Redirect HTTP to HTTPS in production"
    )

    enable_csrf_protection: bool = Field(
        default=True,
        description="Enable CSRF protection for state-changing operations"
    )

    session_cookie_secure: bool = Field(
        default=True,
        description="Set Secure flag on session cookies (HTTPS only)"
    )

    session_cookie_httponly: bool = Field(
        default=True,
        description="Set HttpOnly flag on session cookies"
    )

    session_cookie_samesite: str = Field(
        default="lax",
        description="SameSite cookie attribute (strict, lax, none)"
    )

    # ============================================================================
    # Agent Timeouts (Centralized)
    # ============================================================================

    agent_orchestration_timeout: int = Field(
        default=60,
        description="Orchestration agent operation timeout in seconds"
    )

    agent_planning_timeout: int = Field(
        default=300,
        description="Planning agent operation timeout in seconds"
    )

    agent_verification_timeout: int = Field(
        default=30,
        description="Verification agent operation timeout in seconds"
    )

    agent_execution_timeout: int = Field(
        default=120,
        description="Execution agent operation timeout in seconds"
    )

    agent_retrieval_timeout: int = Field(
        default=30,
        description="Information retrieval agent timeout in seconds"
    )

    agent_conversational_timeout: int = Field(
        default=30,
        description="Conversational agent timeout in seconds"
    )

    # ============================================================================
    # Background Task Intervals
    # ============================================================================

    health_check_interval_seconds: int = Field(
        default=30,
        description="Interval for agent health checks in seconds"
    )

    cleanup_interval_seconds: int = Field(
        default=3600,
        description="Interval for cleanup tasks in seconds (1 hour default)"
    )

    monitoring_interval_seconds: int = Field(
        default=60,
        description="Interval for monitoring tasks in seconds"
    )

    trigger_check_interval_seconds: int = Field(
        default=300,
        description="Interval for CMVL trigger checks in seconds"
    )

    # ============================================================================
    # Monitoring & Observability
    # ============================================================================

    monitoring_enabled: bool = Field(
        default=False,
        description="Enable monitoring and metrics collection"
    )

    prometheus_enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics endpoint"
    )

    prometheus_port: int = Field(
        default=9090,
        description="Prometheus metrics port"
    )

    # Error Tracking (Sentry)
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )

    sentry_environment: Optional[str] = Field(
        default=None,
        description="Sentry environment tag (development, staging, production)"
    )

    sentry_traces_sample_rate: float = Field(
        default=0.1,
        description="Sentry traces sample rate (0.0 to 1.0)"
    )

    # Distributed Tracing
    tracing_enabled: bool = Field(
        default=False,
        description="Enable distributed tracing with OpenTelemetry"
    )

    tracing_endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry collector endpoint"
    )

    # ============================================================================
    # Logging Configuration
    # ============================================================================

    log_file_path: Optional[str] = Field(
        default=None,
        description="Path to log file (None = stdout only)"
    )

    log_rotation_size_mb: int = Field(
        default=100,
        description="Log file rotation size in megabytes"
    )

    log_retention_days: int = Field(
        default=30,
        description="Log retention period in days"
    )

    log_format: str = Field(
        default="json",
        description="Log format (json, text)"
    )

    log_include_correlation_id: bool = Field(
        default=True,
        description="Include correlation IDs in logs"
    )

    # ============================================================================
    # Secrets Management (Optional)
    # ============================================================================

    use_secrets_manager: bool = Field(
        default=False,
        description="Use external secrets manager instead of environment variables"
    )

    secrets_manager_type: str = Field(
        default="env",
        description="Secrets manager type (env, aws, vault, azure, gcp)"
    )

    secrets_manager_path: Optional[str] = Field(
        default=None,
        description="Path/prefix for secrets in secrets manager"
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

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value"""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: Optional[str], info) -> Optional[str]:
        """Validate JWT secret key in production"""
        # Get environment from values if available
        env = info.data.get("environment", "development")

        if env == "production" and not v:
            raise ValueError(
                "jwt_secret_key is REQUIRED in production environment. "
                "Set JWT_SECRET_KEY environment variable."
            )

        # Auto-generate for development if not provided
        if not v and env == "development":
            generated = secrets.token_urlsafe(32)
            print(f"⚠️  AUTO-GENERATED JWT_SECRET_KEY for development: {generated}")
            print("⚠️  DO NOT use this in production! Set JWT_SECRET_KEY environment variable.")
            return generated

        # Warn if secret is too short
        if v and len(v) < 32:
            print(f"⚠️  WARNING: JWT secret key is short ({len(v)} chars). Recommend 32+ chars.")

        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: Optional[str], info) -> Optional[str]:
        """Validate database URL in production"""
        env = info.data.get("environment", "development")

        if env == "production" and not v:
            raise ValueError(
                "database_url is REQUIRED in production environment. "
                "Set DATABASE_URL environment variable."
            )

        return v

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: List[str], info) -> List[str]:
        """Validate CORS origins in production"""
        env = info.data.get("environment", "development")

        if env == "production":
            # Check for wildcards or localhost in production
            for origin in v:
                if origin == "*":
                    raise ValueError("CORS wildcard (*) not allowed in production")
                if "localhost" in origin or "127.0.0.1" in origin:
                    raise ValueError(f"Localhost origins not allowed in production: {origin}")

        return v

    @field_validator("debug_mode")
    @classmethod
    def validate_debug_mode(cls, v: bool, info) -> bool:
        """Force debug_mode to False in production"""
        env = info.data.get("environment", "development")

        if env == "production" and v is True:
            print("⚠️  WARNING: debug_mode forced to False in production")
            return False

        return v

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"

    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return self.environment == "staging"


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
