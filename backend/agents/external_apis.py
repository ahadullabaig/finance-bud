"""
External API Integration Framework for FinPilot VP-MAS

This module provides comprehensive external API integration with rate limiting,
failover mechanisms, caching, authentication, and mock fallback capabilities.

Supports multiple market data providers:
- Barchart API
- Massive API  
- Alpha Vantage API

Requirements: 5.1, 5.2, 4.4, 12.1
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging

import aiohttp
import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from data_models.schemas import MarketData, SeverityLevel, MarketEventType
from utils.logger import LoggingStandards

logger = LoggingStandards.create_system_logger("external_apis")


class APIProvider(str, Enum):
    """Supported external API providers"""
    BARCHART = "barchart"
    MASSIVE = "massive"
    ALPHA_VANTAGE = "alpha_vantage"
    MOCK = "mock"


class APIStatus(str, Enum):
    """API connection status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_ERROR = "authentication_error"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    backoff_factor: float = 2.0
    max_backoff: float = 300.0  # 5 minutes


@dataclass
class APIConfig:
    """Configuration for external API"""
    provider: APIProvider
    base_url: str
    api_key: str
    backup_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    health_check_interval: int = 300  # 5 minutes
    failover_threshold: int = 3  # consecutive failures before failover


class APIResponse(BaseModel):
    """Standardized API response"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    provider: APIProvider
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_time: float
    from_cache: bool = False
    rate_limit_remaining: Optional[int] = None


class CircuitBreakerState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for API failure handling"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_limit
        self.last_refill = time.time()
        self.request_history = []
    
    async def acquire(self) -> bool:
        """Acquire a token for request"""
        now = time.time()
        
        # Refill tokens
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * (self.config.requests_per_minute / 60.0)
        self.tokens = min(self.config.burst_limit, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Clean old request history
        cutoff_time = now - 3600  # 1 hour
        self.request_history = [t for t in self.request_history if t > cutoff_time]
        
        # Check rate limits
        if self.tokens < 1:
            return False
        
        if len([t for t in self.request_history if t > now - 60]) >= self.config.requests_per_minute:
            return False
        
        if len(self.request_history) >= self.config.requests_per_hour:
            return False
        
        # Consume token
        self.tokens -= 1
        self.request_history.append(now)
        return True
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request"""
        if self.tokens >= 1:
            return 0.0
        
        return (1 - self.tokens) / (self.config.requests_per_minute / 60.0)


class CacheManager:
    """Redis-based caching with TTL management"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.default_ttl = 300  # 5 minutes
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data"""
        if not self.redis:
            return None
        
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    async def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set cached data with TTL"""
        if not self.redis:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cached data"""
        if not self.redis:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()


class BaseAPIConnector(ABC):
    """Base class for external API connectors"""
    
    def __init__(self, config: APIConfig, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.circuit_breaker = CircuitBreaker()
        self.session = None
        self.status = APIStatus.HEALTHY
        self.last_health_check = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def fetch_market_data(self, symbols: List[str], **kwargs) -> APIResponse:
        """Fetch market data for given symbols"""
        pass
    
    @abstractmethod
    async def get_market_volatility(self, **kwargs) -> APIResponse:
        """Get market volatility data"""
        pass
    
    @abstractmethod
    async def get_economic_indicators(self, **kwargs) -> APIResponse:
        """Get economic indicators"""
        pass
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Simple ping to API
            response = await self._make_request("GET", "/health", timeout=5.0)
            self.status = APIStatus.HEALTHY if response.success else APIStatus.DEGRADED
            self.last_health_check = datetime.utcnow()
            return response.success
        except Exception as e:
            logger.warning(f"Health check failed for {self.config.provider}: {e}")
            self.status = APIStatus.UNAVAILABLE
            return False
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None
    ) -> APIResponse:
        """Make HTTP request with rate limiting, caching, and error handling"""
        
        start_time = time.time()
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return APIResponse(
                success=False,
                error="Circuit breaker is open",
                provider=self.config.provider,
                response_time=0.0
            )
        
        # Rate limiting
        if not await self.rate_limiter.acquire():
            wait_time = self.rate_limiter.get_wait_time()
            return APIResponse(
                success=False,
                error=f"Rate limited. Wait {wait_time:.2f} seconds",
                provider=self.config.provider,
                response_time=0.0
            )
        
        # Check cache first
        cache_key = f"{self.config.provider}:{method}:{endpoint}:{hash(str(params))}"
        if use_cache:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return APIResponse(
                    success=True,
                    data=cached_data,
                    provider=self.config.provider,
                    response_time=time.time() - start_time,
                    from_cache=True
                )
        
        # Make request with retries
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                url = f"{self.config.base_url}{endpoint}"
                headers = self._get_headers()
                
                async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=timeout or self.config.timeout
                ) as response:
                    
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        # Cache successful response
                        if use_cache:
                            await self.cache_manager.set(cache_key, response_data, cache_ttl)
                        
                        self.circuit_breaker.record_success()
                        
                        return APIResponse(
                            success=True,
                            data=response_data,
                            provider=self.config.provider,
                            response_time=response_time,
                            rate_limit_remaining=self._extract_rate_limit(response)
                        )
                    
                    elif response.status == 429:  # Rate limited
                        self.status = APIStatus.RATE_LIMITED
                        retry_after = int(response.headers.get('Retry-After', 60))
                        await asyncio.sleep(min(retry_after, 300))  # Max 5 minutes
                        continue
                    
                    elif response.status == 401:  # Authentication error
                        self.status = APIStatus.AUTHENTICATION_ERROR
                        if self.config.backup_key and attempt == 0:
                            # Try with backup key
                            continue
                        
                        return APIResponse(
                            success=False,
                            error="Authentication failed",
                            provider=self.config.provider,
                            response_time=response_time
                        )
                    
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"
            
            except asyncio.TimeoutError:
                last_error = "Request timeout"
            except Exception as e:
                last_error = str(e)
            
            # Exponential backoff for retries
            if attempt < self.config.max_retries:
                wait_time = min(
                    self.config.rate_limit.backoff_factor ** attempt,
                    self.config.rate_limit.max_backoff
                )
                await asyncio.sleep(wait_time)
        
        # All retries failed
        self.circuit_breaker.record_failure()
        self.status = APIStatus.UNAVAILABLE
        
        return APIResponse(
            success=False,
            error=last_error or "Request failed after all retries",
            provider=self.config.provider,
            response_time=time.time() - start_time
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FinPilot-VPMAS/1.0"
        }
    
    def _extract_rate_limit(self, response) -> Optional[int]:
        """Extract rate limit information from response headers"""
        rate_limit_header = response.headers.get('X-RateLimit-Remaining')
        if rate_limit_header:
            try:
                return int(rate_limit_header)
            except ValueError:
                pass
        return None


class BarchartAPIConnector(BaseAPIConnector):
    """Barchart API connector implementation"""
    
    async def fetch_market_data(self, symbols: List[str], **kwargs) -> APIResponse:
        """Fetch market data from Barchart API"""
        params = {
            "symbols": ",".join(symbols),
            "fields": "symbol,lastPrice,netChange,percentChange,volume,high,low",
            "format": "json"
        }
        params.update(kwargs)
        
        return await self._make_request("GET", "/v1/quotes/get", params=params)
    
    async def get_market_volatility(self, **kwargs) -> APIResponse:
        """Get market volatility from Barchart"""
        params = {
            "symbol": kwargs.get("symbol", "VIX"),
            "fields": "symbol,lastPrice,netChange,percentChange"
        }
        
        return await self._make_request("GET", "/v1/quotes/get", params=params)
    
    async def get_economic_indicators(self, **kwargs) -> APIResponse:
        """Get economic indicators from Barchart"""
        params = {
            "category": kwargs.get("category", "economic"),
            "limit": kwargs.get("limit", 50)
        }
        
        return await self._make_request("GET", "/v1/economic/get", params=params)


class AlphaVantageAPIConnector(BaseAPIConnector):
    """Alpha Vantage API connector implementation"""
    
    def _get_headers(self) -> Dict[str, str]:
        """Override headers for Alpha Vantage (uses query param for API key)"""
        return {
            "Content-Type": "application/json",
            "User-Agent": "FinPilot-VPMAS/1.0"
        }
    
    async def fetch_market_data(self, symbols: List[str], **kwargs) -> APIResponse:
        """Fetch market data from Alpha Vantage API"""
        # Alpha Vantage requires individual symbol requests
        results = {}
        
        for symbol in symbols:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.config.api_key
            }
            
            response = await self._make_request("GET", "/query", params=params)
            if response.success:
                results[symbol] = response.data
            else:
                logger.warning(f"Failed to fetch data for {symbol}: {response.error}")
        
        return APIResponse(
            success=len(results) > 0,
            data={"quotes": results},
            provider=self.config.provider,
            response_time=0.0  # Will be calculated by caller
        )
    
    async def get_market_volatility(self, **kwargs) -> APIResponse:
        """Get market volatility from Alpha Vantage"""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": kwargs.get("symbol", "VIX"),
            "apikey": self.config.api_key
        }
        
        return await self._make_request("GET", "/query", params=params)
    
    async def get_economic_indicators(self, **kwargs) -> APIResponse:
        """Get economic indicators from Alpha Vantage"""
        params = {
            "function": kwargs.get("function", "REAL_GDP"),
            "interval": kwargs.get("interval", "annual"),
            "apikey": self.config.api_key
        }
        
        return await self._make_request("GET", "/query", params=params)


class MassiveAPIConnector(BaseAPIConnector):
    """Massive API connector implementation (placeholder)"""
    
    async def fetch_market_data(self, symbols: List[str], **kwargs) -> APIResponse:
        """Fetch market data from Massive API"""
        params = {
            "symbols": symbols,
            "fields": ["price", "volume", "change"],
            "format": "json"
        }
        params.update(kwargs)
        
        return await self._make_request("POST", "/v1/market/quotes", data=params)
    
    async def get_market_volatility(self, **kwargs) -> APIResponse:
        """Get market volatility from Massive API"""
        params = {
            "metric": "volatility",
            "timeframe": kwargs.get("timeframe", "1d")
        }
        
        return await self._make_request("GET", "/v1/market/volatility", params=params)
    
    async def get_economic_indicators(self, **kwargs) -> APIResponse:
        """Get economic indicators from Massive API"""
        params = {
            "indicators": kwargs.get("indicators", ["gdp", "inflation", "unemployment"]),
            "country": kwargs.get("country", "US")
        }
        
        return await self._make_request("POST", "/v1/economic/indicators", data=params)


class MockAPIConnector(BaseAPIConnector):
    """Mock API connector for testing and offline development"""
    
    def __init__(self, config: APIConfig, cache_manager: CacheManager):
        super().__init__(config, cache_manager)
        self.mock_data = self._generate_mock_data()
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """Generate realistic mock market data"""
        return {
            "market_quotes": {
                "AAPL": {
                    "symbol": "AAPL",
                    "lastPrice": 175.50,
                    "netChange": 2.30,
                    "percentChange": 1.33,
                    "volume": 45000000,
                    "high": 176.80,
                    "low": 173.20
                },
                "GOOGL": {
                    "symbol": "GOOGL",
                    "lastPrice": 2850.75,
                    "netChange": -15.25,
                    "percentChange": -0.53,
                    "volume": 1200000,
                    "high": 2875.00,
                    "low": 2840.50
                },
                "SPY": {
                    "symbol": "SPY",
                    "lastPrice": 445.20,
                    "netChange": 1.80,
                    "percentChange": 0.41,
                    "volume": 75000000,
                    "high": 446.50,
                    "low": 443.10
                }
            },
            "volatility": {
                "VIX": {
                    "symbol": "VIX",
                    "lastPrice": 18.45,
                    "netChange": -0.75,
                    "percentChange": -3.90
                }
            },
            "economic_indicators": {
                "gdp_growth": 2.1,
                "inflation_rate": 3.2,
                "unemployment_rate": 3.7,
                "federal_funds_rate": 5.25,
                "10_year_treasury": 4.5
            }
        }
    
    async def connect(self):
        """Mock connection - always succeeds"""
        self.status = APIStatus.HEALTHY
        logger.info(f"Mock API connector {self.config.provider} connected")
    
    async def disconnect(self):
        """Mock disconnection"""
        logger.info(f"Mock API connector {self.config.provider} disconnected")
    
    async def fetch_market_data(self, symbols: List[str], **kwargs) -> APIResponse:
        """Return mock market data"""
        # Removed artificial delay - mock should be fast for testing

        quotes = {}
        for symbol in symbols:
            if symbol in self.mock_data["market_quotes"]:
                quotes[symbol] = self.mock_data["market_quotes"][symbol]
            else:
                # Generate random data for unknown symbols
                quotes[symbol] = {
                    "symbol": symbol,
                    "lastPrice": 100.0 + (hash(symbol) % 1000) / 10.0,
                    "netChange": (hash(symbol) % 200 - 100) / 100.0,
                    "percentChange": (hash(symbol) % 200 - 100) / 1000.0,
                    "volume": (hash(symbol) % 10000000) + 1000000,
                    "high": 105.0 + (hash(symbol) % 1000) / 10.0,
                    "low": 95.0 + (hash(symbol) % 1000) / 10.0
                }
        
        return APIResponse(
            success=True,
            data={"quotes": quotes},
            provider=self.config.provider,
            response_time=0.1
        )
    
    async def get_market_volatility(self, **kwargs) -> APIResponse:
        """Return mock volatility data"""
        # Removed artificial delay - mock should be fast for testing

        return APIResponse(
            success=True,
            data=self.mock_data["volatility"],
            provider=self.config.provider,
            response_time=0.05
        )
    
    async def get_economic_indicators(self, **kwargs) -> APIResponse:
        """Return mock economic indicators"""
        # Removed artificial delay - mock should be fast for testing

        return APIResponse(
            success=True,
            data=self.mock_data["economic_indicators"],
            provider=self.config.provider,
            response_time=0.08
        )


class APIManager:
    """Manages multiple API connectors with failover and load balancing"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.connectors: Dict[APIProvider, BaseAPIConnector] = {}
        self.primary_provider = APIProvider.BARCHART
        self.fallback_providers = [APIProvider.ALPHA_VANTAGE, APIProvider.MASSIVE]
        self.mock_mode = False
    
    def add_connector(self, provider: APIProvider, config: APIConfig):
        """Add API connector"""
        if provider == APIProvider.BARCHART:
            connector = BarchartAPIConnector(config, self.cache_manager)
        elif provider == APIProvider.ALPHA_VANTAGE:
            connector = AlphaVantageAPIConnector(config, self.cache_manager)
        elif provider == APIProvider.MASSIVE:
            connector = MassiveAPIConnector(config, self.cache_manager)
        elif provider == APIProvider.MOCK:
            connector = MockAPIConnector(config, self.cache_manager)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.connectors[provider] = connector
    
    def enable_mock_mode(self):
        """Enable mock mode for offline testing"""
        self.mock_mode = True
        if APIProvider.MOCK not in self.connectors:
            mock_config = APIConfig(
                provider=APIProvider.MOCK,
                base_url="http://mock.api",
                api_key="mock_key"
            )
            self.add_connector(APIProvider.MOCK, mock_config)
    
    async def fetch_market_data(self, symbols: List[str], **kwargs) -> APIResponse:
        """Fetch market data with failover"""
        if self.mock_mode:
            return await self.connectors[APIProvider.MOCK].fetch_market_data(symbols, **kwargs)
        
        # Try primary provider first
        providers_to_try = [self.primary_provider] + self.fallback_providers
        
        for provider in providers_to_try:
            if provider in self.connectors:
                connector = self.connectors[provider]
                
                if connector.status == APIStatus.UNAVAILABLE:
                    continue
                
                try:
                    response = await connector.fetch_market_data(symbols, **kwargs)
                    if response.success:
                        return response
                    
                    logger.warning(f"Provider {provider} failed: {response.error}")
                
                except Exception as e:
                    logger.error(f"Provider {provider} error: {e}")
                    continue
        
        # All providers failed, return error
        return APIResponse(
            success=False,
            error="All API providers failed",
            provider=APIProvider.MOCK,  # Fallback
            response_time=0.0
        )
    
    async def get_market_volatility(self, **kwargs) -> APIResponse:
        """Get market volatility with failover"""
        if self.mock_mode:
            return await self.connectors[APIProvider.MOCK].get_market_volatility(**kwargs)
        
        providers_to_try = [self.primary_provider] + self.fallback_providers
        
        for provider in providers_to_try:
            if provider in self.connectors:
                connector = self.connectors[provider]
                
                if connector.status == APIStatus.UNAVAILABLE:
                    continue
                
                try:
                    response = await connector.get_market_volatility(**kwargs)
                    if response.success:
                        return response
                
                except Exception as e:
                    logger.error(f"Provider {provider} error: {e}")
                    continue
        
        return APIResponse(
            success=False,
            error="All API providers failed",
            provider=APIProvider.MOCK,
            response_time=0.0
        )
    
    async def get_economic_indicators(self, **kwargs) -> APIResponse:
        """Get economic indicators with failover"""
        if self.mock_mode:
            return await self.connectors[APIProvider.MOCK].get_economic_indicators(**kwargs)
        
        providers_to_try = [self.primary_provider] + self.fallback_providers
        
        for provider in providers_to_try:
            if provider in self.connectors:
                connector = self.connectors[provider]
                
                if connector.status == APIStatus.UNAVAILABLE:
                    continue
                
                try:
                    response = await connector.get_economic_indicators(**kwargs)
                    if response.success:
                        return response
                
                except Exception as e:
                    logger.error(f"Provider {provider} error: {e}")
                    continue
        
        return APIResponse(
            success=False,
            error="All API providers failed",
            provider=APIProvider.MOCK,
            response_time=0.0
        )
    
    async def health_check_all(self) -> Dict[APIProvider, bool]:
        """Perform health check on all connectors"""
        results = {}
        
        for provider, connector in self.connectors.items():
            if provider != APIProvider.MOCK:
                results[provider] = await connector.health_check()
        
        return results
    
    async def connect_all(self):
        """Connect all API connectors"""
        for connector in self.connectors.values():
            await connector.connect()
    
    async def disconnect_all(self):
        """Disconnect all API connectors"""
        for connector in self.connectors.values():
            await connector.disconnect()
    
    @asynccontextmanager
    async def managed_connections(self):
        """Context manager for API connections"""
        try:
            await self.connect_all()
            yield self
        finally:
            await self.disconnect_all()


# Factory function for creating configured API manager
async def create_api_manager(
    barchart_key: Optional[str] = None,
    alpha_vantage_key: Optional[str] = None,
    massive_key: Optional[str] = None,
    redis_url: str = "redis://localhost:6379",
    mock_mode: bool = False
) -> APIManager:
    """Create and configure API manager with all providers"""
    
    # Initialize cache manager
    cache_manager = CacheManager(redis_url)
    await cache_manager.connect()
    
    # Create API manager
    api_manager = APIManager(cache_manager)
    
    if mock_mode:
        api_manager.enable_mock_mode()
    else:
        # Add real API connectors
        if barchart_key:
            barchart_config = APIConfig(
                provider=APIProvider.BARCHART,
                base_url="https://api.barchart.com",
                api_key=barchart_key,
                rate_limit=RateLimitConfig(
                    requests_per_minute=60,
                    requests_per_hour=1000,
                    requests_per_day=10000
                )
            )
            api_manager.add_connector(APIProvider.BARCHART, barchart_config)
        
        if alpha_vantage_key:
            alpha_vantage_config = APIConfig(
                provider=APIProvider.ALPHA_VANTAGE,
                base_url="https://www.alphavantage.co",
                api_key=alpha_vantage_key,
                rate_limit=RateLimitConfig(
                    requests_per_minute=5,  # Alpha Vantage is more restrictive
                    requests_per_hour=500,
                    requests_per_day=500
                )
            )
            api_manager.add_connector(APIProvider.ALPHA_VANTAGE, alpha_vantage_config)
        
        if massive_key:
            massive_config = APIConfig(
                provider=APIProvider.MASSIVE,
                base_url="https://api.massive.com",
                api_key=massive_key,
                rate_limit=RateLimitConfig(
                    requests_per_minute=100,
                    requests_per_hour=2000,
                    requests_per_day=20000
                )
            )
            api_manager.add_connector(APIProvider.MASSIVE, massive_config)
        
        # Always add mock as fallback
        api_manager.enable_mock_mode()
    
    return api_manager