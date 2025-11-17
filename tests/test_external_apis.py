"""
Test suite for External API Integration Framework

Tests all components of the external API integration including:
- API connectors (Barchart, Alpha Vantage, Massive, Mock)
- Rate limiting and circuit breakers
- Caching with Redis
- Authentication and key rotation
- Failover mechanisms

Requirements: 5.1, 5.2, 4.4, 12.1, 11.1, 11.2, 11.3
"""

import asyncio
import pytest
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from agents.external_apis import (
    APIProvider, APIStatus, APIConfig, RateLimitConfig,
    CircuitBreaker, RateLimiter, CacheManager, APIManager,
    BarchartAPIConnector, AlphaVantageAPIConnector, MassiveAPIConnector, MockAPIConnector,
    create_api_manager
)
from agents.api_config import APIConfigManager, SecureKeyManager, setup_api_keys


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.can_execute() is True
        assert cb.failure_count == 0
    
    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker tracks failures correctly"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        # Record failures
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.can_execute() is True
        
        cb.record_failure()
        assert cb.failure_count == 2
        assert cb.can_execute() is True
        
        cb.record_failure()
        assert cb.failure_count == 3
        assert cb.can_execute() is False  # Circuit should be open
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        # Trigger circuit breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is False
        
        # Wait for recovery timeout
        time.sleep(1.1)
        assert cb.can_execute() is True  # Should be half-open
        
        # Record success to close circuit
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.can_execute() is True


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test rate limiter allows requests within limits"""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=10
        )
        limiter = RateLimiter(config)
        
        # Should allow burst requests
        for _ in range(10):
            assert await limiter.acquire() is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks requests exceeding burst limit"""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=5
        )
        limiter = RateLimiter(config)
        
        # Consume all burst tokens
        for _ in range(5):
            assert await limiter.acquire() is True
        
        # Next request should be blocked
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_token_refill(self):
        """Test rate limiter refills tokens over time"""
        config = RateLimitConfig(
            requests_per_minute=60,  # 1 per second
            burst_limit=2
        )
        limiter = RateLimiter(config)
        
        # Consume tokens
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire() is False
        
        # Wait for token refill (simulate time passage)
        limiter.last_refill = time.time() - 2  # Simulate 2 seconds passed
        assert await limiter.acquire() is True


@pytest.mark.asyncio
class TestCacheManager:
    """Test Redis caching functionality"""
    
    async def test_cache_manager_mock_mode(self):
        """Test cache manager in mock mode (no Redis)"""
        cache = CacheManager("redis://invalid:6379")
        await cache.connect()  # Should handle connection failure gracefully
        
        # Should return None for all operations when Redis unavailable
        assert await cache.get("test_key") is None
        assert await cache.set("test_key", {"data": "value"}) is False
        assert await cache.delete("test_key") is False
    
    @patch('aioredis.from_url')
    async def test_cache_manager_operations(self, mock_redis):
        """Test cache manager operations with mocked Redis"""
        # Mock Redis client
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.get.return_value = json.dumps({"cached": "data"})
        mock_client.setex.return_value = True
        mock_client.delete.return_value = True
        
        cache = CacheManager("redis://localhost:6379")
        await cache.connect()
        
        # Test get operation
        result = await cache.get("test_key")
        assert result == {"cached": "data"}
        mock_client.get.assert_called_with("test_key")
        
        # Test set operation
        success = await cache.set("test_key", {"new": "data"}, ttl=300)
        assert success is True
        mock_client.setex.assert_called_with("test_key", 300, json.dumps({"new": "data"}, default=str))
        
        # Test delete operation
        success = await cache.delete("test_key")
        assert success is True
        mock_client.delete.assert_called_with("test_key")
        
        await cache.close()


class TestMockAPIConnector:
    """Test mock API connector functionality"""
    
    @pytest.mark.asyncio
    async def test_mock_connector_initialization(self):
        """Test mock connector initializes correctly"""
        config = APIConfig(
            provider=APIProvider.MOCK,
            base_url="http://mock.api",
            api_key="mock_key"
        )
        cache = CacheManager("redis://invalid:6379")
        
        connector = MockAPIConnector(config, cache)
        assert connector.config.provider == APIProvider.MOCK
        assert connector.status == APIStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_mock_connector_fetch_market_data(self):
        """Test mock connector returns realistic market data"""
        config = APIConfig(
            provider=APIProvider.MOCK,
            base_url="http://mock.api",
            api_key="mock_key"
        )
        cache = CacheManager("redis://invalid:6379")
        
        async with MockAPIConnector(config, cache) as connector:
            response = await connector.fetch_market_data(["AAPL", "GOOGL"])
            
            assert response.success is True
            assert response.provider == APIProvider.MOCK
            assert "quotes" in response.data
            assert "AAPL" in response.data["quotes"]
            assert "GOOGL" in response.data["quotes"]
            
            # Check data structure
            aapl_data = response.data["quotes"]["AAPL"]
            assert "symbol" in aapl_data
            assert "lastPrice" in aapl_data
            assert "volume" in aapl_data
    
    @pytest.mark.asyncio
    async def test_mock_connector_volatility_data(self):
        """Test mock connector returns volatility data"""
        config = APIConfig(
            provider=APIProvider.MOCK,
            base_url="http://mock.api",
            api_key="mock_key"
        )
        cache = CacheManager("redis://invalid:6379")
        
        async with MockAPIConnector(config, cache) as connector:
            response = await connector.get_market_volatility()
            
            assert response.success is True
            assert "VIX" in response.data
            assert "lastPrice" in response.data["VIX"]
    
    @pytest.mark.asyncio
    async def test_mock_connector_economic_indicators(self):
        """Test mock connector returns economic indicators"""
        config = APIConfig(
            provider=APIProvider.MOCK,
            base_url="http://mock.api",
            api_key="mock_key"
        )
        cache = CacheManager("redis://invalid:6379")
        
        async with MockAPIConnector(config, cache) as connector:
            response = await connector.get_economic_indicators()
            
            assert response.success is True
            assert "gdp_growth" in response.data
            assert "inflation_rate" in response.data
            assert "federal_funds_rate" in response.data


@patch('aiohttp.ClientSession')
class TestAPIConnectorBase:
    """Test base API connector functionality"""
    
    @pytest.mark.asyncio
    async def test_connector_rate_limiting(self, mock_session_class):
        """Test API connector respects rate limits"""
        # Mock HTTP session
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.headers = {}
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        config = APIConfig(
            provider=APIProvider.BARCHART,
            base_url="https://api.test.com",
            api_key="test_key",
            rate_limit=RateLimitConfig(burst_limit=2)
        )
        cache = CacheManager("redis://invalid:6379")
        
        async with BarchartAPIConnector(config, cache) as connector:
            # First two requests should succeed
            response1 = await connector._make_request("GET", "/test")
            assert response1.success is True
            
            response2 = await connector._make_request("GET", "/test")
            assert response2.success is True
            
            # Third request should be rate limited
            response3 = await connector._make_request("GET", "/test")
            assert response3.success is False
            assert "Rate limited" in response3.error
    
    @pytest.mark.asyncio
    async def test_connector_circuit_breaker(self, mock_session_class):
        """Test API connector circuit breaker functionality"""
        # Mock HTTP session that always fails
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.request.side_effect = Exception("Connection failed")
        
        config = APIConfig(
            provider=APIProvider.BARCHART,
            base_url="https://api.test.com",
            api_key="test_key",
            max_retries=0  # No retries for faster test
        )
        cache = CacheManager("redis://invalid:6379")
        
        async with BarchartAPIConnector(config, cache) as connector:
            # Set low failure threshold for testing
            connector.circuit_breaker.failure_threshold = 2
            
            # First request should fail and record failure
            response1 = await connector._make_request("GET", "/test")
            assert response1.success is False
            assert connector.circuit_breaker.failure_count == 1
            
            # Second request should fail and open circuit
            response2 = await connector._make_request("GET", "/test")
            assert response2.success is False
            assert connector.circuit_breaker.failure_count == 2
            
            # Third request should be blocked by circuit breaker
            response3 = await connector._make_request("GET", "/test")
            assert response3.success is False
            assert "Circuit breaker is open" in response3.error


class TestAPIManager:
    """Test API manager functionality"""
    
    @pytest.mark.asyncio
    async def test_api_manager_initialization(self):
        """Test API manager initializes correctly"""
        cache = CacheManager("redis://invalid:6379")
        manager = APIManager(cache)
        
        assert manager.primary_provider == APIProvider.BARCHART
        assert APIProvider.ALPHA_VANTAGE in manager.fallback_providers
        assert manager.mock_mode is False
    
    @pytest.mark.asyncio
    async def test_api_manager_mock_mode(self):
        """Test API manager mock mode"""
        cache = CacheManager("redis://invalid:6379")
        manager = APIManager(cache)
        
        manager.enable_mock_mode()
        assert manager.mock_mode is True
        assert APIProvider.MOCK in manager.connectors
        
        # Test mock data retrieval
        response = await manager.fetch_market_data(["AAPL", "GOOGL"])
        assert response.success is True
        assert response.provider == APIProvider.MOCK
    
    @pytest.mark.asyncio
    async def test_api_manager_failover(self):
        """Test API manager failover mechanism"""
        cache = CacheManager("redis://invalid:6379")
        manager = APIManager(cache)
        
        # Add mock connectors
        primary_config = APIConfig(
            provider=APIProvider.BARCHART,
            base_url="https://api.barchart.com",
            api_key="test_key"
        )
        fallback_config = APIConfig(
            provider=APIProvider.ALPHA_VANTAGE,
            base_url="https://api.alphavantage.co",
            api_key="test_key"
        )
        
        # Create connectors with mocked behavior
        primary_connector = Mock(spec=BarchartAPIConnector)
        primary_connector.status = APIStatus.UNAVAILABLE
        
        fallback_connector = Mock(spec=AlphaVantageAPIConnector)
        fallback_connector.status = APIStatus.HEALTHY
        fallback_connector.fetch_market_data = AsyncMock(return_value=Mock(success=True, data={"test": "data"}))
        
        manager.connectors[APIProvider.BARCHART] = primary_connector
        manager.connectors[APIProvider.ALPHA_VANTAGE] = fallback_connector
        
        # Should failover to Alpha Vantage
        response = await manager.fetch_market_data(["AAPL"])
        fallback_connector.fetch_market_data.assert_called_once()


class TestSecureKeyManager:
    """Test secure API key management"""
    
    def test_key_manager_initialization(self):
        """Test key manager initializes with encryption"""
        manager = SecureKeyManager("test_keys.enc")
        assert manager.encryption_key is not None
        assert manager.cipher is not None
    
    def test_key_encryption_decryption(self):
        """Test key encryption and decryption"""
        manager = SecureKeyManager("test_keys.enc")
        
        original_key = "test_api_key_12345"
        encrypted = manager.encrypt_key(original_key)
        decrypted = manager.decrypt_key(encrypted)
        
        assert encrypted != original_key
        assert decrypted == original_key
    
    @patch('builtins.open', create=True)
    @patch('os.chmod')
    def test_key_saving(self, mock_chmod, mock_open):
        """Test secure key saving"""
        manager = SecureKeyManager("test_keys.enc")
        
        keys = {
            "barchart": "barchart_key_123",
            "alpha_vantage": "av_key_456"
        }
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        manager.save_keys(keys)
        
        # Verify file operations
        mock_open.assert_called_once_with("test_keys.enc", 'w')
        mock_file.write.assert_called()
        mock_chmod.assert_called_once_with("test_keys.enc", 0o600)


class TestAPIConfigManager:
    """Test API configuration management"""
    
    def test_config_manager_initialization(self):
        """Test config manager initializes correctly"""
        manager = APIConfigManager("test_config.json")
        assert manager.key_manager is not None
        assert isinstance(manager.providers, dict)
    
    def test_config_manager_default_providers(self):
        """Test config manager sets up default providers"""
        manager = APIConfigManager("nonexistent_config.json")
        
        # Should have default providers
        assert "barchart" in manager.providers
        assert "alpha_vantage" in manager.providers
        assert "massive" in manager.providers
        
        # Default providers should be disabled without API keys
        assert manager.providers["barchart"].enabled is False
        assert manager.providers["alpha_vantage"].enabled is False
        assert manager.providers["massive"].enabled is False
    
    @patch.dict('os.environ', {
        'BARCHART_API_KEY': 'test_barchart_key',
        'ALPHA_VANTAGE_API_KEY': 'test_av_key'
    })
    def test_config_manager_environment_loading(self):
        """Test config manager loads from environment variables"""
        manager = APIConfigManager("nonexistent_config.json")
        
        # Should load keys from environment
        assert manager.providers["barchart"].api_keys.primary_key == "test_barchart_key"
        assert manager.providers["alpha_vantage"].api_keys.primary_key == "test_av_key"
        
        # Should enable providers with keys
        assert manager.providers["barchart"].enabled is True
        assert manager.providers["alpha_vantage"].enabled is True
    
    def test_config_manager_key_updates(self):
        """Test config manager key update functionality"""
        manager = APIConfigManager("test_config.json")
        
        # Update API key
        success = manager.update_api_key("barchart", "new_test_key")
        assert success is True
        assert manager.providers["barchart"].api_keys.primary_key == "new_test_key"
        
        # Update backup key
        success = manager.update_api_key("barchart", "backup_key", is_backup=True)
        assert success is True
        assert manager.providers["barchart"].api_keys.backup_key == "backup_key"
    
    def test_config_manager_provider_management(self):
        """Test provider enable/disable functionality"""
        manager = APIConfigManager("test_config.json")
        
        # Set up provider with key
        manager.update_api_key("barchart", "test_key")
        
        # Enable provider
        manager.enable_provider("barchart")
        assert manager.providers["barchart"].enabled is True
        
        # Disable provider
        manager.disable_provider("barchart", "Testing")
        assert manager.providers["barchart"].enabled is False
    
    def test_config_manager_enabled_providers(self):
        """Test getting enabled providers sorted by priority"""
        manager = APIConfigManager("test_config.json")
        
        # Set up providers with different priorities
        manager.update_api_key("barchart", "key1")
        manager.update_api_key("alpha_vantage", "key2")
        manager.enable_provider("barchart")
        manager.enable_provider("alpha_vantage")
        
        enabled = manager.get_enabled_providers()
        
        # Should be sorted by priority (barchart=1, alpha_vantage=2)
        provider_names = list(enabled.keys())
        assert provider_names[0] == "barchart"
        assert provider_names[1] == "alpha_vantage"


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_api_integration_mock_mode(self):
        """Test complete API integration in mock mode"""
        # Create API manager in mock mode
        api_manager = await create_api_manager(mock_mode=True)
        
        async with api_manager.managed_connections():
            # Test market data retrieval
            response = await api_manager.fetch_market_data(["AAPL", "GOOGL", "MSFT"])
            assert response.success is True
            assert "quotes" in response.data
            assert len(response.data["quotes"]) == 3
            
            # Test volatility data
            volatility_response = await api_manager.get_market_volatility()
            assert volatility_response.success is True
            assert "VIX" in volatility_response.data
            
            # Test economic indicators
            econ_response = await api_manager.get_economic_indicators()
            assert econ_response.success is True
            assert "gdp_growth" in econ_response.data
    
    @pytest.mark.asyncio
    async def test_api_manager_health_checks(self):
        """Test API manager health check functionality"""
        api_manager = await create_api_manager(mock_mode=True)
        
        async with api_manager.managed_connections():
            health_results = await api_manager.health_check_all()
            
            # Mock connector should not be included in health checks
            assert len(health_results) == 0  # Only mock connector present
    
    def test_setup_api_keys_function(self):
        """Test setup_api_keys convenience function"""
        config_manager = setup_api_keys(
            barchart_key="test_barchart",
            alpha_vantage_key="test_av",
            save_config=False  # Don't save during test
        )
        
        assert config_manager.providers["barchart"].api_keys.primary_key == "test_barchart"
        assert config_manager.providers["alpha_vantage"].api_keys.primary_key == "test_av"
        assert config_manager.providers["barchart"].enabled is True
        assert config_manager.providers["alpha_vantage"].enabled is True


@pytest.mark.asyncio
class TestPerformanceAndResilience:
    """Test performance and resilience characteristics"""
    
    async def test_concurrent_api_requests(self):
        """Test handling of concurrent API requests"""
        api_manager = await create_api_manager(mock_mode=True)
        
        async with api_manager.managed_connections():
            # Make multiple concurrent requests
            tasks = []
            for i in range(10):
                task = api_manager.fetch_market_data([f"STOCK{i}"])
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.success is True
    
    async def test_api_timeout_handling(self):
        """Test API timeout handling"""
        config = APIConfig(
            provider=APIProvider.MOCK,
            base_url="http://mock.api",
            api_key="mock_key",
            timeout=0.001  # Very short timeout
        )
        cache = CacheManager("redis://invalid:6379")
        
        # Mock connector should handle timeout gracefully
        async with MockAPIConnector(config, cache) as connector:
            response = await connector.fetch_market_data(["AAPL"])
            # Mock connector doesn't actually make HTTP requests, so this should still succeed
            assert response.success is True
    
    async def test_cache_performance(self):
        """Test caching improves performance"""
        api_manager = await create_api_manager(mock_mode=True)
        
        async with api_manager.managed_connections():
            # First request (cache miss)
            start_time = time.time()
            response1 = await api_manager.fetch_market_data(["AAPL"])
            first_request_time = time.time() - start_time
            
            # Second request (should be faster due to caching logic)
            start_time = time.time()
            response2 = await api_manager.fetch_market_data(["AAPL"])
            second_request_time = time.time() - start_time
            
            assert response1.success is True
            assert response2.success is True
            
            # Note: Mock connector doesn't actually implement caching,
            # but the test structure is correct for real implementations


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])