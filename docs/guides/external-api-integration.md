# External API Integration Framework

This document describes the external API integration framework for FinPilot VP-MAS, which provides comprehensive market data access with rate limiting, failover, caching, and security features.

## Overview

The external API integration framework supports multiple market data providers with intelligent failover, rate limiting, caching, and secure key management. It's designed to handle production workloads while providing offline development capabilities through mock mode.

## Supported Providers

### Primary Providers
- **Barchart API** - Real-time market data, quotes, and economic indicators
- **Alpha Vantage API** - Stock quotes, technical indicators, and fundamental data
- **Massive API** - Comprehensive financial data and analytics

### Mock Provider
- **Mock API** - Realistic simulated data for offline development and testing

## Features

### 🔒 Security
- **Encrypted API Key Storage** - All API keys are encrypted at rest
- **Key Rotation Support** - Automatic key rotation reminders and management
- **Secure Configuration** - Environment variable and encrypted file support

### ⚡ Performance
- **Redis Caching** - Intelligent caching with TTL management
- **Rate Limiting** - Token bucket algorithm with burst support
- **Connection Pooling** - Efficient HTTP connection management

### 🛡️ Resilience
- **Circuit Breakers** - Automatic failure detection and recovery
- **Failover Support** - Automatic provider switching on failures
- **Retry Logic** - Exponential backoff with configurable limits

### 📊 Monitoring
- **Health Checks** - Continuous provider health monitoring
- **Performance Metrics** - Response time and success rate tracking
- **Usage Analytics** - API usage and quota monitoring

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup API Keys (Interactive)

```bash
python setup_apis.py --interactive
```

### 3. Setup API Keys (Command Line)

```bash
python setup_apis.py \
  --barchart-key YOUR_BARCHART_KEY \
  --alpha-vantage-key YOUR_ALPHA_VANTAGE_KEY \
  --massive-key YOUR_MASSIVE_KEY
```

### 4. Test Connections

```bash
python setup_apis.py --test-connections
```

### 5. Use Mock Mode for Development

```bash
python setup_apis.py --mock-mode
```

## Configuration

### Environment Variables

```bash
# API Keys
export BARCHART_API_KEY="your_barchart_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
export MASSIVE_API_KEY="your_massive_key"

# Backup Keys (optional)
export BARCHART_BACKUP_KEY="your_backup_key"
export ALPHA_VANTAGE_BACKUP_KEY="your_backup_key"
export MASSIVE_BACKUP_KEY="your_backup_key"

# Encryption Key for stored API keys
export FINPILOT_ENCRYPTION_KEY="your_base64_encryption_key"

# Redis Configuration
export REDIS_URL="redis://localhost:6379"
```

### Configuration File

The system uses `api_config.json` for provider configuration:

```json
{
  "providers": {
    "barchart": {
      "base_url": "https://api.barchart.com",
      "rate_limits": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 10000
      },
      "timeout": 30.0,
      "max_retries": 3,
      "priority": 1,
      "enabled": true
    }
  }
}
```

## Usage Examples

### Basic Usage

```python
from agents.external_apis import create_api_manager

# Create API manager
api_manager = await create_api_manager(
    barchart_key="your_key",
    alpha_vantage_key="your_key",
    mock_mode=False
)

# Use with context manager
async with api_manager.managed_connections():
    # Fetch market data
    response = await api_manager.fetch_market_data(["AAPL", "GOOGL"])
    
    if response.success:
        quotes = response.data["quotes"]
        for symbol, data in quotes.items():
            print(f"{symbol}: ${data['lastPrice']}")
    
    # Get market volatility
    volatility = await api_manager.get_market_volatility()
    
    # Get economic indicators
    indicators = await api_manager.get_economic_indicators()
```

### Mock Mode for Testing

```python
# Enable mock mode for offline development
api_manager = await create_api_manager(mock_mode=True)

async with api_manager.managed_connections():
    # Returns realistic mock data
    response = await api_manager.fetch_market_data(["AAPL", "GOOGL"])
    print(response.data)  # Mock market data
```

### Advanced Configuration

```python
from agents.external_apis import APIConfig, RateLimitConfig, APIProvider
from agents.api_config import setup_api_keys

# Setup with custom configuration
config_manager = setup_api_keys(
    barchart_key="your_key",
    save_config=True
)

# Create API manager with custom settings
api_manager = await create_api_manager(
    barchart_key="your_key",
    redis_url="redis://localhost:6379"
)
```

## API Reference

### APIManager Methods

#### `fetch_market_data(symbols: List[str], **kwargs) -> APIResponse`
Fetch real-time market data for given symbols.

**Parameters:**
- `symbols`: List of stock symbols (e.g., ["AAPL", "GOOGL"])
- `**kwargs`: Additional provider-specific parameters

**Returns:**
- `APIResponse` with market quotes data

#### `get_market_volatility(**kwargs) -> APIResponse`
Get market volatility indicators (VIX, etc.).

**Returns:**
- `APIResponse` with volatility data

#### `get_economic_indicators(**kwargs) -> APIResponse`
Get economic indicators (GDP, inflation, etc.).

**Returns:**
- `APIResponse` with economic data

#### `health_check_all() -> Dict[APIProvider, bool]`
Perform health checks on all providers.

**Returns:**
- Dictionary mapping providers to health status

### APIResponse Structure

```python
{
    "success": bool,
    "data": dict,  # Response data
    "error": str,  # Error message if failed
    "provider": APIProvider,  # Which provider served the request
    "timestamp": datetime,
    "response_time": float,  # Response time in seconds
    "from_cache": bool,  # Whether data came from cache
    "rate_limit_remaining": int  # Remaining API calls
}
```

## Rate Limiting

The system implements intelligent rate limiting with the following features:

### Token Bucket Algorithm
- **Burst Capacity**: Allow short bursts of requests
- **Refill Rate**: Steady token replenishment
- **Multiple Time Windows**: Per-minute, per-hour, per-day limits

### Provider-Specific Limits
- **Barchart**: 60/min, 1000/hour, 10000/day
- **Alpha Vantage**: 5/min, 500/hour, 500/day
- **Massive**: 100/min, 2000/hour, 20000/day

### Automatic Backoff
- Exponential backoff on rate limit hits
- Respect `Retry-After` headers
- Maximum backoff of 5 minutes

## Caching Strategy

### Redis-Based Caching
- **TTL Management**: Configurable time-to-live for different data types
- **Cache Keys**: Structured keys with provider and request parameters
- **Fallback**: Graceful degradation when Redis unavailable

### Cache Configuration
```python
# Default TTL values
market_data_ttl = 300  # 5 minutes
volatility_ttl = 600   # 10 minutes
economic_ttl = 3600    # 1 hour
```

## Circuit Breaker Pattern

### States
- **Closed**: Normal operation, requests pass through
- **Open**: Failures detected, requests blocked
- **Half-Open**: Testing recovery, limited requests allowed

### Configuration
```python
failure_threshold = 5      # Failures before opening
recovery_timeout = 60      # Seconds before trying half-open
success_threshold = 2      # Successes needed to close
```

## Error Handling

### Error Types
- **Rate Limited**: 429 status, automatic retry with backoff
- **Authentication**: 401 status, try backup key if available
- **Timeout**: Network timeout, retry with exponential backoff
- **Server Error**: 5xx status, failover to next provider
- **Circuit Open**: Too many failures, block requests temporarily

### Failover Logic
1. Try primary provider (Barchart)
2. On failure, try first fallback (Alpha Vantage)
3. On failure, try second fallback (Massive)
4. On all failures, return error or use mock data

## Security Best Practices

### API Key Management
- **Encryption**: All keys encrypted with Fernet (AES 128)
- **File Permissions**: Key files have 600 permissions (owner read/write only)
- **Environment Variables**: Support for environment-based configuration
- **Key Rotation**: Automatic rotation reminders

### Network Security
- **HTTPS Only**: All API calls use HTTPS
- **Timeout Limits**: Prevent hanging connections
- **User Agent**: Proper identification in requests

## Monitoring and Observability

### Metrics Collected
- **Response Times**: P50, P95, P99 percentiles
- **Success Rates**: Per provider and overall
- **Cache Hit Rates**: Cache effectiveness
- **Rate Limit Usage**: Quota consumption
- **Error Rates**: By error type and provider

### Health Monitoring
- **Provider Health**: Regular health checks
- **Circuit Breaker Status**: Open/closed state tracking
- **Cache Status**: Redis connectivity
- **Key Rotation**: Expiration warnings

## Testing

### Unit Tests
```bash
# Run all external API tests
pytest tests/test_external_apis.py -v

# Run specific test categories
pytest tests/test_external_apis.py::TestCircuitBreaker -v
pytest tests/test_external_apis.py::TestRateLimiter -v
pytest tests/test_external_apis.py::TestMockAPIConnector -v
```

### Integration Tests
```bash
# Test with mock mode
python setup_apis.py --test-mock

# Test with real APIs (requires keys)
python setup_apis.py --test-connections
```

### Performance Tests
```bash
# Run performance benchmarks
pytest tests/test_external_apis.py::TestPerformanceAndResilience -v
```

## Troubleshooting

### Common Issues

#### "Redis connection failed"
- **Cause**: Redis server not running or wrong URL
- **Solution**: Start Redis or update `REDIS_URL`
- **Fallback**: System works without Redis (no caching)

#### "All API providers failed"
- **Cause**: No valid API keys or all providers down
- **Solution**: Check API keys and provider status
- **Fallback**: Enable mock mode for development

#### "Rate limited"
- **Cause**: Exceeded provider rate limits
- **Solution**: Wait for rate limit reset or use different provider
- **Prevention**: Monitor usage and implement request queuing

#### "Circuit breaker is open"
- **Cause**: Too many consecutive failures
- **Solution**: Wait for recovery timeout or fix underlying issue
- **Prevention**: Implement proper error handling and monitoring

### Debug Mode
```python
import logging
logging.getLogger('agents.external_apis').setLevel(logging.DEBUG)
```

### Configuration Validation
```bash
python setup_apis.py --validate
python setup_apis.py --show-config
```

## Production Deployment

### Infrastructure Requirements
- **Redis**: For caching (recommended: Redis 6+)
- **Network**: Outbound HTTPS access to API providers
- **Storage**: Encrypted key storage location
- **Monitoring**: Metrics collection and alerting

### Environment Setup
```bash
# Production environment variables
export FINPILOT_ENV=production
export REDIS_URL=redis://prod-redis:6379
export LOG_LEVEL=INFO

# API keys (use secure secret management)
export BARCHART_API_KEY=$(vault kv get -field=key secret/barchart)
export ALPHA_VANTAGE_API_KEY=$(vault kv get -field=key secret/alphavantage)
```

### Monitoring Setup
- **Health Checks**: Regular provider health monitoring
- **Alerting**: Rate limit and error rate alerts
- **Dashboards**: API usage and performance metrics
- **Log Aggregation**: Centralized logging for debugging

## Contributing

### Adding New Providers
1. Create new connector class inheriting from `BaseAPIConnector`
2. Implement required methods: `fetch_market_data`, `get_market_volatility`, `get_economic_indicators`
3. Add provider configuration to `APIConfigManager`
4. Add tests for the new provider
5. Update documentation

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include error handling and logging

### Testing Requirements
- Unit tests for all new functionality
- Integration tests with mock data
- Performance tests for critical paths
- Security tests for key management

## License

This external API integration framework is part of the FinPilot VP-MAS project and follows the same licensing terms.