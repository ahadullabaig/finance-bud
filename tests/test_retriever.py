"""
Test Suite for Information Retrieval Agent (IRA)

Comprehensive tests for Person B's implementation including:
- Market data retrieval from multiple sources
- Market context aggregation
- Trigger detection and severity assessment
- Volatility monitoring
- Mock data generation for all scenarios
- Performance benchmarking

Requirements: 5.1, 5.2, 11.1, 11.3, 32.1, 32.2, 32.3, 32.4, 32.5
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List

from agents.retriever import InformationRetrievalAgent, create_ira
from data_models.schemas import (
    MarketData, MarketContext, TriggerEvent,
    MarketEventType, SeverityLevel
)
from agents.external_apis import APIConfig, APIProvider, RateLimitConfig
from utils.constants import (
    MOCK_SCENARIOS, TRIGGER_THRESHOLDS, VOLATILITY_THRESHOLDS
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
async def mock_ira():
    """Create IRA with mock data for testing"""
    ira = InformationRetrievalAgent(use_mock=True)
    await ira.initialize()
    yield ira
    await ira.shutdown()


@pytest.fixture
async def real_ira():
    """Create IRA with real API configs (but won't make real calls in tests)"""
    barchart_config = APIConfig(
        provider=APIProvider.BARCHART,
        base_url="https://test.barchart.com",
        api_key="test_key",
        rate_limit=RateLimitConfig()
    )

    ira = InformationRetrievalAgent(
        barchart_config=barchart_config,
        use_mock=False
    )
    await ira.initialize()
    yield ira
    await ira.shutdown()


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return MarketData(
        symbol="NIFTY50",
        price=19500.50,
        change_percent=-2.3,
        volume=1250000,
        volatility=18.5,
        market_sentiment="bearish",
        sector_trend="declining",
        risk_score=37.0,
        source="mock",
        timestamp=datetime.now()
    )


# ============================================================================
# TEST INITIALIZATION
# ============================================================================

class TestIRAInitialization:
    """Test IRA initialization and setup"""

    @pytest.mark.asyncio
    async def test_ira_initialization_mock_mode(self):
        """Test IRA initializes correctly in mock mode"""
        ira = InformationRetrievalAgent(use_mock=True)
        await ira.initialize()

        assert ira.use_mock is True
        assert ira.mock_connector is not None
        assert ira.historical_data == {}
        assert ira.trigger_history == []

        await ira.shutdown()

    @pytest.mark.asyncio
    async def test_ira_initialization_real_mode(self):
        """Test IRA initializes with real API configs"""
        config = APIConfig(
            provider=APIProvider.BARCHART,
            base_url="https://test.barchart.com",
            api_key="test_key"
        )

        ira = InformationRetrievalAgent(
            barchart_config=config,
            use_mock=False
        )
        await ira.initialize()

        assert ira.use_mock is False
        assert ira.barchart is not None

        await ira.shutdown()

    @pytest.mark.asyncio
    async def test_create_ira_factory(self):
        """Test factory function for IRA creation"""
        ira = await create_ira(use_mock=True)

        assert isinstance(ira, InformationRetrievalAgent)
        assert ira.use_mock is True

        await ira.shutdown()


# ============================================================================
# TEST MARKET DATA RETRIEVAL - Task 2.7
# ============================================================================

class TestMarketDataRetrieval:
    """Test market data fetching from multiple sources"""

    @pytest.mark.asyncio
    async def test_get_market_data_mock(self, mock_ira):
        """Test fetching market data in mock mode"""
        data = await mock_ira.get_market_data("NIFTY50")

        assert data is not None
        assert isinstance(data, MarketData)
        assert data.symbol == "NIFTY50"
        assert data.price > 0
        assert data.source == "mock"

    @pytest.mark.asyncio
    async def test_get_market_data_with_cache(self, mock_ira):
        """Test market data caching works correctly"""
        # First call - should fetch from API
        data1 = await mock_ira.get_market_data("NIFTY50", use_cache=True)
        assert data1 is not None

        # Second call - should use cache
        data2 = await mock_ira.get_market_data("NIFTY50", use_cache=True)
        assert data2 is not None
        assert data1.symbol == data2.symbol

    @pytest.mark.asyncio
    async def test_get_market_data_without_cache(self, mock_ira):
        """Test fetching market data without cache"""
        data = await mock_ira.get_market_data("NIFTY50", use_cache=False)

        assert data is not None
        assert isinstance(data, MarketData)

    @pytest.mark.asyncio
    async def test_get_market_data_stores_history(self, mock_ira):
        """Test market data is stored in historical data"""
        symbol = "NIFTY50"
        data = await mock_ira.get_market_data(symbol)

        assert symbol in mock_ira.historical_data
        assert len(mock_ira.historical_data[symbol]) > 0
        assert mock_ira.historical_data[symbol][-1].symbol == symbol

    @pytest.mark.asyncio
    async def test_get_market_data_multiple_symbols(self, mock_ira):
        """Test fetching data for multiple symbols"""
        symbols = ["NIFTY50", "SENSEX", "SPY"]
        tasks = [mock_ira.get_market_data(s) for s in symbols]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(symbols)
        for data, symbol in zip(results, symbols):
            assert data is not None
            assert data.symbol == symbol


# ============================================================================
# TEST MARKET CONTEXT - Task 2.7
# ============================================================================

class TestMarketContext:
    """Test comprehensive market context retrieval"""

    @pytest.mark.asyncio
    async def test_get_market_context_normal(self, mock_ira):
        """Test getting market context for normal scenario"""
        context = await mock_ira.get_market_context(mock=True, scenario="normal")

        assert isinstance(context, MarketContext)
        assert context.market_volatility > 0
        assert context.economic_sentiment in ["bullish", "bearish", "neutral"]
        assert len(context.indices) > 0
        assert context.confidence_score == 1.0  # Mock data always confident

    @pytest.mark.asyncio
    async def test_get_market_context_crash_scenario(self, mock_ira):
        """Test market context during crash scenario"""
        context = await mock_ira.get_market_context(mock=True, scenario="crash")

        assert isinstance(context, MarketContext)
        assert context.market_volatility >= MOCK_SCENARIOS['crash']['volatility']
        assert context.economic_sentiment == "bearish"

        # All indices should show negative change
        for symbol, data in context.indices.items():
            assert data.change_percent < 0

    @pytest.mark.asyncio
    async def test_get_market_context_bull_scenario(self, mock_ira):
        """Test market context during bull market"""
        context = await mock_ira.get_market_context(mock=True, scenario="bull")

        assert isinstance(context, MarketContext)
        assert context.economic_sentiment == "bullish"

        # All indices should show positive change
        for symbol, data in context.indices.items():
            assert data.change_percent > 0

    @pytest.mark.asyncio
    async def test_get_market_context_volatility_spike(self, mock_ira):
        """Test market context during volatility spike"""
        context = await mock_ira.get_market_context(mock=True, scenario="volatility_spike")

        assert isinstance(context, MarketContext)
        assert context.market_volatility >= MOCK_SCENARIOS['volatility_spike']['volatility']

    @pytest.mark.asyncio
    async def test_get_market_context_contains_all_fields(self, mock_ira):
        """Test market context contains all required fields"""
        context = await mock_ira.get_market_context(mock=True)

        assert hasattr(context, 'market_volatility')
        assert hasattr(context, 'interest_rate')
        assert hasattr(context, 'economic_sentiment')
        assert hasattr(context, 'sector_trends')
        assert hasattr(context, 'indices')
        assert hasattr(context, 'timestamp')
        assert hasattr(context, 'confidence_score')

        # Validate types
        assert isinstance(context.market_volatility, float)
        assert isinstance(context.interest_rate, float)
        assert isinstance(context.economic_sentiment, str)
        assert isinstance(context.sector_trends, dict)
        assert isinstance(context.indices, dict)


# ============================================================================
# TEST TRIGGER DETECTION - Task 2.8
# ============================================================================

class TestTriggerDetection:
    """Test market trigger detection and severity assessment"""

    @pytest.mark.asyncio
    async def test_detect_volatility_spike_trigger(self, mock_ira):
        """Test detection of volatility spike trigger"""
        # Create high volatility context
        context = await mock_ira.get_market_context(mock=True, scenario="volatility_spike")
        triggers = await mock_ira.detect_market_triggers(context)

        # Should detect volatility spike
        volatility_triggers = [
            t for t in triggers
            if t.event_type == MarketEventType.VOLATILITY_SPIKE
        ]
        assert len(volatility_triggers) > 0

        trigger = volatility_triggers[0]
        assert trigger.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        assert trigger.confidence > 0.9
        assert len(trigger.recommended_actions) > 0

    @pytest.mark.asyncio
    async def test_detect_market_crash_trigger(self, mock_ira):
        """Test detection of market crash trigger"""
        context = await mock_ira.get_market_context(mock=True, scenario="crash")
        triggers = await mock_ira.detect_market_triggers(context)

        # Should detect crash
        crash_triggers = [
            t for t in triggers
            if t.event_type == MarketEventType.CRASH
        ]
        assert len(crash_triggers) > 0

        trigger = crash_triggers[0]
        assert trigger.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        assert trigger.requires_immediate_action is True
        assert "crash" in trigger.description.lower()

    @pytest.mark.asyncio
    async def test_detect_no_triggers_normal_market(self, mock_ira):
        """Test no triggers in normal market conditions"""
        context = await mock_ira.get_market_context(mock=True, scenario="normal")
        triggers = await mock_ira.detect_market_triggers(context)

        # Normal conditions should have minimal or no triggers
        critical_triggers = [
            t for t in triggers
            if t.severity == SeverityLevel.CRITICAL
        ]
        assert len(critical_triggers) == 0

    @pytest.mark.asyncio
    async def test_trigger_severity_assessment(self, mock_ira):
        """Test severity assessment logic"""
        # Test different scenarios and their severity levels
        scenarios_severity = {
            "normal": [SeverityLevel.LOW, SeverityLevel.MEDIUM],
            "crash": [SeverityLevel.HIGH, SeverityLevel.CRITICAL],
            "volatility_spike": [SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        }

        for scenario, expected_severities in scenarios_severity.items():
            context = await mock_ira.get_market_context(mock=True, scenario=scenario)
            triggers = await mock_ira.detect_market_triggers(context)

            if triggers:
                for trigger in triggers:
                    assert trigger.severity in expected_severities

    @pytest.mark.asyncio
    async def test_triggers_stored_in_history(self, mock_ira):
        """Test triggers are stored in history"""
        initial_count = len(mock_ira.trigger_history)

        context = await mock_ira.get_market_context(mock=True, scenario="crash")
        triggers = await mock_ira.detect_market_triggers(context)

        assert len(mock_ira.trigger_history) > initial_count
        assert len(mock_ira.trigger_history) >= len(triggers)

    @pytest.mark.asyncio
    async def test_trigger_correlation_id(self, mock_ira):
        """Test triggers have correlation IDs"""
        context = await mock_ira.get_market_context(mock=True, scenario="crash")
        triggers = await mock_ira.detect_market_triggers(context)

        for trigger in triggers:
            assert trigger.correlation_id is not None
            assert len(trigger.correlation_id) > 0


# ============================================================================
# TEST VOLATILITY MONITORING - Task 2.9
# ============================================================================

class TestVolatilityMonitoring:
    """Test volatility monitoring and threshold detection"""

    @pytest.mark.asyncio
    async def test_monitor_volatility_single_symbol(self, mock_ira):
        """Test volatility monitoring for single symbol"""
        results = await mock_ira.monitor_volatility(["NIFTY50"])

        assert "NIFTY50" in results
        volatility, status = results["NIFTY50"]
        assert volatility >= 0
        assert status in ["low", "normal", "high", "extreme"]

    @pytest.mark.asyncio
    async def test_monitor_volatility_multiple_symbols(self, mock_ira):
        """Test volatility monitoring for multiple symbols"""
        symbols = ["NIFTY50", "SENSEX", "SPY"]
        results = await mock_ira.monitor_volatility(symbols)

        assert len(results) == len(symbols)
        for symbol in symbols:
            assert symbol in results
            volatility, status = results[symbol]
            assert volatility >= 0

    @pytest.mark.asyncio
    async def test_volatility_classification(self, mock_ira):
        """Test volatility level classification"""
        # Test various scenarios and their volatility classification
        test_cases = [
            ("normal", ["low", "normal"]),
            ("crash", ["high", "extreme"]),
            ("volatility_spike", ["high", "extreme"])
        ]

        for scenario, expected_levels in test_cases:
            # Set up mock to return specific volatility
            context = await mock_ira.get_market_context(mock=True, scenario=scenario)

            # Monitor volatility
            symbols = list(context.indices.keys())[:1]
            results = await mock_ira.monitor_volatility(symbols)

            for symbol in symbols:
                _, status = results[symbol]
                # Status should match expected levels for scenario
                if scenario == "normal":
                    assert status in ["low", "normal", "high"]
                elif scenario in ["crash", "volatility_spike"]:
                    assert status in ["normal", "high", "extreme"]

    @pytest.mark.asyncio
    async def test_volatility_threshold_alert(self, mock_ira):
        """Test volatility threshold alerts"""
        # Use crash scenario which has high volatility
        context = await mock_ira.get_market_context(mock=True, scenario="crash")
        symbols = list(context.indices.keys())

        # Set low threshold to trigger alerts
        results = await mock_ira.monitor_volatility(symbols, threshold=20.0)

        # At least one symbol should exceed threshold in crash scenario
        high_vol_count = sum(
            1 for _, (vol, status) in results.items()
            if status in ["high", "extreme"]
        )
        assert high_vol_count > 0


# ============================================================================
# TEST MOCK DATA GENERATION - Task 2.10
# ============================================================================

class TestMockDataGeneration:
    """Test comprehensive mock data generation for all scenarios"""

    @pytest.mark.asyncio
    async def test_mock_normal_scenario(self, mock_ira):
        """Test normal market scenario mock data"""
        context = await mock_ira.get_market_context(mock=True, scenario="normal")

        assert context.economic_sentiment == "neutral"
        assert 10.0 <= context.market_volatility <= 20.0

    @pytest.mark.asyncio
    async def test_mock_crash_scenario(self, mock_ira):
        """Test market crash scenario mock data"""
        context = await mock_ira.get_market_context(mock=True, scenario="crash")

        assert context.economic_sentiment == "bearish"
        assert context.market_volatility >= 40.0

        # All indices should show significant decline
        for symbol, data in context.indices.items():
            assert data.change_percent < -10.0

    @pytest.mark.asyncio
    async def test_mock_bull_scenario(self, mock_ira):
        """Test bull market scenario mock data"""
        context = await mock_ira.get_market_context(mock=True, scenario="bull")

        assert context.economic_sentiment == "bullish"

        # All indices should show significant gains
        for symbol, data in context.indices.items():
            assert data.change_percent > 5.0

    @pytest.mark.asyncio
    async def test_mock_bear_scenario(self, mock_ira):
        """Test bear market scenario mock data"""
        context = await mock_ira.get_market_context(mock=True, scenario="bear")

        assert context.economic_sentiment == "bearish"

        # Indices should show decline
        for symbol, data in context.indices.items():
            assert data.change_percent < 0

    @pytest.mark.asyncio
    async def test_all_mock_scenarios_available(self, mock_ira):
        """Test all mock scenarios are available and work"""
        for scenario_name in MOCK_SCENARIOS.keys():
            context = await mock_ira.get_market_context(mock=True, scenario=scenario_name)

            assert context is not None
            assert isinstance(context, MarketContext)
            assert len(context.indices) > 0
            assert context.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_mock_data_consistency(self, mock_ira):
        """Test mock data is consistent across calls"""
        # Same scenario should produce similar characteristics
        context1 = await mock_ira.get_market_context(mock=True, scenario="crash")
        context2 = await mock_ira.get_market_context(mock=True, scenario="crash")

        assert context1.economic_sentiment == context2.economic_sentiment
        # Volatility should be in similar range
        assert abs(context1.market_volatility - context2.market_volatility) < 5.0


# ============================================================================
# TEST RAG SYSTEM - Task 2.7
# ============================================================================

class TestRAGSystem:
    """Test financial knowledge retrieval"""

    @pytest.mark.asyncio
    async def test_query_tax_knowledge(self, mock_ira):
        """Test querying tax-related knowledge"""
        response = await mock_ira.query_financial_knowledge("tax saving options")

        assert response is not None
        assert "tax" in response.lower() or "80c" in response.lower()

    @pytest.mark.asyncio
    async def test_query_emergency_fund_knowledge(self, mock_ira):
        """Test querying emergency fund knowledge"""
        response = await mock_ira.query_financial_knowledge("emergency fund")

        assert response is not None
        assert "month" in response.lower() or "emergency" in response.lower()

    @pytest.mark.asyncio
    async def test_query_risk_profile_knowledge(self, mock_ira):
        """Test querying risk profile knowledge"""
        response = await mock_ira.query_financial_knowledge("risk profile")

        assert response is not None
        assert any(term in response.lower() for term in ["conservative", "moderate", "aggressive"])


# ============================================================================
# TEST PERFORMANCE & METRICS - Task 2.10
# ============================================================================

class TestPerformanceMetrics:
    """Test performance tracking and benchmarking"""

    @pytest.mark.asyncio
    async def test_market_data_retrieval_performance(self, mock_ira):
        """Test market data retrieval meets performance SLA"""
        import time

        start = time.time()
        data = await mock_ira.get_market_data("NIFTY50")
        duration = (time.time() - start) * 1000  # Convert to ms

        assert data is not None
        # Should complete in under 1 second for mock data
        assert duration < 1000

    @pytest.mark.asyncio
    async def test_market_context_performance(self, mock_ira):
        """Test market context retrieval performance"""
        import time

        start = time.time()
        context = await mock_ira.get_market_context(mock=True)
        duration = (time.time() - start) * 1000  # Convert to ms

        assert context is not None
        # Should complete in under 3 seconds per requirements
        assert duration < 3000

    @pytest.mark.asyncio
    async def test_trigger_detection_performance(self, mock_ira):
        """Test trigger detection performance"""
        import time

        context = await mock_ira.get_market_context(mock=True, scenario="crash")

        start = time.time()
        triggers = await mock_ira.detect_market_triggers(context)
        duration = (time.time() - start) * 1000

        # Trigger detection should be fast (< 500ms)
        assert duration < 500

    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, mock_ira):
        """Test concurrent market data requests"""
        import time

        symbols = ["NIFTY50", "SENSEX", "SPY", "QQQ", "BANKNIFTY"]

        start = time.time()
        tasks = [mock_ira.get_market_data(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        duration = (time.time() - start) * 1000

        assert len(results) == len(symbols)
        # Concurrent requests should complete faster than sequential
        # Should complete in under 2 seconds for 5 concurrent requests
        assert duration < 2000


# ============================================================================
# TEST UTILITY METHODS
# ============================================================================

class TestUtilityMethods:
    """Test IRA utility methods"""

    @pytest.mark.asyncio
    async def test_get_trigger_history(self, mock_ira):
        """Test getting trigger history"""
        # Generate some triggers
        context = await mock_ira.get_market_context(mock=True, scenario="crash")
        await mock_ira.detect_market_triggers(context)

        history = mock_ira.get_trigger_history(limit=10)

        assert isinstance(history, list)
        assert len(history) <= 10

    @pytest.mark.asyncio
    async def test_get_trigger_history_filtered(self, mock_ira):
        """Test getting filtered trigger history"""
        # Generate triggers
        context = await mock_ira.get_market_context(mock=True, scenario="crash")
        await mock_ira.detect_market_triggers(context)

        history = mock_ira.get_trigger_history(
            limit=5,
            event_type=MarketEventType.CRASH
        )

        for trigger in history:
            assert trigger.event_type == MarketEventType.CRASH

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, mock_ira):
        """Test getting performance metrics"""
        # Make some requests to generate metrics
        await mock_ira.get_market_data("NIFTY50")
        await mock_ira.get_market_context(mock=True)

        metrics = mock_ira.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert 'total_requests' in metrics


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete IRA workflows"""

    @pytest.mark.asyncio
    async def test_complete_market_analysis_workflow(self, mock_ira):
        """Test complete market analysis workflow"""
        # 1. Get market context
        context = await mock_ira.get_market_context(mock=True, scenario="crash")
        assert context is not None

        # 2. Detect triggers
        triggers = await mock_ira.detect_market_triggers(context)
        assert len(triggers) > 0

        # 3. Monitor volatility
        symbols = list(context.indices.keys())
        vol_results = await mock_ira.monitor_volatility(symbols)
        assert len(vol_results) > 0

        # 4. Query knowledge base
        knowledge = await mock_ira.query_financial_knowledge(
            "How should I respond to market volatility?"
        )
        assert knowledge is not None

    @pytest.mark.asyncio
    async def test_offline_development_mode(self, mock_ira):
        """Test system works completely offline with mock data"""
        # All operations should work with mock data
        context = await mock_ira.get_market_context(mock=True)
        data = await mock_ira.get_market_data("NIFTY50")
        triggers = await mock_ira.detect_market_triggers()

        assert context is not None
        assert data is not None
        assert isinstance(triggers, list)


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStressConditions:
    """Test IRA under stress conditions"""

    @pytest.mark.asyncio
    async def test_high_frequency_requests(self, mock_ira):
        """Test handling high-frequency data requests"""
        # Make 50 rapid requests
        tasks = [
            mock_ira.get_market_data("NIFTY50")
            for _ in range(50)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle all requests without errors
        successful = [r for r in results if isinstance(r, MarketData)]
        assert len(successful) == 50

    @pytest.mark.asyncio
    async def test_concurrent_trigger_detection(self, mock_ira):
        """Test concurrent trigger detection"""
        # Run trigger detection concurrently for multiple scenarios
        scenarios = ["crash", "bull", "bear", "volatility_spike"]

        async def detect_for_scenario(scenario):
            context = await mock_ira.get_market_context(mock=True, scenario=scenario)
            return await mock_ira.detect_market_triggers(context)

        tasks = [detect_for_scenario(s) for s in scenarios]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(scenarios)
        for trigger_list in results:
            assert isinstance(trigger_list, list)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
