"""
FinPilot Multi-Agent System - Information Retrieval Agent (IRA)

The Information Retrieval Agent is responsible for:
- Real-time market data fetching from multiple sources
- Volatility monitoring and threshold detection
- Market trigger detection and severity assessment
- RAG system for financial knowledge retrieval
- Regulatory change monitoring
- Multi-source data aggregation and enrichment

Person B - Tasks 2.6, 2.7, 2.8, 2.9, 2.10
Requirements: 5.1, 5.2, 5.3, 2.1, 2.2, 31.1, 31.2, 31.4, 31.5, 43.5
"""

import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from data_models.schemas import (
    MarketData, MarketContext, TriggerEvent, MarketEventType,
    SeverityLevel, PerformanceMetrics
)
from agents.external_apis import (
    BarchartAPIConnector, AlphaVantageAPIConnector,
    MassiveAPIConnector, MockAPIConnector, CacheManager,
    APIConfig, APIProvider, RateLimitConfig, APIResponse
)
from utils.logger import get_logger
from utils.constants import (
    VOLATILITY_THRESHOLDS, MARKET_CHANGE_THRESHOLDS,
    TRIGGER_THRESHOLDS, SEVERITY_SCORES, EVENT_IMPACT_MULTIPLIERS,
    MOCK_SCENARIOS, MOCK_INDICES, CACHE_TTL, API_TIMEOUT_SECONDS
)


class InformationRetrievalAgent:
    """
    Advanced Information Retrieval Agent with multi-source integration.

    Provides comprehensive market data, trigger detection, and intelligence
    gathering capabilities for the FinPilot multi-agent system.

    Requirements: 5.1, 5.2, 5.3, 31.1, 31.2, 31.4, 31.5
    """

    def __init__(
        self,
        barchart_config: Optional[APIConfig] = None,
        alphavantage_config: Optional[APIConfig] = None,
        massive_config: Optional[APIConfig] = None,
        redis_url: str = "redis://localhost:6379",
        use_mock: bool = False,
        log_level: str = "INFO"
    ):
        """
        Initialize the Information Retrieval Agent.

        Args:
            barchart_config: Configuration for Barchart API
            alphavantage_config: Configuration for Alpha Vantage API
            massive_config: Configuration for Massive API
            redis_url: Redis connection URL for caching
            use_mock: Whether to use mock data (for testing)
            log_level: Logging level
        """
        self.logger = get_logger("InformationRetrievalAgent", level=log_level)
        self.use_mock = use_mock

        # Initialize cache manager
        self.cache_manager = CacheManager(redis_url)

        # Initialize API connectors
        self._init_connectors(
            barchart_config,
            alphavantage_config,
            massive_config
        )

        # Historical data for trend analysis
        self.historical_data: Dict[str, List[MarketData]] = {}
        self.trigger_history: List[TriggerEvent] = []

        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []

        self.logger.info("InformationRetrievalAgent initialized", extra_data={
            'use_mock': use_mock,
            'redis_url': redis_url
        })

    def _init_connectors(
        self,
        barchart_config: Optional[APIConfig],
        alphavantage_config: Optional[APIConfig],
        massive_config: Optional[APIConfig]
    ):
        """Initialize API connectors"""
        # Mock connector (always available for testing)
        mock_config = APIConfig(
            provider=APIProvider.MOCK,
            base_url="http://mock",
            api_key="mock_key",
            rate_limit=RateLimitConfig(requests_per_minute=1000)
        )
        self.mock_connector = MockAPIConnector(mock_config, self.cache_manager)

        if self.use_mock:
            self.barchart = self.mock_connector
            self.alphavantage = self.mock_connector
            self.massive = self.mock_connector
            self.logger.info("Using mock connectors for all APIs")
        else:
            # Real API connectors
            self.barchart = BarchartAPIConnector(
                barchart_config, self.cache_manager
            ) if barchart_config else None

            self.alphavantage = AlphaVantageAPIConnector(
                alphavantage_config, self.cache_manager
            ) if alphavantage_config else None

            self.massive = MassiveAPIConnector(
                massive_config, self.cache_manager
            ) if massive_config else None

            # Use mock as fallback if no real APIs configured
            if not any([self.barchart, self.alphavantage, self.massive]):
                self.logger.warning("No real API connectors configured, using mock")
                self.barchart = self.mock_connector
                self.alphavantage = self.mock_connector
                self.massive = self.mock_connector

    async def initialize(self):
        """Initialize async resources (cache, connections)"""
        await self.cache_manager.connect()
        self.logger.info("IRA initialization complete")

    async def shutdown(self):
        """Cleanup resources"""
        await self.cache_manager.close()
        self.logger.info("IRA shutdown complete")

    # ========================================================================
    # CORE MARKET DATA RETRIEVAL - Task 2.7
    # ========================================================================

    async def get_market_data(
        self,
        symbol: str,
        use_cache: bool = True,
        timeout: float = API_TIMEOUT_SECONDS
    ) -> Optional[MarketData]:
        """
        Fetch market data for a symbol from best available source.

        Args:
            symbol: Stock/index symbol (e.g., "NIFTY50", "SPY")
            use_cache: Whether to use cached data
            timeout: Request timeout in seconds

        Returns:
            MarketData object or None if unavailable

        Requirements: 5.1, 5.2, 31.1
        """
        start_time = datetime.now()
        cache_key = f"market_data:{symbol}"

        # Check cache first
        if use_cache:
            cached = await self.cache_manager.get(cache_key)
            if cached:
                self.logger.debug(f"Cache hit for {symbol}")
                return MarketData(**cached)

        # Try primary sources in priority order
        sources = []
        if self.barchart:
            sources.append(("Barchart", self.barchart))
        if self.alphavantage:
            sources.append(("AlphaVantage", self.alphavantage))
        if self.massive:
            sources.append(("Massive", self.massive))

        market_data = None
        for source_name, connector in sources:
            try:
                response: APIResponse = await connector.get_quote(symbol)
                if response.success and response.data:
                    market_data = self._parse_market_data(
                        response.data, symbol, source_name.lower()
                    )
                    break
            except Exception as e:
                self.logger.warning(
                    f"Failed to fetch from {source_name}: {e}"
                )
                continue

        if market_data:
            # Cache the result
            await self.cache_manager.set(
                cache_key,
                market_data.model_dump(),
                ttl=CACHE_TTL['market_data']
            )

            # Store in historical data
            if symbol not in self.historical_data:
                self.historical_data[symbol] = []
            self.historical_data[symbol].append(market_data)

            # Keep only last 100 data points
            if len(self.historical_data[symbol]) > 100:
                self.historical_data[symbol] = self.historical_data[symbol][-100:]

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(
                f"Market data fetched for {symbol}",
                duration_ms=duration_ms
            )

        return market_data

    def _parse_market_data(
        self,
        raw_data: Dict[str, Any],
        symbol: str,
        source: str
    ) -> MarketData:
        """Parse raw API data into MarketData model"""
        # Handle different API response formats
        return MarketData(
            symbol=symbol,
            price=float(raw_data.get('price', raw_data.get('last', 0))),
            change_percent=float(raw_data.get('change_percent',
                                             raw_data.get('changePercent', 0))),
            volume=raw_data.get('volume'),
            volatility=raw_data.get('volatility'),
            market_sentiment=raw_data.get('sentiment', 'neutral'),
            sector_trend=raw_data.get('sector_trend'),
            correlation_indices=raw_data.get('correlation_indices'),
            predicted_volatility=raw_data.get('predicted_volatility'),
            risk_score=raw_data.get('risk_score'),
            source=source,
            timestamp=datetime.now()
        )

    async def get_market_context(
        self,
        mock: bool = False,
        scenario: str = "normal"
    ) -> MarketContext:
        """
        Get enriched market context with comprehensive indicators.

        This is the primary method used by other agents to get market intelligence.
        Aggregates data from multiple sources and adds predictive indicators.

        Args:
            mock: Use mock data (for testing/offline)
            scenario: Mock scenario (crash/bull/bear/volatility_spike)

        Returns:
            MarketContext with comprehensive market intelligence

        Requirements: 31.2, 31.4, 31.5, 43.5
        """
        start_time = datetime.now()
        correlation_id = str(uuid4())
        self.logger.set_correlation_id(correlation_id)

        self.logger.info("Fetching market context", extra_data={
            'mock': mock,
            'scenario': scenario
        })

        if mock or self.use_mock:
            context = await self._get_mock_market_context(scenario)
        else:
            context = await self._get_real_market_context()

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.log_performance(
            "get_market_context",
            duration_ms,
            {'mock': mock, 'scenario': scenario}
        )

        self.logger.clear_correlation_id()
        return context

    async def _get_real_market_context(self) -> MarketContext:
        """Fetch real market context from multiple APIs"""
        # Fetch major indices concurrently
        indices_to_fetch = ["NIFTY50", "SENSEX", "SPY", "QQQ"]
        tasks = [self.get_market_data(symbol) for symbol in indices_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        indices = {}
        for symbol, result in zip(indices_to_fetch, results):
            if isinstance(result, MarketData):
                indices[symbol] = result

        # Calculate aggregate volatility
        volatilities = [
            m.volatility for m in indices.values()
            if m.volatility is not None
        ]
        market_volatility = statistics.mean(volatilities) if volatilities else 15.0

        # Determine economic sentiment
        changes = [m.change_percent for m in indices.values()]
        avg_change = statistics.mean(changes) if changes else 0.0

        if avg_change > 2.0:
            sentiment = "bullish"
        elif avg_change < -2.0:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        # Get sector trends (simplified)
        sector_trends = {
            "technology": "positive" if avg_change > 0 else "negative",
            "financial": "neutral",
            "healthcare": "neutral"
        }

        # Mock regulatory changes (in production, this would come from news APIs)
        regulatory_changes = []

        # Fetch real economic indicators (interest rate and inflation)
        interest_rate = await self._fetch_interest_rate()
        inflation_rate = await self._fetch_inflation_rate()

        return MarketContext(
            market_volatility=market_volatility,
            interest_rate=interest_rate,
            inflation_rate=inflation_rate,
            economic_sentiment=sentiment,
            sector_trends=sector_trends,
            indices=indices,
            regulatory_changes=regulatory_changes,
            timestamp=datetime.now(),
            confidence_score=0.85 if indices else 0.5
        )

    async def _fetch_interest_rate(self) -> float:
        """
        Fetch current interest rate from economic data APIs.

        Uses Alpha Vantage FEDERAL_FUNDS_RATE function with fallback to default.

        Returns:
            Current interest rate as percentage (e.g., 5.33 for 5.33%)
        """
        try:
            # Try Alpha Vantage first
            if self.alphavantage_connector:
                response = await self.alphavantage_connector.get_economic_indicators(
                    function="FEDERAL_FUNDS_RATE"
                )

                if response.success and response.data:
                    # Parse the most recent data point
                    data = response.data.get("data", [])
                    if data and len(data) > 0:
                        # Get the most recent value
                        latest = data[0]
                        rate = float(latest.get("value", 5.33))
                        self.logger.info(f"Fetched interest rate: {rate}%", extra_data={
                            'source': 'AlphaVantage',
                            'date': latest.get("date")
                        })
                        return rate

            # Fallback: Try other connectors or use cached value
            self.logger.warning("Unable to fetch interest rate from API, using fallback value")
            return 5.33  # Current approximate federal funds rate as fallback

        except Exception as e:
            self.logger.error(f"Error fetching interest rate: {str(e)}", extra_data={
                'error_type': type(e).__name__
            })
            return 5.33  # Fallback value

    async def _fetch_inflation_rate(self) -> float:
        """
        Fetch current inflation rate from economic data APIs.

        Uses Alpha Vantage CPI (Consumer Price Index) with fallback to default.

        Returns:
            Current inflation rate as percentage (e.g., 3.4 for 3.4%)
        """
        try:
            # Try Alpha Vantage first
            if self.alphavantage_connector:
                response = await self.alphavantage_connector.get_economic_indicators(
                    function="CPI",
                    interval="monthly"
                )

                if response.success and response.data:
                    # Parse the most recent data point and calculate YoY change
                    data = response.data.get("data", [])
                    if data and len(data) >= 12:
                        # Calculate year-over-year inflation from CPI
                        current_cpi = float(data[0].get("value", 100))
                        year_ago_cpi = float(data[12].get("value", 100))
                        inflation = ((current_cpi - year_ago_cpi) / year_ago_cpi) * 100

                        self.logger.info(f"Fetched inflation rate: {inflation:.2f}%", extra_data={
                            'source': 'AlphaVantage',
                            'current_cpi': current_cpi,
                            'year_ago_cpi': year_ago_cpi
                        })
                        return round(inflation, 2)

            # Fallback: Use approximate current inflation rate
            self.logger.warning("Unable to fetch inflation rate from API, using fallback value")
            return 3.4  # Current approximate US inflation rate as fallback

        except Exception as e:
            self.logger.error(f"Error fetching inflation rate: {str(e)}", extra_data={
                'error_type': type(e).__name__
            })
            return 3.4  # Fallback value

    async def _get_mock_market_context(self, scenario: str = "normal") -> MarketContext:
        """
        Generate mock market context for testing.

        Supports multiple scenarios for comprehensive testing.

        Requirements: 32.1, 32.2, 32.3, 32.4
        """
        scenario_config = MOCK_SCENARIOS.get(scenario, MOCK_SCENARIOS['normal'])

        # Generate mock indices data
        indices = {}
        base_price = 19500.0  # NIFTY50 base

        for symbol in MOCK_INDICES:
            indices[symbol] = MarketData(
                symbol=symbol,
                price=base_price * (1 + scenario_config['market_change'] / 100),
                change_percent=scenario_config['market_change'],
                volume=int(1000000 * (1.5 if scenario == 'crash' else 1.0)),
                volatility=scenario_config['volatility'],
                market_sentiment=scenario_config['sentiment'],
                sector_trend="declining" if scenario_config['market_change'] < 0 else "rising",
                predicted_volatility=scenario_config['volatility'] * 1.1,
                risk_score=min(100, scenario_config['volatility'] * 2),
                source="mock",
                timestamp=datetime.now()
            )

        # Regulatory changes based on scenario
        regulatory_changes = []
        if scenario == "regulatory_change":
            regulatory_changes = [
                "New capital gains tax rules effective Q4 2025",
                "SEBI updated disclosure requirements for mutual funds"
            ]

        return MarketContext(
            market_volatility=scenario_config['volatility'],
            interest_rate=6.5 if scenario != 'crash' else 7.0,
            inflation_rate=5.2 if scenario != 'crash' else 6.5,
            economic_sentiment=scenario_config['sentiment'],
            sector_trends={
                "technology": "negative" if scenario == 'crash' else "positive",
                "financial": "declining" if scenario == 'bear' else "stable",
                "healthcare": "stable"
            },
            indices=indices,
            regulatory_changes=regulatory_changes,
            timestamp=datetime.now(),
            confidence_score=1.0  # Mock data is always confident
        )

    # ========================================================================
    # TRIGGER DETECTION - Tasks 2.8
    # ========================================================================

    async def detect_market_triggers(
        self,
        market_context: Optional[MarketContext] = None
    ) -> List[TriggerEvent]:
        """
        Detect market triggers that should initiate CMVL.

        Analyzes market conditions and detects events requiring replanning:
        - Market crashes (>10% drop)
        - Volatility spikes (VIX > 30)
        - Significant interest rate changes
        - Major sector rotations
        - Regulatory changes

        Args:
            market_context: Current market context (fetches if not provided)

        Returns:
            List of detected trigger events with severity assessment

        Requirements: 2.1, 2.2, 33.1, 33.2, 33.3, 33.4, 33.5
        """
        if market_context is None:
            market_context = await self.get_market_context()

        triggers = []
        correlation_id = str(uuid4())

        # Detect volatility spike
        if market_context.market_volatility > TRIGGER_THRESHOLDS['volatility_spike']:
            severity = self._assess_severity(
                event_type=MarketEventType.VOLATILITY_SPIKE,
                magnitude=market_context.market_volatility
            )

            trigger = TriggerEvent(
                event_id=str(uuid4()),
                event_type=MarketEventType.VOLATILITY_SPIKE,
                severity=severity,
                description=f"Market volatility spike detected: {market_context.market_volatility:.1f}%",
                impact_assessment="High volatility may require portfolio rebalancing and risk reassessment",
                confidence=0.95,
                timestamp=datetime.now(),
                source="InformationRetrievalAgent",
                correlation_id=correlation_id,
                recommended_actions=[
                    "Review portfolio risk exposure",
                    "Consider hedging strategies",
                    "Reassess asset allocation"
                ],
                requires_immediate_action=(severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL])
            )
            triggers.append(trigger)
            self.logger.log_trigger_event(
                event_type="VOLATILITY_SPIKE",
                severity=severity.value,
                description=trigger.description
            )

        # Detect market crash
        market_changes = [
            m.change_percent for m in market_context.indices.values()
        ]
        if market_changes:
            avg_change = statistics.mean(market_changes)

            if avg_change <= TRIGGER_THRESHOLDS['market_crash']:
                severity = self._assess_severity(
                    event_type=MarketEventType.CRASH,
                    magnitude=abs(avg_change)
                )

                trigger = TriggerEvent(
                    event_id=str(uuid4()),
                    event_type=MarketEventType.CRASH,
                    severity=severity,
                    description=f"Market crash detected: {avg_change:.1f}% decline",
                    impact_assessment="Severe market decline requires immediate portfolio review and potential defensive positioning",
                    confidence=0.98,
                    affected_assets=list(market_context.indices.keys()),
                    timestamp=datetime.now(),
                    source="InformationRetrievalAgent",
                    correlation_id=correlation_id,
                    recommended_actions=[
                        "Activate defensive strategy",
                        "Review stop-loss triggers",
                        "Consider moving to safer assets",
                        "Reassess all active plans"
                    ],
                    requires_immediate_action=True
                )
                triggers.append(trigger)
                self.logger.log_trigger_event(
                    event_type="MARKET_CRASH",
                    severity=severity.value,
                    description=trigger.description
                )

        # Detect regulatory changes
        if market_context.regulatory_changes:
            severity = self._assess_severity(
                event_type=MarketEventType.REGULATORY_CHANGE,
                magnitude=len(market_context.regulatory_changes)
            )

            trigger = TriggerEvent(
                event_id=str(uuid4()),
                event_type=MarketEventType.REGULATORY_CHANGE,
                severity=severity,
                description=f"{len(market_context.regulatory_changes)} regulatory changes detected",
                impact_assessment="Regulatory changes may affect tax planning and compliance requirements",
                confidence=1.0,
                timestamp=datetime.now(),
                source="InformationRetrievalAgent",
                correlation_id=correlation_id,
                recommended_actions=[
                    "Review compliance requirements",
                    "Update tax optimization strategies",
                    "Consult regulatory requirements"
                ],
                requires_immediate_action=(severity == SeverityLevel.CRITICAL)
            )
            triggers.append(trigger)
            self.logger.log_trigger_event(
                event_type="REGULATORY_CHANGE",
                severity=severity.value,
                description=trigger.description
            )

        # Store triggers in history
        self.trigger_history.extend(triggers)

        self.logger.info(f"Detected {len(triggers)} market triggers", extra_data={
            'trigger_count': len(triggers),
            'correlation_id': correlation_id
        })

        return triggers

    def _assess_severity(
        self,
        event_type: MarketEventType,
        magnitude: float
    ) -> SeverityLevel:
        """
        Assess severity of a market event.

        Uses event-specific thresholds and impact multipliers.

        Requirements: 33.2, 33.3
        """
        # Apply event-specific multiplier
        multiplier = EVENT_IMPACT_MULTIPLIERS.get(
            event_type.value,
            1.0
        )

        severity_score = magnitude * multiplier

        # Map to severity level
        if severity_score >= SEVERITY_SCORES['critical'][0]:
            return SeverityLevel.CRITICAL
        elif severity_score >= SEVERITY_SCORES['high'][0]:
            return SeverityLevel.HIGH
        elif severity_score >= SEVERITY_SCORES['medium'][0]:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    # ========================================================================
    # VOLATILITY MONITORING - Task 2.9
    # ========================================================================

    async def monitor_volatility(
        self,
        symbols: List[str],
        threshold: float = 25.0
    ) -> Dict[str, Tuple[float, str]]:
        """
        Monitor volatility for multiple symbols.

        Args:
            symbols: List of symbols to monitor
            threshold: Volatility threshold for alerts

        Returns:
            Dict mapping symbol to (volatility, status)
            Status: "low", "normal", "high", "extreme"

        Requirements: 5.2, 31.2
        """
        results = {}

        tasks = [self.get_market_data(symbol) for symbol in symbols]
        market_data_list = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, data in zip(symbols, market_data_list):
            if isinstance(data, MarketData) and data.volatility is not None:
                vol = data.volatility
                status = self._classify_volatility(vol)
                results[symbol] = (vol, status)

                if vol > threshold:
                    self.logger.warning(
                        f"High volatility detected for {symbol}",
                        extra_data={'volatility': vol, 'threshold': threshold}
                    )

        return results

    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < VOLATILITY_THRESHOLDS['low']:
            return "low"
        elif volatility < VOLATILITY_THRESHOLDS['normal']:
            return "normal"
        elif volatility < VOLATILITY_THRESHOLDS['high']:
            return "high"
        else:
            return "extreme"

    # ========================================================================
    # RAG SYSTEM - Task 2.7
    # ========================================================================

    async def query_financial_knowledge(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Query RAG system for financial knowledge.

        This is a simplified implementation. In production, this would
        use a vector database (FAISS/Chroma) and embeddings.

        Args:
            query: Natural language query
            context: Additional context

        Returns:
            Retrieved knowledge/answer

        Requirements: 5.3, 31.4
        """
        # TODO: Implement full RAG (Retrieval-Augmented Generation) system
        #
        # FUTURE IMPLEMENTATION PLAN:
        # ----------------------------
        # 1. Embeddings Model:
        #    - Use sentence-transformers (e.g., all-MiniLM-L6-v2)
        #    - Or OpenAI embeddings API
        #    - Or NVIDIA NeMo embeddings
        #
        # 2. Vector Database:
        #    - Options: Pinecone, Weaviate, ChromaDB, FAISS
        #    - Store financial knowledge base embeddings
        #    - Enable semantic similarity search
        #
        # 3. Knowledge Base Sources:
        #    - Tax regulations and updates
        #    - Investment product documentation
        #    - Market research and analysis
        #    - Regulatory compliance documents
        #    - Best practices and strategies
        #
        # 4. RAG Pipeline:
        #    - Embed user query
        #    - Retrieve top-k relevant documents (k=3-5)
        #    - Pass context to LLM (Ollama/NVIDIA NIM)
        #    - Generate contextual answer
        #    - Include source citations
        #
        # 5. Integration Points:
        #    - ConversationalAgent for natural language responses
        #    - NVIDIA NIM for advanced generation
        #    - Cache frequent queries (Redis)
        #
        # CURRENT STATUS: Basic template-based responses (see below)
        # For production RAG, integrate with vector DB and LLM

        query_lower = query.lower()

        if "tax" in query_lower:
            return "Tax-saving instruments in India include ELSS (80C up to ₹1.5L), PPF, NPS (80CCD1B up to ₹50k), and health insurance (80D up to ₹25k)."
        elif "emergency fund" in query_lower:
            return "Maintain 6-12 months of expenses in liquid funds or high-interest savings accounts for emergencies."
        elif "risk" in query_lower and "profile" in query_lower:
            return "Risk profiles: Conservative (30% equity), Moderate (60% equity), Aggressive (80% equity). Choose based on age, income stability, and goals."
        elif "diversification" in query_lower:
            return "Diversify across asset classes (equity, debt, gold), sectors, and geographies. No single stock should exceed 15% of portfolio."
        else:
            return "Financial planning requires understanding your goals, risk tolerance, time horizon, and current financial situation."

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_trigger_history(
        self,
        limit: int = 10,
        event_type: Optional[MarketEventType] = None
    ) -> List[TriggerEvent]:
        """Get recent trigger event history"""
        history = self.trigger_history

        if event_type:
            history = [t for t in history if t.event_type == event_type]

        return sorted(
            history,
            key=lambda t: t.timestamp,
            reverse=True
        )[:limit]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get IRA performance metrics"""
        if not self.performance_metrics:
            return {
                'total_requests': 0,
                'avg_response_time': 0,
                'cache_hit_rate': 0
            }

        response_times = [m.response_time_ms for m in self.performance_metrics]

        return {
            'total_requests': len(self.performance_metrics),
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'success_rate': statistics.mean([m.success_rate for m in self.performance_metrics])
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_ira(
    use_mock: bool = False,
    redis_url: str = "redis://localhost:6379"
) -> InformationRetrievalAgent:
    """
    Factory function to create and initialize IRA.

    Args:
        use_mock: Use mock data
        redis_url: Redis connection URL

    Returns:
        Initialized InformationRetrievalAgent
    """
    ira = InformationRetrievalAgent(
        use_mock=use_mock,
        redis_url=redis_url
    )
    await ira.initialize()
    return ira


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'InformationRetrievalAgent',
    'create_ira'
]
