"""
Agent Factory Pattern

Centralized agent creation and lifecycle management.
Supports mock/real agent switching via configuration.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

from config import settings


class AgentMode(str, Enum):
    """Agent execution modes"""
    MOCK = "mock"
    REAL = "real"
    HYBRID = "hybrid"  # Mix of mock and real agents


class AgentFactory:
    """
    Factory for creating and managing agent instances.

    Provides centralized agent initialization with support for:
    - Mock vs real agent switching via configuration
    - Lazy initialization
    - Agent health checking
    - Dependency injection
    """

    def __init__(self, mode: AgentMode = AgentMode.HYBRID):
        """
        Initialize agent factory.

        Args:
            mode: Agent execution mode (mock, real, or hybrid)
        """
        self.mode = mode
        self.logger = logging.getLogger("finpilot.agent_factory")
        self._agents: Dict[str, Any] = {}
        self._initialized = False

    def create_all_agents(self) -> Dict[str, Any]:
        """
        Create all agents based on configuration.

        Returns:
            Dictionary of agent instances keyed by agent type
        """
        if self._initialized:
            self.logger.warning("Agents already initialized, returning existing instances")
            return self._agents

        self.logger.info(f"Initializing agents in {self.mode.value} mode...")

        try:
            # Create agents based on configuration
            self._agents["orchestration"] = self._create_orchestration_agent()
            self._agents["planning"] = self._create_planning_agent()
            self._agents["information_retrieval"] = self._create_information_retrieval_agent()
            self._agents["verification"] = self._create_verification_agent()
            self._agents["execution"] = self._create_execution_agent()
            self._agents["conversational"] = self._create_conversational_agent()

            # Log initialization status
            status = self._get_agent_status()
            self.logger.info(f"Agent initialization complete: {status}")
            self._initialized = True

            return self._agents

        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise

    def _create_orchestration_agent(self) -> Any:
        """Create orchestration agent (mock or real based on config)"""
        try:
            if settings.use_mock_orchestration:
                from tests.mocks.mock_interfaces import MockOrchestrationAgent
                self.logger.info("✅ Created mock orchestration agent")
                return MockOrchestrationAgent()
            else:
                # TODO: Import and create real orchestration agent when available
                from agents.orchestration_agent import OrchestrationAgent
                self.logger.info("✅ Created real orchestration agent")
                return OrchestrationAgent()
        except ImportError as e:
            self.logger.error(f"Failed to import orchestration agent: {e}")
            # Fallback to mock
            from tests.mocks.mock_interfaces import MockOrchestrationAgent
            self.logger.warning("⚠️ Falling back to mock orchestration agent")
            return MockOrchestrationAgent()

    def _create_planning_agent(self) -> Any:
        """Create planning agent (mock or real based on config)"""
        try:
            if settings.use_mock_planning:
                from tests.mocks.mock_interfaces import MockPlanningAgent
                self.logger.info("✅ Created mock planning agent")
                return MockPlanningAgent()
            else:
                from agents.planning_agent import PlanningAgent
                self.logger.info("✅ Created real planning agent")
                return PlanningAgent()
        except ImportError as e:
            self.logger.error(f"Failed to import planning agent: {e}")
            from tests.mocks.mock_interfaces import MockPlanningAgent
            self.logger.warning("⚠️ Falling back to mock planning agent")
            return MockPlanningAgent()

    def _create_information_retrieval_agent(self) -> Any:
        """Create information retrieval agent (mock or real based on config)"""
        try:
            if settings.use_mock_retrieval:
                from tests.mocks.mock_interfaces import MockInformationRetrievalAgent
                self.logger.info("✅ Created mock information retrieval agent")
                return MockInformationRetrievalAgent()
            else:
                from agents.retriever import InformationRetrievalAgent
                self.logger.info("✅ Created real information retrieval agent")
                return InformationRetrievalAgent()
        except ImportError as e:
            self.logger.error(f"Failed to import information retrieval agent: {e}")
            from tests.mocks.mock_interfaces import MockInformationRetrievalAgent
            self.logger.warning("⚠️ Falling back to mock information retrieval agent")
            return MockInformationRetrievalAgent()

    def _create_verification_agent(self) -> Any:
        """Create verification agent (typically real for proper validation)"""
        try:
            if settings.use_real_verification:
                from agents.verifier import VerificationAgent
                self.logger.info("✅ Created real verification agent")
                return VerificationAgent()
            else:
                from tests.mocks.mock_interfaces import MockVerificationAgent
                self.logger.info("✅ Created mock verification agent")
                return MockVerificationAgent()
        except ImportError as e:
            self.logger.error(f"Failed to import verification agent: {e}")
            from tests.mocks.mock_interfaces import MockVerificationAgent
            self.logger.warning("⚠️ Falling back to mock verification agent")
            return MockVerificationAgent()

    def _create_execution_agent(self) -> Optional[Any]:
        """Create execution agent (optional)"""
        try:
            from agents.execution_agent import ExecutionAgent
            self.logger.info("✅ Created execution agent")
            return ExecutionAgent()
        except ImportError as e:
            self.logger.warning(f"⚠️ Execution agent not available: {e}")
            try:
                from tests.mocks.mock_interfaces import MockExecutionAgent
                self.logger.info("✅ Created mock execution agent")
                return MockExecutionAgent()
            except ImportError:
                self.logger.warning("⚠️ No execution agent available")
                return None

    def _create_conversational_agent(self) -> Optional[Any]:
        """Create conversational agent (optional, depends on Ollama)"""
        if not settings.conversational_agent_enabled:
            self.logger.info("ℹ️ Conversational agent disabled in configuration")
            return None

        try:
            from agents.conversational_agent import get_conversational_agent
            agent = get_conversational_agent()
            self.logger.info("✅ Created conversational agent")
            return agent
        except ImportError as e:
            self.logger.warning(f"⚠️ Failed to import conversational agent: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize conversational agent: {e}")
            return None

    def _get_agent_status(self) -> Dict[str, str]:
        """Get initialization status of all agents"""
        status = {}
        for name, agent in self._agents.items():
            if agent is not None:
                status[name] = "initialized"
            else:
                status[name] = "unavailable"
        return status

    def get_agent(self, agent_type: str) -> Optional[Any]:
        """
        Get agent by type.

        Args:
            agent_type: Type of agent (orchestration, planning, etc.)

        Returns:
            Agent instance or None if not available
        """
        return self._agents.get(agent_type)

    def get_all_agents(self) -> Dict[str, Any]:
        """Get all initialized agents"""
        return self._agents

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all agents.

        Returns:
            Dictionary with health status for each agent
        """
        health_status = {}

        for name, agent in self._agents.items():
            if agent is None:
                health_status[name] = {
                    "status": "unavailable",
                    "healthy": False
                }
            else:
                try:
                    # Try to get health status from agent if method exists
                    if hasattr(agent, 'get_health_status'):
                        agent_health = agent.get_health_status()
                        health_status[name] = {
                            "status": agent_health,
                            "healthy": agent_health == "healthy"
                        }
                    else:
                        health_status[name] = {
                            "status": "initialized",
                            "healthy": True
                        }
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    health_status[name] = {
                        "status": "error",
                        "healthy": False,
                        "error": str(e)
                    }

        return health_status

    def shutdown(self):
        """Shutdown all agents gracefully"""
        self.logger.info("Shutting down all agents...")

        for name, agent in self._agents.items():
            if agent is not None:
                try:
                    # Call shutdown method if agent has one
                    if hasattr(agent, 'shutdown'):
                        agent.shutdown()
                        self.logger.info(f"✅ Shutdown {name} agent")
                except Exception as e:
                    self.logger.error(f"❌ Error shutting down {name} agent: {e}")

        self._agents.clear()
        self._initialized = False
        self.logger.info("All agents shutdown complete")


# Global factory instance (initialized in main.py)
_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """
    Get the global agent factory instance.

    Returns:
        AgentFactory instance
    """
    global _factory
    if _factory is None:
        _factory = AgentFactory(mode=AgentMode.HYBRID)
    return _factory


def create_agents() -> Dict[str, Any]:
    """
    Convenience function to create all agents.

    Returns:
        Dictionary of agent instances
    """
    factory = get_agent_factory()
    return factory.create_all_agents()
