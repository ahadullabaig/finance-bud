"""
Agent Communication Framework for FinPilot VP-MAS

Provides structured communication protocols, message routing,
and coordination between agents with comprehensive monitoring.

Requirements: 6.1, 6.2, 9.4
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from uuid import uuid4

from data_models.schemas import AgentMessage, MessageType, Priority


class AgentRegistry:
    """Registry for managing agent instances and their capabilities"""
    
    def __init__(self):
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.logger = logging.getLogger("finpilot.communication.registry")
    
    def register_agent(self, agent: 'BaseAgent', capabilities: List[str] = None) -> None:
        """Register an agent with the system"""
        self.agents[agent.agent_id] = agent
        self.agent_capabilities[agent.agent_id] = capabilities or []
        
        self.logger.info(f"Registered agent {agent.agent_id} with capabilities: {capabilities}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the system"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_capabilities[agent_id]
            self.logger.info(f"Unregistered agent {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional['BaseAgent']:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability"""
        return [
            agent_id for agent_id, caps in self.agent_capabilities.items()
            if capability in caps
        ]
    
    def get_all_agents(self) -> Dict[str, 'BaseAgent']:
        """Get all registered agents"""
        return self.agents.copy()


class MessageRouter:
    """Routes messages between agents with load balancing and failover"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.message_history: List[AgentMessage] = []
        self.routing_rules: Dict[str, Callable] = {}
        self.logger = logging.getLogger("finpilot.communication.router")
    
    async def route_message(self, message: AgentMessage) -> bool:
        """Route a message to the appropriate agent(s)"""
        try:
            # Log the routing attempt
            self.logger.info(
                f"Routing {message.message_type} from {message.agent_id} to {message.target_agent_id}",
                extra={'correlation_id': message.correlation_id}
            )
            
            # Store message in history
            self.message_history.append(message)
            
            # Handle broadcast messages
            if message.target_agent_id is None:
                return await self._broadcast_message(message)
            
            # Route to specific agent
            target_agent = self.registry.get_agent(message.target_agent_id)
            if target_agent is None:
                self.logger.error(f"Target agent {message.target_agent_id} not found")
                return False
            
            await target_agent.receive_message(message)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to route message: {str(e)}")
            return False
    
    async def _broadcast_message(self, message: AgentMessage) -> bool:
        """Broadcast a message to all agents except the sender"""
        success_count = 0
        
        for agent_id, agent in self.registry.get_all_agents().items():
            if agent_id != message.agent_id:
                try:
                    await agent.receive_message(message)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to broadcast to {agent_id}: {str(e)}")
        
        return success_count > 0
    
    def add_routing_rule(self, message_type: str, rule: Callable) -> None:
        """Add a custom routing rule for specific message types"""
        self.routing_rules[message_type] = rule
        self.logger.info(f"Added routing rule for {message_type}")
    
    def get_message_history(self, correlation_id: str = None) -> List[AgentMessage]:
        """Get message history, optionally filtered by correlation ID"""
        if correlation_id:
            return [msg for msg in self.message_history if msg.correlation_id == correlation_id]
        return self.message_history.copy()


class CircuitBreaker:
    """Circuit breaker for agent communication to prevent cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.logger = logging.getLogger("finpilot.communication.circuit_breaker")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute a function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                self.logger.info("Circuit breaker moving to half-open state")
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            self.logger.info("Circuit breaker reset to closed state")
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout


class AgentCommunicationFramework:
    """
    Main communication framework that coordinates all agent interactions
    with monitoring, circuit breaking, and comprehensive logging.
    """
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.router = MessageRouter(self.registry)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.correlation_tracker: Dict[str, List[str]] = {}
        self.logger = logging.getLogger("finpilot.communication.framework")
        
        # Performance metrics
        self.total_messages = 0
        self.successful_messages = 0
        self.failed_messages = 0
        self.start_time = datetime.utcnow()
    
    def register_agent(self, agent: 'BaseAgent', capabilities: List[str] = None) -> None:
        """Register an agent with the communication framework"""
        self.registry.register_agent(agent, capabilities)
        
        # Create circuit breaker for this agent
        self.circuit_breakers[agent.agent_id] = CircuitBreaker()
        
        self.logger.info(f"Agent {agent.agent_id} registered with communication framework")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the communication framework"""
        self.total_messages += 1
        
        try:
            # Get circuit breaker for sender
            circuit_breaker = self.circuit_breakers.get(message.agent_id)
            if circuit_breaker:
                success = await circuit_breaker.call(self.router.route_message, message)
            else:
                success = await self.router.route_message(message)
            
            if success:
                self.successful_messages += 1
                
                # Track correlation
                if message.correlation_id not in self.correlation_tracker:
                    self.correlation_tracker[message.correlation_id] = []
                self.correlation_tracker[message.correlation_id].append(message.message_id)
            else:
                self.failed_messages += 1
            
            return success
            
        except Exception as e:
            self.failed_messages += 1
            self.logger.error(f"Communication framework error: {str(e)}")
            return False
    
    def create_message(
        self,
        sender_id: str,
        target_id: str,
        message_type: MessageType,
        payload: Dict,
        correlation_id: str = None,
        session_id: str = None,
        priority: Priority = Priority.MEDIUM
    ) -> AgentMessage:
        """Create a properly formatted agent message"""
        return AgentMessage(
            agent_id=sender_id,
            target_agent_id=target_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id or str(uuid4()),
            session_id=session_id or str(uuid4()),
            priority=priority,
            trace_id=str(uuid4())
        )
    
    def get_system_health(self) -> Dict:
        """Get overall system health metrics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        success_rate = (
            self.successful_messages / max(1, self.total_messages)
        )
        
        agent_health = {}
        for agent_id, agent in self.registry.get_all_agents().items():
            agent_health[agent_id] = agent.get_health_status()
        
        circuit_breaker_status = {}
        for agent_id, cb in self.circuit_breakers.items():
            circuit_breaker_status[agent_id] = {
                "state": cb.state,
                "failure_count": cb.failure_count
            }
        
        return {
            "framework_uptime": uptime,
            "total_messages": self.total_messages,
            "successful_messages": self.successful_messages,
            "failed_messages": self.failed_messages,
            "success_rate": success_rate,
            "registered_agents": len(self.registry.agents),
            "agent_health": agent_health,
            "circuit_breakers": circuit_breaker_status,
            "active_correlations": len(self.correlation_tracker)
        }
    
    def get_correlation_trace(self, correlation_id: str) -> List[AgentMessage]:
        """Get all messages for a specific correlation ID"""
        return self.router.get_message_history(correlation_id)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the communication framework"""
        self.logger.info("Shutting down communication framework...")
        
        # Stop all agents
        for agent in self.registry.get_all_agents().values():
            await agent.stop()
        
        self.logger.info("Communication framework shutdown complete")