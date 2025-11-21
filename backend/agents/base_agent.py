"""
Base Agent Class for FinPilot VP-MAS

Provides common functionality for all agents including communication,
logging, error handling, and performance monitoring.

Requirements: 9.4, 9.5, 11.2
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from data_models.schemas import (
    AgentMessage, MessageType, Priority, ExecutionStatus,
    PerformanceMetrics, ExecutionLog
)


class BaseAgent(ABC):
    """
    Abstract base class for all VP-MAS agents.
    
    Provides common functionality including:
    - Structured communication protocols
    - Performance monitoring
    - Error handling and recovery
    - Comprehensive logging
    - Health monitoring
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = "initializing"
        self.start_time = datetime.utcnow()
        self.message_queue = asyncio.Queue()
        self.performance_metrics = {}
        self.error_count = 0
        self.success_count = 0
        
        # Set up logging
        self.logger = self._setup_logging()
        self.logger.info(f"Initializing {agent_type} agent: {agent_id}")
        
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging for the agent"""
        logger = logging.getLogger(f"finpilot.agents.{self.agent_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    async def start(self) -> None:
        """Start the agent and begin processing messages"""
        self.status = "running"
        self.logger.info(f"Agent {self.agent_id} started successfully")
        
        # Start message processing loop
        asyncio.create_task(self._process_messages())
    
    async def stop(self) -> None:
        """Stop the agent gracefully"""
        self.status = "stopping"
        self.logger.info(f"Agent {self.agent_id} stopping...")
        
        # Wait for current operations to complete
        await asyncio.sleep(0.1)
        
        self.status = "stopped"
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent"""
        start_time = time.time()
        
        try:
            # Add performance tracking
            message.performance_metrics = PerformanceMetrics(
                execution_time=0.0,
                memory_usage=0.0
            )
            
            # Log the message
            self.logger.info(
                f"Sending {message.message_type} to {message.target_agent_id}",
                extra={'correlation_id': message.correlation_id}
            )
            
            # Simulate message sending (in real implementation, this would use HTTP/WebSocket)
            await asyncio.sleep(0.01)  # Simulate network latency
            
            execution_time = time.time() - start_time
            message.performance_metrics.execution_time = execution_time
            
            self.success_count += 1
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(
                f"Failed to send message: {str(e)}",
                extra={'correlation_id': message.correlation_id}
            )
            raise
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent"""
        await self.message_queue.put(message)
        
        self.logger.info(
            f"Received {message.message_type} from {message.agent_id}",
            extra={'correlation_id': message.correlation_id}
        )
    
    async def _process_messages(self) -> None:
        """Process incoming messages"""
        while self.status == "running":
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Process the message
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error processing message: {str(e)}")
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle an incoming message"""
        start_time = time.time()
        
        try:
            # Create execution log
            log_entry = ExecutionLog(
                agent_id=self.agent_id,
                action_type="message_processing",
                operation_name=f"handle_{message.message_type}",
                execution_status=ExecutionStatus.IN_PROGRESS,
                input_data=message.dict(),
                session_id=message.session_id,
                correlation_id=message.correlation_id,
                trace_id=message.trace_id,
                execution_time=0.0
            )
            
            # Process the message using agent-specific logic
            response = await self.process_message(message)
            
            # Update execution log
            execution_time = time.time() - start_time
            log_entry.execution_time = execution_time
            log_entry.execution_status = ExecutionStatus.COMPLETED
            log_entry.output_data = response.dict() if response else {}
            
            self.success_count += 1
            
            # Send response if generated
            if response:
                await self.send_message(response)
                
        except Exception as e:
            execution_time = time.time() - start_time
            log_entry.execution_time = execution_time
            log_entry.execution_status = ExecutionStatus.FAILED
            log_entry.output_data = {"error": str(e)}
            
            self.error_count += 1
            self.logger.error(
                f"Failed to process message: {str(e)}",
                extra={'correlation_id': message.correlation_id}
            )
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message and optionally return a response.
        Must be implemented by each agent.
        """
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the agent"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        success_rate = (
            self.success_count / (self.success_count + self.error_count)
            if (self.success_count + self.error_count) > 0 else 1.0
        )
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "uptime_seconds": uptime,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "queue_size": self.message_queue.qsize(),
            "last_heartbeat": datetime.utcnow().isoformat()
        }
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return PerformanceMetrics(
            execution_time=0.0,  # Will be updated with actual metrics
            memory_usage=0.0,
            api_calls=0,
            cache_hits=0,
            cache_misses=0,
            error_count=self.error_count,
            success_rate=self.success_count / max(1, self.success_count + self.error_count),
            throughput=0.0
        )
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Basic health check - can be extended by subclasses
            return self.status == "running"
        except Exception:
            return False