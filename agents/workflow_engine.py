"""
Advanced Workflow Engine for Multi-Agent Process Management

Manages complex workflows involving multiple agents with state tracking,
error recovery, rollback capabilities, and comprehensive monitoring.

Requirements: 1.4, 2.5, 4.5, 8.4
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
from enum import Enum
import json

from .base_agent import BaseAgent
from .communication import AgentCommunicationFramework
from data_models.schemas import (
    AgentMessage, MessageType, Priority, ExecutionStatus,
    PerformanceMetrics, ExecutionLog, ReasoningTrace
)


class WorkflowState(str, Enum):
    """Workflow execution states"""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLING_BACK = "rolling_back"


class TaskState(str, Enum):
    """Individual task states within workflows"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class WorkflowPriority(str, Enum):
    """Workflow priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WorkflowTask:
    """Individual task within a workflow"""
    
    def __init__(
        self,
        task_id: str,
        task_name: str,
        agent_id: str,
        action: str,
        parameters: Dict[str, Any],
        dependencies: List[str] = None,
        timeout_seconds: int = 300,
        retry_count: int = 3,
        rollback_action: Optional[str] = None
    ):
        self.task_id = task_id
        self.task_name = task_name
        self.agent_id = agent_id
        self.action = action
        self.parameters = parameters
        self.dependencies = dependencies or []
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.rollback_action = rollback_action
        
        # Runtime state
        self.state = TaskState.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.attempts = 0
        self.rollback_data: Optional[Dict[str, Any]] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "agent_id": self.agent_id,
            "action": self.action,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count
        }


class Workflow:
    """Complete workflow definition with tasks and execution logic"""
    
    def __init__(
        self,
        workflow_id: str,
        workflow_name: str,
        workflow_type: str,
        user_id: str,
        priority: WorkflowPriority = WorkflowPriority.MEDIUM,
        timeout_minutes: int = 30,
        rollback_enabled: bool = True
    ):
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.workflow_type = workflow_type
        self.user_id = user_id
        self.priority = priority
        self.timeout_minutes = timeout_minutes
        self.rollback_enabled = rollback_enabled
        
        # Tasks and execution
        self.tasks: Dict[str, WorkflowTask] = {}
        self.task_order: List[str] = []
        self.state = WorkflowState.CREATED
        
        # Runtime tracking
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.session_id = str(uuid4())
        self.correlation_id = str(uuid4())
        
        # Results and metrics
        self.results: Dict[str, Any] = {}
        self.error_log: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.rollback_stack: List[str] = []
        
    def add_task(self, task: WorkflowTask) -> None:
        """Add a task to the workflow"""
        self.tasks[task.task_id] = task
        self.task_order.append(task.task_id)
        
    def get_ready_tasks(self) -> List[WorkflowTask]:
        """Get tasks that are ready to execute (dependencies satisfied)"""
        ready_tasks = []
        
        for task_id in self.task_order:
            task = self.tasks[task_id]
            
            if task.state != TaskState.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_satisfied = all(
                self.tasks[dep_id].state == TaskState.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )
            
            if dependencies_satisfied:
                task.state = TaskState.READY
                ready_tasks.append(task)
                
        return ready_tasks
    
    def is_complete(self) -> bool:
        """Check if workflow is complete"""
        return all(
            task.state in [TaskState.COMPLETED, TaskState.SKIPPED]
            for task in self.tasks.values()
        )
    
    def has_failed(self) -> bool:
        """Check if workflow has failed"""
        return any(
            task.state == TaskState.FAILED and task.attempts >= task.retry_count
            for task in self.tasks.values()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary representation"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_type": self.workflow_type,
            "user_id": self.user_id,
            "priority": self.priority.value,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "task_order": self.task_order,
            "results": self.results,
            "error_log": self.error_log,
            "performance_metrics": self.performance_metrics,
            "progress_percentage": self.calculate_progress()
        }
    
    def calculate_progress(self) -> float:
        """Calculate workflow progress percentage"""
        if not self.tasks:
            return 0.0
            
        completed_tasks = sum(
            1 for task in self.tasks.values()
            if task.state in [TaskState.COMPLETED, TaskState.SKIPPED]
        )
        
        return (completed_tasks / len(self.tasks)) * 100.0


class WorkflowEngine:
    """
    Advanced workflow engine for managing multi-agent processes
    
    Features:
    - Dependency-based task execution
    - Error recovery and retry logic
    - Rollback capabilities
    - Real-time monitoring
    - Performance metrics
    - Concurrent workflow execution
    """
    
    def __init__(self, communication_framework: AgentCommunicationFramework):
        self.communication_framework = communication_framework
        self.logger = logging.getLogger("finpilot.workflow_engine")
        
        # Workflow management
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_history: List[str] = []
        self.workflow_templates: Dict[str, Callable] = {}
        
        # Execution control
        self.max_concurrent_workflows = 10
        self.max_concurrent_tasks = 20
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.engine_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_workflow_time": 0.0,
            "average_task_time": 0.0
        }
        
        # Register built-in workflow templates
        self._register_builtin_templates()
        
    def _register_builtin_templates(self) -> None:
        """Register built-in workflow templates"""
        self.workflow_templates.update({
            "financial_planning": self._create_financial_planning_workflow,
            "portfolio_rebalancing": self._create_portfolio_rebalancing_workflow,
            "compliance_review": self._create_compliance_review_workflow,
            "emergency_response": self._create_emergency_response_workflow,
            "tax_optimization": self._create_tax_optimization_workflow
        })
    
    async def create_workflow(
        self,
        workflow_type: str,
        user_id: str,
        parameters: Dict[str, Any],
        priority: WorkflowPriority = WorkflowPriority.MEDIUM
    ) -> str:
        """Create a new workflow from template"""
        
        if workflow_type not in self.workflow_templates:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        workflow_id = str(uuid4())
        
        # Create workflow from template
        workflow = self.workflow_templates[workflow_type](
            workflow_id, user_id, parameters, priority
        )
        
        # Register workflow
        self.active_workflows[workflow_id] = workflow
        self.engine_metrics["total_workflows"] += 1
        
        self.logger.info(f"Created workflow {workflow_id} of type {workflow_type}")
        
        return workflow_id
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start executing a workflow"""
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if workflow.state != WorkflowState.CREATED:
            raise ValueError(f"Workflow {workflow_id} is not in created state")
        
        try:
            workflow.state = WorkflowState.RUNNING
            workflow.started_at = datetime.utcnow()
            
            # Start workflow execution task
            execution_task = asyncio.create_task(self._execute_workflow(workflow))
            self.running_tasks[workflow_id] = execution_task
            
            self.logger.info(f"Started workflow {workflow_id}")
            return True
            
        except Exception as e:
            workflow.state = WorkflowState.FAILED
            workflow.error_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Failed to start workflow: {str(e)}"
            })
            self.logger.error(f"Failed to start workflow {workflow_id}: {str(e)}")
            return False
    
    async def _execute_workflow(self, workflow: Workflow) -> None:
        """Execute workflow tasks with dependency management"""
        
        try:
            self.logger.info(f"Executing workflow {workflow.workflow_id}")
            
            while not workflow.is_complete() and not workflow.has_failed():
                # Get ready tasks
                ready_tasks = workflow.get_ready_tasks()
                
                if not ready_tasks:
                    # Check if we're stuck (no ready tasks but not complete)
                    if not workflow.is_complete():
                        await asyncio.sleep(1)  # Wait for running tasks
                        continue
                    break
                
                # Execute ready tasks concurrently
                task_futures = []
                for task in ready_tasks[:self.max_concurrent_tasks]:
                    future = asyncio.create_task(self._execute_task(workflow, task))
                    task_futures.append(future)
                
                # Wait for at least one task to complete
                if task_futures:
                    await asyncio.wait(task_futures, return_when=asyncio.FIRST_COMPLETED)
            
            # Determine final workflow state
            if workflow.has_failed():
                workflow.state = WorkflowState.FAILED
                self.engine_metrics["failed_workflows"] += 1
                
                # Initiate rollback if enabled
                if workflow.rollback_enabled:
                    await self._rollback_workflow(workflow)
                    
            elif workflow.is_complete():
                workflow.state = WorkflowState.COMPLETED
                workflow.completed_at = datetime.utcnow()
                self.engine_metrics["successful_workflows"] += 1
                
                # Calculate performance metrics
                execution_time = (workflow.completed_at - workflow.started_at).total_seconds()
                self._update_workflow_metrics(workflow, execution_time)
            
            self.logger.info(f"Workflow {workflow.workflow_id} completed with state: {workflow.state}")
            
        except Exception as e:
            workflow.state = WorkflowState.FAILED
            workflow.error_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Workflow execution failed: {str(e)}"
            })
            self.logger.error(f"Workflow {workflow.workflow_id} execution failed: {str(e)}")
        
        finally:
            # Cleanup
            if workflow.workflow_id in self.running_tasks:
                del self.running_tasks[workflow.workflow_id]
    
    async def _execute_task(self, workflow: Workflow, task: WorkflowTask) -> None:
        """Execute an individual task"""
        
        try:
            task.state = TaskState.RUNNING
            task.started_at = datetime.utcnow()
            task.attempts += 1
            
            self.logger.info(f"Executing task {task.task_id} on agent {task.agent_id}")
            
            # Create agent message
            message = AgentMessage(
                agent_id="workflow_engine",
                target_agent_id=task.agent_id,
                message_type=MessageType.REQUEST,
                payload={
                    "action": task.action,
                    "parameters": task.parameters,
                    "task_id": task.task_id,
                    "workflow_id": workflow.workflow_id
                },
                correlation_id=workflow.correlation_id,
                session_id=workflow.session_id,
                priority=Priority(workflow.priority.value)
            )
            
            # Send message and wait for response
            response = await self._send_task_message(message, task.timeout_seconds)
            
            if response and response.message_type != MessageType.ERROR:
                # Task completed successfully
                task.state = TaskState.COMPLETED
                task.completed_at = datetime.utcnow()
                task.result = response.payload
                
                # Store rollback data if provided
                if task.rollback_action and "rollback_data" in response.payload:
                    task.rollback_data = response.payload["rollback_data"]
                    workflow.rollback_stack.append(task.task_id)
                
                self.engine_metrics["successful_tasks"] += 1
                self.logger.info(f"Task {task.task_id} completed successfully")
                
            else:
                # Task failed
                error_msg = response.payload.get("error", "Unknown error") if response else "No response received"
                await self._handle_task_failure(workflow, task, error_msg)
                
        except asyncio.TimeoutError:
            await self._handle_task_failure(workflow, task, "Task timeout")
        except Exception as e:
            await self._handle_task_failure(workflow, task, str(e))
    
    async def _handle_task_failure(self, workflow: Workflow, task: WorkflowTask, error: str) -> None:
        """Handle task failure with retry logic"""
        
        task.error = error
        self.logger.warning(f"Task {task.task_id} failed: {error}")
        
        # Check if we should retry
        if task.attempts < task.retry_count:
            task.state = TaskState.RETRYING
            self.logger.info(f"Retrying task {task.task_id} (attempt {task.attempts + 1}/{task.retry_count})")
            
            # Wait before retry
            await asyncio.sleep(min(2 ** task.attempts, 30))  # Exponential backoff
            
            # Retry task
            await self._execute_task(workflow, task)
        else:
            # Max retries reached
            task.state = TaskState.FAILED
            task.completed_at = datetime.utcnow()
            
            workflow.error_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task.task_id,
                "error": error,
                "attempts": task.attempts
            })
            
            self.engine_metrics["failed_tasks"] += 1
            self.logger.error(f"Task {task.task_id} failed permanently after {task.attempts} attempts")
    
    async def _send_task_message(self, message: AgentMessage, timeout_seconds: int) -> Optional[AgentMessage]:
        """Send task message to agent and wait for response"""
        
        try:
            # Send message
            success = await self.communication_framework.send_message(message)
            if not success:
                return None
            
            # Wait for response (simplified - in real implementation would use proper response correlation)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Return mock successful response for now
            return AgentMessage(
                agent_id=message.target_agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={"status": "completed", "result": "success"},
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send task message: {str(e)}")
            return None
    
    async def _rollback_workflow(self, workflow: Workflow) -> None:
        """Rollback workflow by executing rollback actions"""
        
        if not workflow.rollback_enabled or not workflow.rollback_stack:
            return
        
        workflow.state = WorkflowState.ROLLING_BACK
        self.logger.info(f"Rolling back workflow {workflow.workflow_id}")
        
        try:
            # Execute rollback actions in reverse order
            for task_id in reversed(workflow.rollback_stack):
                task = workflow.tasks[task_id]
                
                if task.rollback_action and task.rollback_data:
                    await self._execute_rollback_task(workflow, task)
            
            self.logger.info(f"Workflow {workflow.workflow_id} rollback completed")
            
        except Exception as e:
            self.logger.error(f"Rollback failed for workflow {workflow.workflow_id}: {str(e)}")
    
    async def _execute_rollback_task(self, workflow: Workflow, task: WorkflowTask) -> None:
        """Execute rollback action for a task"""
        
        try:
            self.logger.info(f"Rolling back task {task.task_id}")
            
            # Create rollback message
            rollback_message = AgentMessage(
                agent_id="workflow_engine",
                target_agent_id=task.agent_id,
                message_type=MessageType.REQUEST,
                payload={
                    "action": task.rollback_action,
                    "rollback_data": task.rollback_data,
                    "task_id": task.task_id,
                    "workflow_id": workflow.workflow_id
                },
                correlation_id=workflow.correlation_id,
                session_id=workflow.session_id,
                priority=Priority.HIGH
            )
            
            # Send rollback message
            await self._send_task_message(rollback_message, 60)  # 1 minute timeout for rollback
            
        except Exception as e:
            self.logger.error(f"Failed to rollback task {task.task_id}: {str(e)}")
    
    def _update_workflow_metrics(self, workflow: Workflow, execution_time: float) -> None:
        """Update workflow performance metrics"""
        
        # Update average workflow time
        total_workflows = self.engine_metrics["successful_workflows"]
        current_avg = self.engine_metrics["average_workflow_time"]
        
        self.engine_metrics["average_workflow_time"] = (
            (current_avg * (total_workflows - 1) + execution_time) / total_workflows
        )
        
        # Store workflow-specific metrics
        workflow.performance_metrics = {
            "execution_time_seconds": execution_time,
            "total_tasks": len(workflow.tasks),
            "successful_tasks": sum(1 for t in workflow.tasks.values() if t.state == TaskState.COMPLETED),
            "failed_tasks": sum(1 for t in workflow.tasks.values() if t.state == TaskState.FAILED),
            "retry_count": sum(t.attempts - 1 for t in workflow.tasks.values()),
            "average_task_time": execution_time / len(workflow.tasks) if workflow.tasks else 0
        }
    
    # ========================================================================
    # WORKFLOW TEMPLATES
    # ========================================================================
    
    def _create_financial_planning_workflow(
        self,
        workflow_id: str,
        user_id: str,
        parameters: Dict[str, Any],
        priority: WorkflowPriority
    ) -> Workflow:
        """Create financial planning workflow"""
        
        workflow = Workflow(
            workflow_id=workflow_id,
            workflow_name="Financial Planning",
            workflow_type="financial_planning",
            user_id=user_id,
            priority=priority,
            timeout_minutes=30
        )
        
        # Task 1: Parse user goal
        parse_task = WorkflowTask(
            task_id=f"{workflow_id}_parse",
            task_name="Parse User Goal",
            agent_id="orchestration_agent",
            action="parse_goal",
            parameters={"goal_text": parameters.get("goal_text", "")},
            timeout_seconds=60
        )
        workflow.add_task(parse_task)
        
        # Task 2: Generate plan
        plan_task = WorkflowTask(
            task_id=f"{workflow_id}_plan",
            task_name="Generate Financial Plan",
            agent_id="planning_agent",
            action="generate_plan",
            parameters={"planning_request": parameters.get("planning_request", {})},
            dependencies=[parse_task.task_id],
            timeout_seconds=300,
            rollback_action="rollback_plan"
        )
        workflow.add_task(plan_task)
        
        # Task 3: Verify plan
        verify_task = WorkflowTask(
            task_id=f"{workflow_id}_verify",
            task_name="Verify Plan",
            agent_id="verification_agent",
            action="verify_plan",
            parameters={"plan_data": {}},
            dependencies=[plan_task.task_id],
            timeout_seconds=180
        )
        workflow.add_task(verify_task)
        
        # Task 4: Execute plan
        execute_task = WorkflowTask(
            task_id=f"{workflow_id}_execute",
            task_name="Execute Plan",
            agent_id="execution_agent",
            action="execute_plan",
            parameters={"execution_mode": "standard"},
            dependencies=[verify_task.task_id],
            timeout_seconds=600,
            rollback_action="rollback_execution"
        )
        workflow.add_task(execute_task)
        
        return workflow
    
    def _create_portfolio_rebalancing_workflow(
        self,
        workflow_id: str,
        user_id: str,
        parameters: Dict[str, Any],
        priority: WorkflowPriority
    ) -> Workflow:
        """Create portfolio rebalancing workflow"""
        
        workflow = Workflow(
            workflow_id=workflow_id,
            workflow_name="Portfolio Rebalancing",
            workflow_type="portfolio_rebalancing",
            user_id=user_id,
            priority=priority,
            timeout_minutes=20
        )
        
        # Task 1: Analyze current portfolio
        analyze_task = WorkflowTask(
            task_id=f"{workflow_id}_analyze",
            task_name="Analyze Portfolio",
            agent_id="execution_agent",
            action="analyze_portfolio",
            parameters={"user_id": user_id},
            timeout_seconds=120
        )
        workflow.add_task(analyze_task)
        
        # Task 2: Generate rebalancing plan
        rebalance_plan_task = WorkflowTask(
            task_id=f"{workflow_id}_rebalance_plan",
            task_name="Generate Rebalancing Plan",
            agent_id="planning_agent",
            action="generate_rebalancing_plan",
            parameters=parameters,
            dependencies=[analyze_task.task_id],
            timeout_seconds=180
        )
        workflow.add_task(rebalance_plan_task)
        
        # Task 3: Execute rebalancing
        execute_rebalance_task = WorkflowTask(
            task_id=f"{workflow_id}_execute_rebalance",
            task_name="Execute Rebalancing",
            agent_id="execution_agent",
            action="execute_rebalancing",
            parameters={"user_id": user_id},
            dependencies=[rebalance_plan_task.task_id],
            timeout_seconds=300,
            rollback_action="rollback_rebalancing"
        )
        workflow.add_task(execute_rebalance_task)
        
        return workflow
    
    def _create_compliance_review_workflow(
        self,
        workflow_id: str,
        user_id: str,
        parameters: Dict[str, Any],
        priority: WorkflowPriority
    ) -> Workflow:
        """Create compliance review workflow"""
        
        workflow = Workflow(
            workflow_id=workflow_id,
            workflow_name="Compliance Review",
            workflow_type="compliance_review",
            user_id=user_id,
            priority=priority,
            timeout_minutes=15
        )
        
        # Task 1: Check compliance
        compliance_task = WorkflowTask(
            task_id=f"{workflow_id}_compliance",
            task_name="Check Compliance",
            agent_id="verification_agent",
            action="check_compliance",
            parameters={"user_id": user_id, "check_type": parameters.get("check_type", "full")},
            timeout_seconds=180
        )
        workflow.add_task(compliance_task)
        
        # Task 2: Generate report
        report_task = WorkflowTask(
            task_id=f"{workflow_id}_report",
            task_name="Generate Compliance Report",
            agent_id="execution_agent",
            action="generate_compliance_report",
            parameters={"user_id": user_id},
            dependencies=[compliance_task.task_id],
            timeout_seconds=120
        )
        workflow.add_task(report_task)
        
        return workflow
    
    def _create_emergency_response_workflow(
        self,
        workflow_id: str,
        user_id: str,
        parameters: Dict[str, Any],
        priority: WorkflowPriority
    ) -> Workflow:
        """Create emergency response workflow for triggers"""
        
        workflow = Workflow(
            workflow_id=workflow_id,
            workflow_name="Emergency Response",
            workflow_type="emergency_response",
            user_id=user_id,
            priority=WorkflowPriority.CRITICAL,  # Always critical
            timeout_minutes=10,
            rollback_enabled=True
        )
        
        # Task 1: Assess situation
        assess_task = WorkflowTask(
            task_id=f"{workflow_id}_assess",
            task_name="Assess Emergency",
            agent_id="verification_agent",
            action="assess_emergency",
            parameters=parameters,
            timeout_seconds=60
        )
        workflow.add_task(assess_task)
        
        # Task 2: Generate emergency plan
        emergency_plan_task = WorkflowTask(
            task_id=f"{workflow_id}_emergency_plan",
            task_name="Generate Emergency Plan",
            agent_id="planning_agent",
            action="generate_emergency_plan",
            parameters=parameters,
            dependencies=[assess_task.task_id],
            timeout_seconds=120
        )
        workflow.add_task(emergency_plan_task)
        
        # Task 3: Execute emergency actions
        execute_emergency_task = WorkflowTask(
            task_id=f"{workflow_id}_execute_emergency",
            task_name="Execute Emergency Actions",
            agent_id="execution_agent",
            action="execute_emergency_actions",
            parameters={"user_id": user_id},
            dependencies=[emergency_plan_task.task_id],
            timeout_seconds=180,
            rollback_action="rollback_emergency_actions"
        )
        workflow.add_task(execute_emergency_task)
        
        return workflow
    
    def _create_tax_optimization_workflow(
        self,
        workflow_id: str,
        user_id: str,
        parameters: Dict[str, Any],
        priority: WorkflowPriority
    ) -> Workflow:
        """Create tax optimization workflow"""
        
        workflow = Workflow(
            workflow_id=workflow_id,
            workflow_name="Tax Optimization",
            workflow_type="tax_optimization",
            user_id=user_id,
            priority=priority,
            timeout_minutes=25
        )
        
        # Task 1: Analyze tax situation
        tax_analysis_task = WorkflowTask(
            task_id=f"{workflow_id}_tax_analysis",
            task_name="Analyze Tax Situation",
            agent_id="execution_agent",
            action="analyze_tax_situation",
            parameters={"user_id": user_id},
            timeout_seconds=120
        )
        workflow.add_task(tax_analysis_task)
        
        # Task 2: Generate tax optimization strategies
        tax_strategy_task = WorkflowTask(
            task_id=f"{workflow_id}_tax_strategy",
            task_name="Generate Tax Strategies",
            agent_id="planning_agent",
            action="generate_tax_strategies",
            parameters=parameters,
            dependencies=[tax_analysis_task.task_id],
            timeout_seconds=180
        )
        workflow.add_task(tax_strategy_task)
        
        # Task 3: Execute tax optimization
        execute_tax_task = WorkflowTask(
            task_id=f"{workflow_id}_execute_tax",
            task_name="Execute Tax Optimization",
            agent_id="execution_agent",
            action="execute_tax_optimization",
            parameters={"user_id": user_id},
            dependencies=[tax_strategy_task.task_id],
            timeout_seconds=240,
            rollback_action="rollback_tax_optimization"
        )
        workflow.add_task(execute_tax_task)
        
        return workflow
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status and progress"""
        workflow = self.active_workflows.get(workflow_id)
        return workflow.to_dict() if workflow else None
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows"""
        return [workflow.to_dict() for workflow in self.active_workflows.values()]
    
    def get_engine_metrics(self) -> Dict[str, Any]:
        """Get workflow engine performance metrics"""
        return {
            **self.engine_metrics,
            "active_workflows": len(self.active_workflows),
            "running_tasks": len(self.running_tasks),
            "available_templates": list(self.workflow_templates.keys())
        }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return False
        
        workflow.state = WorkflowState.CANCELLED
        
        # Cancel running task if exists
        if workflow_id in self.running_tasks:
            self.running_tasks[workflow_id].cancel()
            del self.running_tasks[workflow_id]
        
        self.logger.info(f"Cancelled workflow {workflow_id}")
        return True
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow or workflow.state != WorkflowState.RUNNING:
            return False
        
        workflow.state = WorkflowState.PAUSED
        self.logger.info(f"Paused workflow {workflow_id}")
        return True
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow or workflow.state != WorkflowState.PAUSED:
            return False
        
        workflow.state = WorkflowState.RUNNING
        
        # Restart execution if needed
        if workflow_id not in self.running_tasks:
            execution_task = asyncio.create_task(self._execute_workflow(workflow))
            self.running_tasks[workflow_id] = execution_task
        
        self.logger.info(f"Resumed workflow {workflow_id}")
        return True
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> int:
        """Clean up old completed workflows"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        workflows_to_remove = []
        
        for workflow_id, workflow in self.active_workflows.items():
            if (workflow.state in [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED] and
                workflow.completed_at and workflow.completed_at < cutoff_time):
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            self.workflow_history.append(workflow_id)
            del self.active_workflows[workflow_id]
        
        self.logger.info(f"Cleaned up {len(workflows_to_remove)} completed workflows")
        return len(workflows_to_remove)