"""
Enhanced Orchestration Agent for FinPilot VP-MAS

The Orchestration Agent serves as the mission control center for the multi-agent system,
coordinating workflows, managing triggers, parsing user goals, and ensuring system reliability
through circuit breakers and comprehensive monitoring.

Requirements: 1.1, 2.2, 4.1, 4.3, 6.1, 6.2, 6.3
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4
from enum import Enum

from .base_agent import BaseAgent
from .communication import AgentCommunicationFramework, CircuitBreaker
from data_models.schemas import (
    AgentMessage, MessageType, Priority, ExecutionStatus,
    EnhancedPlanRequest, TriggerEvent, SeverityLevel, MarketEventType,
    PerformanceMetrics, ExecutionLog, ReasoningTrace, DecisionPoint,
    FinancialState, Constraint, ConstraintType, ConstraintPriority,
    RiskProfile, TaxContext, RegulatoryRequirement, ComplianceStatus
)


class WorkflowState(str, Enum):
    """States of workflow execution"""
    IDLE = "idle"
    PARSING_GOAL = "parsing_goal"
    DELEGATING_TASKS = "delegating_tasks"
    MONITORING_EXECUTION = "monitoring_execution"
    HANDLING_TRIGGERS = "handling_triggers"
    COORDINATING_REPLANNING = "coordinating_replanning"
    FINALIZING_RESULTS = "finalizing_results"
    ERROR_RECOVERY = "error_recovery"


class TriggerType(str, Enum):
    """Types of triggers that can initiate CMVL"""
    MARKET_VOLATILITY = "market_volatility"
    LIFE_EVENT = "life_event"
    REGULATORY_CHANGE = "regulatory_change"
    PORTFOLIO_THRESHOLD = "portfolio_threshold"
    EMERGENCY = "emergency"
    SCHEDULED_REVIEW = "scheduled_review"


class SessionManager:
    """Manages user sessions and correlation tracking"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.correlation_map: Dict[str, str] = {}  # correlation_id -> session_id
        self.session_history: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger("finpilot.orchestration.session_manager")
    
    def create_session(self, user_id: str, initial_goal: str) -> str:
        """Create a new user session"""
        session_id = str(uuid4())
        
        self.active_sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "initial_goal": initial_goal,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "workflow_state": WorkflowState.IDLE,
            "active_correlations": set(),
            "trigger_history": [],
            "plan_versions": [],
            "performance_metrics": {},
            "error_count": 0
        }
        
        self.session_history[session_id] = []
        
        self.logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        return self.active_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str, activity: Dict[str, Any]) -> None:
        """Update session with new activity"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = datetime.utcnow()
            self.session_history[session_id].append({
                "timestamp": datetime.utcnow(),
                "activity": activity
            })
    
    def register_correlation(self, correlation_id: str, session_id: str) -> None:
        """Register correlation ID with session"""
        self.correlation_map[correlation_id] = session_id
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["active_correlations"].add(correlation_id)
    
    def get_session_by_correlation(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Get session by correlation ID"""
        session_id = self.correlation_map.get(correlation_id)
        return self.get_session(session_id) if session_id else None
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> None:
        """Clean up expired sessions"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if session["last_activity"] < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.logger.info(f"Cleaning up expired session {session_id}")
            del self.active_sessions[session_id]
            # Clean up correlation mappings
            correlations_to_remove = [
                corr_id for corr_id, sess_id in self.correlation_map.items()
                if sess_id == session_id
            ]
            for corr_id in correlations_to_remove:
                del self.correlation_map[corr_id]


class GoalParser:
    """Parses natural language financial goals into structured requests"""
    
    def __init__(self):
        self.logger = logging.getLogger("finpilot.orchestration.goal_parser")
        
        # Goal pattern recognition
        self.goal_patterns = {
            "retirement": ["retire", "retirement", "pension", "401k", "ira"],
            "emergency_fund": ["emergency", "rainy day", "safety net", "buffer"],
            "debt_payoff": ["debt", "loan", "payoff", "pay off", "eliminate"],
            "home_purchase": ["house", "home", "mortgage", "down payment"],
            "education": ["college", "education", "tuition", "school"],
            "investment": ["invest", "portfolio", "stocks", "bonds", "mutual fund"],
            "savings": ["save", "saving", "savings", "accumulate"]
        }
        
        # Constraint keywords
        self.constraint_keywords = {
            "time": ["year", "years", "month", "months", "by", "within", "before"],
            "amount": ["$", "dollar", "thousand", "million", "k", "m"],
            "risk": ["conservative", "aggressive", "moderate", "safe", "risky"],
            "tax": ["tax", "after-tax", "pre-tax", "tax-free", "deductible"]
        }
    
    async def parse_goal(self, goal_text: str, user_context: Dict[str, Any]) -> EnhancedPlanRequest:
        """Parse natural language goal into structured planning request"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Parsing goal: {goal_text}")
            
            # Extract goal type
            goal_type = self._identify_goal_type(goal_text.lower())
            
            # Extract constraints
            constraints = self._extract_constraints(goal_text.lower())
            
            # Extract time horizon
            time_horizon = self._extract_time_horizon(goal_text.lower())
            
            # Extract amount if specified
            target_amount = self._extract_amount(goal_text)
            
            # Determine priority based on keywords
            priority = self._determine_priority(goal_text.lower())
            
            # Create structured request
            request = EnhancedPlanRequest(
                user_id=user_context.get("user_id", "unknown"),
                user_goal=goal_text,
                current_state=user_context.get("financial_state", {}),
                constraints=constraints,
                risk_profile=user_context.get("risk_profile", {}),
                regulatory_requirements=user_context.get("regulatory_requirements", []),
                tax_considerations=user_context.get("tax_context", {}),
                time_horizon=time_horizon,
                optimization_preferences={
                    "goal_type": goal_type,
                    "target_amount": target_amount,
                    "extracted_keywords": self._extract_keywords(goal_text)
                },
                correlation_id=str(uuid4()),
                session_id=user_context.get("session_id", str(uuid4())),
                priority=priority
            )
            
            parsing_time = time.time() - start_time
            self.logger.info(f"Goal parsed successfully in {parsing_time:.3f}s: {goal_type}")
            
            return request
            
        except Exception as e:
            self.logger.error(f"Failed to parse goal: {str(e)}")
            raise
    
    def _identify_goal_type(self, goal_text: str) -> str:
        """Identify the primary goal type from text"""
        for goal_type, keywords in self.goal_patterns.items():
            if any(keyword in goal_text for keyword in keywords):
                return goal_type
        return "general_financial_planning"
    
    def _extract_constraints(self, goal_text: str) -> List[Dict[str, Any]]:
        """Extract constraints from goal text"""
        constraints = []
        
        # Time constraints
        if any(keyword in goal_text for keyword in self.constraint_keywords["time"]):
            constraints.append({
                "type": ConstraintType.TIME,
                "priority": ConstraintPriority.HIGH,
                "description": "Time-based constraint extracted from goal",
                "source": "goal_parsing"
            })
        
        # Risk constraints
        risk_keywords = self.constraint_keywords["risk"]
        for keyword in risk_keywords:
            if keyword in goal_text:
                constraints.append({
                    "type": ConstraintType.RISK,
                    "priority": ConstraintPriority.MEDIUM,
                    "description": f"Risk preference: {keyword}",
                    "value": keyword,
                    "source": "goal_parsing"
                })
                break
        
        # Tax constraints
        if any(keyword in goal_text for keyword in self.constraint_keywords["tax"]):
            constraints.append({
                "type": ConstraintType.TAX,
                "priority": ConstraintPriority.MEDIUM,
                "description": "Tax optimization constraint",
                "source": "goal_parsing"
            })
        
        return constraints
    
    def _extract_time_horizon(self, goal_text: str) -> int:
        """Extract time horizon in months from goal text"""
        import re
        
        # Look for patterns like "5 years", "18 months", "by 2030"
        year_pattern = r'(\d+)\s*year'
        month_pattern = r'(\d+)\s*month'
        
        year_match = re.search(year_pattern, goal_text)
        if year_match:
            return int(year_match.group(1)) * 12
        
        month_match = re.search(month_pattern, goal_text)
        if month_match:
            return int(month_match.group(1))
        
        # Default time horizon based on goal type
        if "retirement" in goal_text:
            return 360  # 30 years
        elif "emergency" in goal_text:
            return 12   # 1 year
        elif "house" in goal_text or "home" in goal_text:
            return 60   # 5 years
        else:
            return 36   # 3 years default
    
    def _extract_amount(self, goal_text: str) -> Optional[float]:
        """Extract monetary amount from goal text"""
        import re
        
        # Patterns for different amount formats
        patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $1,000.00
            r'(\d+(?:,\d{3})*)\s*(?:dollars?|k|thousand)',  # 1000 dollars, 50k
            r'(\d+(?:\.\d+)?)\s*(?:million|m)',  # 1.5 million
        ]
        
        for pattern in patterns:
            match = re.search(pattern, goal_text.lower())
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                
                # Apply multipliers
                if 'k' in goal_text.lower() or 'thousand' in goal_text.lower():
                    amount *= 1000
                elif 'million' in goal_text.lower() or 'm' in goal_text.lower():
                    amount *= 1000000
                
                return amount
        
        return None
    
    def _determine_priority(self, goal_text: str) -> Priority:
        """Determine priority based on goal text"""
        high_priority_keywords = ["urgent", "emergency", "asap", "immediately", "critical"]
        low_priority_keywords = ["eventually", "someday", "when possible", "if possible"]
        
        if any(keyword in goal_text for keyword in high_priority_keywords):
            return Priority.HIGH
        elif any(keyword in goal_text for keyword in low_priority_keywords):
            return Priority.LOW
        else:
            return Priority.MEDIUM
    
    def _extract_keywords(self, goal_text: str) -> List[str]:
        """Extract relevant keywords for optimization"""
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "i", "want", "need", "would", "like"}
        words = re.findall(r'\b\w+\b', goal_text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords


class TriggerMonitor:
    """Monitors for triggers that require CMVL initiation"""
    
    def __init__(self, orchestration_agent: 'OrchestrationAgent'):
        self.orchestration_agent = orchestration_agent
        self.active_triggers: Dict[str, TriggerEvent] = {}
        self.trigger_thresholds = {
            TriggerType.MARKET_VOLATILITY: 0.3,
            TriggerType.PORTFOLIO_THRESHOLD: 0.1,
            TriggerType.EMERGENCY: 0.0  # Always trigger
        }
        self.concurrent_trigger_handler = ConcurrentTriggerHandler()
        self.logger = logging.getLogger("finpilot.orchestration.trigger_monitor")
    
    async def register_trigger(self, trigger: TriggerEvent) -> None:
        """Register a new trigger event"""
        self.logger.info(f"Registering trigger: {trigger.trigger_type} - {trigger.severity}")
        
        self.active_triggers[trigger.trigger_id] = trigger
        
        # Check if trigger meets threshold for CMVL initiation
        if self._should_initiate_cmvl(trigger):
            await self._initiate_cmvl(trigger)
    
    async def handle_concurrent_triggers(self, triggers: List[TriggerEvent]) -> None:
        """Handle multiple concurrent triggers with intelligent prioritization"""
        self.logger.warning(f"Handling {len(triggers)} concurrent triggers")
        
        # Sort triggers by priority and severity
        prioritized_triggers = self.concurrent_trigger_handler.prioritize_triggers(triggers)
        
        # Create compound trigger event
        compound_trigger = self._create_compound_trigger(prioritized_triggers)
        
        # Initiate enhanced CMVL for compound scenario
        await self._initiate_compound_cmvl(compound_trigger, prioritized_triggers)
    
    def _should_initiate_cmvl(self, trigger: TriggerEvent) -> bool:
        """Determine if trigger should initiate CMVL"""
        if trigger.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            return True
        
        # Check trigger-specific thresholds
        trigger_type = TriggerType(trigger.trigger_type)
        threshold = self.trigger_thresholds.get(trigger_type, 0.5)
        
        return trigger.impact_score >= threshold
    
    async def _initiate_cmvl(self, trigger: TriggerEvent) -> None:
        """Initiate CMVL process for a trigger"""
        self.logger.info(f"Initiating CMVL for trigger: {trigger.trigger_id}")
        
        # Create CMVL initiation message
        cmvl_message = self.orchestration_agent.communication_framework.create_message(
            sender_id=self.orchestration_agent.agent_id,
            target_id="verification_agent",
            message_type=MessageType.REQUEST,
            payload={
                "action": "initiate_cmvl",
                "trigger": trigger.dict(),
                "priority": "high" if trigger.severity == SeverityLevel.CRITICAL else "medium"
            },
            correlation_id=trigger.correlation_id,
            priority=Priority.HIGH if trigger.severity == SeverityLevel.CRITICAL else Priority.MEDIUM
        )
        
        await self.orchestration_agent.communication_framework.send_message(cmvl_message)
    
    async def _initiate_compound_cmvl(self, compound_trigger: TriggerEvent, individual_triggers: List[TriggerEvent]) -> None:
        """Initiate CMVL for compound trigger scenario"""
        self.logger.critical(f"Initiating compound CMVL for {len(individual_triggers)} triggers")
        
        # Create enhanced CMVL message with rollback capabilities
        cmvl_message = self.orchestration_agent.communication_framework.create_message(
            sender_id=self.orchestration_agent.agent_id,
            target_id="verification_agent",
            message_type=MessageType.REQUEST,
            payload={
                "action": "initiate_compound_cmvl",
                "compound_trigger": compound_trigger.dict(),
                "individual_triggers": [t.dict() for t in individual_triggers],
                "rollback_enabled": True,
                "priority": "critical"
            },
            correlation_id=compound_trigger.correlation_id,
            priority=Priority.CRITICAL
        )
        
        await self.orchestration_agent.communication_framework.send_message(cmvl_message)
    
    def _create_compound_trigger(self, triggers: List[TriggerEvent]) -> TriggerEvent:
        """Create a compound trigger from multiple individual triggers"""
        # Calculate compound impact score
        compound_impact = min(1.0, sum(t.impact_score for t in triggers) * 0.8)  # 80% of sum to avoid over-amplification
        
        # Determine compound severity
        severities = [t.severity for t in triggers]
        if SeverityLevel.CRITICAL in severities:
            compound_severity = SeverityLevel.CRITICAL
        elif SeverityLevel.HIGH in severities:
            compound_severity = SeverityLevel.HIGH
        else:
            compound_severity = SeverityLevel.MEDIUM
        
        return TriggerEvent(
            trigger_type="compound_trigger",
            event_type=MarketEventType.MARKET_CRASH,  # Default for compound
            severity=compound_severity,
            description=f"Compound trigger: {', '.join(t.trigger_type for t in triggers)}",
            source_data={
                "individual_triggers": [t.trigger_id for t in triggers],
                "trigger_types": [t.trigger_type for t in triggers],
                "compound_scenario": True
            },
            impact_score=compound_impact,
            confidence_score=min(t.confidence_score for t in triggers),
            detector_agent_id=self.orchestration_agent.agent_id,
            correlation_id=str(uuid4())
        )


class ConcurrentTriggerHandler:
    """Handles concurrent trigger events with intelligent prioritization"""
    
    def __init__(self):
        self.priority_matrix = {
            # (trigger_type, severity) -> priority_score
            (TriggerType.EMERGENCY, SeverityLevel.CRITICAL): 100,
            (TriggerType.MARKET_VOLATILITY, SeverityLevel.CRITICAL): 90,
            (TriggerType.REGULATORY_CHANGE, SeverityLevel.CRITICAL): 85,
            (TriggerType.PORTFOLIO_THRESHOLD, SeverityLevel.CRITICAL): 80,
            (TriggerType.LIFE_EVENT, SeverityLevel.HIGH): 70,
            (TriggerType.MARKET_VOLATILITY, SeverityLevel.HIGH): 65,
            # ... more combinations
        }
        self.logger = logging.getLogger("finpilot.orchestration.concurrent_handler")
    
    def prioritize_triggers(self, triggers: List[TriggerEvent]) -> List[TriggerEvent]:
        """Prioritize triggers based on type, severity, and impact"""
        def get_priority_score(trigger: TriggerEvent) -> float:
            base_score = self.priority_matrix.get(
                (TriggerType(trigger.trigger_type), trigger.severity), 
                50  # default score
            )
            
            # Adjust by impact score and confidence
            adjusted_score = base_score * trigger.impact_score * trigger.confidence_score
            
            return adjusted_score
        
        prioritized = sorted(triggers, key=get_priority_score, reverse=True)
        
        self.logger.info(f"Prioritized {len(triggers)} triggers")
        return prioritized


class TaskDelegator:
    """Intelligent task delegation system with priority handling"""
    
    def __init__(self, communication_framework: AgentCommunicationFramework):
        self.communication_framework = communication_framework
        self.task_queue: List[Dict[str, Any]] = []
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities = {
            "planning_agent": ["financial_planning", "strategy_optimization", "constraint_solving"],
            "information_retrieval_agent": ["market_data", "regulatory_data", "external_apis"],
            "verification_agent": ["constraint_validation", "compliance_checking", "risk_assessment"],
            "execution_agent": ["portfolio_updates", "transaction_execution", "ledger_management"]
        }
        self.logger = logging.getLogger("finpilot.orchestration.task_delegator")
    
    async def delegate_planning_task(self, request: EnhancedPlanRequest) -> str:
        """Delegate planning task to appropriate agent"""
        task_id = str(uuid4())
        
        # Create planning task
        planning_message = self.communication_framework.create_message(
            sender_id="orchestration_agent",
            target_id="planning_agent",
            message_type=MessageType.REQUEST,
            payload={
                "action": "generate_plan",
                "request": request.dict(),
                "task_id": task_id
            },
            correlation_id=request.correlation_id,
            session_id=request.session_id,
            priority=request.priority
        )
        
        # Track active task
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "task_type": "planning",
            "assigned_agent": "planning_agent",
            "status": "delegated",
            "created_at": datetime.utcnow(),
            "correlation_id": request.correlation_id,
            "priority": request.priority
        }
        
        await self.communication_framework.send_message(planning_message)
        
        self.logger.info(f"Delegated planning task {task_id} to planning_agent")
        return task_id
    
    async def delegate_verification_task(self, plan_data: Dict[str, Any], correlation_id: str) -> str:
        """Delegate verification task to verification agent"""
        task_id = str(uuid4())
        
        verification_message = self.communication_framework.create_message(
            sender_id="orchestration_agent",
            target_id="verification_agent",
            message_type=MessageType.REQUEST,
            payload={
                "action": "verify_plan",
                "plan_data": plan_data,
                "task_id": task_id
            },
            correlation_id=correlation_id,
            priority=Priority.HIGH
        )
        
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "task_type": "verification",
            "assigned_agent": "verification_agent",
            "status": "delegated",
            "created_at": datetime.utcnow(),
            "correlation_id": correlation_id,
            "priority": Priority.HIGH
        }
        
        await self.communication_framework.send_message(verification_message)
        
        self.logger.info(f"Delegated verification task {task_id} to verification_agent")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a delegated task"""
        return self.active_tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Update task status"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = status
            self.active_tasks[task_id]["updated_at"] = datetime.utcnow()
            if result:
                self.active_tasks[task_id]["result"] = result


class OrchestrationAgent(BaseAgent):
    """
    Enhanced Orchestration Agent that serves as mission control for the VP-MAS system.
    
    Coordinates workflows, manages triggers, parses user goals, delegates tasks,
    and ensures system reliability through circuit breakers and comprehensive monitoring.
    """
    
    def __init__(self, agent_id: str = "orchestration_agent"):
        super().__init__(agent_id, "orchestration")
        
        # Initialize core components
        self.communication_framework = AgentCommunicationFramework()
        self.session_manager = SessionManager()
        self.goal_parser = GoalParser()
        self.trigger_monitor = TriggerMonitor(self)
        self.task_delegator = TaskDelegator(self.communication_framework)
        
        # Workflow management
        self.current_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Circuit breakers for external dependencies
        self.circuit_breakers = {
            "planning_agent": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            "verification_agent": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            "information_retrieval_agent": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            "execution_agent": CircuitBreaker(failure_threshold=2, recovery_timeout=45)
        }
        
        # Performance tracking
        self.workflow_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_completion_time": 0.0,
            "trigger_events_handled": 0,
            "concurrent_triggers_handled": 0
        }
        
        self.logger.info("Enhanced Orchestration Agent initialized")
    
    async def start(self) -> None:
        """Start the orchestration agent and all subsystems"""
        await super().start()
        
        # Register with communication framework
        self.communication_framework.register_agent(self, [
            "workflow_coordination",
            "goal_parsing", 
            "task_delegation",
            "trigger_monitoring",
            "session_management"
        ])
        
        # Start background tasks
        asyncio.create_task(self._monitor_system_health())
        asyncio.create_task(self._cleanup_expired_sessions())
        
        self.logger.info("Orchestration Agent started successfully")
    
    async def process_user_goal(self, user_id: str, goal_text: str, user_context: Dict[str, Any] = None) -> str:
        """Process a user's financial goal and initiate planning workflow"""
        start_time = time.time()
        
        try:
            # Create session
            session_id = self.session_manager.create_session(user_id, goal_text)
            
            # Add session context
            if user_context is None:
                user_context = {}
            user_context["session_id"] = session_id
            user_context["user_id"] = user_id
            
            # Parse goal into structured request
            plan_request = await self.goal_parser.parse_goal(goal_text, user_context)
            
            # Register correlation
            self.session_manager.register_correlation(plan_request.correlation_id, session_id)
            
            # Create workflow
            workflow_id = await self._create_workflow(plan_request)
            
            # Update session
            self.session_manager.update_session_activity(session_id, {
                "action": "goal_processed",
                "workflow_id": workflow_id,
                "plan_request_id": plan_request.request_id
            })
            
            # Delegate planning task
            task_id = await self.task_delegator.delegate_planning_task(plan_request)
            
            processing_time = time.time() - start_time
            self.workflow_metrics["total_workflows"] += 1
            
            self.logger.info(f"User goal processed successfully in {processing_time:.3f}s: {workflow_id}")
            
            return workflow_id
            
        except Exception as e:
            self.workflow_metrics["failed_workflows"] += 1
            self.logger.error(f"Failed to process user goal: {str(e)}")
            raise
    
    async def handle_trigger_event(self, trigger: TriggerEvent) -> None:
        """Handle incoming trigger events for CMVL initiation"""
        try:
            self.logger.info(f"Handling trigger event: {trigger.trigger_type} - {trigger.severity}")
            
            # Register trigger with monitor
            await self.trigger_monitor.register_trigger(trigger)
            
            # Update metrics
            self.workflow_metrics["trigger_events_handled"] += 1
            
            # Update session if correlation exists
            session = self.session_manager.get_session_by_correlation(trigger.correlation_id)
            if session:
                session["trigger_history"].append({
                    "trigger_id": trigger.trigger_id,
                    "trigger_type": trigger.trigger_type,
                    "severity": trigger.severity,
                    "timestamp": trigger.detected_at
                })
                
                self.session_manager.update_session_activity(session["session_id"], {
                    "action": "trigger_handled",
                    "trigger_id": trigger.trigger_id
                })
            
        except Exception as e:
            self.logger.error(f"Failed to handle trigger event: {str(e)}")
            raise
    
    async def handle_concurrent_triggers(self, triggers: List[TriggerEvent]) -> None:
        """Handle multiple concurrent trigger events"""
        try:
            self.logger.warning(f"Handling {len(triggers)} concurrent triggers")
            
            # Use trigger monitor to handle concurrent scenario
            await self.trigger_monitor.handle_concurrent_triggers(triggers)
            
            # Update metrics
            self.workflow_metrics["concurrent_triggers_handled"] += 1
            
            # Log compound scenario
            trigger_types = [t.trigger_type for t in triggers]
            self.logger.critical(f"Compound trigger scenario: {', '.join(trigger_types)}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle concurrent triggers: {str(e)}")
            raise
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages from other agents"""
        try:
            payload = message.payload
            action = payload.get("action")
            
            if action == "plan_generated":
                return await self._handle_plan_generated(message)
            elif action == "verification_complete":
                return await self._handle_verification_complete(message)
            elif action == "execution_complete":
                return await self._handle_execution_complete(message)
            elif action == "trigger_detected":
                trigger_data = payload.get("trigger")
                if trigger_data:
                    trigger = TriggerEvent(**trigger_data)
                    await self.handle_trigger_event(trigger)
            elif action == "health_check":
                return await self._handle_health_check(message)
            else:
                self.logger.warning(f"Unknown action received: {action}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return self._create_error_response(message, str(e))
    
    async def _create_workflow(self, plan_request: EnhancedPlanRequest) -> str:
        """Create a new workflow for processing a plan request"""
        workflow_id = str(uuid4())
        
        workflow = {
            "workflow_id": workflow_id,
            "session_id": plan_request.session_id,
            "correlation_id": plan_request.correlation_id,
            "user_id": plan_request.user_id,
            "state": WorkflowState.PARSING_GOAL,
            "created_at": datetime.utcnow(),
            "plan_request": plan_request.dict(),
            "tasks": [],
            "results": {},
            "error_count": 0,
            "retry_count": 0
        }
        
        self.current_workflows[workflow_id] = workflow
        
        return workflow_id
    
    async def _handle_plan_generated(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle plan generation completion"""
        payload = message.payload
        plan_data = payload.get("plan")
        task_id = payload.get("task_id")
        
        if plan_data and task_id:
            # Update task status
            self.task_delegator.update_task_status(task_id, "completed", plan_data)
            
            # Delegate verification task
            verification_task_id = await self.task_delegator.delegate_verification_task(
                plan_data, message.correlation_id
            )
            
            self.logger.info(f"Plan generated, delegated verification: {verification_task_id}")
        
        return None
    
    async def _handle_verification_complete(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle verification completion"""
        payload = message.payload
        verification_result = payload.get("verification_result")
        task_id = payload.get("task_id")
        
        if verification_result and task_id:
            # Update task status
            self.task_delegator.update_task_status(task_id, "completed", verification_result)
            
            # Check if plan was approved
            if verification_result.get("status") == "approved":
                # Delegate execution task
                execution_message = self.communication_framework.create_message(
                    sender_id=self.agent_id,
                    target_id="execution_agent",
                    message_type=MessageType.REQUEST,
                    payload={
                        "action": "execute_plan",
                        "plan_data": verification_result.get("plan"),
                        "verification_report": verification_result
                    },
                    correlation_id=message.correlation_id,
                    priority=Priority.HIGH
                )
                
                await self.communication_framework.send_message(execution_message)
                self.logger.info("Plan approved, delegated execution")
            else:
                # Plan rejected, initiate replanning
                await self._initiate_replanning(message.correlation_id, verification_result)
        
        return None
    
    async def _handle_execution_complete(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle execution completion"""
        payload = message.payload
        execution_result = payload.get("execution_result")
        
        if execution_result:
            # Update workflow metrics
            self.workflow_metrics["successful_workflows"] += 1
            
            # Update session
            session = self.session_manager.get_session_by_correlation(message.correlation_id)
            if session:
                self.session_manager.update_session_activity(session["session_id"], {
                    "action": "workflow_completed",
                    "execution_result": execution_result
                })
            
            self.logger.info("Workflow completed successfully")
        
        return None
    
    async def _initiate_replanning(self, correlation_id: str, verification_result: Dict[str, Any]) -> None:
        """Initiate replanning process when plan is rejected"""
        self.logger.info(f"Initiating replanning for correlation {correlation_id}")
        
        # Create replanning message
        replanning_message = self.communication_framework.create_message(
            sender_id=self.agent_id,
            target_id="planning_agent",
            message_type=MessageType.REQUEST,
            payload={
                "action": "replan",
                "rejection_feedback": verification_result,
                "correlation_id": correlation_id
            },
            correlation_id=correlation_id,
            priority=Priority.HIGH
        )
        
        await self.communication_framework.send_message(replanning_message)
    
    async def _handle_health_check(self, message: AgentMessage) -> AgentMessage:
        """Handle health check requests"""
        health_status = self.get_comprehensive_health_status()
        
        return self.communication_framework.create_message(
            sender_id=self.agent_id,
            target_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "action": "health_check_response",
                "health_status": health_status
            },
            correlation_id=message.correlation_id
        )
    
    def _create_error_response(self, original_message: AgentMessage, error_message: str) -> AgentMessage:
        """Create error response message"""
        return self.communication_framework.create_message(
            sender_id=self.agent_id,
            target_id=original_message.agent_id,
            message_type=MessageType.ERROR,
            payload={
                "error": error_message,
                "original_message_id": original_message.message_id
            },
            correlation_id=original_message.correlation_id
        )
    
    async def _monitor_system_health(self) -> None:
        """Background task to monitor system health"""
        while self.status == "running":
            try:
                # Check circuit breaker states
                for agent_id, circuit_breaker in self.circuit_breakers.items():
                    if circuit_breaker.state == "open":
                        self.logger.warning(f"Circuit breaker open for {agent_id}")
                
                # Check active workflows
                current_time = datetime.utcnow()
                for workflow_id, workflow in self.current_workflows.items():
                    workflow_age = (current_time - workflow["created_at"]).total_seconds()
                    if workflow_age > 300:  # 5 minutes
                        self.logger.warning(f"Long-running workflow detected: {workflow_id}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _cleanup_expired_sessions(self) -> None:
        """Background task to cleanup expired sessions"""
        while self.status == "running":
            try:
                self.session_manager.cleanup_expired_sessions()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {str(e)}")
                await asyncio.sleep(3600)
    
    def get_comprehensive_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status including all subsystems"""
        base_health = self.get_health_status()
        
        # Add orchestration-specific metrics
        base_health.update({
            "workflow_metrics": self.workflow_metrics,
            "active_workflows": len(self.current_workflows),
            "active_sessions": len(self.session_manager.active_sessions),
            "circuit_breaker_status": {
                agent_id: {"state": cb.state, "failure_count": cb.failure_count}
                for agent_id, cb in self.circuit_breakers.items()
            },
            "communication_framework_health": self.communication_framework.get_system_health(),
            "trigger_monitor_status": {
                "active_triggers": len(self.trigger_monitor.active_triggers)
            }
        })
        
        return base_health
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestration agent"""
        self.logger.info("Shutting down Orchestration Agent...")
        
        # Stop communication framework
        await self.communication_framework.shutdown()
        
        # Stop base agent
        await self.stop()
        
        self.logger.info("Orchestration Agent shutdown complete")