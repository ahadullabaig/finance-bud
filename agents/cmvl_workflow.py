"""
CMVL Workflow Infrastructure for FinPilot VP-MAS

This module implements the core CMVL (Continuous Monitoring and Verification Loop) 
workflow infrastructure including:
- WorkflowOrchestrator for session management and agent handoff coordination
- EventClassifier for trigger categorization and priority assignment
- Core workflow coordination and state management

Requirements: 1.1, 1.2, 4.1, 5.1
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from enum import Enum

from .base_agent import BaseAgent
from .communication import AgentCommunicationFramework
from data_models.schemas import (
    TriggerEvent, WorkflowSession, PlanAdjustment, ClassifiedTrigger,
    LifeEventType, WorkflowState, UrgencyLevel, Priority, SeverityLevel,
    MarketEventType, AgentMessage, MessageType, ExecutionStatus
)


class EventClassifier:
    """
    Advanced event classifier for CMVL trigger categorization and priority assignment.
    
    Analyzes incoming trigger events and classifies them based on:
    - Event type and severity
    - User context and financial state
    - Historical patterns and impact assessment
    - Urgency and priority scoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger("finpilot.cmvl.event_classifier")
        
        # Classification rules and weights
        self.life_event_priorities = {
            LifeEventType.JOB_LOSS: 0.9,
            LifeEventType.MEDICAL_EMERGENCY: 0.95,
            LifeEventType.BUSINESS_DISRUPTION: 0.8,
            LifeEventType.INCOME_CHANGE: 0.6,
            LifeEventType.MAJOR_EXPENSE: 0.7,
            LifeEventType.FAMILY_CHANGE: 0.5,
            LifeEventType.CAREER_CHANGE: 0.4,
            LifeEventType.INHERITANCE: 0.3,
            LifeEventType.DIVORCE: 0.8,
            LifeEventType.RETIREMENT: 0.6
        }
        
        self.market_event_priorities = {
            MarketEventType.MARKET_CRASH: 0.9,
            MarketEventType.VOLATILITY_SPIKE: 0.7,
            MarketEventType.INTEREST_RATE_CHANGE: 0.6,
            MarketEventType.REGULATORY_CHANGE: 0.5,
            MarketEventType.SECTOR_ROTATION: 0.4,
            MarketEventType.MARKET_RECOVERY: 0.3
        }
        
        self.severity_multipliers = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.LOW: 0.4,
            SeverityLevel.INFO: 0.2
        }
        
        # Processing time estimates (in seconds)
        self.processing_time_estimates = {
            "life_event": {
                "immediate": 30,
                "high": 60,
                "medium": 180,
                "low": 300
            },
            "market_event": {
                "immediate": 45,
                "high": 90,
                "medium": 240,
                "low": 600
            }
        }
    
    def classify_trigger(self, trigger: TriggerEvent, user_context: Optional[Dict[str, Any]] = None) -> ClassifiedTrigger:
        """
        Classify a trigger event and assign priority and urgency levels.
        
        Args:
            trigger: The trigger event to classify
            user_context: Optional user context for personalized classification
            
        Returns:
            ClassifiedTrigger with classification, priority, and recommended actions
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Classifying trigger: {trigger.trigger_type} - {trigger.severity}")
            
            # Determine base classification
            classification = self._determine_classification(trigger)
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(trigger, user_context)
            
            # Determine urgency level
            urgency_level = self._determine_urgency_level(trigger, priority_score)
            
            # Generate recommended actions
            recommended_actions = self._generate_recommended_actions(trigger, classification, urgency_level)
            
            # Estimate processing time
            estimated_time = self._estimate_processing_time(trigger, urgency_level)
            
            # Check if immediate attention is required
            requires_immediate = self._requires_immediate_attention(trigger, priority_score, urgency_level)
            
            # Calculate classification confidence
            confidence = self._calculate_classification_confidence(trigger, user_context)
            
            classified_trigger = ClassifiedTrigger(
                trigger_event=trigger,
                classification=classification,
                priority_score=priority_score,
                urgency_level=urgency_level,
                recommended_actions=recommended_actions,
                estimated_processing_time=estimated_time,
                requires_immediate_attention=requires_immediate,
                classification_confidence=confidence,
                classified_by_agent="event_classifier",
                classified_at=datetime.utcnow()
            )
            
            classification_time = time.time() - start_time
            self.logger.info(f"Trigger classified in {classification_time:.3f}s: {classification} (priority: {priority_score:.2f})")
            
            return classified_trigger
            
        except Exception as e:
            self.logger.error(f"Failed to classify trigger: {str(e)}")
            raise
    
    def prioritize_multiple_events(self, triggers: List[TriggerEvent], user_context: Optional[Dict[str, Any]] = None) -> List[ClassifiedTrigger]:
        """
        Prioritize multiple concurrent trigger events.
        
        Args:
            triggers: List of trigger events to prioritize
            user_context: Optional user context for personalized prioritization
            
        Returns:
            List of ClassifiedTrigger objects sorted by priority
        """
        try:
            self.logger.info(f"Prioritizing {len(triggers)} concurrent triggers")
            
            # Classify all triggers
            classified_triggers = []
            for trigger in triggers:
                classified = self.classify_trigger(trigger, user_context)
                classified_triggers.append(classified)
            
            # Apply compound scenario adjustments
            if len(classified_triggers) > 1:
                classified_triggers = self._adjust_for_compound_scenario(classified_triggers)
            
            # Sort by priority score (descending) and urgency level
            classified_triggers.sort(
                key=lambda ct: (
                    ct.urgency_level == UrgencyLevel.IMMEDIATE,
                    ct.priority_score,
                    ct.trigger_event.severity == SeverityLevel.CRITICAL
                ),
                reverse=True
            )
            
            self.logger.info(f"Prioritized {len(classified_triggers)} triggers")
            return classified_triggers
            
        except Exception as e:
            self.logger.error(f"Failed to prioritize multiple events: {str(e)}")
            raise
    
    def determine_urgency(self, trigger: ClassifiedTrigger) -> UrgencyLevel:
        """
        Determine urgency level for a classified trigger.
        
        Args:
            trigger: Classified trigger event
            
        Returns:
            UrgencyLevel for the trigger
        """
        return trigger.urgency_level
    
    def _determine_classification(self, trigger: TriggerEvent) -> str:
        """Determine the classification category for a trigger"""
        if trigger.trigger_type == "life_event":
            return f"life_event_{trigger.event_type.value}"
        elif trigger.trigger_type == "market_event":
            return f"market_event_{trigger.event_type.value}"
        elif trigger.trigger_type == "regulatory_change":
            return "regulatory_change"
        elif trigger.trigger_type == "portfolio_threshold":
            return "portfolio_threshold"
        elif trigger.trigger_type == "emergency":
            return "emergency_event"
        else:
            return "general_trigger"
    
    def _calculate_priority_score(self, trigger: TriggerEvent, user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate priority score for a trigger event"""
        base_score = 0.5  # Default base score
        
        # Get base priority from event type
        if trigger.trigger_type == "life_event":
            # Try to map to LifeEventType if possible
            try:
                life_event_type = LifeEventType(trigger.event_type.value)
                base_score = self.life_event_priorities.get(life_event_type, 0.5)
            except (ValueError, AttributeError):
                base_score = 0.5
        elif trigger.trigger_type == "market_event":
            base_score = self.market_event_priorities.get(trigger.event_type, 0.5)
        
        # Apply severity multiplier
        severity_multiplier = self.severity_multipliers.get(trigger.severity, 0.6)
        priority_score = base_score * severity_multiplier
        
        # Apply impact and confidence adjustments
        impact_adjustment = trigger.impact_score * 0.3
        confidence_adjustment = trigger.confidence_score * 0.2
        
        # Combine all factors
        final_score = min(1.0, priority_score + impact_adjustment + confidence_adjustment)
        
        # Apply user context adjustments if available
        if user_context:
            final_score = self._apply_user_context_adjustments(final_score, trigger, user_context)
        
        return final_score
    
    def _determine_urgency_level(self, trigger: TriggerEvent, priority_score: float) -> UrgencyLevel:
        """Determine urgency level based on trigger characteristics and priority score"""
        # Critical severity always gets immediate attention
        if trigger.severity == SeverityLevel.CRITICAL:
            return UrgencyLevel.IMMEDIATE
        
        # High priority score with high severity
        if priority_score >= 0.8 and trigger.severity == SeverityLevel.HIGH:
            return UrgencyLevel.IMMEDIATE
        
        # High priority score
        if priority_score >= 0.7:
            return UrgencyLevel.HIGH
        
        # Medium priority score
        if priority_score >= 0.5:
            return UrgencyLevel.MEDIUM
        
        # Low priority
        return UrgencyLevel.LOW
    
    def _generate_recommended_actions(self, trigger: TriggerEvent, classification: str, urgency: UrgencyLevel) -> List[str]:
        """Generate recommended actions based on trigger classification and urgency"""
        actions = []
        
        # Base actions for all triggers
        actions.append("initiate_cmvl_workflow")
        actions.append("retrieve_current_financial_state")
        
        # Classification-specific actions
        if "life_event" in classification:
            actions.extend([
                "assess_income_impact",
                "evaluate_expense_adjustments",
                "review_emergency_fund_adequacy"
            ])
            
            if "job_loss" in classification:
                actions.extend([
                    "extend_emergency_fund_timeline",
                    "reduce_discretionary_spending",
                    "pause_non_essential_investments"
                ])
            elif "medical_emergency" in classification:
                actions.extend([
                    "prioritize_healthcare_funding",
                    "evaluate_insurance_coverage",
                    "assess_hsa_utilization"
                ])
            elif "business_disruption" in classification:
                actions.extend([
                    "model_irregular_income_scenarios",
                    "adjust_business_expense_projections",
                    "create_contingency_plans"
                ])
        
        elif "market_event" in classification:
            actions.extend([
                "assess_portfolio_exposure",
                "evaluate_rebalancing_opportunities",
                "review_risk_tolerance_alignment"
            ])
            
            if "market_crash" in classification:
                actions.extend([
                    "implement_defensive_positioning",
                    "evaluate_tax_loss_harvesting",
                    "assess_buying_opportunities"
                ])
            elif "volatility_spike" in classification:
                actions.extend([
                    "review_stop_loss_positions",
                    "assess_hedging_strategies",
                    "evaluate_position_sizing"
                ])
        
        # Urgency-specific actions
        if urgency == UrgencyLevel.IMMEDIATE:
            actions.insert(1, "notify_user_immediately")
            actions.insert(2, "escalate_to_human_advisor")
        elif urgency == UrgencyLevel.HIGH:
            actions.insert(1, "notify_user_within_hour")
        
        return actions
    
    def _estimate_processing_time(self, trigger: TriggerEvent, urgency: UrgencyLevel) -> int:
        """Estimate processing time for a trigger based on type and urgency"""
        trigger_category = "life_event" if trigger.trigger_type == "life_event" else "market_event"
        urgency_key = urgency.value
        
        base_time = self.processing_time_estimates.get(trigger_category, {}).get(urgency_key, 300)
        
        # Adjust based on trigger complexity
        complexity_multiplier = 1.0
        if trigger.impact_score > 0.8:
            complexity_multiplier = 1.5
        elif trigger.impact_score < 0.3:
            complexity_multiplier = 0.7
        
        return int(base_time * complexity_multiplier)
    
    def _requires_immediate_attention(self, trigger: TriggerEvent, priority_score: float, urgency: UrgencyLevel) -> bool:
        """Determine if trigger requires immediate attention"""
        return (
            urgency == UrgencyLevel.IMMEDIATE or
            trigger.severity == SeverityLevel.CRITICAL or
            (priority_score >= 0.9 and trigger.impact_score >= 0.8)
        )
    
    def _calculate_classification_confidence(self, trigger: TriggerEvent, user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in the classification"""
        base_confidence = trigger.confidence_score
        
        # Adjust based on data completeness
        if user_context:
            context_completeness = len(user_context) / 10.0  # Assume 10 fields for complete context
            base_confidence = min(1.0, base_confidence + (context_completeness * 0.1))
        
        # Adjust based on trigger data quality
        if trigger.source_data and len(trigger.source_data) > 3:
            base_confidence = min(1.0, base_confidence + 0.1)
        
        return base_confidence
    
    def _apply_user_context_adjustments(self, base_score: float, trigger: TriggerEvent, user_context: Dict[str, Any]) -> float:
        """Apply user-specific context adjustments to priority score"""
        adjusted_score = base_score
        
        # Adjust based on user's financial stability
        emergency_fund_months = user_context.get('emergency_fund_months', 3)
        if emergency_fund_months < 3 and trigger.trigger_type == "life_event":
            adjusted_score = min(1.0, adjusted_score + 0.2)
        
        # Adjust based on user's risk tolerance
        risk_tolerance = user_context.get('risk_tolerance', 'medium')
        if risk_tolerance == 'conservative' and trigger.trigger_type == "market_event":
            adjusted_score = min(1.0, adjusted_score + 0.1)
        
        # Adjust based on user's age and time horizon
        age = user_context.get('age', 40)
        if age > 55 and trigger.trigger_type in ["market_event", "life_event"]:
            adjusted_score = min(1.0, adjusted_score + 0.1)
        
        return adjusted_score
    
    def _adjust_for_compound_scenario(self, classified_triggers: List[ClassifiedTrigger]) -> List[ClassifiedTrigger]:
        """Adjust priority scores for compound scenarios with multiple triggers"""
        if len(classified_triggers) <= 1:
            return classified_triggers
        
        # Increase priority for all triggers in compound scenario
        compound_multiplier = 1.2
        
        for trigger in classified_triggers:
            # Increase priority score but cap at 1.0
            trigger.priority_score = min(1.0, trigger.priority_score * compound_multiplier)
            
            # Upgrade urgency if multiple high-priority triggers
            high_priority_count = sum(1 for t in classified_triggers if t.priority_score >= 0.7)
            if high_priority_count >= 2 and trigger.urgency_level != UrgencyLevel.IMMEDIATE:
                if trigger.urgency_level == UrgencyLevel.HIGH:
                    trigger.urgency_level = UrgencyLevel.IMMEDIATE
                elif trigger.urgency_level == UrgencyLevel.MEDIUM:
                    trigger.urgency_level = UrgencyLevel.HIGH
            
            # Add compound scenario action
            if "handle_compound_scenario" not in trigger.recommended_actions:
                trigger.recommended_actions.insert(0, "handle_compound_scenario")
        
        return classified_triggers


class WorkflowOrchestrator:
    """
    CMVL Workflow Orchestrator for session management and agent handoff coordination.
    
    Manages the complete CMVL workflow lifecycle including:
    - Session creation and tracking
    - Agent handoff coordination
    - State management and timeout handling
    - Error recovery and retry logic
    - Performance monitoring and metrics
    """
    
    def __init__(self, communication_framework: AgentCommunicationFramework):
        self.communication_framework = communication_framework
        self.logger = logging.getLogger("finpilot.cmvl.workflow_orchestrator")
        
        # Session management
        self.active_sessions: Dict[str, WorkflowSession] = {}
        self.session_timeouts: Dict[str, datetime] = {}
        
        # Agent coordination
        self.agent_sequence = [
            "orchestration_agent",
            "information_retrieval_agent", 
            "planning_agent",
            "verification_agent",
            "execution_agent"
        ]
        
        # Configuration
        self.default_timeout_minutes = 30
        self.max_retry_attempts = 3
        self.agent_timeout_seconds = 300  # 5 minutes per agent
        
        # Performance tracking
        self.workflow_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_completion_time": 0.0,
            "agent_handoff_failures": 0,
            "timeout_failures": 0
        }
    
    def initiate_workflow(self, trigger: TriggerEvent, user_id: str, user_context: Optional[Dict[str, Any]] = None) -> WorkflowSession:
        """
        Initiate a new CMVL workflow session.
        
        Args:
            trigger: Trigger event that initiated the workflow
            user_id: User identifier
            user_context: Optional user context for personalized processing
            
        Returns:
            WorkflowSession object tracking the workflow
        """
        try:
            self.logger.info(f"Initiating CMVL workflow for trigger: {trigger.trigger_id}")
            
            # Create workflow session
            session = WorkflowSession(
                session_id=str(uuid4()),
                user_id=user_id,
                trigger_event=trigger,
                current_state=WorkflowState.INITIATED,
                current_agent=None,
                priority=self._determine_workflow_priority(trigger),
                urgency=self._determine_workflow_urgency(trigger),
                timeout_at=datetime.utcnow() + timedelta(minutes=self.default_timeout_minutes)
            )
            
            # Store session
            self.active_sessions[session.session_id] = session
            self.session_timeouts[session.session_id] = session.timeout_at
            
            # Update metrics
            self.workflow_metrics["total_workflows"] += 1
            
            # Start workflow processing
            asyncio.create_task(self._process_workflow(session, user_context))
            
            self.logger.info(f"CMVL workflow initiated: {session.session_id}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to initiate workflow: {str(e)}")
            raise
    
    def track_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Track progress of a workflow session.
        
        Args:
            session_id: Session identifier to track
            
        Returns:
            Dictionary with current workflow status and progress
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        # Calculate progress percentage
        state_progress = {
            WorkflowState.INITIATED: 10,
            WorkflowState.CLASSIFYING_EVENT: 20,
            WorkflowState.RETRIEVING_INFO: 30,
            WorkflowState.GENERATING_ADJUSTMENTS: 50,
            WorkflowState.VALIDATING_PLAN: 70,
            WorkflowState.AWAITING_APPROVAL: 80,
            WorkflowState.EXECUTING_CHANGES: 90,
            WorkflowState.COMPLETED: 100,
            WorkflowState.FAILED: 0,
            WorkflowState.CANCELLED: 0
        }
        
        progress_percentage = state_progress.get(session.current_state, 0)
        
        # Calculate elapsed time
        elapsed_time = (datetime.utcnow() - session.created_at).total_seconds()
        
        return {
            "session_id": session_id,
            "current_state": session.current_state.value,
            "current_agent": session.current_agent,
            "progress_percentage": progress_percentage,
            "elapsed_time_seconds": elapsed_time,
            "error_count": session.error_count,
            "retry_count": session.retry_count,
            "priority": session.priority.value,
            "urgency": session.urgency.value,
            "timeout_at": session.timeout_at.isoformat() if session.timeout_at else None
        }
    
    def handle_agent_failure(self, session_id: str, agent_id: str, error: str) -> bool:
        """
        Handle agent failure with retry and recovery logic.
        
        Args:
            session_id: Session identifier
            agent_id: Failed agent identifier
            error: Error message
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                self.logger.error(f"Session not found for failure handling: {session_id}")
                return False
            
            self.logger.warning(f"Agent failure in session {session_id}: {agent_id} - {error}")
            
            # Update session error tracking
            session.error_count += 1
            session.updated_at = datetime.utcnow()
            
            # Record the failure
            failure_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "error": error,
                "retry_attempt": session.retry_count
            }
            
            # Check if we should retry
            if session.retry_count < self.max_retry_attempts:
                session.retry_count += 1
                self.logger.info(f"Retrying workflow session {session_id} (attempt {session.retry_count})")
                
                # Reset to previous state and retry
                asyncio.create_task(self._retry_workflow_step(session, agent_id))
                return True
            else:
                # Max retries exceeded, mark as failed
                self.logger.error(f"Max retries exceeded for session {session_id}")
                session.current_state = WorkflowState.FAILED
                self.workflow_metrics["failed_workflows"] += 1
                self.workflow_metrics["agent_handoff_failures"] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error handling agent failure: {str(e)}")
            return False
    
    def coordinate_handoffs(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Coordinate agent handoffs for a workflow session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with handoff coordination details
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        current_agent_index = 0
        if session.current_agent:
            try:
                current_agent_index = self.agent_sequence.index(session.current_agent)
            except ValueError:
                pass
        
        # Determine next agent
        next_agent_index = current_agent_index + 1
        if next_agent_index < len(self.agent_sequence):
            next_agent = self.agent_sequence[next_agent_index]
        else:
            next_agent = None
        
        # Record handoff
        handoff_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": session.current_agent,
            "to_agent": next_agent,
            "session_state": session.current_state.value
        }
        
        session.agent_handoffs.append(handoff_record)
        
        return {
            "session_id": session_id,
            "current_agent": session.current_agent,
            "next_agent": next_agent,
            "handoff_record": handoff_record,
            "total_handoffs": len(session.agent_handoffs)
        }
    
    async def _process_workflow(self, session: WorkflowSession, user_context: Optional[Dict[str, Any]]):
        """Process the complete CMVL workflow"""
        try:
            start_time = time.time()
            
            # Step 1: Classify the trigger event
            await self._transition_state(session, WorkflowState.CLASSIFYING_EVENT, "orchestration_agent")
            classified_trigger = await self._classify_trigger_event(session, user_context)
            
            # Step 2: Retrieve information
            await self._transition_state(session, WorkflowState.RETRIEVING_INFO, "information_retrieval_agent")
            context_data = await self._retrieve_context_information(session, classified_trigger)
            
            # Step 3: Generate plan adjustments
            await self._transition_state(session, WorkflowState.GENERATING_ADJUSTMENTS, "planning_agent")
            plan_adjustment = await self._generate_plan_adjustments(session, classified_trigger, context_data)
            
            # Step 4: Validate the plan
            await self._transition_state(session, WorkflowState.VALIDATING_PLAN, "verification_agent")
            validation_result = await self._validate_plan_adjustments(session, plan_adjustment)
            
            # Step 5: Await user approval if required
            if plan_adjustment.requires_user_approval:
                await self._transition_state(session, WorkflowState.AWAITING_APPROVAL, None)
                # In a real implementation, this would wait for user input
                # For now, we'll simulate approval
                await asyncio.sleep(1)
                plan_adjustment.approval_status = "approved"
                plan_adjustment.approved_by = session.user_id
                plan_adjustment.approved_at = datetime.utcnow()
            
            # Step 6: Execute the changes
            await self._transition_state(session, WorkflowState.EXECUTING_CHANGES, "execution_agent")
            execution_result = await self._execute_plan_changes(session, plan_adjustment)
            
            # Step 7: Complete the workflow
            await self._transition_state(session, WorkflowState.COMPLETED, None)
            
            # Update metrics
            total_time = time.time() - start_time
            session.total_processing_time = total_time
            self.workflow_metrics["successful_workflows"] += 1
            
            # Update average completion time
            total_successful = self.workflow_metrics["successful_workflows"]
            current_avg = self.workflow_metrics["average_completion_time"]
            self.workflow_metrics["average_completion_time"] = (
                (current_avg * (total_successful - 1) + total_time) / total_successful
            )
            
            self.logger.info(f"CMVL workflow completed successfully: {session.session_id} in {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Workflow processing failed: {str(e)}")
            session.current_state = WorkflowState.FAILED
            session.error_count += 1
            self.workflow_metrics["failed_workflows"] += 1
    
    async def _transition_state(self, session: WorkflowSession, new_state: WorkflowState, agent_id: Optional[str]):
        """Transition workflow session to a new state"""
        old_state = session.current_state
        old_agent = session.current_agent
        
        session.current_state = new_state
        session.current_agent = agent_id
        session.updated_at = datetime.utcnow()
        
        # Record state transition
        transition_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "from_agent": old_agent,
            "to_agent": agent_id
        }
        
        session.state_transitions.append(transition_record)
        
        self.logger.info(f"Session {session.session_id} transitioned: {old_state.value} -> {new_state.value}")
    
    async def _classify_trigger_event(self, session: WorkflowSession, user_context: Optional[Dict[str, Any]]) -> ClassifiedTrigger:
        """Classify the trigger event using EventClassifier"""
        classifier = EventClassifier()
        classified_trigger = classifier.classify_trigger(session.trigger_event, user_context)
        
        # Store classification results in session
        session.proposed_adjustments = {
            "classification": classified_trigger.classification,
            "priority_score": classified_trigger.priority_score,
            "urgency_level": classified_trigger.urgency_level.value,
            "recommended_actions": classified_trigger.recommended_actions
        }
        
        return classified_trigger
    
    async def _retrieve_context_information(self, session: WorkflowSession, classified_trigger: ClassifiedTrigger) -> Dict[str, Any]:
        """Retrieve context information from information retrieval agent"""
        # Simulate information retrieval
        await asyncio.sleep(0.5)
        
        context_data = {
            "current_financial_state": {
                "total_assets": 150000,
                "total_liabilities": 50000,
                "monthly_income": 8000,
                "monthly_expenses": 6000,
                "emergency_fund": 18000
            },
            "market_conditions": {
                "volatility_index": 0.25,
                "interest_rates": {"federal_funds": 0.05, "10_year_treasury": 0.045}
            },
            "user_preferences": {
                "risk_tolerance": "moderate",
                "investment_horizon": 120  # 10 years
            }
        }
        
        return context_data
    
    async def _generate_plan_adjustments(self, session: WorkflowSession, classified_trigger: ClassifiedTrigger, context_data: Dict[str, Any]) -> PlanAdjustment:
        """Generate plan adjustments using planning agent"""
        # Simulate plan adjustment generation
        await asyncio.sleep(1.0)
        
        # Create a sample plan adjustment
        from data_models.schemas import PlanChange, ImpactAnalysis
        
        plan_changes = [
            PlanChange(
                change_type="modify",
                target_component="emergency_fund",
                original_value={"amount": 18000},
                new_value={"amount": 24000},
                rationale="Increase emergency fund due to job loss risk",
                impact_assessment={"timeline_months": 6, "risk_reduction": 0.3},
                confidence_score=0.85,
                created_by_agent="planning_agent"
            )
        ]
        
        impact_analysis = ImpactAnalysis(
            financial_impact={"net_worth_change": -6000, "monthly_savings_change": -1000},
            risk_impact={"overall_risk_reduction": 0.2},
            timeline_impact={"emergency_preparedness": 6},
            goal_impact={"retirement_delay_months": 3},
            overall_score=0.7,
            confidence_level=0.8
        )
        
        plan_adjustment = PlanAdjustment(
            workflow_session_id=session.session_id,
            trigger_event_id=session.trigger_event.trigger_id,
            user_id=session.user_id,
            original_plan_id="original_plan_123",
            adjustment_type="emergency_response",
            adjustment_scope="partial",
            proposed_changes=plan_changes,
            impact_analysis=impact_analysis,
            adjustment_rationale="Responding to potential job loss by strengthening emergency fund",
            confidence_score=0.82,
            risk_assessment={"implementation_risk": "low", "market_risk": "medium"},
            created_by_agent="planning_agent"
        )
        
        session.proposed_adjustments = plan_adjustment.dict()
        return plan_adjustment
    
    async def _validate_plan_adjustments(self, session: WorkflowSession, plan_adjustment: PlanAdjustment) -> Dict[str, Any]:
        """Validate plan adjustments using verification agent"""
        # Simulate validation
        await asyncio.sleep(0.8)
        
        validation_result = {
            "validation_status": "approved",
            "constraint_violations": [],
            "risk_assessment": "acceptable",
            "compliance_status": "compliant",
            "confidence_score": 0.88
        }
        
        session.validation_results = validation_result
        return validation_result
    
    async def _execute_plan_changes(self, session: WorkflowSession, plan_adjustment: PlanAdjustment) -> Dict[str, Any]:
        """Execute plan changes using execution agent"""
        # Simulate execution
        await asyncio.sleep(0.6)
        
        execution_result = {
            "execution_status": "completed",
            "changes_implemented": len(plan_adjustment.proposed_changes),
            "execution_time": 0.6,
            "success_rate": 1.0
        }
        
        session.execution_results = execution_result
        plan_adjustment.execution_status = ExecutionStatus.COMPLETED
        
        return execution_result
    
    async def _retry_workflow_step(self, session: WorkflowSession, failed_agent: str):
        """Retry a failed workflow step"""
        # Reset to appropriate state based on failed agent
        if failed_agent == "information_retrieval_agent":
            session.current_state = WorkflowState.RETRIEVING_INFO
        elif failed_agent == "planning_agent":
            session.current_state = WorkflowState.GENERATING_ADJUSTMENTS
        elif failed_agent == "verification_agent":
            session.current_state = WorkflowState.VALIDATING_PLAN
        elif failed_agent == "execution_agent":
            session.current_state = WorkflowState.EXECUTING_CHANGES
        
        # Continue processing from the retry point
        # This would involve re-calling the appropriate processing method
        self.logger.info(f"Retrying workflow step for session {session.session_id}")
    
    def _determine_workflow_priority(self, trigger: TriggerEvent) -> Priority:
        """Determine workflow priority based on trigger characteristics"""
        if trigger.severity == SeverityLevel.CRITICAL:
            return Priority.CRITICAL
        elif trigger.severity == SeverityLevel.HIGH and trigger.impact_score >= 0.8:
            return Priority.HIGH
        elif trigger.severity == SeverityLevel.HIGH or trigger.impact_score >= 0.6:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _determine_workflow_urgency(self, trigger: TriggerEvent) -> UrgencyLevel:
        """Determine workflow urgency based on trigger characteristics"""
        if trigger.severity == SeverityLevel.CRITICAL:
            return UrgencyLevel.IMMEDIATE
        elif trigger.severity == SeverityLevel.HIGH and trigger.impact_score >= 0.9:
            return UrgencyLevel.IMMEDIATE
        elif trigger.severity == SeverityLevel.HIGH:
            return UrgencyLevel.HIGH
        elif trigger.impact_score >= 0.7:
            return UrgencyLevel.MEDIUM
        else:
            return UrgencyLevel.LOW