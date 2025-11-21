"""
Mock Agent Interfaces for Independent Development

Provides realistic mock implementations of all VP-MAS agents to enable
parallel development without dependencies on other team members' code.

Requirements: 9.4, 9.5, 11.2
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base_agent import BaseAgent
from data_models.schemas import (
    AgentMessage, MessageType, Priority,
    EnhancedPlanRequest, PlanStep, VerificationReport, VerificationStatus,
    MarketData, TriggerEvent, MarketEventType, SeverityLevel,
    FinancialState, SearchPath, ReasoningTrace, DecisionPoint,
    ExecutionLog, ExecutionStatus
)


class MockOrchestrationAgent(BaseAgent):
    """
    Mock Orchestration Agent for testing and independent development.
    
    Simulates workflow coordination, trigger handling, and agent management
    with realistic response times and decision patterns.
    """
    
    def __init__(self, agent_id: str = "mock_orchestration_001"):
        super().__init__(agent_id, "orchestration")
        self.active_sessions: Dict[str, Dict] = {}
        self.workflow_templates = self._load_workflow_templates()
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages and coordinate workflows"""
        payload = message.payload
        
        if message.message_type == MessageType.REQUEST:
            if "user_goal" in payload:
                return await self._handle_planning_request(message)
            elif "trigger_event" in payload:
                return await self._handle_trigger_event(message)
            elif "health_check" in payload:
                return await self._handle_health_check(message)
        
        return None
    
    async def _handle_planning_request(self, message: AgentMessage) -> AgentMessage:
        """Handle user planning requests"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        user_goal = message.payload["user_goal"]
        session_id = message.session_id
        
        # Create session tracking
        self.active_sessions[session_id] = {
            "goal": user_goal,
            "status": "processing",
            "start_time": datetime.utcnow(),
            "steps_completed": 0,
            "total_steps": 5
        }
        
        # Simulate workflow coordination
        workflow_plan = {
            "workflow_id": str(uuid4()),
            "session_id": session_id,
            "steps": [
                {"agent": "information_retrieval", "action": "fetch_market_data"},
                {"agent": "planning", "action": "generate_plan"},
                {"agent": "verification", "action": "verify_plan"},
                {"agent": "execution", "action": "execute_plan"}
            ],
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=2)).isoformat(),
            "priority": "high" if "emergency" in user_goal.lower() else "medium"
        }
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "status": "workflow_initiated",
                "workflow_plan": workflow_plan,
                "message": f"Processing goal: {user_goal}"
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _handle_trigger_event(self, message: AgentMessage) -> AgentMessage:
        """Handle CMVL trigger events"""
        await asyncio.sleep(0.05)
        
        trigger = message.payload["trigger_event"]
        severity = trigger.get("severity", "medium")
        
        # Simulate CMVL activation
        cmvl_response = {
            "cmvl_activated": True,
            "trigger_id": trigger.get("trigger_id", str(uuid4())),
            "response_time": 0.05,
            "actions_initiated": [
                "market_data_refresh",
                "plan_re_evaluation",
                "risk_assessment_update"
            ],
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=1)).isoformat(),
            "priority_escalation": severity == "critical"
        }
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload=cmvl_response,
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _handle_health_check(self, message: AgentMessage) -> AgentMessage:
        """Handle health check requests"""
        health_status = self.get_health_status()
        health_status.update({
            "active_sessions": len(self.active_sessions),
            "workflow_templates": len(self.workflow_templates),
            "last_trigger_processed": datetime.utcnow().isoformat()
        })
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={"health_status": health_status},
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    def _load_workflow_templates(self) -> Dict[str, Any]:
        """Load predefined workflow templates"""
        return {
            "retirement_planning": {
                "steps": ["market_analysis", "risk_assessment", "plan_generation", "verification"],
                "estimated_time": 120
            },
            "emergency_response": {
                "steps": ["immediate_assessment", "constraint_relaxation", "rapid_planning"],
                "estimated_time": 30
            },
            "investment_optimization": {
                "steps": ["portfolio_analysis", "market_research", "optimization", "verification"],
                "estimated_time": 90
            }
        }


class MockPlanningAgent(BaseAgent):
    """
    Mock Planning Agent with realistic search path generation and ToS simulation.
    
    Generates multiple planning strategies with detailed reasoning traces
    for ReasonGraph visualization testing.
    """
    
    def __init__(self, agent_id: str = "mock_planning_001"):
        super().__init__(agent_id, "planning")
        self.search_strategies = ["conservative", "balanced", "aggressive", "tax_optimized", "growth_focused"]
        self.heuristic_weights = {
            "risk_adjusted_return": 0.3,
            "constraint_satisfaction": 0.25,
            "tax_efficiency": 0.2,
            "liquidity": 0.15,
            "diversification": 0.1
        }
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process planning requests and generate comprehensive plans"""
        if message.message_type == MessageType.REQUEST and "planning_request" in message.payload:
            return await self._generate_financial_plan(message)
        return None
    
    async def _generate_financial_plan(self, message: AgentMessage) -> AgentMessage:
        """Generate a comprehensive financial plan with multiple strategies"""
        # Simulate ToS algorithm processing time
        await asyncio.sleep(0.3)
        
        request_data = message.payload["planning_request"]
        user_goal = request_data.get("user_goal", "Generic financial goal")
        time_horizon = request_data.get("time_horizon", 60)
        
        # Generate multiple search paths
        search_paths = []
        for i, strategy in enumerate(self.search_strategies):
            path = await self._generate_search_path(strategy, user_goal, time_horizon, i)
            search_paths.append(path)
        
        # Select best path based on combined scoring
        best_path = max(search_paths, key=lambda p: p["combined_score"])
        
        # Generate detailed plan steps
        plan_steps = await self._generate_plan_steps(best_path, time_horizon)
        
        # Create reasoning trace
        reasoning_trace = await self._create_reasoning_trace(search_paths, best_path, message)
        
        response_payload = {
            "plan_generated": True,
            "selected_strategy": best_path["strategy"],
            "plan_steps": [step.dict() for step in plan_steps],
            "search_paths": search_paths,
            "reasoning_trace": reasoning_trace.dict(),
            "confidence_score": best_path["combined_score"],
            "alternative_strategies": len(search_paths) - 1,
            "processing_time": 0.3
        }
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload=response_payload,
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _generate_search_path(self, strategy: str, goal: str, horizon: int, index: int) -> Dict:
        """Generate a realistic search path with heuristic scoring"""
        # Simulate different strategy characteristics
        strategy_profiles = {
            "conservative": {"risk": 0.2, "return": 0.06, "tax_efficiency": 0.8},
            "balanced": {"risk": 0.4, "return": 0.08, "tax_efficiency": 0.7},
            "aggressive": {"risk": 0.7, "return": 0.12, "tax_efficiency": 0.6},
            "tax_optimized": {"risk": 0.3, "return": 0.07, "tax_efficiency": 0.9},
            "growth_focused": {"risk": 0.6, "return": 0.11, "tax_efficiency": 0.5}
        }
        
        profile = strategy_profiles.get(strategy, strategy_profiles["balanced"])
        
        # Calculate heuristic scores
        risk_score = 1.0 - profile["risk"]  # Lower risk = higher score
        return_score = min(profile["return"] / 0.15, 1.0)  # Normalize to max 15% return
        tax_score = profile["tax_efficiency"]
        
        combined_score = (
            risk_score * self.heuristic_weights["risk_adjusted_return"] +
            return_score * self.heuristic_weights["risk_adjusted_return"] +
            tax_score * self.heuristic_weights["tax_efficiency"] +
            random.uniform(0.6, 1.0) * self.heuristic_weights["constraint_satisfaction"] +
            random.uniform(0.7, 1.0) * self.heuristic_weights["liquidity"] +
            random.uniform(0.5, 0.9) * self.heuristic_weights["diversification"]
        )
        
        return {
            "path_id": str(uuid4()),
            "strategy": strategy,
            "sequence_steps": [
                {"step": 1, "action": "assess_current_state", "duration": 0.1},
                {"step": 2, "action": "analyze_constraints", "duration": 0.05},
                {"step": 3, "action": "generate_allocations", "duration": 0.1},
                {"step": 4, "action": "optimize_tax_efficiency", "duration": 0.05}
            ],
            "risk_score": profile["risk"],
            "expected_return": profile["return"],
            "tax_efficiency": profile["tax_efficiency"],
            "combined_score": combined_score,
            "exploration_time": 0.3 + index * 0.1,
            "constraint_violations": random.randint(0, 2),
            "status": "explored" if combined_score > 0.6 else "pruned"
        }
    
    async def _generate_plan_steps(self, best_path: Dict, horizon: int) -> List[PlanStep]:
        """Generate detailed plan steps based on selected strategy"""
        steps = []
        strategy = best_path["strategy"]
        
        # Generate steps based on strategy
        if strategy == "conservative":
            steps.extend([
                PlanStep(
                    sequence_number=1,
                    action_type="emergency_fund",
                    description="Build emergency fund to 6 months expenses",
                    amount=Decimal("30000"),
                    target_date=datetime.utcnow() + timedelta(days=90),
                    rationale="Conservative approach prioritizes safety",
                    confidence_score=0.9,
                    risk_level="low"
                ),
                PlanStep(
                    sequence_number=2,
                    action_type="bond_investment",
                    description="Invest in government bonds",
                    amount=Decimal("50000"),
                    target_date=datetime.utcnow() + timedelta(days=180),
                    rationale="Low-risk fixed income for stability",
                    confidence_score=0.85,
                    risk_level="low"
                )
            ])
        elif strategy == "aggressive":
            steps.extend([
                PlanStep(
                    sequence_number=1,
                    action_type="equity_investment",
                    description="Invest in growth stocks",
                    amount=Decimal("75000"),
                    target_date=datetime.utcnow() + timedelta(days=30),
                    rationale="High growth potential for long-term goals",
                    confidence_score=0.7,
                    risk_level="high"
                ),
                PlanStep(
                    sequence_number=2,
                    action_type="options_strategy",
                    description="Implement covered call strategy",
                    amount=Decimal("25000"),
                    target_date=datetime.utcnow() + timedelta(days=60),
                    rationale="Generate additional income from holdings",
                    confidence_score=0.6,
                    risk_level="high"
                )
            ])
        else:  # balanced and other strategies
            steps.extend([
                PlanStep(
                    sequence_number=1,
                    action_type="diversified_portfolio",
                    description="Create balanced portfolio (60/40 stocks/bonds)",
                    amount=Decimal("60000"),
                    target_date=datetime.utcnow() + timedelta(days=60),
                    rationale="Balanced approach for moderate risk tolerance",
                    confidence_score=0.8,
                    risk_level="medium"
                ),
                PlanStep(
                    sequence_number=2,
                    action_type="tax_advantaged_account",
                    description="Maximize 401(k) contributions",
                    amount=Decimal("22500"),
                    target_date=datetime.utcnow() + timedelta(days=365),
                    rationale="Tax-deferred growth for retirement",
                    confidence_score=0.9,
                    risk_level="low"
                )
            ])
        
        return steps
    
    async def _create_reasoning_trace(self, search_paths: List[Dict], best_path: Dict, message: AgentMessage) -> ReasoningTrace:
        """Create detailed reasoning trace for transparency"""
        decision_points = [
            DecisionPoint(
                decision_type="strategy_selection",
                options_considered=[{"strategy": path["strategy"], "score": path["combined_score"]} for path in search_paths],
                chosen_option={"strategy": best_path["strategy"], "score": best_path["combined_score"]},
                rationale=f"Selected {best_path['strategy']} strategy based on highest combined heuristic score",
                confidence_score=best_path["combined_score"]
            ),
            DecisionPoint(
                decision_type="risk_assessment",
                options_considered=[{"level": "low"}, {"level": "medium"}, {"level": "high"}],
                chosen_option={"level": "medium" if best_path["risk_score"] < 0.5 else "high"},
                rationale="Risk level determined by strategy profile and user tolerance",
                confidence_score=0.85
            )
        ]
        
        return ReasoningTrace(
            session_id=message.session_id,
            agent_id=self.agent_id,
            operation_type="financial_planning",
            final_decision=f"Implement {best_path['strategy']} strategy",
            decision_rationale=f"Strategy selected based on combined heuristic score of {best_path['combined_score']:.3f}",
            confidence_score=best_path["combined_score"],
            performance_metrics=self.get_performance_metrics(),
            correlation_id=message.correlation_id,
            decision_points=decision_points
        )


class MockInformationRetrievalAgent(BaseAgent):
    """
    Mock Information Retrieval Agent with realistic market data simulation.
    
    Provides market data, trigger detection, and scenario simulation
    for testing market-driven planning updates.
    """
    
    def __init__(self, agent_id: str = "mock_ira_001"):
        super().__init__(agent_id, "information_retrieval")
        self.market_scenarios = ["normal", "volatile", "crash", "recovery", "bull_market"]
        self.current_scenario = "normal"
        self.data_sources = ["barchart", "alpha_vantage", "massive_api"]
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process market data requests and trigger detection"""
        payload = message.payload
        
        if message.message_type == MessageType.REQUEST:
            if "market_data_request" in payload:
                return await self._fetch_market_data(message)
            elif "trigger_detection" in payload:
                return await self._detect_triggers(message)
            elif "scenario_simulation" in payload:
                return await self._simulate_scenario(message)
        
        return None
    
    async def _fetch_market_data(self, message: AgentMessage) -> AgentMessage:
        """Fetch comprehensive market data"""
        await asyncio.sleep(0.2)  # Simulate API call time
        
        # Generate realistic market data based on current scenario
        scenario_data = self._get_scenario_data(self.current_scenario)
        
        market_data = MarketData(
            source=random.choice(self.data_sources),
            market_volatility=scenario_data["volatility"],
            interest_rates={
                "federal_funds": scenario_data["fed_rate"],
                "10_year_treasury": scenario_data["treasury_10y"],
                "30_year_mortgage": scenario_data["mortgage_30y"]
            },
            sector_trends={
                "technology": scenario_data["tech_trend"],
                "healthcare": scenario_data["health_trend"],
                "financial": scenario_data["finance_trend"],
                "energy": scenario_data["energy_trend"]
            },
            economic_sentiment=scenario_data["sentiment"],
            collection_method="api_aggregation",
            refresh_frequency=300
        )
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "market_data": market_data.dict(),
                "data_quality_score": random.uniform(0.85, 0.98),
                "collection_time": 0.2,
                "scenario": self.current_scenario
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _detect_triggers(self, message: AgentMessage) -> AgentMessage:
        """Detect market triggers based on current conditions"""
        await asyncio.sleep(0.1)
        
        triggers = []
        scenario_data = self._get_scenario_data(self.current_scenario)
        
        # Generate triggers based on scenario
        if scenario_data["volatility"] > 0.3:
            triggers.append(TriggerEvent(
                trigger_type="market_event",
                event_type=MarketEventType.VOLATILITY_SPIKE,
                severity=SeverityLevel.HIGH if scenario_data["volatility"] > 0.5 else SeverityLevel.MEDIUM,
                description=f"Market volatility increased to {scenario_data['volatility']:.1%}",
                source_data={"volatility": scenario_data["volatility"]},
                impact_score=min(scenario_data["volatility"], 1.0),
                confidence_score=0.9,
                detector_agent_id=self.agent_id,
                correlation_id=message.correlation_id
            ))
        
        if scenario_data["sentiment"] < -0.5:
            triggers.append(TriggerEvent(
                trigger_type="sentiment_event",
                event_type=MarketEventType.MARKET_CRASH,
                severity=SeverityLevel.CRITICAL,
                description="Severe negative economic sentiment detected",
                source_data={"sentiment": scenario_data["sentiment"]},
                impact_score=abs(scenario_data["sentiment"]),
                confidence_score=0.85,
                detector_agent_id=self.agent_id,
                correlation_id=message.correlation_id
            ))
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "triggers_detected": len(triggers),
                "triggers": [trigger.dict() for trigger in triggers],
                "monitoring_active": True,
                "next_check": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _simulate_scenario(self, message: AgentMessage) -> AgentMessage:
        """Simulate different market scenarios for testing"""
        scenario = message.payload.get("scenario", "normal")
        self.current_scenario = scenario
        
        scenario_data = self._get_scenario_data(scenario)
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "scenario_activated": scenario,
                "scenario_data": scenario_data,
                "simulation_active": True,
                "estimated_duration": "5 minutes"
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    def _get_scenario_data(self, scenario: str) -> Dict[str, float]:
        """Get market data for different scenarios"""
        scenarios = {
            "normal": {
                "volatility": 0.15,
                "fed_rate": 5.25,
                "treasury_10y": 4.5,
                "mortgage_30y": 7.2,
                "tech_trend": 0.08,
                "health_trend": 0.05,
                "finance_trend": 0.03,
                "energy_trend": -0.02,
                "sentiment": 0.1
            },
            "volatile": {
                "volatility": 0.35,
                "fed_rate": 5.25,
                "treasury_10y": 4.8,
                "mortgage_30y": 7.5,
                "tech_trend": -0.15,
                "health_trend": 0.02,
                "finance_trend": -0.08,
                "energy_trend": 0.12,
                "sentiment": -0.2
            },
            "crash": {
                "volatility": 0.65,
                "fed_rate": 5.25,
                "treasury_10y": 3.8,
                "mortgage_30y": 8.0,
                "tech_trend": -0.35,
                "health_trend": -0.15,
                "finance_trend": -0.45,
                "energy_trend": -0.25,
                "sentiment": -0.8
            },
            "recovery": {
                "volatility": 0.25,
                "fed_rate": 4.75,
                "treasury_10y": 4.2,
                "mortgage_30y": 6.8,
                "tech_trend": 0.25,
                "health_trend": 0.15,
                "finance_trend": 0.18,
                "energy_trend": 0.08,
                "sentiment": 0.4
            },
            "bull_market": {
                "volatility": 0.12,
                "fed_rate": 4.0,
                "treasury_10y": 3.8,
                "mortgage_30y": 6.2,
                "tech_trend": 0.35,
                "health_trend": 0.22,
                "finance_trend": 0.28,
                "energy_trend": 0.15,
                "sentiment": 0.7
            }
        }
        
        return scenarios.get(scenario, scenarios["normal"])


class MockVerificationAgent(BaseAgent):
    """
    Mock Verification Agent with comprehensive constraint checking and CMVL simulation.
    
    Provides realistic verification results, compliance checking,
    and continuous monitoring capabilities.
    """
    
    def __init__(self, agent_id: str = "mock_verification_001"):
        super().__init__(agent_id, "verification")
        self.constraint_rules = self._load_constraint_rules()
        self.cmvl_active = False
        self.verification_history: List[Dict] = []
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process verification requests and CMVL operations"""
        payload = message.payload
        
        if message.message_type == MessageType.REQUEST:
            if "verification_request" in payload:
                return await self._verify_plan(message)
            elif "cmvl_trigger" in payload:
                return await self._handle_cmvl_trigger(message)
            elif "constraint_check" in payload:
                return await self._check_constraints(message)
        
        return None
    
    async def _verify_plan(self, message: AgentMessage) -> AgentMessage:
        """Verify a financial plan against all constraints"""
        await asyncio.sleep(0.15)  # Simulate verification time
        
        plan_data = message.payload["verification_request"]
        plan_steps = plan_data.get("plan_steps", [])
        
        # Perform comprehensive verification
        verification_results = []
        total_violations = 0
        
        for step in plan_steps:
            step_result = await self._verify_plan_step(step)
            verification_results.append(step_result)
            if step_result["violations"]:
                total_violations += len(step_result["violations"])
        
        # Determine overall status
        if total_violations == 0:
            status = VerificationStatus.APPROVED
            rationale = "All constraints satisfied, plan approved for execution"
        elif total_violations <= 2:
            status = VerificationStatus.CONDITIONAL
            rationale = f"Plan approved with {total_violations} minor constraint violations to monitor"
        else:
            status = VerificationStatus.REJECTED
            rationale = f"Plan rejected due to {total_violations} constraint violations"
        
        # Calculate risk score
        risk_score = min(total_violations * 0.1, 1.0)
        
        verification_report = VerificationReport(
            plan_id=plan_data.get("plan_id", str(uuid4())),
            verification_status=status,
            constraints_checked=len(self.constraint_rules) * len(plan_steps),
            constraints_passed=len(self.constraint_rules) * len(plan_steps) - total_violations,
            constraint_violations=[],  # Simplified for mock
            overall_risk_score=risk_score,
            approval_rationale=rationale,
            confidence_score=0.9 - (total_violations * 0.05),
            verification_time=0.15,
            verifier_agent_id=self.agent_id,
            correlation_id=message.correlation_id
        )
        
        # Store in history
        self.verification_history.append({
            "timestamp": datetime.utcnow(),
            "plan_id": verification_report.plan_id,
            "status": status,
            "violations": total_violations
        })
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "verification_report": verification_report.dict(),
                "step_results": verification_results,
                "recommendations": self._generate_recommendations(total_violations, status)
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _verify_plan_step(self, step: Dict) -> Dict:
        """Verify an individual plan step"""
        violations = []
        
        # Check amount constraints
        amount = float(step.get("amount", 0))
        if amount > 100000:
            violations.append({
                "constraint": "max_single_transaction",
                "severity": "medium",
                "description": "Single transaction exceeds $100,000 limit"
            })
        
        # Check risk level constraints
        risk_level = step.get("risk_level", "medium")
        if risk_level == "high" and random.random() < 0.3:
            violations.append({
                "constraint": "risk_tolerance",
                "severity": "high",
                "description": "High-risk investment may exceed user tolerance"
            })
        
        # Check liquidity constraints
        action_type = step.get("action_type", "")
        if "illiquid" in action_type.lower() and random.random() < 0.2:
            violations.append({
                "constraint": "liquidity_requirement",
                "severity": "medium",
                "description": "Investment may impact liquidity requirements"
            })
        
        return {
            "step_id": step.get("step_id", str(uuid4())),
            "action_type": action_type,
            "violations": violations,
            "compliance_score": 1.0 - (len(violations) * 0.2),
            "verification_time": 0.02
        }
    
    async def _handle_cmvl_trigger(self, message: AgentMessage) -> AgentMessage:
        """Handle CMVL trigger and initiate continuous monitoring"""
        self.cmvl_active = True
        trigger_data = message.payload["cmvl_trigger"]
        
        # Simulate CMVL response
        cmvl_response = {
            "cmvl_activated": True,
            "trigger_severity": trigger_data.get("severity", "medium"),
            "monitoring_frequency": "real_time" if trigger_data.get("severity") == "critical" else "5_minutes",
            "verification_actions": [
                "constraint_re_evaluation",
                "risk_assessment_update",
                "compliance_check",
                "plan_validation"
            ],
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=2)).isoformat(),
            "auto_remediation": trigger_data.get("severity") != "critical"
        }
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload=cmvl_response,
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _check_constraints(self, message: AgentMessage) -> AgentMessage:
        """Check specific constraints"""
        constraint_data = message.payload["constraint_check"]
        
        results = []
        for constraint_name in constraint_data.get("constraints", []):
            if constraint_name in self.constraint_rules:
                rule = self.constraint_rules[constraint_name]
                result = {
                    "constraint": constraint_name,
                    "status": "passed" if random.random() > 0.2 else "failed",
                    "rule": rule["description"],
                    "severity": rule["severity"]
                }
                results.append(result)
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "constraint_results": results,
                "overall_compliance": sum(1 for r in results if r["status"] == "passed") / max(len(results), 1)
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    def _load_constraint_rules(self) -> Dict[str, Dict]:
        """Load constraint validation rules"""
        return {
            "emergency_fund": {
                "description": "Maintain emergency fund >= 3 months expenses",
                "severity": "high",
                "validation": "emergency_fund >= monthly_expenses * 3"
            },
            "debt_to_income": {
                "description": "Total debt payments <= 36% of gross income",
                "severity": "high",
                "validation": "debt_payments / gross_income <= 0.36"
            },
            "investment_concentration": {
                "description": "No single investment > 20% of portfolio",
                "severity": "medium",
                "validation": "max_investment_percentage <= 0.20"
            },
            "risk_tolerance": {
                "description": "Portfolio risk level matches user tolerance",
                "severity": "medium",
                "validation": "portfolio_risk <= user_risk_tolerance"
            },
            "liquidity_requirement": {
                "description": "Maintain adequate liquid assets",
                "severity": "medium",
                "validation": "liquid_assets >= monthly_expenses * 2"
            }
        }
    
    def _generate_recommendations(self, violations: int, status: VerificationStatus) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        if status == VerificationStatus.REJECTED:
            recommendations.extend([
                "Consider reducing investment amounts to meet constraint limits",
                "Adjust risk levels to match user tolerance",
                "Ensure adequate emergency fund before investing"
            ])
        elif status == VerificationStatus.CONDITIONAL:
            recommendations.extend([
                "Monitor constraint violations closely",
                "Consider gradual implementation of plan steps",
                "Review and adjust if market conditions change"
            ])
        else:
            recommendations.extend([
                "Plan meets all constraints and can proceed",
                "Continue monitoring for market changes",
                "Review plan quarterly for optimization opportunities"
            ])
        
        return recommendations


class MockExecutionAgent(BaseAgent):
    """
    Mock Execution Agent with realistic transaction simulation and ledger management.
    
    Simulates financial transactions, portfolio updates, and audit trail generation
    for testing execution workflows.
    """
    
    def __init__(self, agent_id: str = "mock_execution_001"):
        super().__init__(agent_id, "execution")
        self.financial_ledger = {}
        self.transaction_history: List[Dict] = []
        self.portfolio_state = self._initialize_portfolio()
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process execution requests and portfolio management"""
        payload = message.payload
        
        if message.message_type == MessageType.REQUEST:
            if "execution_request" in payload:
                return await self._execute_plan(message)
            elif "portfolio_update" in payload:
                return await self._update_portfolio(message)
            elif "ledger_query" in payload:
                return await self._query_ledger(message)
        
        return None
    
    async def _execute_plan(self, message: AgentMessage) -> AgentMessage:
        """Execute approved financial plan steps"""
        await asyncio.sleep(0.2)  # Simulate execution time
        
        plan_data = message.payload["execution_request"]
        plan_steps = plan_data.get("plan_steps", [])
        
        execution_results = []
        total_executed = 0
        
        for step in plan_steps:
            result = await self._execute_step(step, message.session_id)
            execution_results.append(result)
            if result["status"] == "completed":
                total_executed += 1
        
        # Update portfolio state
        await self._update_portfolio_state(execution_results)
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "execution_completed": True,
                "steps_executed": total_executed,
                "total_steps": len(plan_steps),
                "execution_results": execution_results,
                "portfolio_updated": True,
                "execution_time": 0.2
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _execute_step(self, step: Dict, session_id: str) -> Dict:
        """Execute an individual plan step"""
        step_id = step.get("step_id", str(uuid4()))
        action_type = step.get("action_type", "unknown")
        amount = float(step.get("amount", 0))
        
        # Simulate execution with occasional failures
        success_probability = 0.95
        if random.random() < success_probability:
            status = "completed"
            transaction_id = str(uuid4())
            
            # Create transaction record
            transaction = {
                "transaction_id": transaction_id,
                "step_id": step_id,
                "session_id": session_id,
                "action_type": action_type,
                "amount": amount,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
                "fees": amount * 0.001,  # 0.1% fee
                "net_amount": amount * 0.999
            }
            
            self.transaction_history.append(transaction)
            
            return {
                "step_id": step_id,
                "status": "completed",
                "transaction_id": transaction_id,
                "amount_executed": amount,
                "fees": transaction["fees"],
                "execution_time": 0.05
            }
        else:
            return {
                "step_id": step_id,
                "status": "failed",
                "error": "Simulated execution failure",
                "retry_recommended": True,
                "execution_time": 0.02
            }
    
    async def _update_portfolio(self, message: AgentMessage) -> AgentMessage:
        """Update portfolio based on market changes or rebalancing"""
        update_data = message.payload["portfolio_update"]
        
        # Simulate portfolio updates
        updates_applied = []
        for update in update_data.get("updates", []):
            asset = update.get("asset", "unknown")
            change = update.get("change", 0.0)
            
            if asset in self.portfolio_state:
                old_value = self.portfolio_state[asset]
                self.portfolio_state[asset] += change
                
                updates_applied.append({
                    "asset": asset,
                    "old_value": old_value,
                    "new_value": self.portfolio_state[asset],
                    "change": change
                })
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "portfolio_updated": True,
                "updates_applied": updates_applied,
                "current_portfolio": self.portfolio_state,
                "total_value": sum(self.portfolio_state.values())
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _query_ledger(self, message: AgentMessage) -> AgentMessage:
        """Query financial ledger for transaction history"""
        query_params = message.payload["ledger_query"]
        
        # Filter transactions based on query
        filtered_transactions = self.transaction_history
        
        if "session_id" in query_params:
            session_id = query_params["session_id"]
            filtered_transactions = [
                t for t in filtered_transactions 
                if t.get("session_id") == session_id
            ]
        
        if "date_range" in query_params:
            # Simplified date filtering for mock
            filtered_transactions = filtered_transactions[-10:]  # Last 10 transactions
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "transactions": filtered_transactions,
                "total_transactions": len(filtered_transactions),
                "ledger_balance": sum(self.portfolio_state.values()),
                "query_time": 0.01
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    def _initialize_portfolio(self) -> Dict[str, float]:
        """Initialize mock portfolio state"""
        return {
            "cash": 50000.0,
            "stocks": 75000.0,
            "bonds": 25000.0,
            "real_estate": 100000.0,
            "commodities": 5000.0
        }
    
    async def _update_portfolio_state(self, execution_results: List[Dict]) -> None:
        """Update portfolio state based on execution results"""
        for result in execution_results:
            if result["status"] == "completed":
                amount = result.get("amount_executed", 0)
                # Simplified portfolio update logic
                self.portfolio_state["cash"] -= amount
                self.portfolio_state["stocks"] += amount * 0.6
                self.portfolio_state["bonds"] += amount * 0.4