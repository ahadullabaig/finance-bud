"""
Conversational Agent - Phase 6, Task 23
NVIDIA NIM Alternative using Ollama for local LLM inference

Provides natural language understanding and generation for financial planning:
- Natural language goal parsing
- Financial narrative generation
- What-if scenario explanations
- Conversational planning workflow

Requirements: Phase 6, Task 23
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

from data_models.schemas import (
    EnhancedPlanRequest, FinancialState, RiskProfile, TaxContext,
    AgentMessage, MessageType, Priority
)
from agents.base_agent import BaseAgent


class ConversationalAgent(BaseAgent):
    """
    Conversational AI agent for natural language financial planning.

    Uses local LLM (Ollama) as alternative to NVIDIA NIM for:
    - Parsing user goals from natural language
    - Generating financial narratives
    - Explaining complex scenarios
    - Conversational interactions
    """

    def __init__(
        self,
        agent_id: str = "conversational-agent-001",
        model_name: str = "llama3.2:3b",
        use_fallback: bool = True
    ):
        super().__init__(agent_id, "ConversationalAgent")
        self.model_name = model_name
        self.use_fallback = use_fallback
        self.ollama_available = OLLAMA_AVAILABLE

        # Financial domain context
        self.financial_context = self._load_financial_context()

        self.logger.info(
            f"ConversationalAgent initialized with model: {model_name}, "
            f"Ollama available: {self.ollama_available}"
        )

    def _load_financial_context(self) -> Dict[str, Any]:
        """Load financial domain knowledge for better LLM context"""
        return {
            "risk_levels": ["conservative", "moderate", "aggressive"],
            "goal_types": ["retirement", "emergency_fund", "investment", "debt_payoff", "education"],
            "account_types": ["checking", "savings", "401k", "ira", "brokerage", "hsa"],
            "tax_brackets": [10, 12, 22, 24, 32, 35, 37],
            "common_expenses": ["housing", "transportation", "food", "healthcare", "entertainment"]
        }

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages"""
        if message.message_type == MessageType.REQUEST:
            if message.content.get("action") == "parse_goal":
                result = await self.parse_natural_language_goal(
                    message.content.get("user_input", "")
                )
                return AgentMessage(
                    agent_id=self.agent_id,
                    target_agent_id=message.agent_id,
                    message_type=MessageType.RESPONSE,
                    content=result,
                    correlation_id=message.correlation_id,
                    session_id=message.session_id
                )
        return None

    async def parse_natural_language_goal(
        self,
        user_input: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language input into structured financial goal.

        Args:
            user_input: Natural language description of financial goal
            user_context: Optional user context (age, income, etc.)

        Returns:
            Structured goal data compatible with EnhancedPlanRequest
        """
        self.logger.info(f"Parsing natural language goal: {user_input[:100]}...")

        if self.ollama_available:
            try:
                return await self._parse_with_llm(user_input, user_context)
            except Exception as e:
                self.logger.warning(f"LLM parsing failed: {e}. Using fallback.")
                if self.use_fallback:
                    return self._parse_with_rules(user_input, user_context)
                raise
        else:
            self.logger.info("Using rule-based parsing (Ollama not available)")
            return self._parse_with_rules(user_input, user_context)

    async def _parse_with_llm(
        self,
        user_input: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse using LLM (Ollama)"""
        prompt = self._create_parsing_prompt(user_input, user_context)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial planning assistant. Extract structured information from user requests.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                format='json'  # Request JSON output
            )

            # Parse LLM response
            result = json.loads(response['message']['content'])

            # Validate and enrich
            return self._validate_and_enrich_goal(result, user_input)

        except Exception as e:
            self.logger.error(f"LLM parsing error: {e}")
            raise

    def _create_parsing_prompt(
        self,
        user_input: str,
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Create structured prompt for LLM goal parsing"""
        context_str = json.dumps(user_context) if user_context else "None provided"

        return f"""Extract financial planning information from the user's request.

User Request: "{user_input}"

User Context: {context_str}

Extract and return JSON with the following structure:
{{
    "goal_type": "retirement|emergency_fund|investment|debt_payoff|education",
    "target_amount": <number or null>,
    "timeframe_years": <number or null>,
    "risk_tolerance": "conservative|moderate|aggressive",
    "current_age": <number or null>,
    "retirement_age": <number or null>,
    "monthly_contribution": <number or null>,
    "current_savings": <number or null>,
    "annual_income": <number or null>,
    "constraints": ["list of constraints mentioned"],
    "priorities": ["list of priorities mentioned"]
}}

Only include fields that can be extracted from the user's request. Use null for unknown values.
"""

    def _parse_with_rules(
        self,
        user_input: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Rule-based parsing as fallback"""
        user_input_lower = user_input.lower()
        result = {
            "goal_type": "investment",  # default
            "target_amount": None,
            "timeframe_years": None,
            "risk_tolerance": "moderate",
            "constraints": [],
            "priorities": [],
            "raw_input": user_input
        }

        # Extract goal type
        goal_keywords = {
            "retirement": ["retire", "retirement", "retire at"],
            "emergency_fund": ["emergency", "emergency fund", "rainy day"],
            "investment": ["invest", "investment", "grow my money"],
            "debt_payoff": ["debt", "pay off", "payoff", "loan"],
            "education": ["education", "college", "tuition", "school"]
        }

        for goal_type, keywords in goal_keywords.items():
            if any(kw in user_input_lower for kw in keywords):
                result["goal_type"] = goal_type
                break

        # Extract amounts (dollars)
        amount_patterns = [
            r'\$?([\d,]+(?:\.\d{2})?)\s*(?:million|m)',  # $2 million
            r'\$?([\d,]+(?:\.\d{2})?)\s*(?:thousand|k)',  # $500k
            r'\$\s*([\d,]+(?:\.\d{2})?)',  # $100,000
        ]

        for pattern in amount_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)

                if 'million' in user_input_lower or 'm' in match.group(0):
                    amount *= 1_000_000
                elif 'thousand' in user_input_lower or 'k' in match.group(0):
                    amount *= 1_000

                result["target_amount"] = amount
                break

        # Extract age/timeframe
        age_patterns = [
            r'(?:retire\s+at|at\s+age)\s+(\d+)',
            r'(\d+)\s+years?\s+old',
            r'in\s+(\d+)\s+years?'
        ]

        for pattern in age_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                value = int(match.group(1))
                if 'retire at' in match.group(0) or 'age' in match.group(0):
                    result["retirement_age"] = value
                    # Estimate timeframe if current age provided in context
                    if user_context and user_context.get('age'):
                        result["timeframe_years"] = value - user_context['age']
                else:
                    result["timeframe_years"] = value
                break

        # Extract risk tolerance
        if any(kw in user_input_lower for kw in ["safe", "conservative", "low risk"]):
            result["risk_tolerance"] = "conservative"
        elif any(kw in user_input_lower for kw in ["aggressive", "high risk", "growth"]):
            result["risk_tolerance"] = "aggressive"

        return result

    def _validate_and_enrich_goal(
        self,
        goal_data: Dict[str, Any],
        original_input: str
    ) -> Dict[str, Any]:
        """Validate and enrich parsed goal data"""
        # Ensure required fields
        if "goal_type" not in goal_data:
            goal_data["goal_type"] = "investment"

        if "risk_tolerance" not in goal_data:
            goal_data["risk_tolerance"] = "moderate"

        # Add metadata
        goal_data["parsed_at"] = datetime.utcnow().isoformat()
        goal_data["raw_input"] = original_input
        goal_data["parsing_method"] = "llm" if self.ollama_available else "rules"

        return goal_data

    async def generate_financial_narrative(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate human-readable narrative from financial plan.

        Args:
            plan: Structured financial plan
            context: Additional context for narrative

        Returns:
            Natural language narrative explaining the plan
        """
        self.logger.info("Generating financial narrative")

        if self.ollama_available:
            try:
                return await self._generate_narrative_with_llm(plan, context)
            except Exception as e:
                self.logger.warning(f"LLM narrative generation failed: {e}")
                if self.use_fallback:
                    return self._generate_narrative_template(plan, context)
                raise
        else:
            return self._generate_narrative_template(plan, context)

    async def _generate_narrative_with_llm(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate narrative using LLM"""
        prompt = f"""Create a clear, engaging narrative explaining this financial plan:

Plan Details:
{json.dumps(plan, indent=2)}

Context:
{json.dumps(context, indent=2) if context else 'None'}

Generate a narrative that:
1. Summarizes the main financial goal
2. Explains the strategy and approach
3. Highlights key milestones and timeline
4. Mentions important risks or considerations
5. Provides actionable next steps

Write in a professional but friendly tone, as if advising a client."""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a certified financial planner explaining plans to clients.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        return response['message']['content']

    def _generate_narrative_template(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate narrative using templates"""
        goal_type = plan.get('goal_type', 'financial planning')
        target_amount = plan.get('target_amount', 'your target')
        timeframe = plan.get('timeframe_years', 'the specified timeframe')

        # Format target amount properly
        if isinstance(target_amount, (int, float)):
            target_str = f"${target_amount:,.2f}"
        else:
            target_str = str(target_amount)

        narrative = f"""
## Financial Plan Summary

**Goal**: {goal_type.replace('_', ' ').title()}

Based on your objectives, we've created a comprehensive plan to help you achieve {goal_type.replace('_', ' ')}.

**Target**: {target_str}
**Timeline**: {timeframe} years

**Strategy Overview**:
Your plan involves a balanced approach considering your risk tolerance and financial situation.
We've identified key steps and milestones to keep you on track.

**Next Steps**:
1. Review the detailed plan breakdown
2. Set up automatic contributions if applicable
3. Schedule regular reviews to track progress
4. Adjust as life circumstances change

This plan is designed to be flexible and adapt to market conditions and your evolving needs.
"""
        return narrative.strip()

    async def explain_what_if_scenario(
        self,
        scenario: Dict[str, Any],
        impact: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Explain a what-if scenario and its impact.

        Args:
            scenario: Description of the scenario (e.g., market crash, job loss)
            impact: Quantified impact on the plan
            context: Additional context

        Returns:
            Natural language explanation
        """
        self.logger.info(f"Explaining what-if scenario: {scenario.get('type', 'unknown')}")

        if self.ollama_available:
            try:
                return await self._explain_scenario_with_llm(scenario, impact, context)
            except Exception as e:
                self.logger.warning(f"LLM scenario explanation failed: {e}")
                if self.use_fallback:
                    return self._explain_scenario_template(scenario, impact, context)
                raise
        else:
            return self._explain_scenario_template(scenario, impact, context)

    async def _explain_scenario_with_llm(
        self,
        scenario: Dict[str, Any],
        impact: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Explain scenario using LLM"""
        prompt = f"""Explain the impact of this financial scenario:

Scenario:
{json.dumps(scenario, indent=2)}

Impact on Plan:
{json.dumps(impact, indent=2)}

Context:
{json.dumps(context, indent=2) if context else 'None'}

Provide a clear explanation that:
1. Describes what the scenario means
2. Quantifies the impact on financial goals
3. Explains why this impact occurs
4. Suggests potential adjustments or responses
5. Maintains a balanced, reassuring tone"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a financial advisor explaining scenarios to concerned clients.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        return response['message']['content']

    def _explain_scenario_template(
        self,
        scenario: Dict[str, Any],
        impact: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Explain scenario using templates"""
        scenario_type = scenario.get('type', 'market change')
        severity = scenario.get('severity', 'moderate')

        impact_description = []
        if 'target_amount_change' in impact:
            change = impact['target_amount_change']
            impact_description.append(
                f"Your target amount would change by ${abs(change):,.2f} "
                f"({'decrease' if change < 0 else 'increase'})"
            )

        if 'timeframe_change' in impact:
            change = impact['timeframe_change']
            impact_description.append(
                f"Timeline would shift by {abs(change)} "
                f"{'months' if abs(change) > 1 else 'month'} "
                f"({'delay' if change > 0 else 'acceleration'})"
            )

        explanation = f"""
## What-If Scenario Analysis: {scenario_type.replace('_', ' ').title()}

**Scenario Severity**: {severity.title()}

**What This Means**:
{scenario.get('description', 'A significant change in market or personal circumstances.')}

**Impact on Your Plan**:
{' '.join(impact_description) if impact_description else 'We are analyzing the potential impact.'}

**Recommended Response**:
Based on this scenario, we recommend reviewing your plan and considering adjustments
to maintain progress toward your goals. Our system can automatically suggest optimized
alternatives if needed.
"""
        return explanation.strip()


# Singleton instance
_conversational_agent = None


def get_conversational_agent(
    model_name: str = "llama3.2:3b"
) -> ConversationalAgent:
    """Get or create singleton conversational agent instance"""
    global _conversational_agent
    if _conversational_agent is None:
        _conversational_agent = ConversationalAgent(model_name=model_name)
    return _conversational_agent
