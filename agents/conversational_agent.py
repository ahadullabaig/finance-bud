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
    logging.info("Ollama successfully imported and available for LLM operations")
except ImportError as e:
    OLLAMA_AVAILABLE = False
    logging.warning(
        f"Ollama not available - falling back to rule-based processing. "
        f"Import error: {e}. To enable LLM features, install with: pip install ollama"
    )
except Exception as e:
    OLLAMA_AVAILABLE = False
    logging.error(
        f"Unexpected error importing Ollama: {e}. "
        f"Falling back to rule-based processing."
    )

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
        self.ollama_connection_verified = False

        # Financial domain context
        try:
            self.financial_context = self._load_financial_context()
        except Exception as e:
            self.logger.warning(f"Failed to load financial context: {e}. Using minimal context.")
            self.financial_context = {"risk_levels": ["conservative", "moderate", "aggressive"]}

        # Verify Ollama connection if available
        if self.ollama_available:
            self._verify_ollama_connection()

        self.logger.info(
            f"ConversationalAgent initialized successfully - "
            f"Model: {model_name}, Ollama available: {self.ollama_available}, "
            f"Connection verified: {self.ollama_connection_verified}, "
            f"Fallback enabled: {self.use_fallback}"
        )

    def _verify_ollama_connection(self) -> None:
        """Verify Ollama service is running and model is available"""
        if not self.ollama_available:
            return

        try:
            # Test basic connection
            models = ollama.list()
            self.logger.info(f"Ollama service connected. Available models: {len(models.get('models', []))}")
            
            # Check if our model is available
            model_names = [model['name'] for model in models.get('models', [])]
            if self.model_name in model_names:
                self.ollama_connection_verified = True
                self.logger.info(f"Model '{self.model_name}' is available and ready")
            else:
                self.logger.warning(
                    f"Model '{self.model_name}' not found. Available models: {model_names}. "
                    f"Will attempt to pull model on first use."
                )
                # Still mark as verified since we can pull the model
                self.ollama_connection_verified = True
                
        except Exception as e:
            self.logger.warning(
                f"Ollama service connection failed: {e}. "
                f"LLM features will fall back to rule-based processing."
            )
            self.ollama_connection_verified = False

    def _sanitize_user_context(self, user_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize and validate user context data"""
        if not user_context:
            return None
            
        try:
            safe_context = {}
            for key, value in user_context.items():
                if isinstance(key, str) and len(key) < 100:  # Reasonable key length
                    if isinstance(value, (int, float, str, bool)) and str(value) != "":
                        safe_context[key] = value
            return safe_context if safe_context else None
        except Exception as e:
            self.logger.warning(f"Failed to sanitize user context: {e}")
            return None

    def _sanitize_plan_data(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate plan data"""
        try:
            safe_plan = {}
            allowed_keys = {
                'goal_type', 'target_amount', 'timeframe_years', 'risk_tolerance',
                'current_age', 'retirement_age', 'monthly_contribution', 'current_savings',
                'annual_income', 'constraints', 'priorities', 'raw_input', 'parsed_at'
            }
            
            for key, value in plan.items():
                if key in allowed_keys and value is not None:
                    if isinstance(value, (int, float, str, bool, list)):
                        safe_plan[key] = value
                        
            return safe_plan if safe_plan else {"goal_type": "investment"}
        except Exception as e:
            self.logger.warning(f"Failed to sanitize plan data: {e}")
            return {"goal_type": "investment"}

    def _sanitize_scenario_data(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate scenario data"""
        try:
            safe_scenario = {}
            allowed_keys = {'type', 'severity', 'description', 'probability', 'duration'}
            
            for key, value in scenario.items():
                if key in allowed_keys and value is not None:
                    if isinstance(value, (int, float, str, bool)):
                        safe_scenario[key] = value
                        
            return safe_scenario if safe_scenario else {"type": "unknown", "description": "Scenario analysis"}
        except Exception as e:
            self.logger.warning(f"Failed to sanitize scenario data: {e}")
            return {"type": "unknown", "description": "Scenario analysis"}

    def _sanitize_impact_data(self, impact: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate impact data"""
        try:
            safe_impact = {}
            allowed_keys = {
                'target_amount_change', 'timeframe_change', 'risk_change', 
                'probability_success', 'monthly_contribution_change'
            }
            
            for key, value in impact.items():
                if key in allowed_keys and value is not None:
                    if isinstance(value, (int, float)):
                        safe_impact[key] = value
                        
            return safe_impact
        except Exception as e:
            self.logger.warning(f"Failed to sanitize impact data: {e}")
            return {}

    def _create_default_goal(self, reason: str) -> Dict[str, Any]:
        """Create a default goal structure when parsing fails"""
        return {
            "goal_type": "investment",
            "target_amount": None,
            "timeframe_years": None,
            "risk_tolerance": "moderate",
            "constraints": [],
            "priorities": [],
            "raw_input": reason,
            "parsed_at": datetime.utcnow().isoformat(),
            "parsing_method": "default_fallback",
            "error": reason
        }

    def _extract_from_text_response(self, text_response: str, user_input: str) -> Dict[str, Any]:
        """Extract structured data from non-JSON LLM response as fallback"""
        self.logger.info("Attempting to extract structured data from text response")
        
        # Fall back to rule-based parsing if JSON parsing fails
        try:
            return self._parse_with_rules(user_input, None)
        except Exception as e:
            self.logger.warning(f"Text extraction fallback failed: {e}")
            return self._create_default_goal(f"LLM response parsing failed: {text_response[:100]}...")

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
        if not user_input or not user_input.strip():
            self.logger.warning("Empty or invalid user input provided")
            return self._create_default_goal("Empty input provided")

        self.logger.info(f"Parsing natural language goal: {user_input[:100]}...")

        # Sanitize user context
        safe_context = self._sanitize_user_context(user_context)

        if self.ollama_available and self.ollama_connection_verified:
            try:
                return await self._parse_with_llm(user_input, safe_context)
            except Exception as e:
                self.logger.warning(
                    f"LLM parsing failed: {type(e).__name__}: {e}. "
                    f"Falling back to rule-based parsing."
                )
                if self.use_fallback:
                    return self._parse_with_rules(user_input, safe_context)
                else:
                    raise RuntimeError(f"Goal parsing failed and fallback disabled: {e}")
        else:
            reason = "Ollama not available" if not self.ollama_available else "Ollama connection not verified"
            self.logger.info(f"Using rule-based parsing ({reason})")
            return self._parse_with_rules(user_input, safe_context)

    async def _parse_with_llm(
        self,
        user_input: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse using LLM (Ollama) with comprehensive error handling"""
        try:
            prompt = self._create_parsing_prompt(user_input, user_context)
            
            # Attempt to chat with the model
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

            # Validate response structure
            if not response or 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid response structure from Ollama")

            # Parse LLM response
            try:
                result = json.loads(response['message']['content'])
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM JSON response: {e}")
                # Try to extract partial information from text response
                result = self._extract_from_text_response(response['message']['content'], user_input)

            # Validate and enrich
            return self._validate_and_enrich_goal(result, user_input)

        except ollama.ResponseError as e:
            self.logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"LLM service error: {e}")
        except ConnectionError as e:
            self.logger.error(f"Ollama connection error: {e}")
            raise RuntimeError(f"LLM service unavailable: {e}")
        except TimeoutError as e:
            self.logger.error(f"Ollama timeout error: {e}")
            raise RuntimeError(f"LLM service timeout: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected LLM parsing error: {type(e).__name__}: {e}")
            raise RuntimeError(f"LLM parsing failed: {e}")

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
        """Rule-based parsing as fallback with enhanced error handling"""
        try:
            if not user_input or not isinstance(user_input, str):
                return self._create_default_goal("Invalid input for rule-based parsing")
                
            user_input_lower = user_input.lower().strip()
            if not user_input_lower:
                return self._create_default_goal("Empty input for rule-based parsing")
                
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
            try:
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
            except Exception as e:
                self.logger.warning(f"Error extracting goal type: {e}")

            # Extract amounts (dollars)
            try:
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
            except Exception as e:
                self.logger.warning(f"Error extracting target amount: {e}")

            # Extract age/timeframe
            try:
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
            except Exception as e:
                self.logger.warning(f"Error extracting age/timeframe: {e}")

            # Extract risk tolerance
            try:
                if any(kw in user_input_lower for kw in ["safe", "conservative", "low risk"]):
                    result["risk_tolerance"] = "conservative"
                elif any(kw in user_input_lower for kw in ["aggressive", "high risk", "growth"]):
                    result["risk_tolerance"] = "aggressive"
            except Exception as e:
                self.logger.warning(f"Error extracting risk tolerance: {e}")

            return self._validate_and_enrich_goal(result, user_input)
            
        except Exception as e:
            self.logger.error(f"Rule-based parsing failed: {e}")
            return self._create_default_goal(f"Rule-based parsing error: {e}")

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
        if not plan or not isinstance(plan, dict):
            self.logger.warning("Invalid or empty plan provided for narrative generation")
            return "Unable to generate narrative: Invalid plan data provided."

        self.logger.info("Generating financial narrative")

        # Sanitize inputs
        safe_plan = self._sanitize_plan_data(plan)
        safe_context = self._sanitize_user_context(context)

        if self.ollama_available and self.ollama_connection_verified:
            try:
                return await self._generate_narrative_with_llm(safe_plan, safe_context)
            except Exception as e:
                self.logger.warning(
                    f"LLM narrative generation failed: {type(e).__name__}: {e}. "
                    f"Falling back to template generation."
                )
                if self.use_fallback:
                    return self._generate_narrative_template(safe_plan, safe_context)
                else:
                    raise RuntimeError(f"Narrative generation failed and fallback disabled: {e}")
        else:
            reason = "Ollama not available" if not self.ollama_available else "Ollama connection not verified"
            self.logger.info(f"Using template-based narrative generation ({reason})")
            return self._generate_narrative_template(safe_plan, safe_context)

    async def _generate_narrative_with_llm(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate narrative using LLM with comprehensive error handling"""
        try:
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

            # Validate response
            if not response or 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid response structure from Ollama")

            narrative = response['message']['content']
            if not narrative or not narrative.strip():
                raise ValueError("Empty narrative generated by LLM")

            return narrative.strip()

        except ollama.ResponseError as e:
            self.logger.error(f"Ollama API error during narrative generation: {e}")
            raise RuntimeError(f"LLM service error: {e}")
        except ConnectionError as e:
            self.logger.error(f"Ollama connection error during narrative generation: {e}")
            raise RuntimeError(f"LLM service unavailable: {e}")
        except TimeoutError as e:
            self.logger.error(f"Ollama timeout during narrative generation: {e}")
            raise RuntimeError(f"LLM service timeout: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during LLM narrative generation: {type(e).__name__}: {e}")
            raise RuntimeError(f"Narrative generation failed: {e}")

    def _generate_narrative_template(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate narrative using templates with error handling"""
        try:
            goal_type = plan.get('goal_type', 'financial planning')
            target_amount = plan.get('target_amount', 'your target')
            timeframe = plan.get('timeframe_years', 'the specified timeframe')

            # Format target amount properly
            try:
                if isinstance(target_amount, (int, float)) and target_amount > 0:
                    target_str = f"${target_amount:,.2f}"
                else:
                    target_str = "your financial target"
            except Exception:
                target_str = "your financial target"

            # Format timeframe
            try:
                if isinstance(timeframe, (int, float)) and timeframe > 0:
                    timeframe_str = f"{int(timeframe)} years"
                else:
                    timeframe_str = "your specified timeframe"
            except Exception:
                timeframe_str = "your specified timeframe"

            # Safe goal type formatting
            try:
                goal_display = goal_type.replace('_', ' ').title() if isinstance(goal_type, str) else "Financial Planning"
            except Exception:
                goal_display = "Financial Planning"

            narrative = f"""
## Financial Plan Summary

**Goal**: {goal_display}

Based on your objectives, we've created a comprehensive plan to help you achieve your financial goals.

**Target**: {target_str}
**Timeline**: {timeframe_str}

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

        except Exception as e:
            self.logger.error(f"Template narrative generation failed: {e}")
            return """
## Financial Plan Summary

We've created a financial plan based on your objectives. Due to a processing issue, 
we're unable to display the full details at this time. Please review your plan 
data and try again, or contact support for assistance.

**Next Steps**:
1. Verify your plan inputs
2. Try generating the narrative again
3. Contact support if the issue persists
""".strip()

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
        if not scenario or not isinstance(scenario, dict):
            self.logger.warning("Invalid scenario data provided")
            return "Unable to explain scenario: Invalid scenario data provided."

        if not impact or not isinstance(impact, dict):
            self.logger.warning("Invalid impact data provided")
            return "Unable to explain scenario: Invalid impact data provided."

        scenario_type = scenario.get('type', 'unknown')
        self.logger.info(f"Explaining what-if scenario: {scenario_type}")

        # Sanitize inputs
        safe_scenario = self._sanitize_scenario_data(scenario)
        safe_impact = self._sanitize_impact_data(impact)
        safe_context = self._sanitize_user_context(context)

        if self.ollama_available and self.ollama_connection_verified:
            try:
                return await self._explain_scenario_with_llm(safe_scenario, safe_impact, safe_context)
            except Exception as e:
                self.logger.warning(
                    f"LLM scenario explanation failed: {type(e).__name__}: {e}. "
                    f"Falling back to template explanation."
                )
                if self.use_fallback:
                    return self._explain_scenario_template(safe_scenario, safe_impact, safe_context)
                else:
                    raise RuntimeError(f"Scenario explanation failed and fallback disabled: {e}")
        else:
            reason = "Ollama not available" if not self.ollama_available else "Ollama connection not verified"
            self.logger.info(f"Using template-based scenario explanation ({reason})")
            return self._explain_scenario_template(safe_scenario, safe_impact, safe_context)

    async def _explain_scenario_with_llm(
        self,
        scenario: Dict[str, Any],
        impact: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Explain scenario using LLM with comprehensive error handling"""
        try:
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

            # Validate response
            if not response or 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid response structure from Ollama")

            explanation = response['message']['content']
            if not explanation or not explanation.strip():
                raise ValueError("Empty explanation generated by LLM")

            return explanation.strip()

        except ollama.ResponseError as e:
            self.logger.error(f"Ollama API error during scenario explanation: {e}")
            raise RuntimeError(f"LLM service error: {e}")
        except ConnectionError as e:
            self.logger.error(f"Ollama connection error during scenario explanation: {e}")
            raise RuntimeError(f"LLM service unavailable: {e}")
        except TimeoutError as e:
            self.logger.error(f"Ollama timeout during scenario explanation: {e}")
            raise RuntimeError(f"LLM service timeout: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during LLM scenario explanation: {type(e).__name__}: {e}")
            raise RuntimeError(f"Scenario explanation failed: {e}")

    def _explain_scenario_template(
        self,
        scenario: Dict[str, Any],
        impact: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Explain scenario using templates with error handling"""
        try:
            scenario_type = scenario.get('type', 'market change')
            severity = scenario.get('severity', 'moderate')

            # Safe formatting of scenario type and severity
            try:
                scenario_display = scenario_type.replace('_', ' ').title() if isinstance(scenario_type, str) else "Market Change"
                severity_display = severity.title() if isinstance(severity, str) else "Moderate"
            except Exception:
                scenario_display = "Market Change"
                severity_display = "Moderate"

            impact_description = []
            
            # Safe impact processing
            try:
                if 'target_amount_change' in impact and isinstance(impact['target_amount_change'], (int, float)):
                    change = impact['target_amount_change']
                    impact_description.append(
                        f"Your target amount would change by ${abs(change):,.2f} "
                        f"({'decrease' if change < 0 else 'increase'})"
                    )
            except Exception as e:
                self.logger.warning(f"Error processing target amount change: {e}")

            try:
                if 'timeframe_change' in impact and isinstance(impact['timeframe_change'], (int, float)):
                    change = impact['timeframe_change']
                    impact_description.append(
                        f"Timeline would shift by {abs(change)} "
                        f"{'months' if abs(change) > 1 else 'month'} "
                        f"({'delay' if change > 0 else 'acceleration'})"
                    )
            except Exception as e:
                self.logger.warning(f"Error processing timeframe change: {e}")

            # Safe description extraction
            description = scenario.get('description', 'A significant change in market or personal circumstances.')
            if not isinstance(description, str) or not description.strip():
                description = 'A significant change in market or personal circumstances.'

            explanation = f"""
## What-If Scenario Analysis: {scenario_display}

**Scenario Severity**: {severity_display}

**What This Means**:
{description}

**Impact on Your Plan**:
{' '.join(impact_description) if impact_description else 'We are analyzing the potential impact on your financial plan.'}

**Recommended Response**:
Based on this scenario, we recommend reviewing your plan and considering adjustments
to maintain progress toward your goals. Our system can automatically suggest optimized
alternatives if needed.
"""
            return explanation.strip()

        except Exception as e:
            self.logger.error(f"Template scenario explanation failed: {e}")
            return """
## What-If Scenario Analysis

We've analyzed the scenario you provided. Due to a processing issue, 
we're unable to display the full analysis at this time. 

**Recommended Response**:
Please review your scenario inputs and try again, or contact support for assistance.
We recommend reviewing your financial plan regularly to ensure it remains aligned with your goals.
""".strip()


# Singleton instance
_conversational_agent = None


def get_conversational_agent(
    model_name: str = "llama3.2:3b"
) -> ConversationalAgent:
    """Get or create singleton conversational agent instance with error handling"""
    global _conversational_agent
    
    try:
        if _conversational_agent is None:
            logging.info(f"Creating new ConversationalAgent instance with model: {model_name}")
            _conversational_agent = ConversationalAgent(model_name=model_name)
        return _conversational_agent
    except Exception as e:
        logging.error(f"Failed to create ConversationalAgent: {e}")
        # Create a minimal fallback agent
        try:
            _conversational_agent = ConversationalAgent(
                model_name=model_name, 
                use_fallback=True
            )
            logging.warning("Created ConversationalAgent with fallback mode enabled")
            return _conversational_agent
        except Exception as fallback_error:
            logging.critical(f"Failed to create fallback ConversationalAgent: {fallback_error}")
            raise RuntimeError(f"Unable to initialize conversational agent: {e}")
