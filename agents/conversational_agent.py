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
        """Enhanced rule-based parsing with intelligent pattern matching for common financial scenarios"""
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
                "current_age": None,
                "retirement_age": None,
                "monthly_contribution": None,
                "current_savings": None,
                "annual_income": None,
                "constraints": [],
                "priorities": [],
                "raw_input": user_input
            }

            # Enhanced goal type extraction with comprehensive patterns
            try:
                goal_patterns = {
                    "retirement": [
                        "retire", "retirement", "retire at", "retirement planning", "401k", "ira", 
                        "pension", "social security", "golden years", "stop working", "financial independence"
                    ],
                    "emergency_fund": [
                        "emergency", "emergency fund", "rainy day", "unexpected expenses", 
                        "job loss", "medical emergency", "safety net", "peace of mind"
                    ],
                    "investment": [
                        "invest", "investment", "grow my money", "build wealth", "portfolio", 
                        "stocks", "bonds", "mutual funds", "etf", "market", "returns"
                    ],
                    "debt_payoff": [
                        "debt", "pay off", "payoff", "loan", "credit card", "mortgage", 
                        "student loan", "eliminate debt", "debt free", "owe money"
                    ],
                    "education": [
                        "education", "college", "tuition", "school", "university", "529 plan",
                        "student", "degree", "learning", "certification", "training"
                    ]
                }

                # Score each goal type based on keyword matches
                goal_scores = {}
                for goal_type, keywords in goal_patterns.items():
                    score = sum(1 for kw in keywords if kw in user_input_lower)
                    if score > 0:
                        goal_scores[goal_type] = score
                
                # Select goal type with highest score
                if goal_scores:
                    result["goal_type"] = max(goal_scores, key=goal_scores.get)
            except Exception as e:
                self.logger.warning(f"Error extracting goal type: {e}")

            # Enhanced amount extraction with multiple formats
            try:
                amount_patterns = [
                    # Millions
                    (r'(\d+(?:\.\d+)?)\s*(?:million|mil|m)\s*(?:dollars?)?', 1_000_000),
                    # Thousands  
                    (r'(\d+(?:\.\d+)?)\s*(?:thousand|k)\s*(?:dollars?)?', 1_000),
                    # Direct dollar amounts
                    (r'\$\s*([\d,]+(?:\.\d{2})?)', 1),
                    # Numbers followed by dollars
                    (r'([\d,]+(?:\.\d{2})?)\s*dollars?', 1),
                    # Plain numbers (assume dollars if reasonable range)
                    (r'\b(\d{4,})\b', 1)  # 4+ digits likely represents dollars
                ]

                for pattern, multiplier in amount_patterns:
                    match = re.search(pattern, user_input_lower)
                    if match:
                        amount_str = match.group(1).replace(',', '')
                        try:
                            amount = float(amount_str) * multiplier
                            # Reasonable range check
                            if 100 <= amount <= 100_000_000:  # $100 to $100M
                                result["target_amount"] = amount
                                break
                        except ValueError:
                            continue
            except Exception as e:
                self.logger.warning(f"Error extracting target amount: {e}")

            # Enhanced age and timeframe extraction
            try:
                # Current age patterns
                age_patterns = [
                    r'(?:i am|i\'m|age|currently)\s*(\d+)\s*(?:years?\s*old)?',
                    r'(\d+)\s*(?:year\s*old|years?\s*old)',
                    r'age\s*(\d+)'
                ]
                
                for pattern in age_patterns:
                    match = re.search(pattern, user_input_lower)
                    if match:
                        age = int(match.group(1))
                        if 18 <= age <= 100:  # Reasonable age range
                            result["current_age"] = age
                            break

                # Retirement age patterns
                retirement_patterns = [
                    r'retire\s*(?:at|by)\s*(?:age\s*)?(\d+)',
                    r'retirement\s*(?:at|by)\s*(?:age\s*)?(\d+)',
                    r'(?:at|by)\s*(\d+)\s*(?:i want to|plan to|will)\s*retire'
                ]
                
                for pattern in retirement_patterns:
                    match = re.search(pattern, user_input_lower)
                    if match:
                        ret_age = int(match.group(1))
                        if 50 <= ret_age <= 80:  # Reasonable retirement age
                            result["retirement_age"] = ret_age
                            # Calculate timeframe if current age is known
                            if result["current_age"]:
                                result["timeframe_years"] = ret_age - result["current_age"]
                            break

                # General timeframe patterns
                timeframe_patterns = [
                    r'in\s*(\d+)\s*years?',
                    r'over\s*(\d+)\s*years?',
                    r'within\s*(\d+)\s*years?',
                    r'(\d+)\s*year\s*(?:plan|goal|timeline)'
                ]
                
                if not result["timeframe_years"]:  # Only if not already set
                    for pattern in timeframe_patterns:
                        match = re.search(pattern, user_input_lower)
                        if match:
                            years = int(match.group(1))
                            if 1 <= years <= 50:  # Reasonable timeframe
                                result["timeframe_years"] = years
                                break
            except Exception as e:
                self.logger.warning(f"Error extracting age/timeframe: {e}")

            # Enhanced monthly contribution extraction
            try:
                contribution_patterns = [
                    r'(\d+(?:\.\d{2})?)\s*(?:per month|monthly|each month|every month)',
                    r'monthly\s*(?:contribution|payment|saving)?\s*(?:of\s*)?\$?\s*([\d,]+(?:\.\d{2})?)',
                    r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:per month|monthly)'
                ]
                
                for pattern in contribution_patterns:
                    match = re.search(pattern, user_input_lower)
                    if match:
                        amount_str = match.group(1).replace(',', '')
                        try:
                            monthly = float(amount_str)
                            if 10 <= monthly <= 50000:  # Reasonable monthly range
                                result["monthly_contribution"] = monthly
                                break
                        except ValueError:
                            continue
            except Exception as e:
                self.logger.warning(f"Error extracting monthly contribution: {e}")

            # Enhanced current savings extraction
            try:
                savings_patterns = [
                    r'(?:have|saved|currently have|existing)\s*(?:savings?\s*of\s*)?\$?\s*([\d,]+(?:\.\d{2})?)',
                    r'current\s*(?:savings?|balance)\s*(?:is\s*)?\$?\s*([\d,]+(?:\.\d{2})?)',
                    r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:saved|in savings|currently)'
                ]
                
                for pattern in savings_patterns:
                    match = re.search(pattern, user_input_lower)
                    if match:
                        amount_str = match.group(1).replace(',', '')
                        try:
                            savings = float(amount_str)
                            if 0 <= savings <= 50_000_000:  # Reasonable savings range
                                result["current_savings"] = savings
                                break
                        except ValueError:
                            continue
            except Exception as e:
                self.logger.warning(f"Error extracting current savings: {e}")

            # Enhanced income extraction
            try:
                income_patterns = [
                    r'(?:make|earn|income|salary)\s*(?:of\s*)?\$?\s*([\d,]+(?:\.\d{2})?)\s*(?:per year|annually|yearly)?',
                    r'annual\s*(?:income|salary)\s*(?:of\s*)?\$?\s*([\d,]+(?:\.\d{2})?)',
                    r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:per year|annually|yearly|income|salary)'
                ]
                
                for pattern in income_patterns:
                    match = re.search(pattern, user_input_lower)
                    if match:
                        amount_str = match.group(1).replace(',', '')
                        try:
                            income = float(amount_str)
                            if 10000 <= income <= 10_000_000:  # Reasonable income range
                                result["annual_income"] = income
                                break
                        except ValueError:
                            continue
            except Exception as e:
                self.logger.warning(f"Error extracting annual income: {e}")

            # Enhanced risk tolerance extraction
            try:
                risk_keywords = {
                    "conservative": [
                        "safe", "conservative", "low risk", "stable", "secure", "guaranteed",
                        "capital preservation", "don't want to lose", "risk averse", "cautious"
                    ],
                    "aggressive": [
                        "aggressive", "high risk", "growth", "maximum return", "willing to risk",
                        "high reward", "volatile", "speculative", "risky", "adventurous"
                    ],
                    "moderate": [
                        "moderate", "balanced", "medium risk", "some risk", "reasonable risk",
                        "diversified", "mixed", "average", "typical", "standard"
                    ]
                }
                
                # Score risk tolerance based on keyword matches
                risk_scores = {}
                for risk_level, keywords in risk_keywords.items():
                    score = sum(1 for kw in keywords if kw in user_input_lower)
                    if score > 0:
                        risk_scores[risk_level] = score
                
                if risk_scores:
                    result["risk_tolerance"] = max(risk_scores, key=risk_scores.get)
            except Exception as e:
                self.logger.warning(f"Error extracting risk tolerance: {e}")

            # Extract constraints and priorities
            try:
                constraint_keywords = [
                    "can't afford", "limited budget", "tight budget", "constraint", "restriction",
                    "maximum", "no more than", "budget limit", "financial limit"
                ]
                
                priority_keywords = [
                    "priority", "important", "focus on", "main goal", "primary objective",
                    "most important", "key goal", "essential", "critical"
                ]
                
                # Simple keyword matching for constraints and priorities
                for keyword in constraint_keywords:
                    if keyword in user_input_lower:
                        result["constraints"].append(f"Budget constraint mentioned: {keyword}")
                        
                for keyword in priority_keywords:
                    if keyword in user_input_lower:
                        result["priorities"].append(f"Priority mentioned: {keyword}")
            except Exception as e:
                self.logger.warning(f"Error extracting constraints/priorities: {e}")

            # Apply intelligent defaults based on goal type
            result = self._apply_intelligent_defaults(result, user_context)
            
            return self._validate_and_enrich_goal(result, user_input)
            
        except Exception as e:
            self.logger.error(f"Rule-based parsing failed: {e}")
            return self._create_default_goal(f"Rule-based parsing error: {e}")

    def _apply_intelligent_defaults(self, result: Dict[str, Any], user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply intelligent defaults based on goal type and context"""
        try:
            goal_type = result.get("goal_type", "investment")
            
            # Apply context from user_context if available
            if user_context:
                for key in ["current_age", "annual_income", "current_savings"]:
                    if not result.get(key) and user_context.get(key):
                        result[key] = user_context[key]
            
            # Goal-specific intelligent defaults
            if goal_type == "retirement":
                if not result.get("retirement_age"):
                    result["retirement_age"] = 65  # Standard retirement age
                if not result.get("target_amount") and result.get("annual_income"):
                    # Rule of thumb: 10-12x annual income for retirement
                    result["target_amount"] = result["annual_income"] * 10
                if not result.get("timeframe_years") and result.get("current_age") and result.get("retirement_age"):
                    result["timeframe_years"] = result["retirement_age"] - result["current_age"]
                    
            elif goal_type == "emergency_fund":
                if not result.get("target_amount") and result.get("annual_income"):
                    # Rule of thumb: 6 months of expenses (assume 70% of income)
                    monthly_expenses = (result["annual_income"] * 0.7) / 12
                    result["target_amount"] = monthly_expenses * 6
                if not result.get("timeframe_years"):
                    result["timeframe_years"] = 1  # Build emergency fund within 1 year
                    
            elif goal_type == "investment":
                if not result.get("timeframe_years"):
                    result["timeframe_years"] = 10  # Default 10-year investment horizon
                if not result.get("target_amount"):
                    result["target_amount"] = 100000  # Default $100k investment goal
                    
            elif goal_type == "debt_payoff":
                if not result.get("timeframe_years"):
                    result["timeframe_years"] = 3  # Default 3-year debt payoff plan
                    
            elif goal_type == "education":
                if not result.get("target_amount"):
                    result["target_amount"] = 80000  # Average college cost
                if not result.get("timeframe_years"):
                    result["timeframe_years"] = 10  # Default education savings timeline
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Error applying intelligent defaults: {e}")
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
        """Generate comprehensive narrative using enhanced templates with realistic financial advice"""
        try:
            goal_type = plan.get('goal_type', 'investment')
            target_amount = plan.get('target_amount')
            timeframe = plan.get('timeframe_years')
            risk_tolerance = plan.get('risk_tolerance', 'moderate')
            current_age = plan.get('current_age')
            retirement_age = plan.get('retirement_age')
            monthly_contribution = plan.get('monthly_contribution')
            current_savings = plan.get('current_savings')
            annual_income = plan.get('annual_income')

            # Use intelligent pattern matching for comprehensive responses
            if goal_type == 'retirement':
                return self._generate_retirement_narrative(
                    target_amount, timeframe, risk_tolerance, current_age, 
                    retirement_age, monthly_contribution, current_savings, annual_income
                )
            elif goal_type == 'emergency_fund':
                return self._generate_emergency_fund_narrative(
                    target_amount, annual_income, current_savings, monthly_contribution
                )
            elif goal_type == 'investment':
                return self._generate_investment_narrative(
                    target_amount, timeframe, risk_tolerance, monthly_contribution, current_savings
                )
            elif goal_type == 'debt_payoff':
                return self._generate_debt_payoff_narrative(
                    target_amount, timeframe, monthly_contribution
                )
            elif goal_type == 'education':
                return self._generate_education_narrative(
                    target_amount, timeframe, monthly_contribution, current_savings
                )
            else:
                return self._generate_general_financial_narrative(
                    target_amount, timeframe, risk_tolerance, monthly_contribution
                )

        except Exception as e:
            self.logger.error(f"Template narrative generation failed: {e}")
            return self._generate_fallback_narrative()

    def _generate_retirement_narrative(
        self, target_amount, timeframe, risk_tolerance, current_age, 
        retirement_age, monthly_contribution, current_savings, annual_income
    ) -> str:
        """Generate comprehensive retirement planning narrative with realistic calculations"""
        try:
            # Determine retirement scenario based on inputs
            age = current_age or 35
            ret_age = retirement_age or 65
            years_to_retirement = ret_age - age if current_age and retirement_age else timeframe or 30
            target = target_amount or 1000000
            monthly = monthly_contribution or 500
            current = current_savings or 0
            income = annual_income or 75000

            # Calculate realistic projections
            annual_contribution = monthly * 12
            total_contributions = annual_contribution * years_to_retirement
            
            # Assume 7% average return based on risk tolerance
            return_rate = {'conservative': 0.05, 'moderate': 0.07, 'aggressive': 0.09}.get(risk_tolerance, 0.07)
            
            # Future value calculation with compound interest
            if current > 0:
                current_future_value = current * ((1 + return_rate) ** years_to_retirement)
            else:
                current_future_value = 0
                
            # Future value of annuity (monthly contributions)
            if monthly > 0:
                monthly_rate = return_rate / 12
                months = years_to_retirement * 12
                contribution_future_value = monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate)
            else:
                contribution_future_value = 0
                
            projected_total = current_future_value + contribution_future_value
            
            # Determine if on track
            on_track = projected_total >= target * 0.9  # Within 90% is considered on track
            
            # Calculate 4% withdrawal rule
            annual_withdrawal = projected_total * 0.04
            monthly_retirement_income = annual_withdrawal / 12
            
            # Income replacement ratio
            replacement_ratio = (annual_withdrawal / income) * 100 if income > 0 else 0
            
            # Calculate inflation-adjusted value
            inflation_adjusted_value = target / pow(1.03, years_to_retirement)
            
            # Calculate additional monthly contribution needed if not on track
            additional_monthly = int((target - projected_total) / (years_to_retirement * 12)) if not on_track and years_to_retirement > 0 else 0

            narrative = f"""
## Retirement Planning Strategy

**Your Retirement Goal**: Retire at age {ret_age} with ${target:,.0f}

### Current Projection
Based on your current savings of ${current:,.0f} and planned monthly contributions of ${monthly:,.0f}, you're projected to have **${projected_total:,.0f}** by retirement.

### Analysis
{'✅ **You are on track!**' if on_track else '⚠️ **Adjustment needed**'} Your current strategy {'will likely meet' if on_track else 'may fall short of'} your retirement goal.

**Projected Retirement Income**: ${monthly_retirement_income:,.0f}/month (${annual_withdrawal:,.0f}/year)
**Income Replacement**: {replacement_ratio:.0f}% of your current income

### Investment Strategy ({risk_tolerance.title()} Risk Profile)
- **Asset Allocation**: {self._get_asset_allocation(risk_tolerance, years_to_retirement)}
- **Expected Return**: {return_rate*100:.1f}% annually
- **Total Contributions**: ${total_contributions:,.0f} over {years_to_retirement} years

### Key Milestones
- **Age {age + 10}**: Target ${(current_future_value + contribution_future_value * 10/years_to_retirement):,.0f}
- **Age {age + 20}**: Target ${(current_future_value + contribution_future_value * 20/years_to_retirement):,.0f} (halfway point)
- **Age {ret_age - 5}**: Final push - consider increasing contributions by 10-15%

### Actionable Next Steps
1. **Maximize employer 401(k) match** - This is free money (typically 50-100% return)
2. **Consider Roth IRA** - Tax-free growth for retirement ({self._get_roth_ira_advice(income)})
3. **Annual review** - Increase contributions by 3-5% each year or with raises
4. **Rebalance portfolio** - Review asset allocation every 6-12 months
{f'5. **Increase contributions** - Consider adding ${additional_monthly}/month to reach your goal' if not on_track else '5. **Stay the course** - Your current plan is solid'}

### Risk Considerations
- **Market volatility** - Expect 20-30% swings; stay disciplined during downturns
- **Inflation impact** - Your ${target:,.0f} goal equals ~${inflation_adjusted_value:,.0f} in today's purchasing power
- **Healthcare costs** - Budget $300,000+ for medical expenses in retirement

*This projection assumes a {return_rate*100:.1f}% average annual return and doesn't account for taxes or fees. Actual results may vary.*
"""
            return narrative.strip()
            
        except Exception as e:
            self.logger.error(f"Retirement narrative generation failed: {e}")
            return self._generate_basic_retirement_narrative()

    def _generate_emergency_fund_narrative(
        self, target_amount, annual_income, current_savings, monthly_contribution
    ) -> str:
        """Generate emergency fund planning narrative with practical advice"""
        try:
            income = annual_income or 60000
            monthly_income = income / 12
            target = target_amount or (monthly_income * 6)  # 6 months default
            current = current_savings or 0
            monthly = monthly_contribution or 200
            
            months_covered = target / monthly_income
            months_to_goal = max(0, (target - current) / monthly) if monthly > 0 else float('inf')
            
            narrative = f"""
## Emergency Fund Strategy

**Your Goal**: Build a ${target:,.0f} emergency fund ({months_covered:.1f} months of expenses)

### Current Status
- **Current Emergency Savings**: ${current:,.0f}
- **Monthly Contribution**: ${monthly:,.0f}
- **Time to Goal**: {months_to_goal:.0f} months ({months_to_goal/12:.1f} years)

### Why This Matters
An emergency fund protects you from financial setbacks like job loss, medical bills, or major repairs. Financial experts recommend 3-6 months of expenses, with 6+ months for:
- Self-employed individuals
- Single-income households  
- Volatile industries
- Those with dependents

### Optimal Strategy
**Phase 1: Quick Start** (First $1,000)
- Cut discretionary spending temporarily
- Sell unused items
- Use tax refunds or bonuses
- Target: 30-60 days

**Phase 2: Build Momentum** (Next ${min(target-1000, monthly_income*3):,.0f})
- Automate ${monthly:,.0f}/month transfers
- Use high-yield savings account (4-5% APY)
- Direct deposit split: {(monthly/monthly_income)*100:.1f}% to emergency fund

**Phase 3: Complete Fund** (Final ${max(0, target-monthly_income*3-1000):,.0f})
- Maintain consistent contributions
- Resist temptation to use for non-emergencies
- Consider money market or CD ladder

### Smart Storage Options
1. **High-Yield Savings** - 4-5% APY, instant access
2. **Money Market Account** - Slightly higher rates, limited transactions
3. **Short-term CDs** - Higher rates, 3-12 month terms for portion of fund

### What Qualifies as an Emergency
✅ **True Emergencies**:
- Job loss or income reduction
- Medical emergencies not covered by insurance
- Major home/car repairs for safety
- Family emergencies requiring travel

❌ **Not Emergencies**:
- Vacations or holidays
- Shopping sales or "deals"
- Routine maintenance
- Lifestyle upgrades

### Actionable Next Steps
1. **Open high-yield savings account** - Earn 10x more than traditional savings
2. **Automate transfers** - Set up automatic ${monthly:,.0f} monthly transfer
3. **Track progress** - Review monthly, celebrate milestones
4. **Adjust as needed** - Increase contributions with raises or windfalls
5. **Protect the fund** - Only use for true emergencies, replenish immediately

### Milestone Celebrations
- **$1,000**: You're ahead of 40% of Americans! 🎉
- **${monthly_income:,.0f}**: One month covered - major progress!
- **${monthly_income*3:,.0f}**: Three months - you're in the safe zone
- **${target:,.0f}**: Full fund complete - financial security achieved! 🏆

*Remember: This fund is insurance, not an investment. Prioritize accessibility over returns.*
"""
            return narrative.strip()
            
        except Exception as e:
            self.logger.error(f"Emergency fund narrative generation failed: {e}")
            return self._generate_basic_emergency_fund_narrative()

    def _generate_investment_narrative(
        self, target_amount, timeframe, risk_tolerance, monthly_contribution, current_savings
    ) -> str:
        """Generate investment planning narrative with realistic portfolio advice"""
        try:
            target = target_amount or 100000
            years = timeframe or 10
            monthly = monthly_contribution or 500
            current = current_savings or 0
            risk = risk_tolerance or 'moderate'
            
            # Calculate projections
            return_rate = {'conservative': 0.06, 'moderate': 0.08, 'aggressive': 0.10}.get(risk, 0.08)
            annual_contribution = monthly * 12
            
            # Future value calculations
            current_future_value = current * ((1 + return_rate) ** years) if current > 0 else 0
            monthly_rate = return_rate / 12
            months = years * 12
            contribution_future_value = monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate) if monthly > 0 else 0
            projected_total = current_future_value + contribution_future_value
            
            on_track = projected_total >= target * 0.9
            
            # Calculate gap and additional monthly needed
            gap_amount = target - projected_total if not on_track else 0
            additional_monthly_needed = int(gap_amount / months) if not on_track and months > 0 else 0

            narrative = f"""
## Investment Growth Strategy

**Your Goal**: Grow your investments to ${target:,.0f} over {years} years

### Projection Analysis
- **Starting Amount**: ${current:,.0f}
- **Monthly Contributions**: ${monthly:,.0f}
- **Expected Return**: {return_rate*100:.1f}% annually ({risk} risk profile)
- **Projected Value**: ${projected_total:,.0f}

{'✅ **On Track**: Your strategy should meet your goal!' if on_track else f'⚠️ **Gap Alert**: You may fall ${gap_amount:,.0f} short of your goal.'}

### Recommended Portfolio Allocation ({risk.title()} Risk)
{self._get_detailed_asset_allocation(risk, years)}

### Investment Account Strategy
**Tax-Advantaged Accounts (Priority Order)**:
1. **401(k) with match** - Contribute enough for full employer match
2. **Roth IRA** - {self._get_ira_limit()} annual limit, tax-free growth
3. **Traditional IRA** - Tax deduction now, taxed in retirement
4. **Taxable brokerage** - For amounts exceeding retirement account limits

### Dollar-Cost Averaging Benefits
Your ${monthly:,.0f} monthly investment strategy provides:
- **Reduced timing risk** - Automatic buying at various market levels
- **Emotional discipline** - Removes guesswork and fear-based decisions
- **Compound growth** - Earlier investments have more time to grow

### Expected Milestones
- **Year 2**: ~${(current_future_value * pow(1.08, 2) + contribution_future_value * 2/years):,.0f}
- **Year 5**: ~${(current_future_value * pow(1.08, 5) + contribution_future_value * 5/years):,.0f} (halfway point)
- **Year {years-2}**: ~${projected_total * 0.85:,.0f} (final push phase)

### Market Volatility Expectations
- **Normal years**: Expect 6-12% returns
- **Good years**: 15-25% returns (don't get overconfident)
- **Bad years**: -10% to -20% (stay the course!)
- **Historical average**: ~10% for diversified stock portfolios

### Actionable Next Steps
1. **Open investment accounts** - Start with low-cost index funds (expense ratios <0.1%)
2. **Automate investments** - Set up automatic monthly transfers
3. **Diversify globally** - Include international stocks (20-30% allocation)
4. **Rebalance annually** - Maintain target allocation as markets move
5. **Increase contributions** - Boost by 3-5% annually or with raises
{f'6. **Bridge the gap** - Consider increasing monthly contributions by ${additional_monthly_needed} to reach your goal' if not on_track else '6. **Stay disciplined** - Your plan is solid, stick with it through market cycles'}

### Tax Optimization Tips
- **Tax-loss harvesting** - Offset gains with losses in taxable accounts
- **Asset location** - Hold tax-inefficient investments in tax-advantaged accounts
- **Roth conversions** - Consider converting traditional IRA funds during low-income years

### Red Flags to Avoid
❌ **High fees** - Avoid funds with expense ratios >0.5%
❌ **Market timing** - Don't try to predict market movements
❌ **Emotional decisions** - Stick to your plan during market volatility
❌ **Overconcentration** - Don't put >5% in any single stock

*Past performance doesn't guarantee future results. Diversification doesn't eliminate risk.*
"""
            return narrative.strip()
            
        except Exception as e:
            self.logger.error(f"Investment narrative generation failed: {e}")
            return self._generate_basic_investment_narrative()

    def _generate_debt_payoff_narrative(self, target_amount, timeframe, monthly_contribution) -> str:
        """Generate debt payoff strategy narrative"""
        try:
            debt_amount = target_amount or 25000
            months = (timeframe * 12) if timeframe else 36
            monthly_payment = monthly_contribution or 500
            
            # Assume average credit card rate for calculations
            interest_rate = 0.18  # 18% APR
            monthly_rate = interest_rate / 12
            
            # Calculate total interest if paying minimum vs aggressive payoff
            total_interest = (monthly_payment * months) - debt_amount
            
            narrative = f"""
## Debt Elimination Strategy

**Your Goal**: Pay off ${debt_amount:,.0f} in debt over {months/12:.1f} years

### Payoff Analysis
- **Monthly Payment**: ${monthly_payment:,.0f}
- **Estimated Interest**: ${max(0, total_interest):,.0f}
- **Total Paid**: ${debt_amount + max(0, total_interest):,.0f}

### Debt Avalanche vs. Snowball Method
**Avalanche Method** (Mathematically optimal):
- Pay minimums on all debts
- Put extra money toward highest interest rate debt
- Saves the most money overall

**Snowball Method** (Psychologically motivating):
- Pay minimums on all debts  
- Put extra money toward smallest balance
- Builds momentum with quick wins

### Acceleration Strategies
1. **Balance transfers** - Move high-interest debt to 0% APR cards
2. **Debt consolidation loan** - Lower interest rate, fixed payment
3. **Side income** - Direct all extra earnings to debt
4. **Expense cuts** - Redirect savings to debt payments
5. **Windfalls** - Use tax refunds, bonuses for large payments

### Actionable Next Steps
1. **List all debts** - Balance, minimum payment, interest rate
2. **Choose your method** - Avalanche for savings, snowball for motivation
3. **Automate payments** - Never miss a payment (late fees add up)
4. **Track progress** - Celebrate each paid-off account
5. **Avoid new debt** - Cut up credit cards if necessary

### After Debt Freedom
Once debt-free, redirect your ${monthly_payment:,.0f} monthly payment to:
- Emergency fund (if not established)
- Retirement contributions
- Investment accounts
- Other financial goals

*Becoming debt-free is one of the most impactful financial moves you can make.*
"""
            return narrative.strip()
            
        except Exception as e:
            self.logger.error(f"Debt payoff narrative generation failed: {e}")
            return "## Debt Elimination Strategy\n\nWe'll help you create a systematic approach to eliminate your debt efficiently."

    def _generate_education_narrative(self, target_amount, timeframe, monthly_contribution, current_savings) -> str:
        """Generate education funding narrative"""
        try:
            target = target_amount or 80000
            years = timeframe or 10
            monthly = monthly_contribution or 300
            current = current_savings or 0
            
            narrative = f"""
## Education Funding Strategy

**Your Goal**: Save ${target:,.0f} for education expenses over {years} years

### Current Progress
- **Current Savings**: ${current:,.0f}
- **Monthly Contributions**: ${monthly:,.0f}
- **Time Horizon**: {years} years

### Tax-Advantaged Education Accounts
**529 Education Savings Plan**:
- Tax-free growth and withdrawals for qualified expenses
- State tax deductions in many states
- High contribution limits
- Can be used for K-12 tuition (up to $10,000/year)

**Coverdell ESA**:
- $2,000 annual contribution limit
- Tax-free growth for education expenses
- More investment options than 529 plans

### Investment Strategy
For {years}-year timeline:
- **Conservative allocation** (if <5 years): 70% bonds, 30% stocks
- **Moderate allocation** (5-10 years): 50% stocks, 50% bonds  
- **Growth allocation** (>10 years): 70% stocks, 30% bonds

### Cost-Saving Strategies
1. **Community college** - Save 50-70% on first two years
2. **In-state tuition** - Significant savings over out-of-state
3. **Merit scholarships** - Maintain good grades for awards
4. **Work-study programs** - Reduce borrowing needs
5. **AP/dual enrollment** - Earn college credit in high school

### Actionable Next Steps
1. **Open 529 plan** - Research your state's plan for tax benefits
2. **Automate contributions** - Set up monthly transfers
3. **Involve family** - Grandparents can contribute directly
4. **Research schools** - Understand realistic cost expectations
5. **Apply for aid** - Complete FAFSA for federal aid eligibility

*Education is an investment in the future - start early for maximum benefit.*
"""
            return narrative.strip()
            
        except Exception as e:
            self.logger.error(f"Education narrative generation failed: {e}")
            return "## Education Funding Strategy\n\nWe'll help you save systematically for education expenses."

    def _generate_general_financial_narrative(
        self, target_amount, timeframe, risk_tolerance, monthly_contribution
    ) -> str:
        """Generate general financial planning narrative"""
        try:
            target = target_amount or 50000
            years = timeframe or 5
            monthly = monthly_contribution or 400
            risk = risk_tolerance or 'moderate'
            
            # Get expected return range
            return_ranges = {'conservative': '5-7%', 'moderate': '7-9%', 'aggressive': '9-12%'}
            expected_return = return_ranges.get(risk, '7-9%')
            
            narrative = f"""
## Financial Planning Strategy

**Your Goal**: Achieve ${target:,.0f} over {years} years with {risk} risk tolerance

### Strategy Overview
Your {risk} approach balances growth potential with risk management, suitable for your {years}-year timeline.

### Recommended Approach
- **Monthly Contribution**: ${monthly:,.0f}
- **Investment Mix**: {self._get_asset_allocation(risk, years)}
- **Expected Return**: {expected_return} annually

### Key Principles
1. **Start early** - Time is your greatest asset
2. **Stay consistent** - Regular contributions beat timing the market
3. **Diversify** - Don't put all eggs in one basket
4. **Keep costs low** - High fees erode returns
5. **Stay disciplined** - Stick to your plan through market cycles

### Next Steps
1. **Define specific goals** - What is this money for?
2. **Choose appropriate accounts** - Tax-advantaged when possible
3. **Select investments** - Low-cost, diversified options
4. **Automate everything** - Remove emotion from investing
5. **Review regularly** - Annual check-ups and adjustments

*Successful investing is about time in the market, not timing the market.*
"""
            return narrative.strip()
            
        except Exception as e:
            self.logger.error(f"General narrative generation failed: {e}")
            return self._generate_fallback_narrative()

    def _get_asset_allocation(self, risk_tolerance: str, years_to_goal: int) -> str:
        """Get appropriate asset allocation based on risk tolerance and timeline"""
        if years_to_goal <= 3:
            return "70% bonds/cash, 30% stocks (capital preservation focus)"
        elif years_to_goal <= 7:
            allocations = {
                'conservative': "60% bonds, 40% stocks",
                'moderate': "50% bonds, 50% stocks", 
                'aggressive': "40% bonds, 60% stocks"
            }
        else:
            allocations = {
                'conservative': "40% bonds, 60% stocks",
                'moderate': "30% bonds, 70% stocks",
                'aggressive': "20% bonds, 80% stocks"
            }
        return allocations.get(risk_tolerance, "50% bonds, 50% stocks")

    def _get_detailed_asset_allocation(self, risk_tolerance: str, years: int) -> str:
        """Get detailed asset allocation with specific recommendations"""
        if risk_tolerance == 'conservative':
            return """
**Conservative Portfolio (Lower risk, steady growth)**:
- 40% U.S. Total Stock Market Index
- 20% International Developed Markets
- 30% U.S. Aggregate Bond Index  
- 10% High-Yield Savings/CDs

*Expected return: 5-7% annually with lower volatility*"""
        elif risk_tolerance == 'aggressive':
            return """
**Aggressive Portfolio (Higher risk, growth focus)**:
- 50% U.S. Total Stock Market Index
- 25% International Developed Markets
- 10% Emerging Markets
- 10% Real Estate Investment Trusts (REITs)
- 5% U.S. Aggregate Bond Index

*Expected return: 9-12% annually with higher volatility*"""
        else:  # moderate
            return """
**Moderate Portfolio (Balanced approach)**:
- 45% U.S. Total Stock Market Index
- 20% International Developed Markets
- 5% Emerging Markets
- 25% U.S. Aggregate Bond Index
- 5% Real Estate Investment Trusts (REITs)

*Expected return: 7-9% annually with moderate volatility*"""

    def _get_roth_ira_advice(self, income: int) -> str:
        """Get Roth IRA advice based on income level"""
        if income < 125000:
            return "Full $6,500 contribution allowed"
        elif income < 140000:
            return "Partial contribution allowed (phase-out range)"
        else:
            return "Consider backdoor Roth IRA conversion"

    def _get_ira_limit(self) -> str:
        """Get current IRA contribution limit"""
        return "$6,500 (or $7,500 if 50+)"

    def _generate_basic_retirement_narrative(self) -> str:
        """Basic fallback retirement narrative"""
        return """
## Retirement Planning Strategy

Building a secure retirement requires consistent saving and smart investing over time.

### Key Principles
- Start early to benefit from compound growth
- Contribute regularly to retirement accounts
- Diversify your investment portfolio
- Take advantage of employer matching
- Review and adjust your plan annually

### Next Steps
1. Maximize employer 401(k) match
2. Consider opening an IRA
3. Increase contributions with salary raises
4. Review investment allocation regularly

*The earlier you start, the more time your money has to grow.*
"""

    def _generate_basic_emergency_fund_narrative(self) -> str:
        """Basic fallback emergency fund narrative"""
        return """
## Emergency Fund Strategy

An emergency fund provides financial security for unexpected expenses.

### Recommended Amount
- 3-6 months of living expenses
- Higher amounts for variable income
- Start with $1,000 as initial goal

### Next Steps
1. Open high-yield savings account
2. Automate monthly contributions
3. Keep funds easily accessible
4. Only use for true emergencies

*An emergency fund is the foundation of financial security.*
"""

    def _generate_basic_investment_narrative(self) -> str:
        """Basic fallback investment narrative"""
        return """
## Investment Growth Strategy

Building wealth through investing requires patience, consistency, and diversification.

### Key Principles
- Invest regularly regardless of market conditions
- Diversify across asset classes
- Keep costs low with index funds
- Stay disciplined during market volatility

### Next Steps
1. Open investment accounts
2. Choose low-cost, diversified funds
3. Automate monthly contributions
4. Rebalance annually

*Time in the market beats timing the market.*
"""

    def _generate_fallback_narrative(self) -> str:
        """Ultimate fallback narrative when all else fails"""
        return """
## Financial Plan Summary

We've created a financial plan based on your objectives. 

### Next Steps
1. Review the detailed plan breakdown
2. Set up automatic contributions if applicable  
3. Schedule regular reviews to track progress
4. Adjust as life circumstances change

This plan is designed to be flexible and adapt to market conditions and your evolving needs.
"""

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
        """Explain scenario using comprehensive templates with realistic financial analysis"""
        try:
            scenario_type = scenario.get('type', 'market_change')
            severity = scenario.get('severity', 'moderate')
            description = scenario.get('description', '')

            # Use intelligent pattern matching for comprehensive scenario explanations
            if 'market' in scenario_type.lower() or 'crash' in scenario_type.lower():
                return self._explain_market_scenario(scenario_type, severity, impact, description)
            elif 'job' in scenario_type.lower() or 'unemployment' in scenario_type.lower():
                return self._explain_job_loss_scenario(severity, impact, description)
            elif 'inflation' in scenario_type.lower():
                return self._explain_inflation_scenario(severity, impact, description)
            elif 'interest' in scenario_type.lower() or 'rate' in scenario_type.lower():
                return self._explain_interest_rate_scenario(severity, impact, description)
            elif 'recession' in scenario_type.lower():
                return self._explain_recession_scenario(severity, impact, description)
            elif 'health' in scenario_type.lower() or 'medical' in scenario_type.lower():
                return self._explain_health_scenario(severity, impact, description)
            else:
                return self._explain_general_scenario(scenario_type, severity, impact, description)

        except Exception as e:
            self.logger.error(f"Template scenario explanation failed: {e}")
            return self._generate_fallback_scenario_explanation()

    def _explain_market_scenario(self, scenario_type: str, severity: str, impact: Dict[str, Any], description: str) -> str:
        """Explain market crash or volatility scenarios with detailed analysis"""
        try:
            # Determine market decline percentage based on severity
            decline_map = {
                'mild': ('5-10%', '2-6 months', 'normal market correction'),
                'moderate': ('15-25%', '6-18 months', 'significant market downturn'),
                'severe': ('30-50%', '1-3 years', 'major market crash similar to 2008')
            }
            
            decline_pct, recovery_time, scenario_desc = decline_map.get(severity.lower(), decline_map['moderate'])
            
            # Extract impact data safely
            target_change = impact.get('target_amount_change', 0)
            timeframe_change = impact.get('timeframe_change', 0)
            probability_success = impact.get('probability_success', 75)
            
            explanation = f"""
## What-If Analysis: Market Downturn ({severity.title()})

### Scenario Overview
**Market Decline**: {decline_pct} drop in stock market values
**Expected Duration**: {recovery_time}
**Historical Context**: This represents a {scenario_desc}

### Impact on Your Portfolio
- **Immediate Effect**: Portfolio value would decline by approximately {decline_pct}
- **Recovery Timeline**: {recovery_time} for markets to return to previous highs
- **Success Probability**: {probability_success}% chance of meeting long-term goals

### What This Means for You
**Short-term (0-2 years)**:
- Your account balances will show paper losses
- Don't panic - these are unrealized losses until you sell
- Continue regular contributions to buy at lower prices

**Medium-term (2-5 years)**:
- Markets typically recover within this timeframe
- Dollar-cost averaging works in your favor during recovery
- Rebalancing opportunities as markets recover

**Long-term (5+ years)**:
- Historical data shows markets recover and reach new highs
- Your regular contributions during the downturn will show significant gains
- Staying invested is crucial for long-term success

### Historical Perspective
**Major Market Declines & Recoveries**:
- **2020 COVID Crash**: -34% decline, recovered in 5 months
- **2008 Financial Crisis**: -57% decline, recovered in 4 years  
- **2000 Dot-com Crash**: -49% decline, recovered in 7 years
- **1987 Black Monday**: -22% in one day, recovered in 2 years

### Recommended Actions
**✅ DO**:
1. **Stay the course** - Continue your regular investment schedule
2. **Rebalance if needed** - Sell high-performing bonds, buy discounted stocks
3. **Increase contributions** - If possible, invest more during the downturn
4. **Review timeline** - Ensure you don't need the money for 5+ years
5. **Focus on fundamentals** - Companies continue to operate and grow

**❌ DON'T**:
1. **Panic sell** - This locks in losses and misses the recovery
2. **Stop contributing** - You'll miss buying opportunities
3. **Try to time the bottom** - Impossible to predict market timing
4. **Check balances daily** - Reduces emotional stress
5. **Make major changes** - Stick to your long-term plan

### Stress Testing Your Plan
If this scenario occurred:
- **Portfolio Impact**: Temporary decline of {decline_pct}
- **Recovery Strategy**: Continue contributions, rebalance annually
- **Timeline Adjustment**: {f'May extend timeline by {abs(timeframe_change)} months' if timeframe_change > 0 else 'Timeline likely unchanged with continued contributions'}
- **Risk Mitigation**: Diversification helps reduce impact

### Emotional Preparation
Market downturns test your discipline more than your portfolio. Remember:
- **Volatility is normal** - It's the price of long-term growth
- **Time heals wounds** - Every major decline has been followed by recovery
- **Opportunity in crisis** - Lower prices mean better future returns
- **Stay focused** - Your goals haven't changed, just the path to get there

*Historical market data shows that staying invested through downturns is crucial for long-term success.*
"""
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Market scenario explanation failed: {e}")
            return self._generate_basic_market_scenario()

    def _explain_job_loss_scenario(self, severity: str, impact: Dict[str, Any], description: str) -> str:
        """Explain job loss scenarios with practical financial guidance"""
        try:
            # Determine unemployment duration based on severity
            duration_map = {
                'mild': ('1-3 months', 'brief job transition'),
                'moderate': ('3-6 months', 'typical job search period'),
                'severe': ('6-12+ months', 'extended unemployment period')
            }
            
            duration, scenario_desc = duration_map.get(severity.lower(), duration_map['moderate'])
            
            explanation = f"""
## What-If Analysis: Job Loss ({severity.title()})

### Scenario Overview
**Unemployment Duration**: {duration}
**Situation**: {scenario_desc}
**Income Impact**: Complete loss of primary income source

### Immediate Financial Impact
**Monthly Budget Changes**:
- **Lost Income**: Primary salary no longer available
- **Reduced Expenses**: Commuting, work clothes, meals out
- **New Expenses**: COBRA health insurance, job search costs
- **Emergency Fund Usage**: Primary source of income replacement

### Financial Survival Strategy

**Phase 1: Immediate Actions (Week 1)**
1. **File for unemployment** - Apply immediately, benefits take 2-4 weeks
2. **Review emergency fund** - Calculate months of coverage available
3. **Contact creditors** - Explain situation, request payment deferrals
4. **Cut non-essential spending** - Cancel subscriptions, reduce discretionary expenses
5. **Apply for COBRA** - Maintain health insurance (expensive but necessary)

**Phase 2: Stabilization (Weeks 2-4)**
1. **Create bare-bones budget** - Focus on housing, food, utilities, insurance
2. **Explore income sources** - Freelancing, part-time work, gig economy
3. **Network actively** - Contact professional connections, update LinkedIn
4. **Skill development** - Use time for certifications or training
5. **Consider relocation** - If job market is better elsewhere

**Phase 3: Extended Period (Months 2+)**
1. **Expand job search** - Consider different roles, industries, locations
2. **Manage stress** - Job loss affects mental health and decision-making
3. **Preserve retirement funds** - Avoid early 401(k) withdrawals if possible
4. **Consider career pivot** - Use time to explore new opportunities
5. **Maintain professional image** - Stay active in industry events

### Financial Priorities During Unemployment
**Essential Expenses (Pay First)**:
1. Housing (rent/mortgage)
2. Utilities (electricity, water, gas)
3. Food and basic necessities
4. Health insurance
5. Transportation (if needed for job search)
6. Minimum debt payments

**Non-Essential Expenses (Pause if Needed)**:
- Dining out and entertainment
- Gym memberships and subscriptions
- Investment contributions (temporarily)
- Travel and vacations
- Major purchases

### Emergency Fund Strategy
**3-Month Fund**: Covers basic scenario, provides breathing room
**6-Month Fund**: Handles moderate job search timeline
**12-Month Fund**: Provides security for extended unemployment or career change

**Stretching Your Emergency Fund**:
- Apply for unemployment benefits immediately
- Reduce expenses by 30-50% from normal budget
- Consider temporary income sources
- Negotiate payment deferrals with creditors

### Income Replacement Options
**Unemployment Benefits**: Typically 40-50% of previous salary for 26 weeks
**Freelance/Consulting**: Use existing skills for project-based income
**Part-time Work**: Maintain some income while job searching
**Gig Economy**: Uber, DoorDash, TaskRabbit for flexible income
**Severance Package**: Negotiate if possible during layoff

### Protecting Your Financial Future
**Retirement Accounts**:
- **Don't withdraw early** - 10% penalty plus taxes
- **Consider loan option** - If 401(k) allows, better than withdrawal
- **Roll over to IRA** - Maintain tax-advantaged status

**Health Insurance**:
- **COBRA continuation** - Expensive but maintains coverage
- **ACA marketplace** - May be cheaper, especially with subsidies
- **Spouse's plan** - If available, often most cost-effective

### Actionable Recovery Plan
**Week 1-2**: File unemployment, assess finances, cut expenses
**Week 3-4**: Intensive job search, networking, skill assessment
**Month 2**: Expand search criteria, consider temporary income
**Month 3+**: Evaluate career pivot options, consider relocation

### Emotional and Practical Support
**Professional Resources**:
- Career counseling services
- Industry networking groups
- Professional development courses
- Job placement agencies

**Personal Support**:
- Maintain routines and structure
- Stay physically and mentally healthy
- Communicate with family about situation
- Consider counseling if stress becomes overwhelming

### Prevention for the Future
**Build Stronger Emergency Fund**: Target 6-12 months of expenses
**Diversify Income**: Develop side hustles or passive income
**Maintain Network**: Stay connected even when employed
**Keep Skills Current**: Continuous learning and development
**Update Resume Regularly**: Don't wait until you need it

*Job loss is temporary, but the financial habits you build during this time can strengthen your long-term security.*
"""
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Job loss scenario explanation failed: {e}")
            return self._generate_basic_job_loss_scenario()

    def _explain_inflation_scenario(self, severity: str, impact: Dict[str, Any], description: str) -> str:
        """Explain inflation scenarios with purchasing power analysis"""
        try:
            # Determine inflation rates based on severity
            inflation_map = {
                'mild': ('4-6%', 'elevated but manageable'),
                'moderate': ('7-10%', 'significantly above normal'),
                'severe': ('10%+', 'high inflation period')
            }
            
            inflation_rate, scenario_desc = inflation_map.get(severity.lower(), inflation_map['moderate'])
            
            explanation = f"""
## What-If Analysis: High Inflation ({severity.title()})

### Scenario Overview
**Inflation Rate**: {inflation_rate} annually
**Context**: {scenario_desc} inflation environment
**Historical Reference**: Similar to 1970s-1980s inflation periods

### Impact on Your Finances
**Purchasing Power Erosion**:
- Your money buys less over time
- Fixed incomes lose value
- Cash savings lose real value
- Debt becomes cheaper to repay (if fixed-rate)

**Price Increases Across Categories**:
- **Housing**: Rent and home prices typically rise with inflation
- **Food**: Grocery costs increase, eating out becomes more expensive
- **Energy**: Gas, electricity, heating costs rise significantly
- **Healthcare**: Medical costs often outpace general inflation
- **Education**: Tuition and fees continue increasing

### Investment Strategy Adjustments
**Inflation-Protected Assets**:
1. **TIPS (Treasury Inflation-Protected Securities)** - Principal adjusts with inflation
2. **Real Estate** - Property values and rents typically rise with inflation
3. **Commodities** - Gold, oil, agricultural products hedge against inflation
4. **Stocks** - Companies can raise prices, protecting real returns
5. **I Bonds** - Government bonds that adjust for inflation

**Assets to Avoid**:
- **Long-term bonds** - Fixed payments lose purchasing power
- **Cash equivalents** - Savings accounts don't keep up with inflation
- **Fixed annuities** - Payments become worth less over time

### Budget Adaptation Strategies
**Income Protection**:
- Negotiate salary increases tied to inflation
- Seek jobs with cost-of-living adjustments
- Develop inflation-resistant income streams
- Consider variable-rate income sources

**Expense Management**:
- **Lock in fixed costs** - Fixed-rate mortgages, long-term contracts
- **Reduce discretionary spending** - Focus on necessities
- **Buy in bulk** - Stock up on non-perishables when prices are lower
- **Energy efficiency** - Reduce utility costs through conservation

### Debt Strategy in High Inflation
**Fixed-Rate Debt** (Advantage):
- Your mortgage payment stays the same while your income (hopefully) rises
- You're paying back with "cheaper" dollars
- Don't rush to pay off low fixed-rate debt

**Variable-Rate Debt** (Disadvantage):
- Credit card rates will increase
- Variable mortgages become more expensive
- Prioritize paying off variable-rate debt

### Long-Term Financial Planning
**Retirement Planning Adjustments**:
- Increase contribution rates to maintain purchasing power
- Focus on growth investments over fixed income
- Plan for higher healthcare costs in retirement
- Consider delaying retirement if savings are insufficient

**Emergency Fund Considerations**:
- May need larger emergency fund due to higher costs
- Keep emergency fund in high-yield accounts
- Consider short-term CDs or money market accounts
- Don't let cash sit in low-yield savings

### Historical Context and Lessons
**1970s-1980s Inflation Period**:
- Peak inflation reached 14.8% in 1980
- Fed raised interest rates to 20% to combat inflation
- Stocks initially struggled but eventually recovered
- Real estate and commodities performed well

**Successful Strategies from History**:
- Investors who stayed in stocks did well long-term
- Real estate owners benefited from rising property values
- Fixed-rate mortgage holders paid off loans with cheaper dollars
- Those who panicked and sold stocks missed the recovery

### Actionable Steps
**Immediate (0-3 months)**:
1. **Review and adjust budget** - Account for rising costs
2. **Negotiate salary increase** - Present inflation-based case
3. **Refinance to fixed rates** - Lock in current rates if variable
4. **Increase investment contributions** - Maintain real purchasing power

**Medium-term (3-12 months)**:
1. **Rebalance portfolio** - Increase inflation-hedged assets
2. **Consider real estate** - If financially feasible
3. **Develop additional income** - Side hustles, skills development
4. **Review insurance coverage** - Ensure adequate protection

**Long-term (1+ years)**:
1. **Stay disciplined with investments** - Don't panic sell
2. **Focus on real returns** - Returns above inflation rate
3. **Maintain emergency fund** - Adjust for higher costs
4. **Plan for continued inflation** - Don't assume it's temporary

### Psychological Preparation
**Managing Inflation Anxiety**:
- Focus on what you can control (spending, investing, income)
- Remember that inflation periods eventually end
- Don't make drastic changes based on fear
- Maintain long-term perspective on investments

*Inflation is challenging but manageable with proper planning and disciplined execution.*
"""
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Inflation scenario explanation failed: {e}")
            return self._generate_basic_inflation_scenario()

    def _explain_interest_rate_scenario(self, severity: str, impact: Dict[str, Any], description: str) -> str:
        """Explain interest rate change scenarios"""
        try:
            rate_map = {
                'mild': ('1-2%', 'gradual adjustment'),
                'moderate': ('2-4%', 'significant change'),
                'severe': ('4%+', 'dramatic shift')
            }
            
            rate_change, scenario_desc = rate_map.get(severity.lower(), rate_map['moderate'])
            
            explanation = f"""
## What-If Analysis: Interest Rate Changes ({severity.title()})

### Scenario Overview
**Rate Change**: {rate_change} increase in interest rates
**Context**: {scenario_desc} in monetary policy
**Timeline**: Changes typically occur over 6-18 months

### Impact on Different Financial Areas

**Borrowing Costs (Negative Impact)**:
- **Credit cards**: Rates increase immediately for variable balances
- **Mortgages**: New loans more expensive, refinancing less attractive
- **Auto loans**: Higher monthly payments for new purchases
- **Personal loans**: Increased cost of borrowing

**Savings Returns (Positive Impact)**:
- **High-yield savings**: Rates increase, better returns on cash
- **CDs**: New certificates offer higher rates
- **Money market**: Better yields on liquid savings
- **Bond funds**: New bonds offer higher yields

### Investment Portfolio Effects
**Bond Investments**:
- **Existing bonds**: Values decrease as rates rise
- **New bonds**: Higher yields available for new purchases
- **Bond funds**: Short-term losses, long-term gains from higher yields
- **Duration risk**: Longer-term bonds more affected

**Stock Market Impact**:
- **Growth stocks**: Often decline due to higher discount rates
- **Value stocks**: May be less affected or benefit
- **Dividend stocks**: Become less attractive vs. bonds
- **Financial sector**: Banks often benefit from higher rates

### Strategic Adjustments
**Debt Management**:
1. **Pay down variable debt** - Prioritize credit cards, HELOCs
2. **Consider refinancing** - Lock in rates before further increases
3. **Avoid new debt** - Unless absolutely necessary
4. **Fixed vs. variable** - Choose fixed rates for new borrowing

**Investment Strategy**:
1. **Shorten bond duration** - Reduce interest rate sensitivity
2. **Consider bond ladders** - Systematic reinvestment at higher rates
3. **Rebalance gradually** - Don't make dramatic changes
4. **Focus on quality** - Higher rates stress weaker companies

### Opportunities in Rising Rate Environment
**Savings Strategy**:
- **Shop for better rates** - Banks compete more aggressively
- **Consider CDs** - Lock in higher rates for guaranteed returns
- **Build cash reserves** - Higher returns make cash more attractive
- **Emergency fund growth** - Earn more on required cash holdings

**Investment Opportunities**:
- **New bond purchases** - Higher yields for income investors
- **Financial sector stocks** - Banks, insurance companies may benefit
- **Value investing** - Growth premiums may compress
- **International diversification** - Different rate environments globally

### Timeline and Expectations
**Short-term (0-6 months)**:
- Immediate impact on variable-rate debt
- Bond portfolio values decline
- Savings rates begin to increase
- Market volatility as investors adjust

**Medium-term (6-18 months)**:
- Full impact on borrowing costs realized
- Savings rates reach new equilibrium
- Investment portfolios adjust to new environment
- Economic effects become apparent

**Long-term (18+ months)**:
- New normal established for rates
- Investment returns reflect new rate environment
- Economic growth may slow due to higher borrowing costs
- Inflation may moderate due to tighter monetary policy

### Actionable Response Plan
**Immediate Actions**:
1. **Review variable-rate debt** - Calculate increased payments
2. **Shop savings rates** - Move money to higher-yielding accounts
3. **Assess refinancing** - Act quickly if beneficial
4. **Don't panic sell bonds** - Consider your timeline and needs

**Medium-term Adjustments**:
1. **Rebalance bond portfolio** - Shorter duration, higher quality
2. **Increase emergency fund yield** - Take advantage of higher rates
3. **Review mortgage strategy** - Fixed vs. variable considerations
4. **Adjust investment timeline** - Account for potential volatility

### Historical Perspective
**Previous Rate Cycles**:
- **2004-2006**: Fed raised rates from 1% to 5.25%
- **2015-2018**: Gradual increases from 0% to 2.5%
- **2022-2023**: Rapid increases from 0% to 5%+

**Lessons Learned**:
- Markets eventually adjust to new rate environments
- Quality investments tend to outperform during transitions
- Patience and discipline are rewarded
- Opportunities emerge for prepared investors

*Interest rate changes create both challenges and opportunities - the key is strategic adaptation.*
"""
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Interest rate scenario explanation failed: {e}")
            return "## Interest Rate Analysis\n\nRising interest rates affect both borrowing costs and investment returns. We'll help you navigate these changes strategically."

    def _explain_recession_scenario(self, severity: str, impact: Dict[str, Any], description: str) -> str:
        """Explain recession scenarios with economic context"""
        try:
            recession_map = {
                'mild': ('6-12 months', '2-5%', 'brief economic slowdown'),
                'moderate': ('12-18 months', '5-10%', 'typical recession'),
                'severe': ('18+ months', '10%+', 'severe economic downturn')
            }
            
            duration, unemployment_rate, scenario_desc = recession_map.get(severity.lower(), recession_map['moderate'])
            
            explanation = f"""
## What-If Analysis: Economic Recession ({severity.title()})

### Scenario Overview
**Duration**: {duration}
**Unemployment Rate**: {unemployment_rate}
**Economic Impact**: {scenario_desc}
**GDP Decline**: Negative growth for 2+ consecutive quarters

### Multi-Faceted Impact on Your Finances

**Employment and Income**:
- **Job security concerns** - Layoffs and hiring freezes increase
- **Wage stagnation** - Salary increases become rare
- **Reduced hours** - Part-time work or furloughs possible
- **Bonus cuts** - Performance-based compensation decreases

**Investment Portfolio Effects**:
- **Stock market decline** - Typically 20-50% from peak
- **Bond performance** - Government bonds often perform well
- **Real estate impact** - Property values may decline
- **Retirement accounts** - Significant paper losses likely

**Credit and Lending**:
- **Tighter lending standards** - Harder to qualify for loans
- **Higher credit requirements** - Banks become more selective
- **Reduced credit limits** - Existing lines may be cut
- **Business credit crunch** - Affects economic recovery

### Recession-Proof Financial Strategy

**Employment Protection**:
1. **Enhance job security** - Become indispensable at work
2. **Develop recession-proof skills** - Healthcare, utilities, essential services
3. **Build professional network** - Maintain relationships across industries
4. **Create multiple income streams** - Reduce dependence on single employer
5. **Update emergency skills** - Stay current with technology and trends

**Investment Approach**:
1. **Stay invested** - Don't try to time the market
2. **Continue contributions** - Dollar-cost average through the downturn
3. **Rebalance opportunistically** - Buy stocks when they're on sale
4. **Focus on quality** - Strong companies survive and thrive
5. **Maintain diversification** - Don't concentrate in any single area

**Cash Management**:
1. **Build larger emergency fund** - Target 9-12 months of expenses
2. **Preserve liquidity** - Keep cash accessible
3. **Avoid major purchases** - Delay non-essential spending
4. **Negotiate payment terms** - Work with creditors if needed
5. **Maintain good credit** - You'll need it for opportunities

### Sector-Specific Impacts
**Recession-Resistant Sectors**:
- **Healthcare** - People still need medical care
- **Utilities** - Essential services remain in demand
- **Consumer staples** - Food, household goods, basic necessities
- **Government** - Public sector jobs often more stable
- **Discount retail** - Value-focused businesses may thrive

**Recession-Vulnerable Sectors**:
- **Luxury goods** - Discretionary spending declines first
- **Travel and hospitality** - People cut vacation and entertainment
- **Construction** - New projects get delayed or cancelled
- **Automotive** - Major purchases are postponed
- **Technology** - Growth companies face funding challenges

### Historical Recession Analysis
**Recent Recessions and Recovery Times**:
- **2020 COVID Recession**: 2 months duration, rapid recovery with stimulus
- **2008 Financial Crisis**: 18 months duration, 4+ years for full recovery
- **2001 Dot-com Recession**: 8 months duration, 2+ years for recovery
- **1990-1991 Recession**: 8 months duration, gradual recovery

**Key Lessons from History**:
- **Recessions are temporary** - All have ended eventually
- **Markets recover** - Often before the economy shows improvement
- **Opportunities emerge** - Great companies go on sale
- **Preparation matters** - Those with cash can take advantage

### Opportunity Recognition
**Investment Opportunities**:
- **Quality stocks at discounts** - Blue-chip companies on sale
- **Real estate deals** - Motivated sellers, lower prices
- **Business acquisitions** - Competitors may struggle
- **Skill development** - Use downtime for education and training
- **Career pivots** - Economic shifts create new opportunities

**Financial Opportunities**:
- **Refinancing** - Interest rates often drop during recessions
- **Debt consolidation** - Better terms may be available
- **Roth conversions** - Convert retirement funds at lower values
- **Tax-loss harvesting** - Offset gains with investment losses

### Recession Survival Checklist
**Before Recession Hits**:
- [ ] Build 6-12 month emergency fund
- [ ] Diversify income sources
- [ ] Maintain good credit score
- [ ] Update resume and professional skills
- [ ] Review and optimize all insurance coverage

**During Recession**:
- [ ] Maintain investment contributions if possible
- [ ] Cut non-essential expenses
- [ ] Communicate with creditors if struggling
- [ ] Look for opportunities to advance career
- [ ] Stay informed but don't panic

**Recovery Phase**:
- [ ] Gradually increase risk tolerance
- [ ] Take advantage of investment opportunities
- [ ] Rebuild emergency fund if depleted
- [ ] Position for post-recession growth
- [ ] Learn from the experience

### Psychological Resilience
**Managing Recession Stress**:
- **Focus on controllables** - Your actions, not market movements
- **Maintain perspective** - Recessions are temporary economic cycles
- **Stay connected** - Social support is crucial during difficult times
- **Limit news consumption** - Constant negative news increases anxiety
- **Plan for recovery** - Recessions create the foundation for future growth

### Long-Term Wealth Building
**Recession as Opportunity**:
- **Forced savings** - Reduced spending can increase savings rate
- **Investment discipline** - Learn to invest during uncertainty
- **Career development** - Economic pressure drives innovation and growth
- **Financial education** - Crisis motivates better financial habits
- **Relationship building** - Mutual support creates lasting connections

*Recessions test financial resilience but also create opportunities for those who are prepared and disciplined.*
"""
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Recession scenario explanation failed: {e}")
            return "## Recession Analysis\n\nRecessions are challenging but temporary economic cycles. We'll help you navigate the downturn and position for recovery."

    def _explain_health_scenario(self, severity: str, impact: Dict[str, Any], description: str) -> str:
        """Explain health-related financial scenarios"""
        try:
            health_map = {
                'mild': ('$5,000-15,000', 'minor health issue'),
                'moderate': ('$15,000-50,000', 'significant health event'),
                'severe': ('$50,000+', 'major health crisis')
            }
            
            cost_range, scenario_desc = health_map.get(severity.lower(), health_map['moderate'])
            
            explanation = f"""
## What-If Analysis: Health Emergency ({severity.title()})

### Scenario Overview
**Estimated Costs**: {cost_range}
**Situation**: {scenario_desc}
**Timeline**: Immediate to several months of treatment

### Financial Impact Areas
**Direct Medical Costs**:
- Hospital stays and procedures
- Specialist consultations
- Prescription medications
- Medical equipment and supplies
- Rehabilitation and therapy

**Indirect Financial Effects**:
- Lost income during recovery
- Increased insurance premiums
- Transportation to medical appointments
- Home modifications for accessibility
- Caregiver costs

### Insurance Strategy
**Health Insurance Optimization**:
1. **Understand your coverage** - Deductibles, co-pays, out-of-network costs
2. **Maximize HSA contributions** - Triple tax advantage for medical expenses
3. **Review plan annually** - Ensure adequate coverage for your needs
4. **Keep detailed records** - Track all medical expenses for taxes

**Supplemental Coverage**:
- **Disability insurance** - Replaces income if unable to work
- **Critical illness insurance** - Lump sum for major diagnoses
- **Long-term care insurance** - Covers extended care needs
- **Life insurance** - Protects family from financial hardship

### Emergency Response Plan
**Immediate Actions**:
1. **Contact insurance** - Verify coverage and get pre-authorization
2. **Negotiate payment plans** - Most providers offer payment options
3. **Apply for financial aid** - Hospitals often have assistance programs
4. **Use HSA/FSA funds** - Tax-advantaged accounts for medical expenses
5. **Document everything** - Keep records for insurance and taxes

**Financial Triage**:
1. **Prioritize essential expenses** - Housing, utilities, food, medications
2. **Communicate with creditors** - Explain situation, request deferrals
3. **Access emergency funds** - This is exactly what they're for
4. **Explore assistance programs** - Government and nonprofit support
5. **Consider family support** - Don't be afraid to ask for help

### Long-Term Financial Planning
**Recovery Budget**:
- Account for ongoing medical costs
- Plan for reduced income during recovery
- Budget for lifestyle modifications
- Include caregiver or assistance costs

**Rebuilding Strategy**:
1. **Replenish emergency fund** - Make this a priority after recovery
2. **Increase health savings** - Boost HSA contributions
3. **Review insurance needs** - Adjust coverage based on experience
4. **Update estate planning** - Ensure documents reflect current situation
5. **Focus on prevention** - Invest in preventive care and healthy lifestyle

### Actionable Next Steps
**Prevention and Preparation**:
1. **Build robust emergency fund** - Target 6-12 months of expenses
2. **Maximize HSA contributions** - $4,150 individual, $8,300 family (2024)
3. **Review insurance coverage** - Ensure adequate health and disability protection
4. **Maintain healthy lifestyle** - Prevention is the best financial strategy
5. **Create health directive** - Plan for decision-making if incapacitated

*Health emergencies are unpredictable, but financial preparation can reduce their impact significantly.*
"""
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Health scenario explanation failed: {e}")
            return "## Health Emergency Analysis\n\nHealth crises can have significant financial impacts. We'll help you prepare and respond effectively."

    def _explain_general_scenario(self, scenario_type: str, severity: str, impact: Dict[str, Any], description: str) -> str:
        """Explain general scenarios with basic analysis"""
        try:
            scenario_display = scenario_type.replace('_', ' ').title()
            severity_display = severity.title()
            
            explanation = f"""
## What-If Analysis: {scenario_display} ({severity_display})

### Scenario Overview
**Type**: {scenario_display}
**Severity**: {severity_display}
**Description**: {description or 'A significant change in your financial circumstances'}

### Impact Assessment
Based on the scenario parameters, we've analyzed the potential effects on your financial plan.

### Recommended Response Strategy
1. **Assess the situation** - Understand the full scope of impact
2. **Review your options** - Consider all available responses
3. **Prioritize actions** - Focus on most critical areas first
4. **Monitor progress** - Track how changes affect your goals
5. **Adjust as needed** - Modify your plan based on results

### Key Principles
- **Stay calm and rational** - Emotional decisions often backfire
- **Focus on what you can control** - Your actions and responses
- **Maintain long-term perspective** - Most challenges are temporary
- **Seek professional advice** - Consider consulting experts
- **Learn from the experience** - Use insights to strengthen future planning

### Next Steps
1. Review the detailed impact analysis
2. Consider adjustments to your financial plan
3. Implement recommended changes gradually
4. Monitor results and adjust as needed
5. Update your emergency preparedness

*Every financial challenge is also an opportunity to strengthen your financial resilience.*
"""
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"General scenario explanation failed: {e}")
            return self._generate_fallback_scenario_explanation()

    def _generate_basic_market_scenario(self) -> str:
        """Basic fallback for market scenarios"""
        return """
## Market Downturn Analysis

Market volatility is a normal part of investing. During downturns:

### Key Principles
- Stay invested for long-term goals
- Continue regular contributions
- Avoid emotional decisions
- Focus on quality investments
- Rebalance when appropriate

### Historical Context
Markets have recovered from every major downturn in history. Staying disciplined during volatility is crucial for long-term success.
"""

    def _generate_basic_job_loss_scenario(self) -> str:
        """Basic fallback for job loss scenarios"""
        return """
## Job Loss Analysis

Unemployment can significantly impact your finances. Key strategies:

### Immediate Actions
- File for unemployment benefits
- Reduce non-essential expenses
- Use emergency fund strategically
- Network actively for new opportunities
- Consider temporary income sources

### Financial Protection
An emergency fund covering 3-6 months of expenses provides crucial protection during job transitions.
"""

    def _generate_basic_inflation_scenario(self) -> str:
        """Basic fallback for inflation scenarios"""
        return """
## Inflation Impact Analysis

High inflation erodes purchasing power over time. Protection strategies:

### Investment Adjustments
- Consider inflation-protected securities
- Maintain stock allocations for growth
- Avoid long-term fixed-rate investments
- Focus on real returns above inflation

### Budget Management
- Negotiate salary increases
- Lock in fixed costs where possible
- Focus on essential expenses
- Build inflation expectations into planning
"""

    def _generate_fallback_scenario_explanation(self) -> str:
        """Ultimate fallback for scenario explanations"""
        return """
## Scenario Analysis

We've analyzed the potential impact of this scenario on your financial plan.

### General Recommendations
1. Review your current financial position
2. Consider adjustments to your strategy
3. Maintain emergency fund adequacy
4. Stay focused on long-term goals
5. Seek professional advice if needed

### Next Steps
- Assess the specific impacts on your situation
- Implement appropriate adjustments
- Monitor progress regularly
- Update your plan as circumstances change

*Financial planning is about preparing for various scenarios and maintaining flexibility to adapt.*
"""


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
