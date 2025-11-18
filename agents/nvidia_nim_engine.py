"""
NVIDIA NIM Generative AI Financial Narrative Engine
Advanced conversational AI for natural language financial planning

STATUS: ALTERNATIVE IMPLEMENTATION - NOT CURRENTLY INTEGRATED
---------------------------------------------------------------
This module provides a fully-featured NVIDIA NIM integration for
advanced conversational AI capabilities. It is an alternative to
the Ollama-based ConversationalAgent currently in use.

INTEGRATION STATUS:
- Fully implemented with all core features
- Imported by AIIntegrationHub but not used by main agent workflow
- Ready for integration if NVIDIA NIM API access is available

TO USE THIS MODULE:
1. Obtain NVIDIA NIM API key from https://build.nvidia.com
2. Configure NIMConfiguration with your API key
3. Replace ConversationalAgent's Ollama backend with NIM engine
   OR use as additional LLM option in multi-model setup

FEATURES:
- Natural language goal parsing with structured extraction
- Financial narrative generation with domain-specific prompts
- What-if scenario analysis
- Conversational planning interface
- Multi-model support (Llama 70B/8B, Nemotron, Mixtral)
- Conversation context management
- Retry logic and error handling

CURRENT ALTERNATIVE: ConversationalAgent uses Ollama (llama3.2:3b)
Location: /agents/conversational_agent.py
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

import aiohttp
import openai
from pydantic import BaseModel, Field

from data_models.schemas import (
    FinancialState, PlanRequest, PlanStep, 
    MarketData, RiskProfile, TaxContext
)

logger = logging.getLogger(__name__)

class NIMModelType(str, Enum):
    """NVIDIA NIM model types for different financial tasks"""
    LLAMA_70B = "meta/llama-3.1-70b-instruct"
    LLAMA_8B = "meta/llama-3.1-8b-instruct"
    NEMOTRON_70B = "nvidia/nemotron-4-340b-instruct"
    MIXTRAL_8X7B = "mistralai/mixtral-8x7b-instruct-v0.1"

class ConversationContext(BaseModel):
    """Context for maintaining conversation state"""
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    financial_context: Optional[FinancialState] = None
    current_goals: List[str] = Field(default_factory=list)
    risk_profile: Optional[RiskProfile] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)

class NIMResponse(BaseModel):
    """Response from NVIDIA NIM API"""
    content: str
    model_used: str
    tokens_used: int
    confidence_score: float
    processing_time: float
    structured_data: Optional[Dict[str, Any]] = None

class FinancialNarrativeRequest(BaseModel):
    """Request for financial narrative generation"""
    user_input: str
    context: ConversationContext
    narrative_type: str  # "goal_parsing", "plan_explanation", "scenario_analysis"
    financial_data: Optional[Dict[str, Any]] = None
    market_context: Optional[MarketData] = None

@dataclass
class NIMConfiguration:
    """Configuration for NVIDIA NIM API"""
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    default_model: NIMModelType = NIMModelType.LLAMA_70B
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3

class NVIDIANIMEngine:
    """
    NVIDIA NIM Generative AI Financial Narrative Engine
    
    Provides advanced conversational AI capabilities for:
    - Natural language goal parsing
    - Financial story narrative generation
    - What-if scenario explanations
    - Conversational planning workflows
    """
    
    def __init__(self, config: NIMConfiguration):
        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Financial domain prompts
        self.system_prompts = {
            "goal_parsing": self._load_goal_parsing_prompt(),
            "plan_explanation": self._load_plan_explanation_prompt(),
            "scenario_analysis": self._load_scenario_analysis_prompt(),
            "risk_assessment": self._load_risk_assessment_prompt(),
            "tax_optimization": self._load_tax_optimization_prompt()
        }
        
    def _load_goal_parsing_prompt(self) -> str:
        """Load system prompt for natural language goal parsing"""
        return """
        You are an expert financial advisor AI specializing in parsing natural language financial goals into structured, actionable plans.
        
        Your role:
        1. Parse complex financial goals from natural language
        2. Identify key constraints, timelines, and priorities
        3. Extract risk tolerance and preferences
        4. Structure goals into actionable components
        5. Identify potential conflicts or unrealistic expectations
        
        Always respond with:
        - Clear understanding of the user's goals
        - Structured breakdown of objectives
        - Timeline analysis
        - Risk considerations
        - Next steps for planning
        
        Be empathetic, clear, and financially sound in your advice.
        """
    
    def _load_plan_explanation_prompt(self) -> str:
        """Load system prompt for financial plan explanations"""
        return """
        You are an expert financial advisor AI specializing in explaining complex financial plans in simple, understandable terms.
        
        Your role:
        1. Explain financial strategies clearly and simply
        2. Highlight key benefits and risks
        3. Provide rationale for recommendations
        4. Address potential concerns
        5. Suggest alternatives when appropriate
        
        Always:
        - Use clear, jargon-free language
        - Provide specific examples
        - Explain the "why" behind recommendations
        - Acknowledge risks and limitations
        - Encourage questions and clarification
        """
    
    def _load_scenario_analysis_prompt(self) -> str:
        """Load system prompt for what-if scenario analysis"""
        return """
        You are an expert financial advisor AI specializing in scenario analysis and stress testing financial plans.
        
        Your role:
        1. Analyze "what-if" scenarios thoroughly
        2. Assess impact on financial goals
        3. Identify risks and opportunities
        4. Suggest adaptive strategies
        5. Provide confidence intervals for outcomes
        
        Consider:
        - Market volatility impacts
        - Life event implications
        - Regulatory changes
        - Economic cycles
        - Personal circumstances
        
        Provide balanced, realistic assessments with actionable insights.
        """
    
    def _load_risk_assessment_prompt(self) -> str:
        """Load system prompt for risk assessment"""
        return """
        You are an expert financial risk analyst AI specializing in comprehensive risk assessment for financial planning.
        
        Your role:
        1. Identify and quantify financial risks
        2. Assess risk tolerance alignment
        3. Suggest risk mitigation strategies
        4. Explain risk-return tradeoffs
        5. Provide stress test scenarios
        
        Consider all risk types:
        - Market risk
        - Credit risk
        - Liquidity risk
        - Inflation risk
        - Regulatory risk
        - Personal/life event risk
        """
    
    def _load_tax_optimization_prompt(self) -> str:
        """Load system prompt for tax optimization"""
        return """
        You are an expert tax optimization AI specializing in tax-efficient financial planning strategies.
        
        Your role:
        1. Identify tax optimization opportunities
        2. Explain tax implications of strategies
        3. Suggest tax-efficient alternatives
        4. Consider multi-year tax planning
        5. Ensure compliance with tax regulations
        
        Focus on:
        - Tax-advantaged accounts
        - Tax-loss harvesting
        - Asset location strategies
        - Timing of income/deductions
        - Estate planning considerations
        """

    async def parse_natural_language_goal(
        self, 
        user_input: str, 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Parse natural language financial goals into structured format
        
        Args:
            user_input: Natural language description of financial goals
            context: Conversation context and user information
            
        Returns:
            Structured goal data with parsed components
        """
        try:
            # Prepare context for goal parsing
            context_str = self._prepare_context_string(context)
            
            prompt = f"""
            Parse the following financial goal into structured components:
            
            User Input: "{user_input}"
            
            Context: {context_str}
            
            Please provide a structured analysis including:
            1. Primary financial objectives
            2. Timeline and milestones
            3. Risk tolerance indicators
            4. Constraints and limitations
            5. Success metrics
            6. Potential challenges
            
            Format your response as JSON with clear categories.
            """
            
            response = await self._call_nim_api(
                prompt=prompt,
                system_prompt=self.system_prompts["goal_parsing"],
                model=self.config.default_model
            )
            
            # Extract structured data from response
            structured_goal = self._extract_structured_data(response.content)
            
            # Update conversation context
            context.current_goals.append(user_input)
            context.conversation_history.append({
                "type": "goal_parsing",
                "input": user_input,
                "output": response.content,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "parsed_goal": structured_goal,
                "narrative_explanation": response.content,
                "confidence_score": response.confidence_score,
                "processing_metrics": {
                    "tokens_used": response.tokens_used,
                    "processing_time": response.processing_time,
                    "model_used": response.model_used
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing natural language goal: {e}")
            raise

    async def generate_financial_narrative(
        self, 
        request: FinancialNarrativeRequest
    ) -> NIMResponse:
        """
        Generate financial story narratives and explanations
        
        Args:
            request: Financial narrative generation request
            
        Returns:
            Generated narrative with context and metrics
        """
        try:
            # Select appropriate system prompt
            system_prompt = self.system_prompts.get(
                request.narrative_type, 
                self.system_prompts["plan_explanation"]
            )
            
            # Prepare comprehensive context
            context_data = {
                "user_input": request.user_input,
                "financial_context": request.financial_data,
                "market_context": request.market_context.dict() if request.market_context else None,
                "conversation_history": request.context.conversation_history[-5:],  # Last 5 exchanges
                "user_preferences": request.context.preferences
            }
            
            prompt = f"""
            Generate a comprehensive financial narrative for the following request:
            
            Request Type: {request.narrative_type}
            User Input: {request.user_input}
            
            Context: {json.dumps(context_data, indent=2, default=str)}
            
            Please provide:
            1. Clear, engaging narrative explanation
            2. Key insights and recommendations
            3. Risk considerations
            4. Next steps or action items
            5. Confidence assessment
            
            Make the explanation accessible while maintaining financial accuracy.
            """
            
            response = await self._call_nim_api(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.config.default_model
            )
            
            # Update conversation context
            request.context.conversation_history.append({
                "type": request.narrative_type,
                "input": request.user_input,
                "output": response.content,
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating financial narrative: {e}")
            raise

    async def analyze_what_if_scenario(
        self, 
        scenario_description: str,
        current_plan: Dict[str, Any],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Generate detailed what-if scenario analysis
        
        Args:
            scenario_description: Description of the scenario to analyze
            current_plan: Current financial plan data
            context: Conversation context
            
        Returns:
            Comprehensive scenario analysis with recommendations
        """
        try:
            prompt = f"""
            Analyze the following "what-if" scenario and its impact on the current financial plan:
            
            Scenario: "{scenario_description}"
            
            Current Plan: {json.dumps(current_plan, indent=2, default=str)}
            
            User Context: {self._prepare_context_string(context)}
            
            Please provide:
            1. Scenario impact assessment
            2. Quantitative analysis where possible
            3. Risk implications
            4. Adaptive strategies
            5. Timeline considerations
            6. Confidence intervals for outcomes
            7. Alternative approaches
            
            Be specific about financial implications and provide actionable insights.
            """
            
            response = await self._call_nim_api(
                prompt=prompt,
                system_prompt=self.system_prompts["scenario_analysis"],
                model=self.config.default_model
            )
            
            # Extract structured scenario analysis
            scenario_analysis = self._extract_structured_data(response.content)
            
            return {
                "scenario_description": scenario_description,
                "impact_analysis": scenario_analysis,
                "narrative_explanation": response.content,
                "confidence_score": response.confidence_score,
                "recommendations": self._extract_recommendations(response.content),
                "processing_metrics": {
                    "tokens_used": response.tokens_used,
                    "processing_time": response.processing_time,
                    "model_used": response.model_used
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing what-if scenario: {e}")
            raise

    async def create_conversational_interface(
        self, 
        user_message: str,
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Handle conversational financial planning interface
        
        Args:
            user_message: User's conversational input
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Conversational response with context management
        """
        try:
            # Get or create conversation context
            context = self.conversation_contexts.get(session_id)
            if not context:
                context = ConversationContext(
                    session_id=session_id,
                    user_id=user_id
                )
                self.conversation_contexts[session_id] = context
            
            # Determine conversation intent
            intent = await self._classify_conversation_intent(user_message, context)
            
            # Generate appropriate response based on intent
            if intent == "goal_setting":
                response_data = await self.parse_natural_language_goal(user_message, context)
            elif intent == "plan_inquiry":
                response_data = await self._handle_plan_inquiry(user_message, context)
            elif intent == "scenario_question":
                response_data = await self._handle_scenario_question(user_message, context)
            else:
                response_data = await self._handle_general_conversation(user_message, context)
            
            # Update context
            context.last_updated = datetime.now()
            
            return {
                "response": response_data,
                "intent": intent,
                "context_updated": True,
                "session_id": session_id,
                "conversation_state": {
                    "total_exchanges": len(context.conversation_history),
                    "current_goals": len(context.current_goals),
                    "last_updated": context.last_updated.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in conversational interface: {e}")
            raise

    async def _call_nim_api(
        self, 
        prompt: str, 
        system_prompt: str,
        model: NIMModelType,
        temperature: Optional[float] = None
    ) -> NIMResponse:
        """
        Call NVIDIA NIM API with retry logic and error handling
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            model: NIM model to use
            temperature: Sampling temperature
            
        Returns:
            NIM API response with metrics
        """
        start_time = datetime.now()
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=model.value,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                    timeout=self.config.timeout
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return NIMResponse(
                    content=response.choices[0].message.content,
                    model_used=model.value,
                    tokens_used=response.usage.total_tokens,
                    confidence_score=self._calculate_confidence_score(response),
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.warning(f"NIM API attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _prepare_context_string(self, context: ConversationContext) -> str:
        """Prepare context information for prompts"""
        context_parts = []
        
        if context.financial_context:
            context_parts.append(f"Financial State: {context.financial_context.dict()}")
        
        if context.risk_profile:
            context_parts.append(f"Risk Profile: {context.risk_profile.dict()}")
        
        if context.current_goals:
            context_parts.append(f"Current Goals: {context.current_goals}")
        
        if context.preferences:
            context_parts.append(f"Preferences: {context.preferences}")
        
        return "\n".join(context_parts)

    def _extract_structured_data(self, response_content: str) -> Dict[str, Any]:
        """Extract structured data from NIM response"""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback to parsing key sections
        return {
            "content": response_content,
            "parsed": False,
            "extraction_method": "fallback"
        }

    def _extract_recommendations(self, response_content: str) -> List[str]:
        """Extract actionable recommendations from response"""
        recommendations = []
        lines = response_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'consider']):
                recommendations.append(line)
        
        return recommendations[:5]  # Top 5 recommendations

    def _calculate_confidence_score(self, response) -> float:
        """Calculate confidence score based on response characteristics"""
        # Simplified confidence calculation
        # In production, this would use more sophisticated metrics
        base_confidence = 0.8
        
        # Adjust based on response length and structure
        content_length = len(response.choices[0].message.content)
        if content_length > 500:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

    async def _classify_conversation_intent(
        self, 
        message: str, 
        context: ConversationContext
    ) -> str:
        """Classify the intent of a conversational message"""
        # Simplified intent classification
        # In production, this would use a trained classifier
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['goal', 'want', 'plan', 'save', 'invest']):
            return "goal_setting"
        elif any(word in message_lower for word in ['what if', 'scenario', 'happen', 'change']):
            return "scenario_question"
        elif any(word in message_lower for word in ['plan', 'strategy', 'recommendation']):
            return "plan_inquiry"
        else:
            return "general_conversation"

    async def _handle_plan_inquiry(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle plan-related inquiries"""
        request = FinancialNarrativeRequest(
            user_input=message,
            context=context,
            narrative_type="plan_explanation"
        )
        
        response = await self.generate_financial_narrative(request)
        
        return {
            "type": "plan_inquiry",
            "response": response.content,
            "confidence": response.confidence_score
        }

    async def _handle_scenario_question(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle scenario-related questions"""
        # Extract scenario from message and analyze
        scenario_analysis = await self.analyze_what_if_scenario(
            scenario_description=message,
            current_plan={},  # Would get from context in production
            context=context
        )
        
        return {
            "type": "scenario_analysis",
            "response": scenario_analysis["narrative_explanation"],
            "analysis": scenario_analysis["impact_analysis"]
        }

    async def _handle_general_conversation(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle general conversational messages"""
        request = FinancialNarrativeRequest(
            user_input=message,
            context=context,
            narrative_type="plan_explanation"
        )
        
        response = await self.generate_financial_narrative(request)
        
        return {
            "type": "general_conversation",
            "response": response.content,
            "confidence": response.confidence_score
        }

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up conversation context for a session"""
        if session_id in self.conversation_contexts:
            del self.conversation_contexts[session_id]
            logger.info(f"Cleaned up conversation context for session {session_id}")

# Example usage and configuration
def create_nim_engine() -> NVIDIANIMEngine:
    """Create and configure NVIDIA NIM engine"""
    config = NIMConfiguration(
        api_key="your-nvidia-nim-api-key",  # Replace with actual API key
        base_url="https://integrate.api.nvidia.com/v1",
        default_model=NIMModelType.LLAMA_70B,
        max_tokens=4096,
        temperature=0.7
    )
    
    return NVIDIANIMEngine(config)

# Integration with existing agent system
class NIMIntegratedAgent:
    """Integration wrapper for NVIDIA NIM with existing agent system"""
    
    def __init__(self, nim_engine: NVIDIANIMEngine):
        self.nim_engine = nim_engine
        
    async def process_user_goal(self, goal: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user goal through NIM engine"""
        context = ConversationContext(
            session_id=user_context.get("session_id", "default"),
            user_id=user_context.get("user_id", "anonymous"),
            financial_context=user_context.get("financial_state"),
            risk_profile=user_context.get("risk_profile")
        )
        
        return await self.nim_engine.parse_natural_language_goal(goal, context)
    
    async def explain_plan(self, plan_data: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """Generate plan explanation through NIM engine"""
        context = ConversationContext(
            session_id=user_context.get("session_id", "default"),
            user_id=user_context.get("user_id", "anonymous")
        )
        
        request = FinancialNarrativeRequest(
            user_input="Explain this financial plan",
            context=context,
            narrative_type="plan_explanation",
            financial_data=plan_data
        )
        
        response = await self.nim_engine.generate_financial_narrative(request)
        return response.content