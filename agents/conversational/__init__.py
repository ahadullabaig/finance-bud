"""
Conversational Agent Module

Natural language processing for financial planning using local LLM (Ollama).

Current structure: Main implementation in parent conversational_agent.py
Future refactoring: Split into:
- llm_service.py - Ollama integration and LLM communication
- fallback_rules.py - Rule-based processing when LLM unavailable
- templates.py - Response templates and prompts
- core.py - Main ConversationalAgent class
"""

# Import from parent directory for backward compatibility
from ..conversational_agent import (
    ConversationalAgent,
    get_conversational_agent,
    OLLAMA_AVAILABLE
)

__all__ = [
    "ConversationalAgent",
    "get_conversational_agent",
    "OLLAMA_AVAILABLE",
]
