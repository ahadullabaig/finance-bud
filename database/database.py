"""
Database operations for FinPilot VP-MAS using Supabase

Provides high-level database operations for all financial data,
agent communications, and system operations.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from uuid import UUID, uuid4
from decimal import Decimal

from .config import get_supabase_client
# Import will be handled at runtime to avoid circular imports
# from data_models.schemas import (
#     FinancialState, EnhancedPlanRequest, PlanStep, VerificationReport,
#     MarketData, TriggerEvent, AgentMessage, ExecutionLog, ReasoningTrace
# )


class FinancialStateDB:
    """Database operations for financial states"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    async def create_financial_state(self, user_id: str, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new financial state record"""
        try:
            result = self.client.client.table('financial_states').insert({
                'user_id': user_id,
                'total_assets': float(state_data.get('total_assets', 0)),
                'total_liabilities': float(state_data.get('total_liabilities', 0)),
                'monthly_income': float(state_data.get('monthly_income', 0)),
                'monthly_expenses': float(state_data.get('monthly_expenses', 0)),
                'risk_tolerance': state_data.get('risk_tolerance', 'moderate'),
                'tax_filing_status': state_data.get('tax_filing_status', 'single'),
                'estimated_tax_rate': float(state_data.get('estimated_tax_rate', 0.22))
            }).execute()
            
            return {"success": True, "data": result.data[0]}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_financial_state(self, user_id: str) -> Dict[str, Any]:
        """Get the latest financial state for a user"""
        try:
            result = self.client.client.table('financial_states')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if result.data:
                return {"success": True, "data": result.data[0]}
            else:
                return {"success": False, "error": "No financial state found"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def update_financial_state(self, state_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing financial state"""
        try:
            result = self.client.client.table('financial_states')\
                .update(updates)\
                .eq('id', state_id)\
                .execute()
            
            return {"success": True, "data": result.data[0]}
        except Exception as e:
            return {"success": False, "error": str(e)}


class PlansDB:
    """Database operations for financial plans"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    async def create_plan(self, user_id: str, plan_data: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
        """Create a new financial plan"""
        try:
            result = self.client.client.table('plans').insert({
                'user_id': user_id,
                'plan_data': plan_data,
                'correlation_id': correlation_id,
                'session_id': plan_data.get('session_id'),
                'confidence_score': float(plan_data.get('confidence_score', 0.0))
            }).execute()
            
            return {"success": True, "data": result.data[0]}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_plan(self, plan_id: str) -> Dict[str, Any]:
        """Get a specific plan by ID"""
        try:
            result = self.client.client.table('plans')\
                .select('*')\
                .eq('id', plan_id)\
                .execute()
            
            if result.data:
                return {"success": True, "data": result.data[0]}
            else:
                return {"success": False, "error": "Plan not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_user_plans(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get all plans for a user"""
        try:
            result = self.client.client.table('plans')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def update_plan_status(self, plan_id: str, status: str) -> Dict[str, Any]:
        """Update plan status"""
        try:
            result = self.client.client.table('plans')\
                .update({'status': status})\
                .eq('id', plan_id)\
                .execute()
            
            return {"success": True, "data": result.data[0]}
        except Exception as e:
            return {"success": False, "error": str(e)}


class PlanStepsDB:
    """Database operations for plan steps"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    async def create_plan_steps(self, plan_id: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create multiple plan steps"""
        try:
            steps_data = []
            for step in steps:
                steps_data.append({
                    'plan_id': plan_id,
                    'step_id': step.get('step_id', str(uuid4())),
                    'sequence_number': step['sequence_number'],
                    'action_type': step['action_type'],
                    'description': step['description'],
                    'amount': float(step['amount']),
                    'target_date': step['target_date'],
                    'rationale': step.get('rationale', ''),
                    'confidence_score': float(step.get('confidence_score', 0.0)),
                    'risk_level': step.get('risk_level', 'medium')
                })
            
            result = self.client.client.table('plan_steps').insert(steps_data).execute()
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_plan_steps(self, plan_id: str) -> Dict[str, Any]:
        """Get all steps for a plan"""
        try:
            result = self.client.client.table('plan_steps')\
                .select('*')\
                .eq('plan_id', plan_id)\
                .order('sequence_number')\
                .execute()
            
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}


class MarketDataDB:
    """Database operations for market data"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    async def store_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store market data"""
        try:
            result = self.client.client.table('market_data').insert({
                'data_id': market_data.get('data_id', str(uuid4())),
                'source': market_data['source'],
                'market_volatility': float(market_data.get('market_volatility', 0)),
                'interest_rates': market_data.get('interest_rates', {}),
                'sector_trends': market_data.get('sector_trends', {}),
                'economic_sentiment': float(market_data.get('economic_sentiment', 0)),
                'collection_method': market_data.get('collection_method', ''),
                'refresh_frequency': int(market_data.get('refresh_frequency', 300))
            }).execute()
            
            return {"success": True, "data": result.data[0]}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_latest_market_data(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Get the latest market data"""
        try:
            query = self.client.client.table('market_data').select('*')
            
            if source:
                query = query.eq('source', source)
            
            result = query.order('timestamp', desc=True).limit(1).execute()
            
            if result.data:
                return {"success": True, "data": result.data[0]}
            else:
                return {"success": False, "error": "No market data found"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class AgentMessagesDB:
    """Database operations for agent messages"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    async def store_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Store an agent message"""
        try:
            result = self.client.client.table('agent_messages').insert({
                'message_id': message.get('message_id', str(uuid4())),
                'agent_id': message['agent_id'],
                'target_agent_id': message.get('target_agent_id'),
                'message_type': message['message_type'],
                'payload': message['payload'],
                'correlation_id': message['correlation_id'],
                'session_id': message.get('session_id'),
                'priority': message.get('priority', 'medium'),
                'trace_id': message.get('trace_id'),
                'retry_count': int(message.get('retry_count', 0)),
                'expires_at': message.get('expires_at')
            }).execute()
            
            return {"success": True, "data": result.data[0]}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_messages_by_correlation(self, correlation_id: str) -> Dict[str, Any]:
        """Get all messages for a correlation ID"""
        try:
            result = self.client.client.table('agent_messages')\
                .select('*')\
                .eq('correlation_id', correlation_id)\
                .order('created_at')\
                .execute()
            
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ExecutionLogsDB:
    """Database operations for execution logs"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    async def log_execution(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log an execution event"""
        try:
            result = self.client.client.table('execution_logs').insert({
                'log_id': log_data.get('log_id', str(uuid4())),
                'agent_id': log_data['agent_id'],
                'action_type': log_data['action_type'],
                'operation_name': log_data['operation_name'],
                'execution_status': log_data['execution_status'],
                'input_data': log_data.get('input_data', {}),
                'output_data': log_data.get('output_data', {}),
                'execution_time': float(log_data.get('execution_time', 0)),
                'session_id': log_data.get('session_id'),
                'correlation_id': log_data['correlation_id'],
                'trace_id': log_data.get('trace_id')
            }).execute()
            
            return {"success": True, "data": result.data[0]}
        except Exception as e:
            return {"success": False, "error": str(e)}


class VerificationReportsDB:
    """Database operations for verification reports"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    async def store_verification_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Store a verification report"""
        try:
            result = self.client.client.table('verification_reports').insert({
                'report_id': report.get('report_id', str(uuid4())),
                'plan_id': report['plan_id'],
                'verification_status': report['verification_status'],
                'constraints_checked': int(report.get('constraints_checked', 0)),
                'constraints_passed': int(report.get('constraints_passed', 0)),
                'constraint_violations': report.get('constraint_violations', []),
                'overall_risk_score': float(report.get('overall_risk_score', 0)),
                'approval_rationale': report.get('approval_rationale', ''),
                'confidence_score': float(report.get('confidence_score', 0)),
                'verification_time': float(report.get('verification_time', 0)),
                'verifier_agent_id': report['verifier_agent_id'],
                'correlation_id': report['correlation_id']
            }).execute()
            
            return {"success": True, "data": result.data[0]}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Database manager class
class DatabaseManager:
    """Central database manager for all operations"""
    
    def __init__(self):
        self.financial_states = FinancialStateDB()
        self.plans = PlansDB()
        self.plan_steps = PlanStepsDB()
        self.market_data = MarketDataDB()
        self.agent_messages = AgentMessagesDB()
        self.execution_logs = ExecutionLogsDB()
        self.verification_reports = VerificationReportsDB()
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        client = get_supabase_client()
        return await client.health_check()


# Global database manager instance
db = DatabaseManager()