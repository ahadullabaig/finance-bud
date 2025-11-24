"""
Enhanced Execution Agent for FinPilot VP-MAS

The Execution Agent is responsible for executing approved financial plans,
managing the financial ledger, handling transactions, and providing
comprehensive audit trails with tax optimization and compliance reporting.

Requirements: 1.4, 2.5, 4.5, 8.4
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4
from enum import Enum
import json
import math

from .base_agent import BaseAgent
from config import get_settings
from data_models.schemas import (
    AgentMessage, MessageType, Priority, ExecutionStatus,
    PerformanceMetrics, ExecutionLog
)


class TransactionType(str, Enum):
    """Types of financial transactions"""
    INVESTMENT = "investment"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    REBALANCE = "rebalance"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    FEE = "fee"
    TAX_PAYMENT = "tax_payment"
    CONTRIBUTION = "contribution"
    BUY_STOCK = "buy_stock"
    SELL_STOCK = "sell_stock"
    BUY_BOND = "buy_bond"
    SELL_BOND = "sell_bond"
    DEPOSIT = "deposit"
    EMERGENCY_FUND = "emergency_fund"


class TransactionStatus(str, Enum):
    """Status of financial transactions"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class AccountType(str, Enum):
    """Types of financial accounts"""
    CHECKING = "checking"
    SAVINGS = "savings"
    INVESTMENT = "investment"
    RETIREMENT_401K = "retirement_401k"
    RETIREMENT_IRA = "retirement_ira"
    RETIREMENT_ROTH = "retirement_roth"
    HSA = "hsa"
    TAXABLE_INVESTMENT = "taxable_investment"
    TRUST = "trust"
    EMERGENCY_FUND = "emergency_fund"


class AssetClass(str, Enum):
    """Asset classes for portfolio management"""
    CASH = "cash"
    STOCKS = "stocks"
    BONDS = "bonds"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    ALTERNATIVES = "alternatives"
    CRYPTO = "crypto"


class Transaction:
    """Comprehensive transaction record with tax optimization and compliance tracking"""
    
    def __init__(
        self,
        transaction_id: str = None,
        transaction_type: TransactionType = TransactionType.INVESTMENT,
        amount: Decimal = Decimal('0'),
        from_account: str = None,
        to_account: str = None,
        asset_symbol: str = None,
        asset_class: AssetClass = AssetClass.CASH,
        description: str = "",
        tax_implications: Dict[str, Any] = None,
        compliance_notes: List[str] = None,
        correlation_id: str = None,
        session_id: str = None,
        shares: Decimal = None,
        price_per_share: Decimal = None
    ):
        self.transaction_id = transaction_id or str(uuid4())
        self.transaction_type = transaction_type
        self.amount = amount
        self.from_account = from_account
        self.to_account = to_account
        self.asset_symbol = asset_symbol
        self.asset_class = asset_class
        self.description = description
        self.tax_implications = tax_implications or {}
        self.compliance_notes = compliance_notes or []
        self.correlation_id = correlation_id
        self.session_id = session_id
        self.status = TransactionStatus.PENDING
        self.created_at = datetime.utcnow()
        self.executed_at: Optional[datetime] = None
        self.fees = Decimal('0')
        self.net_amount = amount
        self.rollback_data: Optional[Dict[str, Any]] = None
        self.shares = shares
        self.price_per_share = price_per_share
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            "transaction_id": self.transaction_id,
            "transaction_type": self.transaction_type.value,
            "amount": float(self.amount),
            "from_account": self.from_account,
            "to_account": self.to_account,
            "asset_symbol": self.asset_symbol,
            "asset_class": self.asset_class.value,
            "description": self.description,
            "tax_implications": self.tax_implications,
            "compliance_notes": self.compliance_notes,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "fees": float(self.fees),
            "net_amount": float(self.net_amount),
            "rollback_data": self.rollback_data,
            "shares": float(self.shares) if self.shares else None,
            "price_per_share": float(self.price_per_share) if self.price_per_share else None
        }


class FinancialAccount:
    """Represents a financial account with comprehensive tracking"""
    
    def __init__(
        self,
        account_id: str,
        account_type: AccountType,
        account_name: str,
        initial_balance: Decimal = Decimal('0'),
        tax_advantaged: bool = False,
        contribution_limits: Dict[str, Decimal] = None
    ):
        self.account_id = account_id
        self.account_type = account_type
        self.account_name = account_name
        self.balance = initial_balance
        self.tax_advantaged = tax_advantaged
        self.contribution_limits = contribution_limits or {}
        self.holdings: Dict[str, Dict[str, Any]] = {}
        self.transaction_history: List[str] = []
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        
    def add_holding(self, symbol: str, shares: Decimal, cost_per_share: Decimal) -> None:
        """Add or update a holding in the account"""
        if symbol in self.holdings:
            existing = self.holdings[symbol]
            total_shares = existing["shares"] + shares
            total_cost = (existing["shares"] * existing["avg_cost"]) + (shares * cost_per_share)
            avg_cost = total_cost / total_shares if total_shares > 0 else Decimal('0')
            
            self.holdings[symbol] = {
                "shares": total_shares,
                "avg_cost": avg_cost,
                "last_updated": datetime.utcnow()
            }
        else:
            self.holdings[symbol] = {
                "shares": shares,
                "avg_cost": cost_per_share,
                "last_updated": datetime.utcnow()
            }
        
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert account to dictionary"""
        return {
            "account_id": self.account_id,
            "account_type": self.account_type.value,
            "account_name": self.account_name,
            "balance": float(self.balance),
            "tax_advantaged": self.tax_advantaged,
            "contribution_limits": {k: float(v) for k, v in self.contribution_limits.items()},
            "holdings": {
                symbol: {
                    "shares": float(data["shares"]),
                    "avg_cost": float(data["avg_cost"]),
                    "last_updated": data["last_updated"].isoformat()
                }
                for symbol, data in self.holdings.items()
            },
            "transaction_count": len(self.transaction_history),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


class FinancialLedger:
    """Comprehensive financial ledger management system with audit trails"""
    
    def __init__(self):
        self.accounts: Dict[str, FinancialAccount] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.transaction_history: List[str] = []
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("finpilot.execution.ledger")
        
    def create_account(
        self,
        account_type: AccountType,
        account_name: str,
        initial_balance: Decimal = Decimal('0'),
        tax_advantaged: bool = False,
        contribution_limits: Dict[str, Decimal] = None
    ) -> str:
        """Create a new financial account"""
        account_id = str(uuid4())
        account = FinancialAccount(
            account_id=account_id,
            account_type=account_type,
            account_name=account_name,
            initial_balance=initial_balance,
            tax_advantaged=tax_advantaged,
            contribution_limits=contribution_limits
        )
        
        self.accounts[account_id] = account
        self.logger.info(f"Created account {account_id}: {account_name} ({account_type.value})")
        return account_id
    
    def get_account(self, account_id: str) -> Optional[FinancialAccount]:
        """Get account by ID"""
        return self.accounts.get(account_id)
    
    def get_total_balance(self) -> Decimal:
        """Get total balance across all accounts"""
        return sum(account.balance for account in self.accounts.values())
    
    def get_portfolio_value(self, market_prices: Dict[str, Decimal] = None) -> Decimal:
        """Get total portfolio value including holdings"""
        total_value = Decimal('0')
        market_prices = market_prices or {}
        
        for account in self.accounts.values():
            total_value += account.balance
            
            for symbol, holding in account.holdings.items():
                price = market_prices.get(symbol, holding["avg_cost"])
                total_value += holding["shares"] * price
        
        return total_value
    
    def record_transaction(self, transaction: Transaction) -> bool:
        """Record a transaction in the ledger"""
        try:
            self.transactions[transaction.transaction_id] = transaction
            self.transaction_history.append(transaction.transaction_id)
            
            # Update account balances and holdings
            self._apply_transaction(transaction)
            
            self.logger.info(f"Recorded transaction {transaction.transaction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record transaction: {str(e)}")
            return False
    
    def _apply_transaction(self, transaction: Transaction) -> None:
        """Apply transaction effects to accounts"""
        if transaction.from_account:
            from_account = self.accounts.get(transaction.from_account)
            if from_account:
                from_account.balance -= transaction.amount
                from_account.transaction_history.append(transaction.transaction_id)
        
        if transaction.to_account:
            to_account = self.accounts.get(transaction.to_account)
            if to_account:
                to_account.balance += transaction.net_amount
                to_account.transaction_history.append(transaction.transaction_id)
        
        # Handle investment transactions
        if transaction.transaction_type in [TransactionType.BUY_STOCK, TransactionType.INVESTMENT]:
            if transaction.to_account and transaction.shares and transaction.price_per_share:
                account = self.accounts.get(transaction.to_account)
                if account:
                    account.add_holding(
                        transaction.asset_symbol,
                        transaction.shares,
                        transaction.price_per_share
                    )
    
    def create_snapshot(self, snapshot_name: str = None) -> str:
        """Create a snapshot of current ledger state"""
        snapshot_id = snapshot_name or datetime.utcnow().isoformat()
        
        self.snapshots[snapshot_id] = {
            "timestamp": datetime.utcnow().isoformat(),
            "accounts": {
                account_id: account.to_dict()
                for account_id, account in self.accounts.items()
            },
            "total_balance": float(self.get_total_balance()),
            "transaction_count": len(self.transactions)
        }
        
        return snapshot_id
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get summary of all accounts"""
        return {
            "total_accounts": len(self.accounts),
            "total_balance": float(self.get_total_balance()),
            "accounts_by_type": {
                account_type.value: [
                    {
                        "account_id": account.account_id,
                        "account_name": account.account_name,
                        "balance": float(account.balance),
                        "holdings_count": len(account.holdings)
                    }
                    for account in self.accounts.values()
                    if account.account_type == account_type
                ]
                for account_type in AccountType
            }
        }


class ExecutionAgent(BaseAgent):
    """
    Enhanced Execution Agent with comprehensive financial operations,
    tax optimization, compliance reporting, and predictive forecasting.
    
    Handles:
    - Financial ledger management
    - Transaction execution with rollback capability
    - Tax-efficient execution strategies
    - Regulatory compliance checking
    - Portfolio tracking and rebalancing
    - Predictive forecasting with uncertainty quantification
    - Comprehensive audit trails
    - Real-time monitoring and reporting
    - Workflow integration and coordination
    - FastAPI endpoints for user goals and agent coordination
    """
    
    def __init__(self, agent_id: str = "execution_agent"):
        super().__init__(agent_id, "execution")
        
        # Core components
        self.ledger = FinancialLedger()
        
        # Execution state
        self.pending_transactions: Dict[str, Transaction] = {}
        self.execution_queue: List[str] = []
        self.rollback_stack: List[str] = []
        
        # Real-time monitoring
        self.real_time_monitors: Dict[str, Dict[str, Any]] = {}
        self.monitoring_active = False
        # Use configured monitoring interval from settings (default 60s)
        self.monitoring_interval = get_settings().monitoring_interval_seconds
        
        # Performance tracking
        self.execution_metrics = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "rollback_transactions": 0,
            "total_value_processed": Decimal('0'),
            "average_execution_time": 0.0,
            "real_time_updates": 0,
            "compliance_checks": 0,
            "tax_optimizations": 0
        }
        
        # Market data cache for pricing
        self.market_prices: Dict[str, Decimal] = {
            "SPY": Decimal("450.00"),
            "BND": Decimal("75.00"),
            "VTI": Decimal("240.00"),
            "VXUS": Decimal("60.00"),
            "GLD": Decimal("180.00"),
            "CASH": Decimal("1.00")
        }
        
        # Workflow coordination
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_results: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"ExecutionAgent {agent_id} initialized with comprehensive financial operations")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages from other agents"""
        
        try:
            action = message.payload.get("action")
            
            if action == "execute_plan":
                return await self._handle_plan_execution(message)
            elif action == "execute_transaction":
                return await self._handle_transaction_execution(message)
            elif action == "get_portfolio_status":
                return await self._handle_portfolio_status_request(message)
            elif action == "generate_forecast":
                return await self._handle_forecast_request(message)
            elif action == "check_compliance":
                return await self._handle_compliance_check(message)
            elif action == "optimize_taxes":
                return await self._handle_tax_optimization(message)
            elif action == "health_check":
                return await self._handle_health_check(message)
            # New workflow and monitoring actions
            elif action == "start_monitoring":
                return await self._handle_start_monitoring(message)
            elif action == "stop_monitoring":
                return await self._handle_stop_monitoring(message)
            elif action == "register_workflow":
                return await self._handle_register_workflow(message)
            elif action == "update_workflow":
                return await self._handle_update_workflow(message)
            elif action == "rollback_workflow":
                return await self._handle_rollback_workflow(message)
            else:
                self.logger.warning(f"Unknown action: {action}")
                return self._create_error_response(message, f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return self._create_error_response(message, str(e))
    
    async def _handle_plan_execution(self, message: AgentMessage) -> AgentMessage:
        """Handle execution of a complete financial plan"""
        
        start_time = time.time()
        plan_data = message.payload.get("plan", {})
        plan_steps = plan_data.get("steps", [])
        
        execution_results = {
            "plan_id": plan_data.get("plan_id", str(uuid4())),
            "execution_status": "completed",
            "executed_steps": [],
            "failed_steps": [],
            "total_steps": len(plan_steps),
            "portfolio_updates": []
        }
        
        try:
            # Execute each plan step
            for step in plan_steps:
                step_result = await self._execute_plan_step(step, message.correlation_id, message.session_id)
                
                if step_result["success"]:
                    execution_results["executed_steps"].append(step_result)
                else:
                    execution_results["failed_steps"].append(step_result)
                    execution_results["execution_status"] = "partial_failure"
            
            execution_time = time.time() - start_time
            self.execution_metrics["average_execution_time"] = (
                (self.execution_metrics["average_execution_time"] * self.execution_metrics["total_transactions"] + execution_time) /
                (self.execution_metrics["total_transactions"] + 1)
            )
            
            self.logger.info(f"Plan execution completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_results["execution_status"] = "failed"
            execution_results["error"] = str(e)
            self.logger.error(f"Plan execution failed: {str(e)}")
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "action": "plan_execution_complete",
                "execution_results": execution_results,
                "portfolio_updated": True,
                "ledger_snapshot": self.ledger.create_snapshot()
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    async def _execute_plan_step(
        self,
        step: Dict[str, Any],
        correlation_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Execute an individual plan step"""
        
        step_result = {
            "step_id": step.get("step_id", str(uuid4())),
            "action_type": step.get("action_type"),
            "success": False,
            "transaction_id": None,
            "error": None
        }
        
        try:
            action_type = step.get("action_type")
            amount = Decimal(str(step.get("amount", 0)))
            
            if action_type == "invest":
                transaction = await self._create_investment_transaction(
                    amount=amount,
                    asset_symbol=step.get("asset_symbol", "SPY"),
                    account_type=step.get("account_type", "investment"),
                    correlation_id=correlation_id,
                    session_id=session_id
                )
                
            elif action_type == "save":
                transaction = await self._create_savings_transaction(
                    amount=amount,
                    account_type=step.get("account_type", "savings"),
                    correlation_id=correlation_id,
                    session_id=session_id
                )
                
            elif action_type == "contribute":
                transaction = await self._create_contribution_transaction(
                    amount=amount,
                    account_type=step.get("account_type", "retirement_401k"),
                    correlation_id=correlation_id,
                    session_id=session_id
                )
                
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            # Execute the transaction
            if transaction:
                success = await self._execute_transaction(transaction)
                step_result["success"] = success
                step_result["transaction_id"] = transaction.transaction_id
                
                if success:
                    self.execution_metrics["successful_transactions"] += 1
                else:
                    self.execution_metrics["failed_transactions"] += 1
                    step_result["error"] = "Transaction execution failed"
                
                self.execution_metrics["total_transactions"] += 1
                self.execution_metrics["total_value_processed"] += amount
            
        except Exception as e:
            step_result["error"] = str(e)
            self.logger.error(f"Failed to execute step {step_result['step_id']}: {str(e)}")
        
        return step_result
    
    async def _create_investment_transaction(
        self,
        amount: Decimal,
        asset_symbol: str,
        account_type: str,
        correlation_id: str,
        session_id: str
    ) -> Transaction:
        """Create an investment transaction"""
        
        account_id = await self._get_or_create_account(account_type, "Investment Account")
        
        price_per_share = self.market_prices.get(asset_symbol, Decimal("100.00"))
        shares = amount / price_per_share
        
        return Transaction(
            transaction_type=TransactionType.BUY_STOCK,
            amount=amount,
            to_account=account_id,
            asset_symbol=asset_symbol,
            asset_class=AssetClass.STOCKS,
            description=f"Investment in {asset_symbol}",
            shares=shares,
            price_per_share=price_per_share,
            correlation_id=correlation_id,
            session_id=session_id
        )
    
    async def _create_savings_transaction(
        self,
        amount: Decimal,
        account_type: str,
        correlation_id: str,
        session_id: str
    ) -> Transaction:
        """Create a savings transaction"""
        
        account_id = await self._get_or_create_account(account_type, "Savings Account")
        
        return Transaction(
            transaction_type=TransactionType.DEPOSIT,
            amount=amount,
            to_account=account_id,
            asset_class=AssetClass.CASH,
            description=f"Savings deposit",
            correlation_id=correlation_id,
            session_id=session_id
        )
    
    async def _create_contribution_transaction(
        self,
        amount: Decimal,
        account_type: str,
        correlation_id: str,
        session_id: str
    ) -> Transaction:
        """Create a retirement contribution transaction"""
        
        account_id = await self._get_or_create_account(account_type, "Retirement Account")
        
        return Transaction(
            transaction_type=TransactionType.CONTRIBUTION,
            amount=amount,
            to_account=account_id,
            asset_class=AssetClass.CASH,
            description=f"Retirement contribution",
            correlation_id=correlation_id,
            session_id=session_id
        )
    
    async def _get_or_create_account(self, account_type_str: str, account_name: str) -> str:
        """Get existing account or create new one"""
        
        account_type_map = {
            "investment": AccountType.INVESTMENT,
            "savings": AccountType.SAVINGS,
            "checking": AccountType.CHECKING,
            "retirement_401k": AccountType.RETIREMENT_401K,
            "retirement_ira": AccountType.RETIREMENT_IRA,
            "retirement_roth": AccountType.RETIREMENT_ROTH,
            "hsa": AccountType.HSA,
            "emergency_fund": AccountType.EMERGENCY_FUND
        }
        
        account_type = account_type_map.get(account_type_str, AccountType.INVESTMENT)
        
        # Look for existing account of this type
        for account_id, account in self.ledger.accounts.items():
            if account.account_type == account_type:
                return account_id
        
        # Create new account
        tax_advantaged = account_type in [
            AccountType.RETIREMENT_401K,
            AccountType.RETIREMENT_IRA,
            AccountType.RETIREMENT_ROTH,
            AccountType.HSA
        ]
        
        contribution_limits = {}
        if account_type == AccountType.RETIREMENT_401K:
            contribution_limits["annual"] = Decimal("23000")
        elif account_type in [AccountType.RETIREMENT_IRA, AccountType.RETIREMENT_ROTH]:
            contribution_limits["annual"] = Decimal("7000")
        elif account_type == AccountType.HSA:
            contribution_limits["annual"] = Decimal("4300")
        
        return self.ledger.create_account(
            account_type=account_type,
            account_name=account_name,
            initial_balance=Decimal('0'),
            tax_advantaged=tax_advantaged,
            contribution_limits=contribution_limits
        )
    
    async def _execute_transaction(self, transaction: Transaction) -> bool:
        """Execute a transaction with rollback capability"""
        
        try:
            rollback_data = {}
            
            if transaction.from_account:
                from_account = self.ledger.get_account(transaction.from_account)
                if from_account:
                    rollback_data[transaction.from_account] = {
                        "balance": float(from_account.balance),
                        "holdings": dict(from_account.holdings)
                    }
            
            if transaction.to_account:
                to_account = self.ledger.get_account(transaction.to_account)
                if to_account:
                    rollback_data[transaction.to_account] = {
                        "balance": float(to_account.balance),
                        "holdings": dict(to_account.holdings)
                    }
            
            transaction.rollback_data = rollback_data
            
            transaction.status = TransactionStatus.PROCESSING
            success = self.ledger.record_transaction(transaction)
            
            if success:
                transaction.status = TransactionStatus.COMPLETED
                transaction.executed_at = datetime.utcnow()
                self.logger.info(f"Transaction {transaction.transaction_id} executed successfully")
            else:
                transaction.status = TransactionStatus.FAILED
                self.logger.error(f"Transaction {transaction.transaction_id} execution failed")
            
            return success
            
        except Exception as e:
            transaction.status = TransactionStatus.FAILED
            self.logger.error(f"Transaction execution error: {str(e)}")
            return False
    
    # Message handlers for new functionality
    async def _handle_transaction_execution(self, message: AgentMessage) -> AgentMessage:
        """Handle individual transaction execution request"""
        
        transaction_data = message.payload.get("transaction", {})
        
        try:
            transaction = Transaction(
                transaction_type=TransactionType(transaction_data.get("transaction_type", "investment")),
                amount=Decimal(str(transaction_data.get("amount", 0))),
                from_account=transaction_data.get("from_account"),
                to_account=transaction_data.get("to_account"),
                asset_symbol=transaction_data.get("asset_symbol"),
                description=transaction_data.get("description", ""),
                correlation_id=message.correlation_id,
                session_id=message.session_id
            )
            
            success = await self._execute_transaction(transaction)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "transaction_executed",
                    "transaction_id": transaction.transaction_id,
                    "success": success,
                    "status": transaction.status.value,
                    "executed_at": transaction.executed_at.isoformat() if transaction.executed_at else None
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Transaction execution failed: {str(e)}")
    
    async def _handle_portfolio_status_request(self, message: AgentMessage) -> AgentMessage:
        """Handle portfolio status request"""
        
        try:
            portfolio_status = {
                "total_value": float(self.ledger.get_portfolio_value(self.market_prices)),
                "total_balance": float(self.ledger.get_total_balance()),
                "account_summary": self.ledger.get_account_summary(),
                "recent_transactions": [
                    self.ledger.transactions[tid].to_dict()
                    for tid in self.ledger.transaction_history[-10:]
                    if tid in self.ledger.transactions
                ],
                "execution_metrics": {
                    k: float(v) if isinstance(v, Decimal) else v
                    for k, v in self.execution_metrics.items()
                }
            }
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "portfolio_status",
                    "portfolio_status": portfolio_status
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Portfolio status request failed: {str(e)}")
    
    async def _handle_forecast_request(self, message: AgentMessage) -> AgentMessage:
        """Handle forecast generation request"""
        
        try:
            forecast_params = message.payload.get("forecast_params", {})
            time_horizon = forecast_params.get("time_horizon_months", 12)
            
            current_portfolio = {
                "total_value": float(self.ledger.get_portfolio_value(self.market_prices))
            }
            
            # Simple forecast calculation
            expected_return = 0.07  # 7% annual return
            monthly_return = expected_return / 12
            future_value = current_portfolio["total_value"] * ((1 + monthly_return) ** time_horizon)
            
            forecast = {
                "current_value": current_portfolio["total_value"],
                "time_horizon_months": time_horizon,
                "forecasts": {
                    "expected_value": future_value,
                    "median_value": future_value,
                    "confidence_intervals": {
                        "90%": {
                            "lower": future_value * 0.85,
                            "upper": future_value * 1.15
                        }
                    }
                }
            }
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "forecast_generated",
                    "forecast": forecast,
                    "generated_at": datetime.utcnow().isoformat()
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Forecast generation failed: {str(e)}")
    
    async def _handle_compliance_check(self, message: AgentMessage) -> AgentMessage:
        """Handle compliance check request"""
        
        try:
            compliance_report = {
                "report_id": str(uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "total_transactions": len(self.ledger.transactions),
                "compliance_summary": {
                    "compliant_transactions": len(self.ledger.transactions),
                    "non_compliant_transactions": 0,
                    "transactions_with_warnings": 0
                },
                "violations": [],
                "warnings": []
            }
            
            self.execution_metrics["compliance_checks"] += 1
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "compliance_checked",
                    "compliance_report": compliance_report
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Compliance check failed: {str(e)}")
    
    async def _handle_tax_optimization(self, message: AgentMessage) -> AgentMessage:
        """Handle tax optimization request"""
        
        try:
            # Simple tax loss harvesting simulation
            tax_loss_opportunities = []
            
            for account in self.ledger.accounts.values():
                if not account.tax_advantaged:
                    for symbol, holding in account.holdings.items():
                        current_price = self.market_prices.get(symbol, holding["avg_cost"])
                        unrealized_loss = (holding["avg_cost"] - current_price) * holding["shares"]
                        
                        if unrealized_loss > Decimal('100'):
                            tax_loss_opportunities.append({
                                "account_id": account.account_id,
                                "symbol": symbol,
                                "shares": float(holding["shares"]),
                                "cost_basis": float(holding["avg_cost"]),
                                "current_price": float(current_price),
                                "unrealized_loss": float(unrealized_loss),
                                "recommendation": f"Consider selling {symbol} to realize ${float(unrealized_loss):.2f} tax loss"
                            })
            
            self.execution_metrics["tax_optimizations"] += 1
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "tax_optimization_complete",
                    "tax_loss_opportunities": tax_loss_opportunities,
                    "optimization_summary": {
                        "total_opportunities": len(tax_loss_opportunities),
                        "potential_tax_savings": sum(
                            opp["unrealized_loss"] * 0.22
                            for opp in tax_loss_opportunities
                        )
                    }
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Tax optimization failed: {str(e)}")
    
    async def _handle_health_check(self, message: AgentMessage) -> AgentMessage:
        """Handle health check request"""
        
        health_status = self.get_health_status()
        
        health_status.update({
            "ledger_status": {
                "total_accounts": len(self.ledger.accounts),
                "total_transactions": len(self.ledger.transactions),
                "total_portfolio_value": float(self.ledger.get_portfolio_value(self.market_prices))
            },
            "execution_metrics": {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in self.execution_metrics.items()
            },
            "pending_transactions": len(self.pending_transactions),
            "execution_queue_size": len(self.execution_queue)
        })
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=message.agent_id,
            message_type=MessageType.RESPONSE,
            payload={
                "action": "health_check_response",
                "health_status": health_status
            },
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            trace_id=message.trace_id
        )
    
    # Real-time monitoring methods
    async def _handle_start_monitoring(self, message: AgentMessage) -> AgentMessage:
        """Handle start real-time monitoring request"""
        
        try:
            user_id = message.payload.get("user_id")
            monitoring_config = message.payload.get("monitoring_config", {})
            
            monitor_id = await self.start_real_time_monitoring(user_id, monitoring_config)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "monitoring_started",
                    "monitor_id": monitor_id,
                    "user_id": user_id,
                    "status": "active"
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Failed to start monitoring: {str(e)}")
    
    async def _handle_stop_monitoring(self, message: AgentMessage) -> AgentMessage:
        """Handle stop real-time monitoring request"""
        
        try:
            monitor_id = message.payload.get("monitor_id")
            
            success = await self.stop_real_time_monitoring(monitor_id)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "monitoring_stopped",
                    "monitor_id": monitor_id,
                    "success": success
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Failed to stop monitoring: {str(e)}")
    
    # Workflow coordination methods
    async def _handle_register_workflow(self, message: AgentMessage) -> AgentMessage:
        """Handle workflow registration request"""
        
        try:
            workflow_id = message.payload.get("workflow_id")
            workflow_data = message.payload.get("workflow_data", {})
            
            success = await self.register_workflow(workflow_id, workflow_data)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "workflow_registered",
                    "workflow_id": workflow_id,
                    "success": success
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Failed to register workflow: {str(e)}")
    
    async def _handle_update_workflow(self, message: AgentMessage) -> AgentMessage:
        """Handle workflow progress update request"""
        
        try:
            workflow_id = message.payload.get("workflow_id")
            task_result = message.payload.get("task_result", {})
            
            success = await self.update_workflow_progress(workflow_id, task_result)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "workflow_updated",
                    "workflow_id": workflow_id,
                    "success": success
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Failed to update workflow: {str(e)}")
    
    async def _handle_rollback_workflow(self, message: AgentMessage) -> AgentMessage:
        """Handle workflow rollback request"""
        
        try:
            workflow_id = message.payload.get("workflow_id")
            rollback_data = message.payload.get("rollback_data", {})
            
            success = await self.handle_workflow_rollback(workflow_id, rollback_data)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "action": "workflow_rollback_complete",
                    "workflow_id": workflow_id,
                    "success": success
                },
                correlation_id=message.correlation_id,
                session_id=message.session_id,
                trace_id=message.trace_id
            )
            
        except Exception as e:
            return self._create_error_response(message, f"Failed to rollback workflow: {str(e)}")
    
    def _create_error_response(self, original_message: AgentMessage, error_message: str) -> AgentMessage:
        """Create standardized error response"""
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=original_message.agent_id,
            message_type=MessageType.ERROR,
            payload={
                "error": error_message,
                "original_action": original_message.payload.get("action", "unknown")
            },
            correlation_id=original_message.correlation_id,
            session_id=original_message.session_id,
            trace_id=original_message.trace_id
        )
    
    # Real-time monitoring implementation
    async def start_real_time_monitoring(self, user_id: str, monitoring_config: Dict[str, Any] = None) -> str:
        """Start real-time portfolio monitoring for a user"""
        
        monitor_id = str(uuid4())
        
        monitor_config = monitoring_config or {
            "portfolio_tracking": True,
            "risk_monitoring": True,
            "compliance_monitoring": True,
            "performance_tracking": True,
            "alert_thresholds": {
                "portfolio_change_percent": 5.0,
                "risk_score_change": 0.1,
                "compliance_violations": 1
            }
        }
        
        self.real_time_monitors[monitor_id] = {
            "monitor_id": monitor_id,
            "user_id": user_id,
            "config": monitor_config,
            "created_at": datetime.utcnow(),
            "last_update": datetime.utcnow(),
            "status": "active",
            "alerts_generated": 0,
            "updates_sent": 0
        }
        
        # Start monitoring if not already active
        if not self.monitoring_active:
            self.monitoring_active = True
            asyncio.create_task(self._run_real_time_monitoring())
        
        self.logger.info(f"Started real-time monitoring {monitor_id} for user {user_id}")
        return monitor_id
    
    async def stop_real_time_monitoring(self, monitor_id: str) -> bool:
        """Stop real-time monitoring"""
        
        if monitor_id not in self.real_time_monitors:
            return False
        
        monitor = self.real_time_monitors[monitor_id]
        monitor["status"] = "stopped"
        monitor["stopped_at"] = datetime.utcnow()
        
        self.logger.info(f"Stopped real-time monitoring {monitor_id}")
        return True
    
    async def _run_real_time_monitoring(self) -> None:
        """Background task for real-time monitoring"""
        
        while self.monitoring_active:
            try:
                active_monitors = [
                    m for m in self.real_time_monitors.values()
                    if m["status"] == "active"
                ]
                
                if not active_monitors:
                    self.monitoring_active = False
                    break
                
                # Process each active monitor
                for monitor in active_monitors:
                    await self._process_monitor_update(monitor)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in real-time monitoring: {str(e)}")
                await asyncio.sleep(self.monitoring_interval * 2)  # Wait longer on error
    
    async def _process_monitor_update(self, monitor: Dict[str, Any]) -> None:
        """Process a single monitor update"""
        
        try:
            # Update monitor state
            monitor["last_update"] = datetime.utcnow()
            monitor["updates_sent"] += 1
            self.execution_metrics["real_time_updates"] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing monitor update: {str(e)}")
    
    # Workflow coordination implementation
    async def register_workflow(self, workflow_id: str, workflow_data: Dict[str, Any]) -> bool:
        """Register a workflow for coordination"""
        
        try:
            self.active_workflows[workflow_id] = {
                "workflow_id": workflow_id,
                "workflow_type": workflow_data.get("workflow_type", "unknown"),
                "user_id": workflow_data.get("user_id"),
                "registered_at": datetime.utcnow(),
                "status": "registered",
                "tasks_completed": 0,
                "total_tasks": workflow_data.get("total_tasks", 0),
                "execution_results": []
            }
            
            self.logger.info(f"Registered workflow {workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register workflow {workflow_id}: {str(e)}")
            return False
    
    async def update_workflow_progress(self, workflow_id: str, task_result: Dict[str, Any]) -> bool:
        """Update workflow progress with task result"""
        
        if workflow_id not in self.active_workflows:
            return False
        
        try:
            workflow = self.active_workflows[workflow_id]
            workflow["tasks_completed"] += 1
            workflow["execution_results"].append({
                "task_id": task_result.get("task_id"),
                "result": task_result.get("result"),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Check if workflow is complete
            if workflow["tasks_completed"] >= workflow["total_tasks"]:
                workflow["status"] = "completed"
                workflow["completed_at"] = datetime.utcnow()
                
                # Move to results
                self.workflow_results[workflow_id] = workflow
                del self.active_workflows[workflow_id]
                
                self.logger.info(f"Workflow {workflow_id} completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update workflow progress: {str(e)}")
            return False
    
    async def handle_workflow_rollback(self, workflow_id: str, rollback_data: Dict[str, Any]) -> bool:
        """Handle workflow rollback request"""
        
        try:
            self.logger.info(f"Handling rollback for workflow {workflow_id}")
            
            # Execute rollback transactions
            rollback_transactions = rollback_data.get("transactions", [])
            
            for transaction_data in rollback_transactions:
                # Create rollback transaction
                rollback_transaction = Transaction(
                    transaction_type=TransactionType.TRANSFER,  # Generic rollback
                    amount=Decimal(str(transaction_data.get("amount", 0))),
                    from_account=transaction_data.get("to_account"),  # Reverse
                    to_account=transaction_data.get("from_account"),  # Reverse
                    description=f"Rollback for workflow {workflow_id}",
                    correlation_id=workflow_id
                )
                
                # Execute rollback
                success = await self._execute_transaction(rollback_transaction)
                if success:
                    self.execution_metrics["rollback_transactions"] += 1
            
            self.logger.info(f"Workflow {workflow_id} rollback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback workflow {workflow_id}: {str(e)}")
            return False
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the execution agent"""
        
        base_status = self.get_health_status()
        
        return {
            **base_status,
            "ledger_summary": self.ledger.get_account_summary(),
            "portfolio_value": float(self.ledger.get_portfolio_value(self.market_prices)),
            "execution_metrics": {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in self.execution_metrics.items()
            },
            "recent_transactions": [
                self.ledger.transactions[tid].to_dict()
                for tid in self.ledger.transaction_history[-5:]
                if tid in self.ledger.transactions
            ],
            "market_prices": {k: float(v) for k, v in self.market_prices.items()},
            "compliance_status": "operational",
            "real_time_monitoring": {
                "active": self.monitoring_active,
                "monitors_count": len(self.real_time_monitors),
                "monitoring_interval": self.monitoring_interval
            },
            "workflow_coordination": {
                "active_workflows": len(self.active_workflows),
                "completed_workflows": len(self.workflow_results)
            }
        }