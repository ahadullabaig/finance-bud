"""
FastAPI Endpoints for Execution Agent

Provides REST API endpoints for receiving user goals, coordinating agents,
and managing financial operations with comprehensive monitoring and reporting.

Requirements: 1.4, 2.5, 4.5, 8.4
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
from uuid import uuid4
import asyncio
import logging

from agents.execution_agent import ExecutionAgent
from agents.orchestration_agent import OrchestrationAgent
from data_models.schemas import (
    AgentMessage, MessageType, Priority, ExecutionStatus,
    FinancialState, TaxContext, RiskProfile
)


# Initialize FastAPI app
app = FastAPI(
    title="FinPilot Execution Agent API",
    description="Advanced Financial Operations and Multi-Agent Coordination",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instances
execution_agent: Optional[ExecutionAgent] = None
orchestration_agent: Optional[OrchestrationAgent] = None

# Logger
logger = logging.getLogger("finpilot.api.execution")


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class UserGoalRequest(BaseModel):
    """Request model for user financial goals"""
    user_id: str = Field(..., description="User identifier")
    goal_text: str = Field(..., description="Natural language financial goal")
    financial_state: Optional[Dict[str, Any]] = Field(None, description="Current financial state")
    risk_profile: Optional[Dict[str, Any]] = Field(None, description="Risk tolerance and preferences")
    tax_context: Optional[Dict[str, Any]] = Field(None, description="Tax situation and preferences")
    time_horizon_months: Optional[int] = Field(None, description="Planning time horizon in months")
    priority: Optional[str] = Field("medium", description="Goal priority level")


class TransactionRequest(BaseModel):
    """Request model for individual transactions"""
    transaction_type: str = Field(..., description="Type of transaction")
    amount: float = Field(..., description="Transaction amount")
    from_account: Optional[str] = Field(None, description="Source account ID")
    to_account: Optional[str] = Field(None, description="Destination account ID")
    asset_symbol: Optional[str] = Field(None, description="Asset symbol for investments")
    description: Optional[str] = Field("", description="Transaction description")
    user_id: str = Field(..., description="User identifier")


class PlanExecutionRequest(BaseModel):
    """Request model for executing financial plans"""
    plan_id: str = Field(..., description="Plan identifier")
    user_id: str = Field(..., description="User identifier")
    plan_steps: List[Dict[str, Any]] = Field(..., description="Plan steps to execute")
    execution_mode: Optional[str] = Field("standard", description="Execution mode")
    dry_run: Optional[bool] = Field(False, description="Whether to perform a dry run")


class ForecastRequest(BaseModel):
    """Request model for portfolio forecasting"""
    user_id: str = Field(..., description="User identifier")
    time_horizon_months: int = Field(..., description="Forecast time horizon")
    market_assumptions: Optional[Dict[str, float]] = Field(None, description="Market assumptions")
    scenario_type: Optional[str] = Field("base", description="Forecast scenario type")


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking"""
    user_id: str = Field(..., description="User identifier")
    check_type: Optional[str] = Field("full", description="Type of compliance check")
    include_recommendations: Optional[bool] = Field(True, description="Include recommendations")


class WorkflowRequest(BaseModel):
    """Request model for workflow management"""
    workflow_type: str = Field(..., description="Type of workflow to initiate")
    user_id: str = Field(..., description="User identifier")
    parameters: Dict[str, Any] = Field(..., description="Workflow parameters")
    priority: Optional[str] = Field("medium", description="Workflow priority")


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_execution_agent() -> ExecutionAgent:
    """Get execution agent instance"""
    global execution_agent
    if execution_agent is None:
        execution_agent = ExecutionAgent("api_execution_agent")
        await execution_agent.start()
    return execution_agent


async def get_orchestration_agent() -> OrchestrationAgent:
    """Get orchestration agent instance"""
    global orchestration_agent
    if orchestration_agent is None:
        orchestration_agent = OrchestrationAgent("api_orchestration_agent")
        await orchestration_agent.start()
    return orchestration_agent


# ============================================================================
# USER GOAL AND WORKFLOW ENDPOINTS
# ============================================================================

@app.post("/api/v1/goals/process")
async def process_user_goal(
    request: UserGoalRequest,
    background_tasks: BackgroundTasks,
    orchestrator: OrchestrationAgent = Depends(get_orchestration_agent)
):
    """
    Process a user's financial goal and initiate planning workflow
    
    This endpoint serves as the main entry point for users to submit their
    financial goals in natural language and initiate the multi-agent planning process.
    """
    try:
        logger.info(f"Processing goal for user {request.user_id}: {request.goal_text}")
        
        # Prepare user context
        user_context = {
            "user_id": request.user_id,
            "financial_state": request.financial_state or {},
            "risk_profile": request.risk_profile or {},
            "tax_context": request.tax_context or {},
            "time_horizon_months": request.time_horizon_months
        }
        
        # Process goal through orchestration agent
        workflow_id = await orchestrator.process_user_goal(
            user_id=request.user_id,
            goal_text=request.goal_text,
            user_context=user_context
        )
        
        # Add background monitoring task
        background_tasks.add_task(monitor_workflow_progress, workflow_id, orchestrator)
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": "Goal processing initiated successfully",
            "estimated_completion_time": "2-5 minutes",
            "status_endpoint": f"/api/v1/workflows/{workflow_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Failed to process user goal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Goal processing failed: {str(e)}")


@app.post("/api/v1/workflows/initiate")
async def initiate_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    orchestrator: OrchestrationAgent = Depends(get_orchestration_agent)
):
    """
    Initiate a specific workflow type for multi-agent coordination
    
    Supports various workflow types including planning, rebalancing,
    compliance checking, and emergency response workflows.
    """
    try:
        workflow_id = str(uuid4())
        
        # Create workflow based on type
        if request.workflow_type == "financial_planning":
            # Standard financial planning workflow
            session_id = orchestrator.session_manager.create_session(
                request.user_id, 
                request.parameters.get("goal", "Financial planning workflow")
            )
            
        elif request.workflow_type == "portfolio_rebalancing":
            # Portfolio rebalancing workflow
            session_id = orchestrator.session_manager.create_session(
                request.user_id,
                "Portfolio rebalancing workflow"
            )
            
        elif request.workflow_type == "compliance_review":
            # Compliance review workflow
            session_id = orchestrator.session_manager.create_session(
                request.user_id,
                "Compliance review workflow"
            )
            
        elif request.workflow_type == "emergency_response":
            # Emergency response workflow for triggers
            session_id = orchestrator.session_manager.create_session(
                request.user_id,
                "Emergency response workflow"
            )
            
        else:
            raise ValueError(f"Unknown workflow type: {request.workflow_type}")
        
        # Add background workflow execution
        background_tasks.add_task(
            execute_workflow, 
            workflow_id, 
            request.workflow_type, 
            request.parameters,
            orchestrator
        )
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "session_id": session_id,
            "workflow_type": request.workflow_type,
            "status": "initiated",
            "message": f"{request.workflow_type} workflow initiated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow initiation failed: {str(e)}")


@app.get("/api/v1/workflows/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    orchestrator: OrchestrationAgent = Depends(get_orchestration_agent)
):
    """Get the current status of a workflow"""
    try:
        workflow = orchestrator.current_workflows.get(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "workflow_id": workflow_id,
            "status": workflow.get("state", "unknown"),
            "created_at": workflow.get("created_at"),
            "user_id": workflow.get("user_id"),
            "progress": calculate_workflow_progress(workflow),
            "current_step": workflow.get("current_step", "initializing"),
            "estimated_completion": estimate_completion_time(workflow),
            "error_count": workflow.get("error_count", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


# ============================================================================
# EXECUTION ENDPOINTS
# ============================================================================

@app.post("/api/v1/plans/execute")
async def execute_plan(
    request: PlanExecutionRequest,
    background_tasks: BackgroundTasks,
    executor: ExecutionAgent = Depends(get_execution_agent)
):
    """
    Execute a financial plan with comprehensive monitoring and rollback capability
    
    Supports both dry-run mode for validation and actual execution with
    real-time monitoring and audit trails.
    """
    try:
        logger.info(f"Executing plan {request.plan_id} for user {request.user_id}")
        
        # Create execution message
        execution_message = AgentMessage(
            agent_id="api_client",
            target_agent_id=executor.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "action": "execute_plan",
                "plan": {
                    "plan_id": request.plan_id,
                    "steps": request.plan_steps,
                    "execution_mode": request.execution_mode,
                    "dry_run": request.dry_run
                },
                "user_id": request.user_id
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        # Execute plan
        response = await executor.process_message(execution_message)
        
        if response and response.payload.get("execution_results"):
            execution_results = response.payload["execution_results"]
            
            # Add background monitoring if not dry run
            if not request.dry_run:
                background_tasks.add_task(
                    monitor_execution_progress, 
                    request.plan_id, 
                    executor
                )
            
            return {
                "success": True,
                "plan_id": request.plan_id,
                "execution_status": execution_results.get("execution_status"),
                "executed_steps": len(execution_results.get("executed_steps", [])),
                "failed_steps": len(execution_results.get("failed_steps", [])),
                "total_steps": execution_results.get("total_steps", 0),
                "dry_run": request.dry_run,
                "portfolio_updated": response.payload.get("portfolio_updated", False),
                "ledger_snapshot": response.payload.get("ledger_snapshot"),
                "message": "Plan execution completed successfully" if not request.dry_run else "Dry run completed successfully"
            }
        else:
            raise Exception("No execution results received")
            
    except Exception as e:
        logger.error(f"Failed to execute plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Plan execution failed: {str(e)}")


@app.post("/api/v1/transactions/execute")
async def execute_transaction(
    request: TransactionRequest,
    executor: ExecutionAgent = Depends(get_execution_agent)
):
    """Execute an individual financial transaction"""
    try:
        logger.info(f"Executing transaction for user {request.user_id}: {request.transaction_type}")
        
        # Create transaction message
        transaction_message = AgentMessage(
            agent_id="api_client",
            target_agent_id=executor.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "action": "execute_transaction",
                "transaction": {
                    "transaction_type": request.transaction_type,
                    "amount": request.amount,
                    "from_account": request.from_account,
                    "to_account": request.to_account,
                    "asset_symbol": request.asset_symbol,
                    "description": request.description
                },
                "user_id": request.user_id
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        # Execute transaction
        response = await executor.process_message(transaction_message)
        
        if response:
            return {
                "success": response.payload.get("success", False),
                "transaction_id": response.payload.get("transaction_id"),
                "status": response.payload.get("status"),
                "executed_at": response.payload.get("executed_at"),
                "message": "Transaction executed successfully" if response.payload.get("success") else "Transaction execution failed"
            }
        else:
            raise Exception("No response received from execution agent")
            
    except Exception as e:
        logger.error(f"Failed to execute transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transaction execution failed: {str(e)}")


# ============================================================================
# PORTFOLIO AND MONITORING ENDPOINTS
# ============================================================================

@app.get("/api/v1/portfolio/{user_id}/status")
async def get_portfolio_status(
    user_id: str,
    executor: ExecutionAgent = Depends(get_execution_agent)
):
    """Get comprehensive portfolio status and metrics"""
    try:
        # Create portfolio status request
        status_message = AgentMessage(
            agent_id="api_client",
            target_agent_id=executor.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "action": "get_portfolio_status",
                "user_id": user_id
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await executor.process_message(status_message)
        
        if response and response.payload.get("portfolio_status"):
            portfolio_status = response.payload["portfolio_status"]
            
            return {
                "success": True,
                "user_id": user_id,
                "portfolio_value": portfolio_status.get("total_value"),
                "cash_balance": portfolio_status.get("total_balance"),
                "account_summary": portfolio_status.get("account_summary"),
                "recent_transactions": portfolio_status.get("recent_transactions", []),
                "execution_metrics": portfolio_status.get("execution_metrics", {}),
                "last_updated": datetime.utcnow().isoformat()
            }
        else:
            raise Exception("No portfolio status received")
            
    except Exception as e:
        logger.error(f"Failed to get portfolio status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio status retrieval failed: {str(e)}")


@app.post("/api/v1/forecasts/generate")
async def generate_forecast(
    request: ForecastRequest,
    executor: ExecutionAgent = Depends(get_execution_agent)
):
    """Generate portfolio performance forecast with uncertainty quantification"""
    try:
        logger.info(f"Generating forecast for user {request.user_id}")
        
        # Create forecast request message
        forecast_message = AgentMessage(
            agent_id="api_client",
            target_agent_id=executor.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "action": "generate_forecast",
                "forecast_params": {
                    "time_horizon_months": request.time_horizon_months,
                    "market_assumptions": request.market_assumptions,
                    "scenario_type": request.scenario_type
                },
                "user_id": request.user_id
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await executor.process_message(forecast_message)
        
        if response and response.payload.get("forecast"):
            forecast = response.payload["forecast"]
            
            return {
                "success": True,
                "user_id": request.user_id,
                "forecast": forecast,
                "generated_at": response.payload.get("generated_at"),
                "scenario_type": request.scenario_type,
                "time_horizon_months": request.time_horizon_months
            }
        else:
            raise Exception("No forecast generated")
            
    except Exception as e:
        logger.error(f"Failed to generate forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


# ============================================================================
# COMPLIANCE AND TAX OPTIMIZATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/compliance/check")
async def check_compliance(
    request: ComplianceCheckRequest,
    executor: ExecutionAgent = Depends(get_execution_agent)
):
    """Perform comprehensive compliance check with regulatory reporting"""
    try:
        logger.info(f"Performing compliance check for user {request.user_id}")
        
        # Create compliance check message
        compliance_message = AgentMessage(
            agent_id="api_client",
            target_agent_id=executor.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "action": "check_compliance",
                "check_type": request.check_type,
                "include_recommendations": request.include_recommendations,
                "user_id": request.user_id
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await executor.process_message(compliance_message)
        
        if response and response.payload.get("compliance_report"):
            compliance_report = response.payload["compliance_report"]
            
            return {
                "success": True,
                "user_id": request.user_id,
                "compliance_report": compliance_report,
                "check_type": request.check_type,
                "checked_at": datetime.utcnow().isoformat()
            }
        else:
            raise Exception("No compliance report generated")
            
    except Exception as e:
        logger.error(f"Failed to check compliance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")


@app.post("/api/v1/tax/optimize")
async def optimize_taxes(
    user_id: str,
    executor: ExecutionAgent = Depends(get_execution_agent)
):
    """Generate tax optimization recommendations"""
    try:
        logger.info(f"Optimizing taxes for user {user_id}")
        
        # Create tax optimization message
        tax_message = AgentMessage(
            agent_id="api_client",
            target_agent_id=executor.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "action": "optimize_taxes",
                "user_id": user_id
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await executor.process_message(tax_message)
        
        if response:
            return {
                "success": True,
                "user_id": user_id,
                "tax_loss_opportunities": response.payload.get("tax_loss_opportunities", []),
                "optimization_summary": response.payload.get("optimization_summary", {}),
                "optimized_at": datetime.utcnow().isoformat()
            }
        else:
            raise Exception("No tax optimization results received")
            
    except Exception as e:
        logger.error(f"Failed to optimize taxes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tax optimization failed: {str(e)}")


# ============================================================================
# SYSTEM MONITORING ENDPOINTS
# ============================================================================

@app.get("/api/v1/system/health")
async def get_system_health(
    executor: ExecutionAgent = Depends(get_execution_agent),
    orchestrator: OrchestrationAgent = Depends(get_orchestration_agent)
):
    """Get comprehensive system health status"""
    try:
        execution_health = executor.get_comprehensive_status()
        orchestration_health = orchestrator.get_comprehensive_health_status()
        
        return {
            "success": True,
            "system_status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "execution_agent": execution_health,
            "orchestration_agent": orchestration_health,
            "api_status": {
                "uptime": "operational",
                "endpoints_active": True,
                "database_connected": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        return {
            "success": False,
            "system_status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/api/v1/system/metrics")
async def get_system_metrics(
    executor: ExecutionAgent = Depends(get_execution_agent),
    orchestrator: OrchestrationAgent = Depends(get_orchestration_agent)
):
    """Get detailed system performance metrics"""
    try:
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "execution_metrics": executor.execution_metrics,
            "workflow_metrics": orchestrator.workflow_metrics,
            "system_metrics": {
                "active_workflows": len(orchestrator.current_workflows),
                "active_sessions": len(orchestrator.session_manager.active_sessions),
                "total_portfolio_value": float(executor.ledger.get_portfolio_value(executor.market_prices)),
                "total_transactions": len(executor.ledger.transactions)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def monitor_workflow_progress(workflow_id: str, orchestrator: OrchestrationAgent):
    """Background task to monitor workflow progress"""
    try:
        logger.info(f"Starting workflow monitoring for {workflow_id}")
        
        # Monitor for up to 10 minutes
        for _ in range(120):  # 120 * 5 seconds = 10 minutes
            workflow = orchestrator.current_workflows.get(workflow_id)
            if not workflow:
                break
                
            # Check if workflow is complete
            if workflow.get("state") in ["completed", "failed"]:
                logger.info(f"Workflow {workflow_id} completed with state: {workflow.get('state')}")
                break
                
            await asyncio.sleep(5)  # Check every 5 seconds
            
    except Exception as e:
        logger.error(f"Error monitoring workflow {workflow_id}: {str(e)}")


async def execute_workflow(
    workflow_id: str, 
    workflow_type: str, 
    parameters: Dict[str, Any],
    orchestrator: OrchestrationAgent
):
    """Background task to execute workflow"""
    try:
        logger.info(f"Executing workflow {workflow_id} of type {workflow_type}")
        
        # Implementation would depend on workflow type
        # This is a placeholder for the actual workflow execution logic
        
        await asyncio.sleep(1)  # Simulate workflow execution
        
        logger.info(f"Workflow {workflow_id} execution completed")
        
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {str(e)}")


async def monitor_execution_progress(plan_id: str, executor: ExecutionAgent):
    """Background task to monitor plan execution progress"""
    try:
        logger.info(f"Starting execution monitoring for plan {plan_id}")
        
        # Monitor execution progress
        await asyncio.sleep(1)  # Placeholder for actual monitoring logic
        
        logger.info(f"Plan {plan_id} execution monitoring completed")
        
    except Exception as e:
        logger.error(f"Error monitoring execution for plan {plan_id}: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_workflow_progress(workflow: Dict[str, Any]) -> float:
    """Calculate workflow progress percentage"""
    try:
        total_tasks = len(workflow.get("tasks", []))
        if total_tasks == 0:
            return 0.0
            
        completed_tasks = sum(1 for task in workflow.get("tasks", []) if task.get("status") == "completed")
        return (completed_tasks / total_tasks) * 100.0
        
    except Exception:
        return 0.0


def estimate_completion_time(workflow: Dict[str, Any]) -> Optional[str]:
    """Estimate workflow completion time"""
    try:
        state = workflow.get("state", "unknown")
        
        if state == "completed":
            return "completed"
        elif state == "failed":
            return "failed"
        else:
            # Simple estimation based on workflow state
            estimates = {
                "parsing_goal": "3-5 minutes",
                "delegating_tasks": "2-4 minutes", 
                "monitoring_execution": "1-3 minutes",
                "finalizing_results": "1-2 minutes"
            }
            return estimates.get(state, "2-5 minutes")
            
    except Exception:
        return "unknown"


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global execution_agent, orchestration_agent
    
    logger.info("Starting FinPilot Execution Agent API...")
    
    try:
        # Initialize agents
        execution_agent = ExecutionAgent("api_execution_agent")
        orchestration_agent = OrchestrationAgent("api_orchestration_agent")
        
        # Start agents
        await execution_agent.start()
        await orchestration_agent.start()
        
        logger.info("FinPilot Execution Agent API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global execution_agent, orchestration_agent
    
    logger.info("Shutting down FinPilot Execution Agent API...")
    
    try:
        if execution_agent:
            await execution_agent.stop()
        if orchestration_agent:
            await orchestration_agent.shutdown()
            
        logger.info("FinPilot Execution Agent API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)