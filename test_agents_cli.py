#!/usr/bin/env python
"""
CLI Agent Testing Tool

Interactive command-line interface for testing FinPilot agents.
Allows testing verification, planning, and CMVL workflows.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from agents.verifier import VerificationAgent
from agents.mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent
)
from data_models.schemas import (
    AgentMessage, MessageType, Priority,
    TriggerEvent, MarketEventType, SeverityLevel
)
from utils.reason_graph_mapper import ReasonGraphMapper


class AgentCLI:
    """Command-line interface for testing agents"""
    
    def __init__(self):
        print("🚀 Initializing FinPilot Agents...")
        self.orchestration = MockOrchestrationAgent()
        self.planning = MockPlanningAgent()
        self.information_retrieval = MockInformationRetrievalAgent()
        self.verification = VerificationAgent()
        self.mapper = ReasonGraphMapper()
        print("✅ All agents initialized!\n")
    
    async def test_verification(self, plan_steps: list = None):
        """Test verification agent with sample plan"""
        print("\n" + "="*60)
        print("🔍 TESTING VERIFICATION AGENT")
        print("="*60 + "\n")
        
        if plan_steps is None:
            plan_steps = [
                {
                    "step_id": "step_1",
                    "sequence_number": 1,
                    "action_type": "emergency_fund",
                    "description": "Build emergency fund to 6 months expenses",
                    "amount": 30000,
                    "risk_level": "low",
                    "confidence_score": 0.9
                },
                {
                    "step_id": "step_2",
                    "sequence_number": 2,
                    "action_type": "equity_investment",
                    "description": "Invest in diversified portfolio",
                    "amount": 50000,
                    "risk_level": "medium",
                    "confidence_score": 0.8
                }
            ]
        
        print("📋 Plan Steps:")
        for step in plan_steps:
            print(f"  • {step['action_type']}: ${step['amount']:,} ({step['risk_level']} risk)")
        
        message = AgentMessage(
            agent_id="cli_tester",
            target_agent_id=self.verification.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "verification_request": {
                    "plan_id": "cli_test_001",
                    "plan_steps": plan_steps,
                    "verification_level": "comprehensive"
                }
            },
            correlation_id="cli_test_001",
            session_id="cli_session_001",
            trace_id="cli_trace_001"
        )
        
        print("\n⏳ Running verification...")
        response = await self.verification.process_message(message)
        
        if response:
            report = response.payload["verification_report"]
            print(f"\n✅ Verification Complete!")
            print(f"   Status: {report['verification_status']}")
            print(f"   Risk Score: {report['overall_risk_score']:.2%}")
            print(f"   Confidence: {report['confidence_score']:.2%}")
            print(f"   Constraints Checked: {report['constraints_checked']}")
            print(f"   Constraints Passed: {report['constraints_passed']}")
            print(f"   Verification Time: {report['verification_time']:.3f}s")
            
            if response.payload.get("step_results"):
                print("\n📊 Step Results:")
                for result in response.payload["step_results"]:
                    status_icon = "✅" if result["status"] == "passed" else "❌"
                    print(f"   {status_icon} {result['action_type']}: {result['compliance_score']:.1%} compliance")
                    if result.get("violations"):
                        for violation in result["violations"]:
                            print(f"      ⚠️  {violation['description']}")
            
            if response.payload.get("recommendations"):
                print("\n💡 Recommendations:")
                for rec in response.payload["recommendations"]:
                    print(f"   • {rec}")
            
            return response.payload
        
        return None
    
    async def test_cmvl(self, severity: str = "medium"):
        """Test CMVL trigger"""
        print("\n" + "="*60)
        print(f"🚨 TESTING CMVL TRIGGER ({severity.upper()} SEVERITY)")
        print("="*60 + "\n")
        
        severity_map = {
            "low": SeverityLevel.LOW,
            "medium": SeverityLevel.MEDIUM,
            "high": SeverityLevel.HIGH,
            "critical": SeverityLevel.CRITICAL
        }
        
        trigger_event = {
            "trigger_type": "market_event",
            "event_type": "volatility_spike",
            "severity": severity,
            "description": f"Simulated {severity} severity market volatility event",
            "source_data": {"volatility": 0.45},
            "impact_score": 0.8,
            "confidence_score": 0.9,
            "detector_agent_id": "ira_001",
            "correlation_id": "cli_cmvl_001"
        }
        
        print(f"📡 Trigger Event: {trigger_event['description']}")
        print(f"   Impact Score: {trigger_event['impact_score']:.1%}")
        print(f"   Confidence: {trigger_event['confidence_score']:.1%}")
        
        message = AgentMessage(
            agent_id="cli_tester",
            target_agent_id=self.verification.agent_id,
            message_type=MessageType.REQUEST,
            payload={"cmvl_trigger": trigger_event},
            correlation_id="cli_cmvl_001",
            session_id="cli_session_002",
            trace_id="cli_trace_002"
        )
        
        print("\n⏳ Activating CMVL...")
        response = await self.verification.process_message(message)
        
        if response:
            cmvl_data = response.payload
            print(f"\n✅ CMVL Activated!")
            print(f"   CMVL ID: {cmvl_data['cmvl_id']}")
            print(f"   Monitoring Frequency: {cmvl_data['monitoring_frequency']}")
            print(f"   Auto-Remediation: {'Enabled' if cmvl_data['auto_remediation'] else 'Disabled'}")
            print(f"   Next Check: {cmvl_data['next_check']}")
            
            print("\n🔧 Verification Actions:")
            for action in cmvl_data['verification_actions']:
                print(f"   • {action.replace('_', ' ').title()}")
            
            return cmvl_data
        
        return None
    
    async def test_complete_workflow(self, user_goal: str = None):
        """Test complete workflow from goal to verification"""
        print("\n" + "="*60)
        print("🎯 TESTING COMPLETE WORKFLOW")
        print("="*60 + "\n")
        
        if user_goal is None:
            user_goal = "Save $100,000 for retirement in 10 years"
        
        print(f"💭 User Goal: {user_goal}\n")
        
        # Step 1: Orchestration
        print("1️⃣  Orchestration Agent...")
        orch_message = AgentMessage(
            agent_id="cli_tester",
            target_agent_id=self.orchestration.agent_id,
            message_type=MessageType.REQUEST,
            payload={"user_goal": user_goal},
            correlation_id="cli_workflow_001",
            session_id="cli_workflow_session",
            trace_id="cli_workflow_trace"
        )
        orch_response = await self.orchestration.process_message(orch_message)
        print(f"   ✅ Workflow initiated: {orch_response.payload['status']}")
        
        # Step 2: Planning
        print("\n2️⃣  Planning Agent...")
        planning_message = AgentMessage(
            agent_id="cli_tester",
            target_agent_id=self.planning.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": user_goal,
                    "time_horizon": 120
                }
            },
            correlation_id="cli_workflow_001",
            session_id="cli_workflow_session",
            trace_id="cli_workflow_trace"
        )
        planning_response = await self.planning.process_message(planning_message)
        print(f"   ✅ Plan generated: {planning_response.payload['selected_strategy']}")
        print(f"   📊 {len(planning_response.payload['plan_steps'])} steps, "
              f"{len(planning_response.payload['search_paths'])} strategies explored")
        
        # Step 3: Verification
        print("\n3️⃣  Verification Agent...")
        verification_message = AgentMessage(
            agent_id="cli_tester",
            target_agent_id=self.verification.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "verification_request": {
                    "plan_id": "cli_workflow_plan",
                    "plan_steps": planning_response.payload["plan_steps"],
                    "verification_level": "comprehensive"
                }
            },
            correlation_id="cli_workflow_001",
            session_id="cli_workflow_session",
            trace_id="cli_workflow_trace"
        )
        verification_response = await self.verification.process_message(verification_message)
        report = verification_response.payload["verification_report"]
        print(f"   ✅ Verification complete: {report['verification_status']}")
        print(f"   🎯 Risk Score: {report['overall_risk_score']:.2%}")
        
        # Step 4: Generate ReasonGraph
        print("\n4️⃣  Generating ReasonGraph...")
        planning_graph = self.mapper.map_planning_trace(planning_response.payload)
        verification_graph = self.mapper.map_verification_trace(
            verification_response.payload, 
            planning_graph
        )
        final_graph = self.mapper.merge_graphs(planning_graph, verification_graph)
        print(f"   ✅ ReasonGraph generated: {len(final_graph['nodes'])} nodes, "
              f"{len(final_graph['edges'])} edges")
        
        # Summary
        print("\n" + "="*60)
        print("📈 WORKFLOW SUMMARY")
        print("="*60)
        print(f"Goal: {user_goal}")
        print(f"Strategy: {planning_response.payload['selected_strategy']}")
        print(f"Steps: {len(planning_response.payload['plan_steps'])}")
        print(f"Status: {report['verification_status']}")
        print(f"Risk: {report['overall_risk_score']:.2%}")
        print(f"Confidence: {report['confidence_score']:.2%}")
        
        return {
            "orchestration": orch_response.payload,
            "planning": planning_response.payload,
            "verification": verification_response.payload,
            "reason_graph": final_graph
        }
    
    async def test_high_risk_plan(self):
        """Test verification with high-risk plan that should be rejected"""
        print("\n" + "="*60)
        print("⚠️  TESTING HIGH-RISK PLAN (SHOULD BE REJECTED)")
        print("="*60 + "\n")
        
        risky_steps = [
            {
                "step_id": "step_1",
                "action_type": "high_risk_investment",
                "description": "Invest in speculative options",
                "amount": 150000,  # Exceeds limit
                "risk_level": "high"
            },
            {
                "step_id": "step_2",
                "action_type": "leveraged_trading",
                "description": "Use margin for trading",
                "amount": 200000,  # Way over limit
                "risk_level": "high"
            }
        ]
        
        return await self.test_verification(risky_steps)
    
    def save_results(self, results: Dict[str, Any], filename: str = "test_results.json"):
        """Save test results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {filename}")


async def main():
    """Main CLI interface"""
    cli = AgentCLI()
    
    print("="*60)
    print("🤖 FINPILOT AGENT CLI TESTER")
    print("="*60)
    print("\nAvailable Tests:")
    print("  1. Test Verification Agent (valid plan)")
    print("  2. Test Verification Agent (high-risk plan)")
    print("  3. Test CMVL Trigger (medium severity)")
    print("  4. Test CMVL Trigger (critical severity)")
    print("  5. Test Complete Workflow")
    print("  6. Run All Tests")
    print("  0. Exit")
    
    while True:
        print("\n" + "-"*60)
        choice = input("\nSelect test (0-6): ").strip()
        
        if choice == "0":
            print("\n👋 Goodbye!")
            break
        
        elif choice == "1":
            await cli.test_verification()
        
        elif choice == "2":
            await cli.test_high_risk_plan()
        
        elif choice == "3":
            await cli.test_cmvl("medium")
        
        elif choice == "4":
            await cli.test_cmvl("critical")
        
        elif choice == "5":
            goal = input("\nEnter financial goal (or press Enter for default): ").strip()
            results = await cli.test_complete_workflow(goal if goal else None)
            save = input("\nSave results to file? (y/n): ").strip().lower()
            if save == 'y':
                cli.save_results(results)
        
        elif choice == "6":
            print("\n🚀 Running all tests...\n")
            await cli.test_verification()
            await cli.test_high_risk_plan()
            await cli.test_cmvl("medium")
            await cli.test_cmvl("critical")
            results = await cli.test_complete_workflow()
            cli.save_results(results, "all_tests_results.json")
            print("\n✅ All tests completed!")
        
        else:
            print("❌ Invalid choice. Please select 0-6.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
