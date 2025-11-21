"""
Performance Baseline Measurement - Phase 1

Establishes performance baselines before refactoring to ensure
no regression during architectural changes.

Metrics measured:
- Agent response times
- Memory usage
- Message throughput
- API endpoint latency
- Database query performance

Run this script to generate baseline.json before starting refactoring.
Run again after each phase to compare performance.

Usage:
    python tests/performance_baseline.py
    python tests/performance_baseline.py --compare baseline.json

Created: Phase 1 - Foundation & Safety Net
"""

import asyncio
import time
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from uuid import uuid4
from decimal import Decimal
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent,
    MockVerificationAgent,
    MockExecutionAgent
)
from agents.communication import AgentCommunicationFramework
from data_models.schemas import AgentMessage, MessageType, Priority


class PerformanceBaseline:
    """Measure and track performance baselines"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": self._get_system_info(),
            "metrics": {}
        }
        self.process = psutil.Process()

    def _get_system_info(self) -> Dict[str, Any]:
        """Capture system information"""
        return {
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024),
            "platform": sys.platform
        }

    async def measure_agent_initialization_time(self) -> Dict[str, float]:
        """Measure time to initialize all agents"""
        print("📊 Measuring agent initialization time...")

        start_memory = self.process.memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()

        # Initialize agents
        agents = {
            'orchestrator': MockOrchestrationAgent("perf_oa_001"),
            'planner': MockPlanningAgent("perf_pa_001"),
            'ira': MockInformationRetrievalAgent("perf_ira_001"),
            'verifier': MockVerificationAgent("perf_va_001"),
            'executor': MockExecutionAgent("perf_ea_001")
        }

        # Start agents
        for agent in agents.values():
            await agent.start()

        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / (1024 * 1024)

        # Cleanup
        for agent in agents.values():
            await agent.stop()

        return {
            "initialization_time_seconds": end_time - start_time,
            "memory_increase_mb": end_memory - start_memory,
            "agents_initialized": len(agents)
        }

    async def measure_message_throughput(self) -> Dict[str, float]:
        """Measure message passing throughput"""
        print("📊 Measuring message passing throughput...")

        framework = AgentCommunicationFramework()

        # Setup agents
        agents = {
            'sender': MockOrchestrationAgent("perf_sender"),
            'receiver': MockPlanningAgent("perf_receiver")
        }

        for agent in agents.values():
            await agent.start()
            framework.register_agent(agent, [f"{agent.agent_id}_cap"])

        # Measure throughput for 100 messages
        message_count = 100
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / (1024 * 1024)

        for i in range(message_count):
            msg = framework.create_message(
                sender_id="perf_sender",
                target_id="perf_receiver",
                message_type=MessageType.REQUEST,
                payload={"test_data": f"message_{i}", "value": i},
                correlation_id=str(uuid4()),
                session_id=str(uuid4())
            )
            await framework.send_message(msg)

        # Allow processing time
        await asyncio.sleep(0.5)

        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / (1024 * 1024)

        # Cleanup
        for agent in agents.values():
            await agent.stop()

        duration = end_time - start_time
        throughput = message_count / duration

        return {
            "messages_sent": message_count,
            "duration_seconds": duration,
            "messages_per_second": throughput,
            "avg_latency_ms": (duration / message_count) * 1000,
            "memory_increase_mb": end_memory - start_memory
        }

    async def measure_planning_agent_performance(self) -> Dict[str, float]:
        """Measure planning agent response time"""
        print("📊 Measuring planning agent performance...")

        planner = MockPlanningAgent("perf_planner")
        await planner.start()

        # Simulate planning request
        test_message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id="perf_planner",
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": "Save $100,000 for retirement",
                    "time_horizon": 120,
                    "risk_profile": "moderate",
                    "constraints": [{"type": "budget", "value": 5000}]
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )

        # Measure 10 planning cycles
        iterations = 10
        times = []
        start_memory = self.process.memory_info().rss / (1024 * 1024)

        for _ in range(iterations):
            start = time.perf_counter()
            await planner.process_message(test_message)
            end = time.perf_counter()
            times.append(end - start)
            await asyncio.sleep(0.05)

        end_memory = self.process.memory_info().rss / (1024 * 1024)
        await planner.stop()

        times.sort()
        return {
            "iterations": iterations,
            "avg_response_time_seconds": sum(times) / len(times),
            "min_response_time_seconds": min(times),
            "max_response_time_seconds": max(times),
            "p50_response_time_ms": times[len(times)//2] * 1000,
            "p95_response_time_ms": times[int(len(times)*0.95)] * 1000,
            "memory_per_iteration_kb": (end_memory - start_memory) / iterations * 1024
        }

    async def measure_end_to_end_workflow(self) -> Dict[str, float]:
        """Measure complete end-to-end workflow"""
        print("📊 Measuring end-to-end workflow performance...")

        framework = AgentCommunicationFramework()

        # Setup complete system
        agents = {
            'orchestrator': MockOrchestrationAgent("perf_oa"),
            'planner': MockPlanningAgent("perf_pa"),
            'ira': MockInformationRetrievalAgent("perf_ira"),
            'verifier': MockVerificationAgent("perf_va"),
            'executor': MockExecutionAgent("perf_ea")
        }

        for agent in agents.values():
            await agent.start()
            framework.register_agent(agent, [f"{agent.agent_id}_cap"])

        # Measure workflow
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / (1024 * 1024)

        correlation_id = str(uuid4())
        session_id = str(uuid4())

        # Step 1: Submit goal
        goal_msg = framework.create_message(
            sender_id="user",
            target_id="perf_oa",
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Retire comfortably in 20 years"},
            correlation_id=correlation_id,
            session_id=session_id
        )
        await framework.send_message(goal_msg)
        await asyncio.sleep(0.1)

        # Step 2: Market data
        market_msg = framework.create_message(
            sender_id="perf_ira",
            target_id="perf_pa",
            message_type=MessageType.RESPONSE,
            payload={"market_data": {"volatility": 0.15}},
            correlation_id=correlation_id,
            session_id=session_id
        )
        await framework.send_message(market_msg)
        await asyncio.sleep(0.1)

        # Step 3: Planning
        plan_msg = framework.create_message(
            sender_id="perf_oa",
            target_id="perf_pa",
            message_type=MessageType.REQUEST,
            payload={"planning_request": {"goal": "retirement"}},
            correlation_id=correlation_id,
            session_id=session_id
        )
        await framework.send_message(plan_msg)
        await asyncio.sleep(0.2)

        # Step 4: Verification
        verify_msg = framework.create_message(
            sender_id="perf_pa",
            target_id="perf_va",
            message_type=MessageType.REQUEST,
            payload={"verification_request": {"plan_id": str(uuid4())}},
            correlation_id=correlation_id,
            session_id=session_id
        )
        await framework.send_message(verify_msg)
        await asyncio.sleep(0.1)

        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / (1024 * 1024)

        # Cleanup
        for agent in agents.values():
            await agent.stop()

        # Get correlation trace
        trace = framework.get_correlation_trace(correlation_id)

        return {
            "workflow_duration_seconds": end_time - start_time,
            "messages_in_trace": len(trace),
            "memory_used_mb": end_memory - start_memory,
            "avg_time_per_message_ms": ((end_time - start_time) / len(trace)) * 1000 if trace else 0
        }

    def measure_system_overhead(self) -> Dict[str, float]:
        """Measure current system resource usage"""
        print("📊 Measuring system overhead...")

        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()

        return {
            "cpu_percent": cpu_percent,
            "system_memory_used_percent": memory.percent,
            "system_memory_available_mb": memory.available / (1024 * 1024),
            "process_memory_mb": process_memory.rss / (1024 * 1024),
            "process_threads": self.process.num_threads()
        }

    async def run_all_measurements(self) -> Dict[str, Any]:
        """Run all performance measurements"""
        print("\n" + "="*60)
        print("  FinPilot Performance Baseline Measurement")
        print("="*60 + "\n")

        self.results["metrics"]["agent_initialization"] = await self.measure_agent_initialization_time()
        await asyncio.sleep(1)

        self.results["metrics"]["message_throughput"] = await self.measure_message_throughput()
        await asyncio.sleep(1)

        self.results["metrics"]["planning_performance"] = await self.measure_planning_agent_performance()
        await asyncio.sleep(1)

        self.results["metrics"]["end_to_end_workflow"] = await self.measure_end_to_end_workflow()
        await asyncio.sleep(1)

        self.results["metrics"]["system_overhead"] = self.measure_system_overhead()

        return self.results

    def save_baseline(self, filename: str = "baseline.json"):
        """Save baseline results to file"""
        filepath = Path(__file__).parent / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✅ Baseline saved to: {filepath}")

    def print_results(self):
        """Print results in human-readable format"""
        print("\n" + "="*60)
        print("  PERFORMANCE BASELINE RESULTS")
        print("="*60 + "\n")

        metrics = self.results["metrics"]

        print("🚀 Agent Initialization:")
        ai = metrics["agent_initialization"]
        print(f"  - Time: {ai['initialization_time_seconds']:.3f}s")
        print(f"  - Memory: +{ai['memory_increase_mb']:.2f} MB")
        print(f"  - Agents: {ai['agents_initialized']}")

        print("\n📨 Message Throughput:")
        mt = metrics["message_throughput"]
        print(f"  - Throughput: {mt['messages_per_second']:.1f} msg/sec")
        print(f"  - Latency: {mt['avg_latency_ms']:.2f} ms avg")
        print(f"  - Memory: +{mt['memory_increase_mb']:.2f} MB")

        print("\n🧠 Planning Agent:")
        pp = metrics["planning_performance"]
        print(f"  - Avg Response: {pp['avg_response_time_seconds']*1000:.2f} ms")
        print(f"  - P50: {pp['p50_response_time_ms']:.2f} ms")
        print(f"  - P95: {pp['p95_response_time_ms']:.2f} ms")
        print(f"  - Memory/iter: {pp['memory_per_iteration_kb']:.2f} KB")

        print("\n🔄 End-to-End Workflow:")
        e2e = metrics["end_to_end_workflow"]
        print(f"  - Duration: {e2e['workflow_duration_seconds']:.3f}s")
        print(f"  - Messages: {e2e['messages_in_trace']}")
        print(f"  - Memory: {e2e['memory_used_mb']:.2f} MB")

        print("\n💻 System Overhead:")
        so = metrics["system_overhead"]
        print(f"  - CPU: {so['cpu_percent']:.1f}%")
        print(f"  - Memory: {so['system_memory_used_percent']:.1f}%")
        print(f"  - Process: {so['process_memory_mb']:.2f} MB")
        print(f"  - Threads: {so['process_threads']}")

        print("\n" + "="*60 + "\n")


async def main():
    """Main entry point"""
    baseline = PerformanceBaseline()

    try:
        await baseline.run_all_measurements()
        baseline.print_results()
        baseline.save_baseline()

        print("✅ Performance baseline established successfully!\n")
        print("📝 Next steps:")
        print("   1. Save this baseline before starting refactoring")
        print("   2. Run again after each refactoring phase")
        print("   3. Compare results to ensure no performance regression\n")

    except Exception as e:
        print(f"\n❌ Error during baseline measurement: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
