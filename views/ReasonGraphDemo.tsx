/**
 * ReasonGraph Demo Page
 * 
 * Interactive demonstration of the ReasonGraph visualization system.
 * Shows complete workflow from goal submission to verification.
 * 
 * Person D - Task 22 & 31
 */

import React, { useState } from 'react';
import ReasonGraph from '../components/ReasonGraph';

interface WorkflowResponse {
  orchestration: any;
  planning: any;
  verification: any;
  reason_graph: any;
  session_id: string;
}

export const ReasonGraphDemo: React.FC = () => {
  const [userGoal, setUserGoal] = useState('Save $100,000 for retirement in 10 years');
  const [loading, setLoading] = useState(false);
  const [workflowData, setWorkflowData] = useState<WorkflowResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runWorkflow = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/v1/demo/complete-workflow', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_goal: userGoal }),
      });
      
      const result = await response.json();
      
      if (result.success) {
        setWorkflowData(result.data);
      } else {
        setError(result.error || 'Workflow failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const triggerCMVL = async (severity: string) => {
    if (!workflowData) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/verification/cmvl/trigger', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          cmvl_trigger: {
            trigger_type: 'market_event',
            event_type: 'volatility_spike',
            severity: severity,
            description: `Simulated ${severity} severity market event`,
            source_data: { volatility: 0.45 },
            impact_score: 0.8,
            confidence_score: 0.9,
            detector_agent_id: 'ira_001',
            correlation_id: `cmvl_${Date.now()}`
          },
          session_id: workflowData.session_id
        }),
      });
      
      const result = await response.json();
      
      if (result.success) {
        // Update reason graph with CMVL data
        const graphResponse = await fetch('http://localhost:8000/api/v1/reasongraph/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            planning_response: workflowData.planning,
            verification_response: workflowData.verification,
            cmvl_response: result.data
          }),
        });
        
        const graphResult = await graphResponse.json();
        
        if (graphResult.success) {
          setWorkflowData({
            ...workflowData,
            reason_graph: graphResult.data
          });
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'CMVL trigger failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="reason-graph-demo p-6 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">ReasonGraph Visualization Demo</h1>
      
      <div className="demo-controls mb-8 p-6 bg-white rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Financial Goal</h2>
        
        <div className="mb-4">
          <textarea
            value={userGoal}
            onChange={(e) => setUserGoal(e.target.value)}
            className="w-full p-3 border rounded-lg"
            rows={3}
            placeholder="Enter your financial goal..."
          />
        </div>
        
        <div className="flex gap-4">
          <button
            onClick={runWorkflow}
            disabled={loading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Processing...' : 'Generate Plan & Verify'}
          </button>
          
          {workflowData && (
            <>
              <button
                onClick={() => triggerCMVL('medium')}
                disabled={loading}
                className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:bg-gray-400"
              >
                Trigger CMVL (Medium)
              </button>
              
              <button
                onClick={() => triggerCMVL('critical')}
                disabled={loading}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400"
              >
                Trigger CMVL (Critical)
              </button>
            </>
          )}
        </div>
        
        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
      </div>
      
      {workflowData && (
        <>
          <div className="workflow-summary mb-8 grid grid-cols-3 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold mb-2">Orchestration</h3>
              <p className="text-sm text-gray-700">
                Status: {workflowData.orchestration.status}
              </p>
              <p className="text-sm text-gray-700">
                Workflow ID: {workflowData.orchestration.workflow_plan?.workflow_id?.substring(0, 8)}...
              </p>
            </div>
            
            <div className="p-4 bg-green-50 rounded-lg">
              <h3 className="font-semibold mb-2">Planning</h3>
              <p className="text-sm text-gray-700">
                Strategy: {workflowData.planning.selected_strategy}
              </p>
              <p className="text-sm text-gray-700">
                Steps: {workflowData.planning.plan_steps?.length || 0}
              </p>
              <p className="text-sm text-gray-700">
                Confidence: {((workflowData.planning.confidence_score || 0) * 100).toFixed(1)}%
              </p>
            </div>
            
            <div className="p-4 bg-purple-50 rounded-lg">
              <h3 className="font-semibold mb-2">Verification</h3>
              <p className="text-sm text-gray-700">
                Status: {workflowData.verification.verification_report?.verification_status}
              </p>
              <p className="text-sm text-gray-700">
                Risk Score: {((workflowData.verification.verification_report?.overall_risk_score || 0) * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-gray-700">
                Violations: {workflowData.verification.verification_report?.constraint_violations?.length || 0}
              </p>
            </div>
          </div>
          
          <div className="reason-graph-container mb-8">
            <h2 className="text-2xl font-semibold mb-4">Reasoning Trace Visualization</h2>
            <ReasonGraph
              data={workflowData.reason_graph}
              width={1200}
              height={800}
              onNodeClick={(node) => console.log('Node clicked:', node)}
            />
          </div>
          
          <div className="plan-details grid grid-cols-2 gap-6">
            <div className="p-6 bg-white rounded-lg shadow">
              <h3 className="text-xl font-semibold mb-4">Plan Steps</h3>
              <div className="space-y-3">
                {workflowData.planning.plan_steps?.map((step: any, index: number) => (
                  <div key={index} className="p-3 border rounded">
                    <div className="font-semibold">{step.action_type}</div>
                    <div className="text-sm text-gray-600">{step.description}</div>
                    <div className="text-sm mt-1">
                      Amount: ${step.amount?.toLocaleString()} | Risk: {step.risk_level}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="p-6 bg-white rounded-lg shadow">
              <h3 className="text-xl font-semibold mb-4">Verification Results</h3>
              <div className="space-y-3">
                {workflowData.verification.step_results?.map((result: any, index: number) => (
                  <div key={index} className="p-3 border rounded">
                    <div className="font-semibold">{result.action_type}</div>
                    <div className="text-sm">
                      Status: <span className={result.status === 'passed' ? 'text-green-600' : 'text-red-600'}>
                        {result.status}
                      </span>
                    </div>
                    <div className="text-sm">
                      Compliance: {(result.compliance_score * 100).toFixed(1)}%
                    </div>
                    {result.violations?.length > 0 && (
                      <div className="text-sm text-red-600 mt-1">
                        {result.violations.length} violation(s)
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ReasonGraphDemo;
