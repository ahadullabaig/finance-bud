import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Card } from '../components/ui/card';
import { CheckCircle2, XCircle, Clock, Circle } from 'lucide-react';

/**
 * ReasonGraphLive - Real-time ToS Visualization Component
 *
 * CURRENT STATUS: Demo mode with simulated animation
 *
 * FOR PRODUCTION REAL-TIME UPDATES:
 * ----------------------------------------
 * 1. WebSocket Integration:
 *    - Connect to backend WebSocket endpoint (e.g., ws://localhost:8000/ws/reasoning)
 *    - Subscribe to agent events by session_id
 *    - Receive real-time updates as agents progress through ToS
 *
 * 2. Server-Sent Events (SSE) Alternative:
 *    - Use EventSource for one-way real-time updates
 *    - Simpler than WebSocket if bidirectional communication not needed
 *
 * 3. Backend Implementation Needed:
 *    - Add WebSocket endpoint in FastAPI/Flask
 *    - Agents emit events when decision points occur
 *    - Broadcast events to connected clients
 *
 * 4. Data Structure:
 *    - Each event should include: node_id, status, timestamp, agent_id
 *    - Support incremental graph building as search progresses
 *
 * 5. Enhanced Props:
 *    interface ReasonGraphLiveProps {
 *      sessionId?: string;           // Connect to specific session
 *      realtimeData?: GraphNode[];   // Dynamic node data from backend
 *      webSocketUrl?: string;        // WebSocket endpoint
 *      enableRealtime?: boolean;     // Toggle real-time vs demo mode
 *    }
 */

interface ReasonGraphLiveProps {
  isProcessing: boolean;
  triggerType: 'market' | 'lifeevent';
  sessionId?: string;              // Session to monitor (for future real-time)
  realtimeData?: GraphNode[];      // Real-time graph data (for future integration)
  enableRealtime?: boolean;        // Enable WebSocket real-time updates
}

interface GraphNode {
  id: string;
  label: string;
  agent: string;
  status: 'pending' | 'active' | 'complete' | 'rejected' | 'approved';
  type: 'trigger' | 'planning' | 'verification' | 'execution' | 'completion';
  x: number;
  y: number;
  children?: string[];
}

export function ReasonGraphLive({
  isProcessing,
  triggerType,
  sessionId,
  realtimeData,
  enableRealtime = false
}: ReasonGraphLiveProps) {
  const [activeNodes, setActiveNodes] = useState<Set<string>>(new Set());
  const [completedNodes, setCompletedNodes] = useState<Set<string>>(new Set());

  // Demo data - used when realtimeData is not provided
  const demoNodes: GraphNode[] = [
    { id: '1', label: 'CMVL_INIT', agent: 'OA', status: 'complete', type: 'trigger', x: 50, y: 10, children: ['2', '3'] },
    { id: '2', label: 'RETRIEVE_MARKET', agent: 'IRA', status: 'complete', type: 'planning', x: 20, y: 30, children: ['4'] },
    { id: '3', label: 'FETCH_CONSTRAINTS', agent: 'IRA', status: 'complete', type: 'planning', x: 80, y: 30, children: ['4'] },
    { id: '4', label: 'GSM_SEARCH_INIT', agent: 'PA', status: 'complete', type: 'planning', x: 50, y: 50, children: ['5', '6', '7'] },
    { id: '5', label: 'PATH_A_CONSERVATIVE', agent: 'PA', status: 'rejected', type: 'planning', x: 15, y: 70, children: ['8'] },
    { id: '6', label: 'PATH_B_MODERATE', agent: 'PA', status: 'approved', type: 'planning', x: 50, y: 70, children: ['9'] },
    { id: '7', label: 'PATH_C_MINIMAL', agent: 'PA', status: 'rejected', type: 'planning', x: 85, y: 70, children: ['8'] },
    { id: '8', label: 'CONSTRAINT_FAIL', agent: 'VA', status: 'complete', type: 'verification', x: 20, y: 90, children: [] },
    { id: '9', label: 'VERIFY_PATH_B', agent: 'VA', status: 'approved', type: 'verification', x: 50, y: 90, children: ['10'] },
    { id: '10', label: 'EXECUTE_PLAN', agent: 'EA', status: 'complete', type: 'execution', x: 50, y: 110, children: ['11'] },
    { id: '11', label: 'PLAN_VALIDATED', agent: 'OA', status: 'complete', type: 'completion', x: 50, y: 130, children: [] },
  ];

  // Use real-time data if provided, otherwise use demo data
  const nodes = realtimeData && realtimeData.length > 0 ? realtimeData : demoNodes;

  // TODO: WebSocket integration for real-time updates
  // useEffect(() => {
  //   if (!enableRealtime || !sessionId) return;
  //
  //   const ws = new WebSocket(`ws://localhost:8000/ws/reasoning/${sessionId}`);
  //
  //   ws.onmessage = (event) => {
  //     const update = JSON.parse(event.data);
  //     // Update activeNodes and completedNodes based on real-time events
  //     if (update.type === 'node_active') {
  //       setActiveNodes(new Set([update.nodeId]));
  //     }
  //     if (update.type === 'node_complete') {
  //       setCompletedNodes(prev => new Set(prev).add(update.nodeId));
  //     }
  //   };
  //
  //   return () => ws.close();
  // }, [enableRealtime, sessionId]);

  useEffect(() => {
    if (!isProcessing) return;

    const sequence = [
      { time: 0, nodes: ['1'] },
      { time: 400, nodes: ['2', '3'] },
      { time: 800, nodes: ['4'] },
      { time: 1200, nodes: ['5', '6', '7'] },
      { time: 1600, nodes: ['8'] },
      { time: 2000, nodes: ['9'] },
      { time: 2400, nodes: ['10'] },
      { time: 2800, nodes: ['11'] },
    ];

    sequence.forEach(({ time, nodes: nodeIds }) => {
      setTimeout(() => {
        setActiveNodes(new Set(nodeIds));
        setCompletedNodes(prev => {
          const next = new Set(prev);
          nodeIds.forEach(id => next.add(id));
          return next;
        });
      }, time);
    });
  }, [isProcessing]);

  const getAgentColor = (agent: string) => {
    switch(agent) {
      case 'OA': return 'var(--color-blueprint)';
      case 'IRA': return 'var(--color-cyan)';
      case 'PA': return 'var(--color-amber)';
      case 'VA': return 'var(--color-emerald)';
      case 'EA': return 'var(--color-vermillion)';
      default: return 'var(--color-ink)';
    }
  };

  const getStatusIcon = (node: GraphNode) => {
    if (node.status === 'rejected') return <XCircle className="w-4 h-4" style={{ color: 'var(--color-vermillion)' }} />;
    if (node.status === 'approved') return <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--color-emerald)' }} />;
    if (completedNodes.has(node.id)) return <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--color-emerald)' }} />;
    if (activeNodes.has(node.id)) return <Circle className="w-4 h-4 pulse-indicator" style={{ color: 'var(--color-cyan)' }} />;
    return <Clock className="w-4 h-4" style={{ color: 'var(--color-grid)' }} />;
  };

  return (
    <Card className="border-2 border-[var(--color-ink)] bg-white p-8 blueprint-corners">
      <div className="mb-6">
        <div className="data-label mb-2">EXECUTION TRACE</div>
        <h2 className="mb-2" style={{ fontFamily: 'var(--font-display)' }}>ReasonGraph: Live Monitoring</h2>
        <p className="text-sm text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)' }}>
          Real-time visualization of ToS path exploration and verification checkpoints
        </p>
      </div>

      {/* SVG Graph */}
      <div className="relative w-full" style={{ height: '600px' }}>
        <svg className="w-full h-full" viewBox="0 0 100 140" preserveAspectRatio="xMidYMid meet">
          {/* Draw connections */}
          {nodes.map(node => 
            node.children?.map(childId => {
              const child = nodes.find(n => n.id === childId);
              if (!child) return null;
              
              const isActive = completedNodes.has(node.id) || activeNodes.has(node.id);
              
              return (
                <motion.line
                  key={`${node.id}-${childId}`}
                  x1={node.x}
                  y1={node.y}
                  x2={child.x}
                  y2={child.y}
                  stroke={isActive ? 'var(--color-cyan)' : 'var(--color-grid)'}
                  strokeWidth="0.3"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ 
                    pathLength: isActive ? 1 : 0,
                    opacity: isActive ? 1 : 0.3
                  }}
                  transition={{ duration: 0.5 }}
                />
              );
            })
          )}

          {/* Draw nodes */}
          {nodes.map(node => {
            const isActive = activeNodes.has(node.id);
            const isComplete = completedNodes.has(node.id);
            const isRejected = node.status === 'rejected';
            
            return (
              <g key={node.id}>
                <motion.circle
                  cx={node.x}
                  cy={node.y}
                  r="3"
                  fill={isRejected ? 'var(--color-vermillion)' : isActive ? 'var(--color-cyan)' : isComplete ? 'var(--color-emerald)' : 'white'}
                  stroke={getAgentColor(node.agent)}
                  strokeWidth="0.4"
                  initial={{ scale: 0 }}
                  animate={{ scale: isActive ? [1, 1.3, 1] : 1 }}
                  transition={{ duration: 0.5, repeat: isActive ? Infinity : 0 }}
                />
                <text
                  x={node.x}
                  y={node.y - 4}
                  textAnchor="middle"
                  style={{ 
                    fontSize: '2.5px', 
                    fontFamily: 'var(--font-code)',
                    fill: 'var(--color-ink)',
                    fontWeight: isActive ? '600' : '400'
                  }}
                >
                  {node.label}
                </text>
                <text
                  x={node.x}
                  y={node.y + 5}
                  textAnchor="middle"
                  style={{ 
                    fontSize: '1.8px', 
                    fontFamily: 'var(--font-code)',
                    fill: getAgentColor(node.agent)
                  }}
                >
                  {node.agent}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-8 pt-6 border-t-2 border-[var(--color-grid)]">
        <div className="data-label mb-4">AGENT_LEGEND</div>
        <div className="grid grid-cols-5 gap-4">
          {['OA', 'IRA', 'PA', 'VA', 'EA'].map(agent => (
            <div key={agent} className="flex items-center gap-2">
              <div 
                className="w-3 h-3 border-2"
                style={{ borderColor: getAgentColor(agent), backgroundColor: 'white' }}
              ></div>
              <span className="text-sm" style={{ fontFamily: 'var(--font-code)', color: getAgentColor(agent) }}>
                {agent}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
