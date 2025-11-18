/**
 * ReasonGraph Visualization Component
 * 
 * Advanced interactive visualization for financial planning reasoning traces.
 * Shows decision paths, verification points, and alternative strategies.
 * 
 * Person D - Task 21 & 22
 */

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface DecisionNode {
  id: string;
  type: 'planning' | 'verification' | 'decision' | 'alternative';
  label: string;
  status: 'approved' | 'rejected' | 'conditional' | 'explored' | 'pruned';
  confidence?: number;
  rationale?: string;
  timestamp?: string;
  metadata?: Record<string, any>;
}

interface DecisionEdge {
  source: string;
  target: string;
  label?: string;
  weight?: number;
}

interface ReasonGraphData {
  nodes: DecisionNode[];
  edges: DecisionEdge[];
  session_id: string;
  plan_id?: string;
}

interface ReasonGraphProps {
  data: ReasonGraphData;
  width?: number;
  height?: number;
  onNodeClick?: (node: DecisionNode) => void;
}

export const ReasonGraph: React.FC<ReasonGraphProps> = ({
  data,
  width = 1200,
  height = 800,
  onNodeClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<DecisionNode | null>(null);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    if (!svgRef.current || !data) return;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove();

    renderGraph();
  }, [data, filter]);

  const renderGraph = () => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const g = svg.append('g');

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Filter nodes based on current filter
    const filteredNodes = filter === 'all' 
      ? data.nodes 
      : data.nodes.filter(n => n.status === filter);

    const filteredEdges = data.edges.filter(e => 
      filteredNodes.some(n => n.id === e.source) &&
      filteredNodes.some(n => n.id === e.target)
    );

    // Create force simulation
    const simulation = d3.forceSimulation(filteredNodes as any)
      .force('link', d3.forceLink(filteredEdges)
        .id((d: any) => d.id)
        .distance(150))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(50));

    // Draw edges
    const link = g.append('g')
      .selectAll('line')
      .data(filteredEdges)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: any) => Math.sqrt(d.weight || 1));

    // Draw nodes
    const node = g.append('g')
      .selectAll('g')
      .data(filteredNodes)
      .enter()
      .append('g')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    // Add circles for nodes
    node.append('circle')
      .attr('r', 20)
      .attr('fill', (d: DecisionNode) => getNodeColor(d))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .on('click', (event, d) => {
        setSelectedNode(d);
        onNodeClick?.(d);
      })
      .on('mouseover', function() {
        d3.select(this).attr('r', 25);
      })
      .on('mouseout', function() {
        d3.select(this).attr('r', 20);
      });

    // Add labels
    node.append('text')
      .text((d: DecisionNode) => d.label)
      .attr('x', 25)
      .attr('y', 5)
      .attr('font-size', '12px')
      .attr('fill', '#333');

    // Add confidence badges
    node.filter((d: DecisionNode) => d.confidence !== undefined)
      .append('text')
      .text((d: DecisionNode) => `${Math.round((d.confidence || 0) * 100)}%`)
      .attr('x', 0)
      .attr('y', 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#fff')
      .attr('font-weight', 'bold');

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  };

  const getNodeColor = (node: DecisionNode): string => {
    const colors = {
      approved: '#10b981',    // green
      rejected: '#ef4444',    // red
      conditional: '#f59e0b', // yellow
      explored: '#3b82f6',    // blue
      pruned: '#9ca3af'       // gray
    };
    return colors[node.status] || '#6b7280';
  };

  return (
    <div className="reason-graph-container">
      <div className="controls mb-4 flex gap-2">
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded ${filter === 'all' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
        >
          All Nodes
        </button>
        <button
          onClick={() => setFilter('approved')}
          className={`px-4 py-2 rounded ${filter === 'approved' ? 'bg-green-500 text-white' : 'bg-gray-200'}`}
        >
          Approved
        </button>
        <button
          onClick={() => setFilter('rejected')}
          className={`px-4 py-2 rounded ${filter === 'rejected' ? 'bg-red-500 text-white' : 'bg-gray-200'}`}
        >
          Rejected
        </button>
        <button
          onClick={() => setFilter('conditional')}
          className={`px-4 py-2 rounded ${filter === 'conditional' ? 'bg-yellow-500 text-white' : 'bg-gray-200'}`}
        >
          Conditional
        </button>
      </div>

      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="border border-gray-300 rounded-lg bg-white"
      />

      {selectedNode && (
        <div className="node-details mt-4 p-4 border rounded-lg bg-gray-50">
          <h3 className="text-lg font-bold mb-2">{selectedNode.label}</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div><strong>Type:</strong> {selectedNode.type}</div>
            <div><strong>Status:</strong> {selectedNode.status}</div>
            {selectedNode.confidence && (
              <div><strong>Confidence:</strong> {(selectedNode.confidence * 100).toFixed(1)}%</div>
            )}
            {selectedNode.timestamp && (
              <div><strong>Time:</strong> {new Date(selectedNode.timestamp).toLocaleString()}</div>
            )}
          </div>
          {selectedNode.rationale && (
            <div className="mt-2">
              <strong>Rationale:</strong>
              <p className="text-sm text-gray-700 mt-1">{selectedNode.rationale}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ReasonGraph;
