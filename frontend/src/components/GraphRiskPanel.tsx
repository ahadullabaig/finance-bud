import React, { useEffect, useState } from 'react';
import ReasonGraph from './ReasonGraph';

type Anomaly = {
  node_id?: string;
  anomaly_score?: number;
  reason?: string;
  features?: Record<string, any>;
  edge?: string; // optional edge representation like 'u -> v'
};

type FraudIndicator = {
  type?: string;
  reason?: string;
  severity?: string;
  fraud_confidence?: number;
  transaction?: any;
};

export default function GraphRiskPanel({ defaultUserId = 'demo_user' }: { defaultUserId?: string }) {
  const API_BASE = '';
  const [userId, setUserId] = useState<string>(defaultUserId);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [graphStats, setGraphStats] = useState<any>(null);
  const [overallRisk, setOverallRisk] = useState<any>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [fraudIndicators, setFraudIndicators] = useState<FraudIndicator[]>([]);
  const [reasonGraphData, setReasonGraphData] = useState<any | null>(null);

  useEffect(() => {
    // Initial quick load for demo user
    fetchAnalysis();
  }, []);

  const fetchAnalysis = async (overrideUser?: string) => {
    setLoading(true);
    setError(null);
    const uid = overrideUser || userId;

    try {
      const res = await fetch(`${API_BASE}/api/risk/analyze-transactions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: uid, lookback_days: 90 })
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Server error ${res.status}: ${body}`);
      }

      const data = await res.json();

      // Map backend response fields
      setOverallRisk({ score: data.overall_risk_score, level: data.risk_level, tx_count: data.transaction_count });
      setGraphStats(data.graph_stats || null);
      setAnomalies(data.anomalies || []);
      setFraudIndicators(data.fraud_indicators || []);

      // Build minimal ReasonGraph from anomalies + fraud indicators
      const nodesMap: Record<string, any> = {};
      const edges: Array<{ source: string; target: string; weight?: number; label?: string }> = [];

      // Create nodes for anomalies
      (data.anomalies || []).forEach((a: Anomaly, idx: number) => {
        const id = a.node_id || `anomaly_${idx}`;
        nodesMap[id] = {
          id,
          type: 'decision',
          label: `${id}`,
          status: a.anomaly_score && a.anomaly_score > 0.6 ? 'conditional' : 'explored',
          confidence: Math.min(1, a.anomaly_score || 0),
          rationale: a.reason || undefined,
          metadata: a.features || {}
        };

        if (a.edge && typeof a.edge === 'string' && a.edge.includes('->')) {
          const parts = a.edge.split('->').map((s) => s.trim());
          if (parts.length === 2) {
            edges.push({ source: parts[0], target: parts[1], weight: a.anomaly_score || 1, label: 'anomaly' });
            // ensure source/target nodes exist
            nodesMap[parts[0]] = nodesMap[parts[0]] || { id: parts[0], type: 'decision', label: parts[0], status: 'explored' };
            nodesMap[parts[1]] = nodesMap[parts[1]] || { id: parts[1], type: 'decision', label: parts[1], status: 'explored' };
          }
        }
      });

      // Add fraud indicators as nodes (linked to transactions if available)
      (data.fraud_indicators || []).forEach((f: FraudIndicator, idx: number) => {
        const id = f.transaction?.id ? `txn_${f.transaction.id}` : `fraud_${idx}`;
        nodesMap[id] = nodesMap[id] || {
          id,
          type: 'verification',
          label: f.transaction?.merchant ? `${f.transaction.merchant}` : id,
          status: f.fraud_confidence && f.fraud_confidence > 0.7 ? 'rejected' : 'conditional',
          confidence: f.fraud_confidence || 0,
          rationale: f.reason || undefined,
          metadata: { severity: f.severity }
        };
      });

      const nodes = Object.values(nodesMap).map((n) => ({
        id: n.id,
        type: n.type || 'decision',
        label: n.label || n.id,
        status: n.status || 'explored',
        confidence: n.confidence,
        rationale: n.rationale,
        metadata: n.metadata || {}
      }));

      const reasonGraph = {
        nodes,
        edges: edges.map((e) => ({ source: e.source, target: e.target, label: e.label || '', weight: e.weight })),
        session_id: `risk_${uid}_${Date.now()}`,
        plan_id: undefined
      };

      setReasonGraphData(reasonGraph);

    } catch (e: any) {
      const errMsg = e?.message || String(e);
      setError(errMsg);

      // Fallback demo data so the UI shows something useful when backend is unavailable
      const demoOverallRisk = { score: 0.23, level: 'Low', tx_count: 12 };
      const demoGraphStats = { nodes: 5, edges: 6, density: 0.300 };
      const demoAnomalies: Anomaly[] = [
        { node_id: 'acct_123', anomaly_score: 0.85, reason: 'Spike in outgoing transactions', features: { avg_amount: 1200 } },
        { node_id: 'acct_456', anomaly_score: 0.62, reason: 'New merchant linkage', features: { merchants: ['AcmeStore'] } }
      ];
      const demoFraudIndicators: FraudIndicator[] = [
        { type: 'high_velocity', reason: 'Rapid sequence of transactions', severity: 'medium', fraud_confidence: 0.72, transaction: { id: 'tx_789', merchant: 'AcmeStore', amount: 499.99 } }
      ];

      setOverallRisk(demoOverallRisk);
      setGraphStats(demoGraphStats);
      setAnomalies(demoAnomalies);
      setFraudIndicators(demoFraudIndicators);

      const nodes = [
        { id: 'acct_123', type: 'decision', label: 'Acct 123', status: 'conditional', confidence: 0.85, rationale: 'Spike in outgoing transactions', metadata: demoAnomalies[0].features },
        { id: 'acct_456', type: 'decision', label: 'Acct 456', status: 'conditional', confidence: 0.62, rationale: 'New merchant linkage', metadata: demoAnomalies[1].features },
        { id: 'txn_tx_789', type: 'verification', label: 'AcmeStore', status: 'rejected', confidence: 0.72, rationale: demoFraudIndicators[0].reason, metadata: { amount: 499.99 } }
      ];

      const edges = [
        { source: 'acct_123', target: 'acct_456', label: 'transfer', weight: 1 },
        { source: 'txn_tx_789', target: 'acct_123', label: 'purchase', weight: 0.72 }
      ];

      setReasonGraphData({ nodes, edges, session_id: `demo_${uid}_${Date.now()}`, plan_id: undefined });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-2 border-[var(--color-ink)] p-4 bg-white angular-card max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="data-label">GRAPH RISK ANALYSIS</div>
          <div className="text-sm text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)' }}>
            Live graph features, anomalies and GNN-ready output
          </div>
        </div>
        <div className="text-xs text-[var(--color-blueprint)]">Network • GNN • Explainability</div>
      </div>

      <div className="mb-4 flex gap-2">
        <input value={userId} onChange={(e) => setUserId(e.target.value)} className="px-3 py-2 border rounded flex-1" />
        <button onClick={() => fetchAnalysis()} disabled={loading} className="px-4 py-2 bg-[var(--color-ink)] text-[var(--color-cyan)] rounded">
          {loading ? 'Running...' : 'Run Analysis'}
        </button>
      </div>

      {error && <div className="mb-3 text-red-600">Error: {error}</div>}

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="p-3 border rounded">
          <div className="text-sm text-gray-600">Overall Risk</div>
          <div className="text-xl font-bold">{overallRisk ? `${overallRisk.level} (${(overallRisk.score*100).toFixed(0)}%)` : '—'}</div>
          <div className="text-xs text-gray-500">Tx Count: {overallRisk?.tx_count ?? '—'}</div>
        </div>

        <div className="p-3 border rounded">
          <div className="text-sm text-gray-600">Graph Stats</div>
          <div className="text-lg">Nodes: {graphStats?.nodes ?? '—'}</div>
          <div className="text-lg">Edges: {graphStats?.edges ?? '—'}</div>
          <div className="text-sm text-gray-500">Density: {graphStats?.density ? graphStats.density.toFixed(3) : '—'}</div>
        </div>

        <div className="p-3 border rounded">
          <div className="text-sm text-gray-600">Anomalies</div>
          <div className="text-lg">{anomalies.length}</div>
          <div className="text-xs text-gray-500">Fraud Indicators: {fraudIndicators.length}</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          {reasonGraphData ? (
            <ReasonGraph data={reasonGraphData} width={900} height={600} />
          ) : (
            <div className="p-6 border rounded text-center text-gray-500">Run analysis to populate graph visualization.</div>
          )}
        </div>

        <div>
          <div className="mb-4">
            <h4 className="font-bold">Anomalies</h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {anomalies.length === 0 && <div className="text-sm text-gray-500">No anomalies detected.</div>}
              {anomalies.map((a, i) => (
                <div key={i} className="p-2 border rounded">
                  <div className="text-sm font-medium">{a.node_id ?? `anomaly_${i}`}</div>
                  <div className="text-xs text-gray-600">Score: {(a.anomaly_score || 0).toFixed(3)}</div>
                  <div className="text-xs text-gray-600">{a.reason}</div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="font-bold">Fraud Indicators</h4>
            <div className="space-y-2 max-h-64 overflow-y-auto mt-2">
              {fraudIndicators.length === 0 && <div className="text-sm text-gray-500">No indicators found.</div>}
              {fraudIndicators.map((f, i) => (
                <div key={i} className="p-2 border rounded">
                  <div className="text-sm font-medium">{f.type ?? `indicator_${i}`}</div>
                  <div className="text-xs text-gray-600">Confidence: {(f.fraud_confidence || 0).toFixed(2)}</div>
                  <div className="text-xs text-gray-600">{f.reason}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
