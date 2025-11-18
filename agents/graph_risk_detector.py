"""
Graph Risk Detector - Phase 6, Task 24
NVIDIA cuGraph/GNN Alternative using NetworkX for graph-based risk detection

Provides graph-based analysis for:
- Transaction pattern analysis
- Asset correlation risk detection
- Fraud pattern identification
- Systemic risk detection

Requirements: Phase 6, Task 24
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. Install with: pip install networkx")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available")

from agents.base_agent import BaseAgent
from data_models.schemas import AgentMessage, MessageType


class GraphRiskDetector(BaseAgent):
    """
    Graph-based risk detection agent using NetworkX.

    Analyzes:
    - Transaction patterns for anomalies
    - Asset correlations for systemic risk
    - Fraud indicators in spending behavior
    - Hidden risk through graph topology
    """

    def __init__(
        self,
        agent_id: str = "graph-risk-detector-001",
        fraud_threshold: float = 0.75,
        risk_threshold: float = 0.65
    ):
        super().__init__(agent_id, "GraphRiskDetector")
        self.fraud_threshold = fraud_threshold
        self.risk_threshold = risk_threshold

        # Initialize graphs
        self.transaction_graph = nx.DiGraph()
        self.asset_correlation_graph = nx.Graph()

        # Cache for analysis results
        self.risk_scores = {}
        self.fraud_indicators = {}

        self.logger.info(
            f"GraphRiskDetector initialized. "
            f"NetworkX available: {NETWORKX_AVAILABLE}, "
            f"sklearn available: {SKLEARN_AVAILABLE}"
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages"""
        if message.message_type == MessageType.REQUEST:
            action = message.content.get("action")

            if action == "analyze_transaction_risk":
                result = await self.analyze_transaction_risk(
                    message.content.get("user_id"),
                    message.content.get("lookback_days", 90)
                )
                return AgentMessage(
                    agent_id=self.agent_id,
                    target_agent_id=message.agent_id,
                    message_type=MessageType.RESPONSE,
                    content=result,
                    correlation_id=message.correlation_id,
                    session_id=message.session_id
                )
        return None

    async def build_transaction_graph(
        self,
        transactions: List[Dict[str, Any]],
        user_id: str
    ) -> nx.DiGraph:
        """
        Build directed graph from transaction data.

        Nodes: accounts, merchants, categories
        Edges: transactions with amounts and timestamps

        Args:
            transactions: List of transaction dictionaries
            user_id: User identifier

        Returns:
            NetworkX directed graph
        """
        self.logger.info(f"Building transaction graph for user {user_id}")

        G = nx.DiGraph()

        for txn in transactions:
            source = txn.get('from_account', f"user_{user_id}")
            target = txn.get('to_account') or txn.get('merchant', 'unknown')
            amount = txn.get('amount', 0)
            timestamp = txn.get('timestamp', datetime.utcnow().isoformat())
            category = txn.get('category', 'general')

            # Add nodes with attributes
            if not G.has_node(source):
                G.add_node(source, node_type='account', user_id=user_id)

            if not G.has_node(target):
                G.add_node(
                    target,
                    node_type='merchant' if 'merchant' in txn else 'account',
                    category=category
                )

            # Add edge with transaction details
            if G.has_edge(source, target):
                # Update existing edge
                edge_data = G[source][target]
                edge_data['total_amount'] += amount
                edge_data['transaction_count'] += 1
                edge_data['last_transaction'] = timestamp
            else:
                # Create new edge
                G.add_edge(
                    source,
                    target,
                    amount=amount,
                    total_amount=amount,
                    transaction_count=1,
                    first_transaction=timestamp,
                    last_transaction=timestamp,
                    category=category
                )

        self.transaction_graph = G
        self.logger.info(
            f"Transaction graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

        return G

    async def build_asset_correlation_graph(
        self,
        assets: List[Dict[str, Any]],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ) -> nx.Graph:
        """
        Build undirected graph of asset correlations.

        Nodes: assets/stocks
        Edges: correlation strength (weighted)

        Args:
            assets: List of assets in portfolio
            correlation_matrix: Optional pre-computed correlation matrix

        Returns:
            NetworkX undirected graph
        """
        self.logger.info(f"Building asset correlation graph for {len(assets)} assets")

        G = nx.Graph()

        # Add asset nodes
        for asset in assets:
            symbol = asset.get('symbol')
            G.add_node(
                symbol,
                asset_type=asset.get('type', 'stock'),
                sector=asset.get('sector', 'unknown'),
                allocation=asset.get('allocation', 0.0)
            )

        # Add correlation edges
        if correlation_matrix:
            for asset1 in correlation_matrix:
                for asset2, correlation in correlation_matrix[asset1].items():
                    if asset1 != asset2 and abs(correlation) > 0.3:  # Threshold
                        G.add_edge(
                            asset1,
                            asset2,
                            weight=abs(correlation),
                            correlation=correlation
                        )
        else:
            # Create simple connections based on same sector
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    if asset1.get('sector') == asset2.get('sector'):
                        G.add_edge(
                            asset1['symbol'],
                            asset2['symbol'],
                            weight=0.5,  # Default correlation for same sector
                            correlation=0.5
                        )

        self.asset_correlation_graph = G
        self.logger.info(
            f"Asset correlation graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

        return G

    async def detect_anomalous_patterns(
        self,
        graph: nx.Graph,
        method: str = "isolation_forest"
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalous patterns in transaction graph.

        Args:
            graph: Transaction graph
            method: Detection method ('isolation_forest', 'centrality', 'clustering')

        Returns:
            List of anomaly reports
        """
        self.logger.info(f"Detecting anomalous patterns using {method}")

        anomalies = []

        if method == "isolation_forest" and SKLEARN_AVAILABLE:
            anomalies = self._detect_with_isolation_forest(graph)
        elif method == "centrality":
            anomalies = self._detect_with_centrality(graph)
        elif method == "clustering":
            anomalies = self._detect_with_clustering(graph)
        else:
            # Fallback to simple statistical methods
            anomalies = self._detect_with_statistics(graph)

        self.logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies

    def _detect_with_isolation_forest(
        self,
        graph: nx.Graph
    ) -> List[Dict[str, Any]]:
        """Use Isolation Forest for anomaly detection"""
        if not SKLEARN_AVAILABLE:
            return []

        # Extract features from nodes
        features = []
        node_ids = []

        for node in graph.nodes():
            node_ids.append(node)

            # Calculate node features
            degree = graph.degree(node)
            in_degree = graph.in_degree(node) if graph.is_directed() else 0
            out_degree = graph.out_degree(node) if graph.is_directed() else 0

            # Transaction volumes
            total_amount = sum(
                graph[node][neighbor].get('total_amount', 0)
                for neighbor in graph.neighbors(node)
            ) if graph.is_directed() else 0

            features.append([degree, in_degree, out_degree, total_amount])

        if len(features) < 2:
            return []

        # Normalize and detect
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        clf = IsolationForest(contamination=0.1, random_state=42)
        predictions = clf.fit_predict(features_normalized)
        scores = clf.score_samples(features_normalized)

        # Extract anomalies
        anomalies = []
        for idx, (node_id, pred, score) in enumerate(zip(node_ids, predictions, scores)):
            if pred == -1:  # Anomaly
                anomalies.append({
                    "node_id": node_id,
                    "anomaly_score": float(-score),  # Convert to positive score
                    "reason": "Unusual transaction pattern detected",
                    "features": {
                        "degree": features[idx][0],
                        "in_degree": features[idx][1],
                        "out_degree": features[idx][2],
                        "total_amount": features[idx][3]
                    }
                })

        return anomalies

    def _detect_with_centrality(
        self,
        graph: nx.Graph
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using centrality measures"""
        anomalies = []

        if graph.number_of_nodes() == 0:
            return anomalies

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(graph)
        betweenness = nx.betweenness_centrality(graph)

        # Find outliers (> 2 std dev from mean)
        if degree_centrality:
            values = list(degree_centrality.values())
            mean = np.mean(values) if SKLEARN_AVAILABLE else sum(values) / len(values)
            std = np.std(values) if SKLEARN_AVAILABLE else 0

            threshold = mean + 2 * std

            for node, centrality in degree_centrality.items():
                if centrality > threshold:
                    anomalies.append({
                        "node_id": node,
                        "anomaly_score": centrality,
                        "reason": "Unusually high degree centrality",
                        "metrics": {
                            "degree_centrality": centrality,
                            "betweenness": betweenness.get(node, 0)
                        }
                    })

        return anomalies

    def _detect_with_clustering(
        self,
        graph: nx.Graph
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using community detection"""
        anomalies = []

        if graph.number_of_nodes() < 3:
            return anomalies

        # Convert to undirected for community detection
        G_undirected = graph.to_undirected() if graph.is_directed() else graph

        try:
            # Detect communities
            communities = community.greedy_modularity_communities(G_undirected)

            # Find singleton communities or unusual patterns
            for idx, comm in enumerate(communities):
                if len(comm) == 1:  # Isolated node
                    node = list(comm)[0]
                    anomalies.append({
                        "node_id": node,
                        "anomaly_score": 0.8,
                        "reason": "Isolated from main transaction patterns",
                        "community_size": 1,
                        "community_id": idx
                    })

        except Exception as e:
            self.logger.warning(f"Community detection failed: {e}")

        return anomalies

    def _detect_with_statistics(
        self,
        graph: nx.Graph
    ) -> List[Dict[str, Any]]:
        """Simple statistical anomaly detection"""
        anomalies = []

        if graph.number_of_edges() == 0:
            return anomalies

        # Analyze edge weights (transaction amounts)
        amounts = []
        for u, v, data in graph.edges(data=True):
            amount = data.get('total_amount', data.get('amount', 0))
            amounts.append((u, v, amount))

        if not amounts:
            return anomalies

        # Find statistical outliers
        values = [amt for _, _, amt in amounts]
        mean_amt = sum(values) / len(values)
        threshold = mean_amt * 3  # 3x mean

        for u, v, amount in amounts:
            if amount > threshold:
                anomalies.append({
                    "edge": f"{u} -> {v}",
                    "anomaly_score": amount / mean_amt,
                    "reason": f"Unusually large transaction amount: ${amount:,.2f}",
                    "amount": amount
                })

        return anomalies

    async def calculate_systemic_risk(
        self,
        correlation_graph: Optional[nx.Graph] = None
    ) -> Dict[str, Any]:
        """
        Calculate systemic risk from asset correlation graph.

        Uses graph topology to identify:
        - Highly connected assets (contagion risk)
        - Dense clusters (sector concentration)
        - Bridge assets (systemic importance)

        Returns:
            Risk assessment with scores and recommendations
        """
        G = correlation_graph or self.asset_correlation_graph

        if G.number_of_nodes() == 0:
            return {"risk_score": 0.0, "message": "No assets to analyze"}

        self.logger.info("Calculating systemic risk from correlation graph")

        # Calculate risk metrics
        risk_metrics = {}

        # 1. Density (high density = high correlation = high risk)
        density = nx.density(G)
        risk_metrics['density'] = density

        # 2. Average clustering (how interconnected are neighbors)
        clustering = nx.average_clustering(G) if G.number_of_nodes() > 1 else 0
        risk_metrics['clustering_coefficient'] = clustering

        # 3. Identify critical nodes (high PageRank = high systemic importance)
        if G.number_of_nodes() > 1:
            pagerank = nx.pagerank(G, weight='weight')
            top_critical = sorted(
                pagerank.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            risk_metrics['critical_assets'] = [
                {"symbol": symbol, "importance": score}
                for symbol, score in top_critical
            ]
        else:
            risk_metrics['critical_assets'] = []

        # 4. Detect communities (sector clustering)
        communities = list(community.greedy_modularity_communities(G))
        risk_metrics['num_communities'] = len(communities)
        risk_metrics['largest_community_size'] = max(
            len(c) for c in communities
        ) if communities else 0

        # Calculate overall risk score (0-1)
        risk_score = (
            density * 0.4 +
            clustering * 0.3 +
            (1.0 if len(communities) < 3 else 0.5) * 0.3  # Lack of diversification
        )

        risk_assessment = {
            "overall_risk_score": min(risk_score, 1.0),
            "risk_level": self._categorize_risk(risk_score),
            "metrics": risk_metrics,
            "recommendations": self._generate_risk_recommendations(risk_metrics, risk_score)
        }

        return risk_assessment

    def _categorize_risk(self, score: float) -> str:
        """Categorize risk score into levels"""
        if score >= 0.75:
            return "CRITICAL"
        elif score >= 0.5:
            return "HIGH"
        elif score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_risk_recommendations(
        self,
        metrics: Dict[str, Any],
        score: float
    ) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []

        if score > 0.7:
            recommendations.append(
                "Portfolio shows high systemic risk. Consider diversification across uncorrelated assets."
            )

        if metrics.get('density', 0) > 0.6:
            recommendations.append(
                "High asset correlation detected. Add low-correlation assets to reduce contagion risk."
            )

        if metrics.get('num_communities', 0) < 3:
            recommendations.append(
                "Portfolio is concentrated in few sectors. Diversify across more sectors."
            )

        critical_assets = metrics.get('critical_assets', [])
        if critical_assets and critical_assets[0]['importance'] > 0.3:
            recommendations.append(
                f"Asset {critical_assets[0]['symbol']} has high systemic importance. "
                "Monitor closely and consider rebalancing."
            )

        if not recommendations:
            recommendations.append("Portfolio risk profile is well-balanced.")

        return recommendations

    async def find_fraud_indicators(
        self,
        transactions: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify potential fraud indicators.

        Checks for:
        - Unusual spending patterns
        - Geographic anomalies
        - Velocity checks (rapid transactions)
        - Amount anomalies

        Returns:
            List of fraud indicators with confidence scores
        """
        self.logger.info(f"Analyzing {len(transactions)} transactions for fraud")

        fraud_indicators = []

        # Build temporal analysis
        temporal_patterns = self._analyze_temporal_patterns(transactions)

        # Check for velocity anomalies
        velocity_alerts = self._check_velocity(transactions)
        fraud_indicators.extend(velocity_alerts)

        # Check for amount anomalies
        amount_alerts = self._check_amount_anomalies(transactions, user_profile)
        fraud_indicators.extend(amount_alerts)

        # Score and rank indicators
        for indicator in fraud_indicators:
            indicator['fraud_confidence'] = self._calculate_fraud_confidence(indicator)

        # Filter by threshold
        fraud_indicators = [
            ind for ind in fraud_indicators
            if ind['fraud_confidence'] >= self.fraud_threshold
        ]

        self.logger.info(f"Found {len(fraud_indicators)} potential fraud indicators")
        return fraud_indicators

    def _analyze_temporal_patterns(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze time-based transaction patterns"""
        if not transactions:
            return {}

        hour_distribution = defaultdict(int)
        day_distribution = defaultdict(int)

        for txn in transactions:
            timestamp = txn.get('timestamp')
            if timestamp:
                # Parse timestamp and extract hour/day
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp

                hour_distribution[dt.hour] += 1
                day_distribution[dt.weekday()] += 1

        return {
            "hour_distribution": dict(hour_distribution),
            "day_distribution": dict(day_distribution),
            "total_transactions": len(transactions)
        }

    def _check_velocity(
        self,
        transactions: List[Dict[str, Any]],
        time_window_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Check for rapid succession of transactions"""
        alerts = []

        # Sort by timestamp
        sorted_txns = sorted(
            transactions,
            key=lambda x: x.get('timestamp', '')
        )

        for i in range(len(sorted_txns) - 1):
            t1 = sorted_txns[i].get('timestamp')
            t2 = sorted_txns[i + 1].get('timestamp')

            if t1 and t2:
                if isinstance(t1, str):
                    t1 = datetime.fromisoformat(t1.replace('Z', '+00:00'))
                if isinstance(t2, str):
                    t2 = datetime.fromisoformat(t2.replace('Z', '+00:00'))

                time_diff = (t2 - t1).total_seconds() / 60

                if time_diff < time_window_minutes:
                    alerts.append({
                        "type": "velocity_alert",
                        "reason": f"Two transactions within {time_diff:.1f} minutes",
                        "transactions": [sorted_txns[i], sorted_txns[i + 1]],
                        "severity": "high" if time_diff < 5 else "medium"
                    })

        return alerts

    def _check_amount_anomalies(
        self,
        transactions: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check for unusual transaction amounts"""
        alerts = []

        amounts = [txn.get('amount', 0) for txn in transactions if txn.get('amount')]

        if not amounts:
            return alerts

        mean_amount = sum(amounts) / len(amounts)
        threshold = mean_amount * 5  # 5x average

        for txn in transactions:
            amount = txn.get('amount', 0)
            if amount > threshold:
                alerts.append({
                    "type": "amount_anomaly",
                    "reason": f"Transaction amount ${amount:,.2f} is {amount/mean_amount:.1f}x average",
                    "transaction": txn,
                    "severity": "high"
                })

        return alerts

    def _calculate_fraud_confidence(self, indicator: Dict[str, Any]) -> float:
        """Calculate fraud confidence score (0-1)"""
        base_score = 0.5

        # Adjust based on severity
        severity = indicator.get('severity', 'low')
        if severity == 'critical':
            base_score += 0.4
        elif severity == 'high':
            base_score += 0.3
        elif severity == 'medium':
            base_score += 0.2

        # Cap at 1.0
        return min(base_score, 1.0)

    async def analyze_transaction_risk(
        self,
        user_id: str,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Comprehensive transaction risk analysis.

        Args:
            user_id: User identifier
            lookback_days: Days of history to analyze

        Returns:
            Complete risk analysis report
        """
        self.logger.info(f"Analyzing transaction risk for user {user_id}")

        # This would fetch real data from database
        # For MVP, using mock data
        mock_transactions = self._generate_mock_transactions(user_id, lookback_days)

        # Build graphs
        txn_graph = await self.build_transaction_graph(mock_transactions, user_id)

        # Detect anomalies
        anomalies = await self.detect_anomalous_patterns(txn_graph)

        # Check fraud indicators
        fraud_indicators = await self.find_fraud_indicators(mock_transactions)

        # Calculate overall risk
        risk_score = len(anomalies) * 0.1 + len(fraud_indicators) * 0.2
        risk_score = min(risk_score, 1.0)

        return {
            "user_id": user_id,
            "analysis_period_days": lookback_days,
            "overall_risk_score": risk_score,
            "risk_level": self._categorize_risk(risk_score),
            "transaction_count": len(mock_transactions),
            "graph_stats": {
                "nodes": txn_graph.number_of_nodes(),
                "edges": txn_graph.number_of_edges(),
                "density": nx.density(txn_graph)
            },
            "anomalies": anomalies,
            "fraud_indicators": fraud_indicators,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _generate_mock_transactions(
        self,
        user_id: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """Generate mock transaction data for testing"""
        import random

        transactions = []
        base_time = datetime.utcnow() - timedelta(days=days)

        merchants = ['Amazon', 'Grocery Store', 'Gas Station', 'Restaurant', 'Utilities']
        categories = ['shopping', 'food', 'transportation', 'bills', 'entertainment']

        for i in range(random.randint(20, 50)):
            transactions.append({
                "id": f"txn_{i}",
                "from_account": f"user_{user_id}_checking",
                "merchant": random.choice(merchants),
                "amount": random.uniform(10, 500),
                "category": random.choice(categories),
                "timestamp": (base_time + timedelta(days=random.randint(0, days))).isoformat()
            })

        return transactions


# Singleton instance
_graph_risk_detector = None


def get_graph_risk_detector() -> GraphRiskDetector:
    """Get or create singleton graph risk detector instance"""
    global _graph_risk_detector
    if _graph_risk_detector is None:
        _graph_risk_detector = GraphRiskDetector()
    return _graph_risk_detector
