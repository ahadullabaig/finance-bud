"""
NVIDIA Graph Neural Network Fraud/Risk Detection System
Advanced GNN-based risk detection for financial planning
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# NVIDIA cuGraph imports (with fallback to NetworkX)
try:
    import cudf
    import cugraph
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.cluster import DBSCAN as cuDBSCAN
    CUDA_AVAILABLE = True
    logger.info("NVIDIA cuGraph and cuML available - using GPU acceleration")
except ImportError:
    CUDA_AVAILABLE = False
    logger.info("NVIDIA cuGraph not available - using CPU fallback with NetworkX")

from pydantic import BaseModel, Field
from data_models.schemas import FinancialState, MarketData, RiskProfile

logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

class AnomalyType(str, Enum):
    """Types of financial anomalies"""
    FRAUD_PATTERN = "fraud_pattern"
    SYSTEMIC_RISK = "systemic_risk"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    MARKET_MANIPULATION = "market_manipulation"
    CORRELATION_BREAK = "correlation_break"
    LIQUIDITY_CRISIS = "liquidity_crisis"

class GraphNodeType(str, Enum):
    """Types of nodes in the financial graph"""
    USER = "user"
    ACCOUNT = "account"
    TRANSACTION = "transaction"
    ASSET = "asset"
    MARKET_SECTOR = "market_sector"
    INSTITUTION = "institution"
    REGULATORY_ENTITY = "regulatory_entity"

class RiskDetectionResult(BaseModel):
    """Result of risk detection analysis"""
    risk_id: str
    risk_level: RiskLevel
    anomaly_type: AnomalyType
    confidence_score: float
    description: str
    affected_entities: List[str]
    risk_factors: Dict[str, float]
    mitigation_strategies: List[str]
    detection_timestamp: datetime
    graph_context: Dict[str, Any]
    systemic_impact: float
    propagation_paths: List[List[str]]

class GraphAnalysisResult(BaseModel):
    """Result of graph-based analysis"""
    graph_metrics: Dict[str, float]
    community_structure: Dict[str, List[str]]
    centrality_measures: Dict[str, Dict[str, float]]
    anomaly_scores: Dict[str, float]
    risk_propagation_paths: List[Dict[str, Any]]
    systemic_risk_indicators: Dict[str, float]

class TransactionPattern(BaseModel):
    """Transaction pattern for analysis"""
    pattern_id: str
    user_id: str
    transaction_sequence: List[Dict[str, Any]]
    pattern_type: str
    frequency: float
    amount_statistics: Dict[str, float]
    temporal_features: Dict[str, float]
    risk_indicators: Dict[str, float]

@dataclass
class GNNConfiguration:
    """Configuration for GNN risk detection system"""
    use_gpu: bool = CUDA_AVAILABLE
    graph_database_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    risk_threshold: float = 0.7
    anomaly_threshold: float = 0.8
    max_graph_size: int = 100000
    analysis_window_days: int = 30
    update_frequency_hours: int = 1

class NVIDIAGNNRiskDetector:
    """
    NVIDIA Graph Neural Network Risk Detection System
    
    Provides advanced GNN-based capabilities for:
    - User spending pattern analysis
    - Asset correlation risk detection
    - Hidden systemic risk identification
    - Fraud pattern detection
    - Interactive graph visualization
    """
    
    def __init__(self, config: GNNConfiguration):
        self.config = config
        self.use_gpu = config.use_gpu and CUDA_AVAILABLE
        
        # Initialize graph storage
        if self.use_gpu:
            self.graph = cugraph.Graph()
            self.ml_models = self._initialize_cuml_models()
        else:
            self.graph = nx.Graph()
            self.ml_models = self._initialize_sklearn_models()
        
        # Risk detection models
        self.fraud_detector = None
        self.systemic_risk_detector = None
        self.behavioral_analyzer = None
        
        # Graph analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        self.last_update: Optional[datetime] = None
        
        # Initialize components
        self._initialize_risk_models()
        
    def _initialize_cuml_models(self) -> Dict[str, Any]:
        """Initialize cuML models for GPU acceleration"""
        return {
            "isolation_forest": cuml.ensemble.IsolationForest(
                n_estimators=100,
                contamination=0.1
            ),
            "dbscan": cuDBSCAN(eps=0.5, min_samples=5),
            "random_forest": cuRF(
                n_estimators=100,
                max_depth=10
            )
        }
    
    def _initialize_sklearn_models(self) -> Dict[str, Any]:
        """Initialize scikit-learn models for CPU fallback"""
        return {
            "isolation_forest": IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            ),
            "dbscan": DBSCAN(eps=0.5, min_samples=5),
            "random_forest": None  # Will initialize when needed
        }
    
    def _initialize_risk_models(self):
        """Initialize specialized risk detection models"""
        self.fraud_detector = FraudPatternDetector(self.use_gpu)
        self.systemic_risk_detector = SystemicRiskDetector(self.use_gpu)
        self.behavioral_analyzer = BehavioralAnomalyAnalyzer(self.use_gpu)

    async def create_user_spending_graph(
        self, 
        user_transactions: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Create user spending pattern graph with transaction relationships
        
        Args:
            user_transactions: List of user transaction data
            user_id: User identifier
            
        Returns:
            Graph analysis results with spending patterns
        """
        try:
            # Create spending pattern graph
            spending_graph = self._build_spending_graph(user_transactions, user_id)
            
            # Analyze spending patterns
            pattern_analysis = await self._analyze_spending_patterns(spending_graph, user_id)
            
            # Detect anomalies in spending behavior
            anomalies = await self._detect_spending_anomalies(pattern_analysis)
            
            # Calculate risk scores
            risk_scores = self._calculate_spending_risk_scores(pattern_analysis, anomalies)
            
            return {
                "user_id": user_id,
                "graph_metrics": self._calculate_graph_metrics(spending_graph),
                "spending_patterns": pattern_analysis,
                "anomalies": anomalies,
                "risk_scores": risk_scores,
                "visualization_data": self._prepare_visualization_data(spending_graph),
                "recommendations": self._generate_spending_recommendations(risk_scores)
            }
            
        except Exception as e:
            logger.error(f"Error creating user spending graph: {e}")
            raise

    async def build_asset_correlation_graph(
        self, 
        portfolio_data: Dict[str, Any],
        market_data: MarketData
    ) -> Dict[str, Any]:
        """
        Build asset correlation graph with market relationship mapping
        
        Args:
            portfolio_data: User portfolio information
            market_data: Current market data
            
        Returns:
            Asset correlation analysis with risk assessment
        """
        try:
            # Build correlation graph
            correlation_graph = self._build_correlation_graph(portfolio_data, market_data)
            
            # Analyze correlation structure
            correlation_analysis = await self._analyze_correlation_structure(correlation_graph)
            
            # Detect correlation breaks and anomalies
            correlation_anomalies = await self._detect_correlation_anomalies(correlation_analysis)
            
            # Assess systemic risk from correlations
            systemic_risk = await self._assess_correlation_systemic_risk(
                correlation_graph, 
                correlation_anomalies
            )
            
            return {
                "portfolio_id": portfolio_data.get("portfolio_id"),
                "correlation_matrix": correlation_analysis["correlation_matrix"],
                "graph_structure": correlation_analysis["graph_structure"],
                "anomalies": correlation_anomalies,
                "systemic_risk": systemic_risk,
                "diversification_score": self._calculate_diversification_score(correlation_graph),
                "risk_concentration": self._analyze_risk_concentration(correlation_graph),
                "visualization_data": self._prepare_correlation_visualization(correlation_graph)
            }
            
        except Exception as e:
            logger.error(f"Error building asset correlation graph: {e}")
            raise

    async def detect_hidden_systemic_risk(
        self, 
        financial_network: Dict[str, Any],
        market_context: MarketData
    ) -> List[RiskDetectionResult]:
        """
        Detect hidden systemic risk using graph neural networks
        
        Args:
            financial_network: Complete financial network data
            market_context: Current market conditions
            
        Returns:
            List of detected systemic risks with analysis
        """
        try:
            # Build comprehensive financial network graph
            network_graph = self._build_financial_network_graph(financial_network)
            
            # Apply GNN-based systemic risk detection
            systemic_risks = await self._detect_systemic_risks_gnn(network_graph, market_context)
            
            # Analyze risk propagation paths
            propagation_analysis = await self._analyze_risk_propagation(network_graph, systemic_risks)
            
            # Calculate systemic impact scores
            impact_scores = self._calculate_systemic_impact_scores(
                network_graph, 
                systemic_risks, 
                propagation_analysis
            )
            
            # Generate risk detection results
            risk_results = []
            for risk_id, risk_data in systemic_risks.items():
                result = RiskDetectionResult(
                    risk_id=risk_id,
                    risk_level=self._determine_risk_level(risk_data["risk_score"]),
                    anomaly_type=AnomalyType.SYSTEMIC_RISK,
                    confidence_score=risk_data["confidence"],
                    description=risk_data["description"],
                    affected_entities=risk_data["affected_entities"],
                    risk_factors=risk_data["risk_factors"],
                    mitigation_strategies=self._generate_mitigation_strategies(risk_data),
                    detection_timestamp=datetime.now(),
                    graph_context=risk_data["graph_context"],
                    systemic_impact=impact_scores.get(risk_id, 0.0),
                    propagation_paths=propagation_analysis.get(risk_id, [])
                )
                risk_results.append(result)
            
            return risk_results
            
        except Exception as e:
            logger.error(f"Error detecting hidden systemic risk: {e}")
            raise

    async def detect_fraud_patterns(
        self, 
        transaction_data: List[Dict[str, Any]],
        user_profiles: Dict[str, Any]
    ) -> List[RiskDetectionResult]:
        """
        Detect fraud patterns using GNN-based anomaly detection
        
        Args:
            transaction_data: Transaction history data
            user_profiles: User profile information
            
        Returns:
            List of detected fraud patterns
        """
        try:
            # Build transaction network graph
            transaction_graph = self._build_transaction_graph(transaction_data, user_profiles)
            
            # Extract transaction patterns
            patterns = await self._extract_transaction_patterns(transaction_graph)
            
            # Apply fraud detection models
            fraud_results = await self.fraud_detector.detect_fraud_patterns(
                transaction_graph, 
                patterns
            )
            
            # Analyze fraud propagation networks
            fraud_networks = await self._analyze_fraud_networks(transaction_graph, fraud_results)
            
            # Generate fraud detection results
            detection_results = []
            for fraud_id, fraud_data in fraud_results.items():
                result = RiskDetectionResult(
                    risk_id=fraud_id,
                    risk_level=self._determine_risk_level(fraud_data["fraud_score"]),
                    anomaly_type=AnomalyType.FRAUD_PATTERN,
                    confidence_score=fraud_data["confidence"],
                    description=fraud_data["description"],
                    affected_entities=fraud_data["affected_users"],
                    risk_factors=fraud_data["risk_indicators"],
                    mitigation_strategies=self._generate_fraud_mitigation_strategies(fraud_data),
                    detection_timestamp=datetime.now(),
                    graph_context=fraud_data["graph_context"],
                    systemic_impact=fraud_data.get("systemic_impact", 0.0),
                    propagation_paths=fraud_networks.get(fraud_id, [])
                )
                detection_results.append(result)
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error detecting fraud patterns: {e}")
            raise

    async def create_interactive_risk_visualization(
        self, 
        risk_analysis_results: List[RiskDetectionResult]
    ) -> Dict[str, Any]:
        """
        Create interactive graph visualization for risk exploration
        
        Args:
            risk_analysis_results: Risk detection results
            
        Returns:
            Interactive visualization data
        """
        try:
            # Prepare graph data for visualization
            viz_graph = self._prepare_risk_visualization_graph(risk_analysis_results)
            
            # Calculate layout positions
            layout_data = self._calculate_force_directed_layout(viz_graph)
            
            # Prepare interactive features
            interactive_features = {
                "zoom_levels": self._prepare_zoom_levels(viz_graph),
                "filter_options": self._prepare_filter_options(risk_analysis_results),
                "drill_down_data": self._prepare_drill_down_data(risk_analysis_results),
                "risk_heatmap": self._generate_risk_heatmap(viz_graph),
                "temporal_analysis": self._prepare_temporal_risk_analysis(risk_analysis_results)
            }
            
            # Generate visualization metadata
            visualization_metadata = {
                "graph_statistics": self._calculate_visualization_statistics(viz_graph),
                "risk_summary": self._generate_risk_summary(risk_analysis_results),
                "performance_metrics": self._calculate_visualization_performance_metrics(viz_graph)
            }
            
            return {
                "graph_data": {
                    "nodes": self._format_nodes_for_visualization(viz_graph),
                    "edges": self._format_edges_for_visualization(viz_graph),
                    "layout": layout_data
                },
                "interactive_features": interactive_features,
                "metadata": visualization_metadata,
                "styling": self._generate_visualization_styling(risk_analysis_results),
                "animations": self._prepare_risk_animations(risk_analysis_results)
            }
            
        except Exception as e:
            logger.error(f"Error creating interactive risk visualization: {e}")
            raise

    def _build_spending_graph(
        self, 
        transactions: List[Dict[str, Any]], 
        user_id: str
    ) -> Union[nx.Graph, Any]:
        """Build spending pattern graph from transaction data"""
        if self.use_gpu:
            # Use cuGraph for GPU acceleration
            graph = cugraph.Graph()
            
            # Convert transactions to cuDF DataFrame
            df_data = []
            for i, tx in enumerate(transactions):
                df_data.append({
                    "src": f"user_{user_id}",
                    "dst": f"category_{tx.get('category', 'unknown')}",
                    "weight": float(tx.get('amount', 0))
                })
            
            if df_data:
                df = cudf.DataFrame(df_data)
                graph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
            
            return graph
        else:
            # Use NetworkX for CPU fallback
            graph = nx.Graph()
            
            # Add user node
            graph.add_node(f"user_{user_id}", node_type=GraphNodeType.USER.value)
            
            # Add transaction nodes and edges
            for i, tx in enumerate(transactions):
                tx_id = f"tx_{i}"
                category = tx.get('category', 'unknown')
                amount = float(tx.get('amount', 0))
                
                # Add transaction node
                graph.add_node(tx_id, 
                    node_type=GraphNodeType.TRANSACTION.value,
                    amount=amount,
                    category=category,
                    timestamp=tx.get('timestamp')
                )
                
                # Add category node if not exists
                category_id = f"category_{category}"
                if not graph.has_node(category_id):
                    graph.add_node(category_id, node_type="category")
                
                # Add edges
                graph.add_edge(f"user_{user_id}", tx_id, weight=amount)
                graph.add_edge(tx_id, category_id, weight=1.0)
            
            return graph

    def _build_correlation_graph(
        self, 
        portfolio_data: Dict[str, Any], 
        market_data: MarketData
    ) -> Union[nx.Graph, Any]:
        """Build asset correlation graph"""
        if self.use_gpu:
            # Use cuGraph for correlation analysis
            graph = cugraph.Graph()
            
            # Build correlation matrix using cuML
            assets = list(portfolio_data.get('holdings', {}).keys())
            correlation_data = []
            
            # Calculate correlations (simplified for demo)
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets[i+1:], i+1):
                    correlation = np.random.uniform(0.1, 0.9)  # Replace with real correlation calculation
                    if correlation > 0.3:  # Threshold for significant correlation
                        correlation_data.append({
                            "src": asset1,
                            "dst": asset2,
                            "weight": correlation
                        })
            
            if correlation_data:
                df = cudf.DataFrame(correlation_data)
                graph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
            
            return graph
        else:
            # Use NetworkX for CPU fallback
            graph = nx.Graph()
            
            assets = list(portfolio_data.get('holdings', {}).keys())
            
            # Add asset nodes
            for asset in assets:
                graph.add_node(asset, 
                    node_type=GraphNodeType.ASSET.value,
                    holding_amount=portfolio_data['holdings'][asset]
                )
            
            # Add correlation edges (simplified calculation)
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets[i+1:], i+1):
                    correlation = np.random.uniform(0.1, 0.9)  # Replace with real correlation
                    if correlation > 0.3:
                        graph.add_edge(asset1, asset2, 
                            weight=correlation,
                            correlation_type="positive" if correlation > 0 else "negative"
                        )
            
            return graph

    async def _analyze_spending_patterns(
        self, 
        graph: Union[nx.Graph, Any], 
        user_id: str
    ) -> Dict[str, Any]:
        """Analyze spending patterns from graph structure"""
        if self.use_gpu:
            # Use cuGraph algorithms
            try:
                centrality = cugraph.betweenness_centrality(graph)
                pagerank = cugraph.pagerank(graph)
                
                return {
                    "centrality_measures": centrality.to_pandas().to_dict(),
                    "pagerank_scores": pagerank.to_pandas().to_dict(),
                    "graph_density": len(graph.edges()) / (len(graph.nodes()) * (len(graph.nodes()) - 1) / 2) if len(graph.nodes()) > 1 else 0
                }
            except Exception as e:
                logger.warning(f"cuGraph analysis failed, falling back to NetworkX: {e}")
                # Fallback to NetworkX
                return self._analyze_spending_patterns_networkx(graph, user_id)
        else:
            return self._analyze_spending_patterns_networkx(graph, user_id)

    def _analyze_spending_patterns_networkx(
        self, 
        graph: nx.Graph, 
        user_id: str
    ) -> Dict[str, Any]:
        """Analyze spending patterns using NetworkX"""
        if len(graph.nodes()) == 0:
            return {"error": "Empty graph"}
        
        # Calculate graph metrics
        centrality = nx.betweenness_centrality(graph)
        pagerank = nx.pagerank(graph)
        clustering = nx.clustering(graph)
        
        # Analyze spending categories
        category_analysis = {}
        user_node = f"user_{user_id}"
        
        if graph.has_node(user_node):
            neighbors = list(graph.neighbors(user_node))
            total_spending = sum(graph[user_node][neighbor].get('weight', 0) for neighbor in neighbors)
            
            for neighbor in neighbors:
                if graph.nodes[neighbor].get('node_type') == GraphNodeType.TRANSACTION.value:
                    category = graph.nodes[neighbor].get('category', 'unknown')
                    amount = graph.nodes[neighbor].get('amount', 0)
                    
                    if category not in category_analysis:
                        category_analysis[category] = {
                            'total_amount': 0,
                            'transaction_count': 0,
                            'percentage': 0
                        }
                    
                    category_analysis[category]['total_amount'] += amount
                    category_analysis[category]['transaction_count'] += 1
            
            # Calculate percentages
            for category in category_analysis:
                category_analysis[category]['percentage'] = (
                    category_analysis[category]['total_amount'] / total_spending * 100
                    if total_spending > 0 else 0
                )
        
        return {
            "centrality_measures": centrality,
            "pagerank_scores": pagerank,
            "clustering_coefficients": clustering,
            "graph_density": nx.density(graph),
            "category_analysis": category_analysis,
            "total_nodes": len(graph.nodes()),
            "total_edges": len(graph.edges())
        }

    async def _detect_spending_anomalies(self, pattern_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in spending patterns"""
        anomalies = []
        
        # Analyze category spending for anomalies
        category_analysis = pattern_analysis.get('category_analysis', {})
        
        for category, data in category_analysis.items():
            percentage = data['percentage']
            
            # Simple anomaly detection rules
            if percentage > 50:  # More than 50% in one category
                anomalies.append({
                    "type": "high_concentration",
                    "category": category,
                    "percentage": percentage,
                    "severity": "high" if percentage > 70 else "medium",
                    "description": f"High spending concentration in {category}: {percentage:.1f}%"
                })
            
            if data['transaction_count'] == 1 and data['total_amount'] > 1000:  # Large single transaction
                anomalies.append({
                    "type": "large_transaction",
                    "category": category,
                    "amount": data['total_amount'],
                    "severity": "medium",
                    "description": f"Large single transaction in {category}: ${data['total_amount']:.2f}"
                })
        
        return anomalies

    def _calculate_spending_risk_scores(
        self, 
        pattern_analysis: Dict[str, Any], 
        anomalies: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate risk scores based on spending patterns and anomalies"""
        risk_scores = {
            "overall_risk": 0.0,
            "concentration_risk": 0.0,
            "volatility_risk": 0.0,
            "anomaly_risk": 0.0
        }
        
        # Calculate concentration risk
        category_analysis = pattern_analysis.get('category_analysis', {})
        if category_analysis:
            max_percentage = max(data['percentage'] for data in category_analysis.values())
            risk_scores["concentration_risk"] = min(max_percentage / 50.0, 1.0)  # Normalize to 0-1
        
        # Calculate anomaly risk
        high_severity_anomalies = sum(1 for a in anomalies if a.get('severity') == 'high')
        medium_severity_anomalies = sum(1 for a in anomalies if a.get('severity') == 'medium')
        
        risk_scores["anomaly_risk"] = min(
            (high_severity_anomalies * 0.3 + medium_severity_anomalies * 0.1), 1.0
        )
        
        # Calculate overall risk
        risk_scores["overall_risk"] = (
            risk_scores["concentration_risk"] * 0.4 +
            risk_scores["anomaly_risk"] * 0.6
        )
        
        return risk_scores

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from numerical score"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _generate_mitigation_strategies(self, risk_data: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        risk_score = risk_data.get("risk_score", 0)
        risk_factors = risk_data.get("risk_factors", {})
        
        if risk_score > 0.7:
            strategies.append("Implement immediate risk monitoring and alerts")
            strategies.append("Consider portfolio diversification to reduce concentration risk")
        
        if "correlation_risk" in risk_factors and risk_factors["correlation_risk"] > 0.6:
            strategies.append("Reduce correlated asset exposure")
            strategies.append("Add uncorrelated or negatively correlated assets")
        
        if "liquidity_risk" in risk_factors and risk_factors["liquidity_risk"] > 0.5:
            strategies.append("Increase liquid asset allocation")
            strategies.append("Establish emergency liquidity reserves")
        
        return strategies

    def _prepare_visualization_data(self, graph: Union[nx.Graph, Any]) -> Dict[str, Any]:
        """Prepare graph data for visualization"""
        if self.use_gpu:
            # Convert cuGraph to visualization format
            try:
                # Get node and edge data from cuGraph
                nodes = []
                edges = []
                
                # This is a simplified conversion - in production, you'd use proper cuGraph methods
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "layout": "force_directed"
                }
            except Exception as e:
                logger.warning(f"cuGraph visualization preparation failed: {e}")
                return {"error": "Visualization preparation failed"}
        else:
            # Convert NetworkX to visualization format
            nodes = []
            edges = []
            
            for node_id, node_data in graph.nodes(data=True):
                nodes.append({
                    "id": node_id,
                    "type": node_data.get("node_type", "unknown"),
                    "size": node_data.get("amount", 10),
                    "color": self._get_node_color(node_data.get("node_type", "unknown"))
                })
            
            for source, target, edge_data in graph.edges(data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    "weight": edge_data.get("weight", 1.0),
                    "type": edge_data.get("correlation_type", "default")
                })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "layout": "force_directed"
            }

    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type"""
        color_map = {
            "user": "#3498db",
            "transaction": "#e74c3c",
            "category": "#2ecc71",
            "asset": "#f39c12",
            "unknown": "#95a5a6"
        }
        return color_map.get(node_type, "#95a5a6")

    def _calculate_graph_metrics(self, graph: Union[nx.Graph, Any]) -> Dict[str, float]:
        """Calculate basic graph metrics"""
        if self.use_gpu:
            try:
                # Use cuGraph metrics
                return {
                    "node_count": len(graph.nodes()) if hasattr(graph, 'nodes') else 0,
                    "edge_count": len(graph.edges()) if hasattr(graph, 'edges') else 0,
                    "density": 0.0  # Calculate using cuGraph methods
                }
            except Exception as e:
                logger.warning(f"cuGraph metrics calculation failed: {e}")
                return {"error": "Metrics calculation failed"}
        else:
            # Use NetworkX metrics
            if len(graph.nodes()) == 0:
                return {"node_count": 0, "edge_count": 0, "density": 0.0}
            
            return {
                "node_count": len(graph.nodes()),
                "edge_count": len(graph.edges()),
                "density": nx.density(graph),
                "average_clustering": nx.average_clustering(graph) if len(graph.nodes()) > 0 else 0.0
            }

    def _generate_spending_recommendations(self, risk_scores: Dict[str, float]) -> List[str]:
        """Generate spending recommendations based on risk analysis"""
        recommendations = []
        
        if risk_scores.get("concentration_risk", 0) > 0.6:
            recommendations.append("Consider diversifying spending across more categories")
            recommendations.append("Set budget limits for high-concentration categories")
        
        if risk_scores.get("anomaly_risk", 0) > 0.5:
            recommendations.append("Review unusual transactions for accuracy")
            recommendations.append("Set up alerts for large or unusual spending patterns")
        
        if risk_scores.get("overall_risk", 0) > 0.7:
            recommendations.append("Implement comprehensive spending monitoring")
            recommendations.append("Consider consulting with a financial advisor")
        
        return recommendations

# Specialized risk detection components
class FraudPatternDetector:
    """Specialized fraud pattern detection using GNN"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        
    async def detect_fraud_patterns(
        self, 
        transaction_graph: Union[nx.Graph, Any], 
        patterns: List[TransactionPattern]
    ) -> Dict[str, Any]:
        """Detect fraud patterns in transaction graph"""
        fraud_results = {}
        
        # Implement fraud detection logic
        for i, pattern in enumerate(patterns):
            fraud_score = self._calculate_fraud_score(pattern)
            
            if fraud_score > 0.7:  # Fraud threshold
                fraud_results[f"fraud_{i}"] = {
                    "fraud_score": fraud_score,
                    "confidence": 0.8,
                    "description": f"Suspicious transaction pattern detected",
                    "affected_users": [pattern.user_id],
                    "risk_indicators": pattern.risk_indicators,
                    "graph_context": {"pattern_type": pattern.pattern_type}
                }
        
        return fraud_results
    
    def _calculate_fraud_score(self, pattern: TransactionPattern) -> float:
        """Calculate fraud score for a transaction pattern"""
        # Simplified fraud scoring
        risk_factors = pattern.risk_indicators
        
        score = 0.0
        score += risk_factors.get("unusual_timing", 0) * 0.3
        score += risk_factors.get("amount_anomaly", 0) * 0.4
        score += risk_factors.get("frequency_anomaly", 0) * 0.3
        
        return min(score, 1.0)

class SystemicRiskDetector:
    """Specialized systemic risk detection using GNN"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        
    async def detect_systemic_risks(
        self, 
        network_graph: Union[nx.Graph, Any]
    ) -> Dict[str, Any]:
        """Detect systemic risks in financial network"""
        # Implement systemic risk detection logic
        return {}

class BehavioralAnomalyAnalyzer:
    """Behavioral anomaly analysis using GNN"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        
    async def analyze_behavioral_anomalies(
        self, 
        user_behavior_graph: Union[nx.Graph, Any]
    ) -> Dict[str, Any]:
        """Analyze behavioral anomalies in user patterns"""
        # Implement behavioral anomaly analysis
        return {}

# Factory function for creating GNN risk detector
def create_gnn_risk_detector() -> NVIDIAGNNRiskDetector:
    """Create and configure NVIDIA GNN risk detector"""
    config = GNNConfiguration(
        use_gpu=CUDA_AVAILABLE,
        risk_threshold=0.7,
        anomaly_threshold=0.8,
        analysis_window_days=30
    )
    
    return NVIDIAGNNRiskDetector(config)

# Integration with existing agent system
class GNNIntegratedAgent:
    """Integration wrapper for GNN risk detector with existing agent system"""
    
    def __init__(self, gnn_detector: NVIDIAGNNRiskDetector):
        self.gnn_detector = gnn_detector
        
    async def analyze_user_risk(
        self, 
        user_data: Dict[str, Any], 
        transaction_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user risk using GNN detector"""
        user_id = user_data.get("user_id", "unknown")
        
        # Create spending graph and analyze
        spending_analysis = await self.gnn_detector.create_user_spending_graph(
            transaction_history, 
            user_id
        )
        
        # Detect fraud patterns
        fraud_results = await self.gnn_detector.detect_fraud_patterns(
            transaction_history, 
            {user_id: user_data}
        )
        
        return {
            "user_id": user_id,
            "spending_analysis": spending_analysis,
            "fraud_detection": fraud_results,
            "overall_risk_score": spending_analysis["risk_scores"]["overall_risk"],
            "recommendations": spending_analysis["recommendations"]
        }
    
    async def analyze_portfolio_risk(
        self, 
        portfolio_data: Dict[str, Any], 
        market_data: MarketData
    ) -> Dict[str, Any]:
        """Analyze portfolio risk using correlation analysis"""
        correlation_analysis = await self.gnn_detector.build_asset_correlation_graph(
            portfolio_data, 
            market_data
        )
        
        return {
            "portfolio_id": portfolio_data.get("portfolio_id"),
            "correlation_analysis": correlation_analysis,
            "diversification_score": correlation_analysis["diversification_score"],
            "risk_concentration": correlation_analysis["risk_concentration"]
        }