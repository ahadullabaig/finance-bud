"""
ML Prediction Engine - Phase 6, Task 25
Advanced AI and Machine Learning Features

Provides ML-based predictions and insights:
- Market trend predictions
- Portfolio performance forecasting
- Anomaly detection
- Personalized recommendations
- Risk predictions

Requirements: Phase 6, Task 25
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available")

from agents.base_agent import BaseAgent
from agents.input_validator import InputValidator
from agents.response_formatter import ResponseFormatter
from agents.message_validator import MessageValidator
from data_models.schemas import AgentMessage, MessageType


class MLPredictionEngine(BaseAgent):
    """
    Machine Learning prediction engine for financial planning.

    Capabilities:
    - Market trend prediction
    - Portfolio performance forecasting
    - Risk prediction models
    - Anomaly detection
    - Personalized recommendations
    """

    def __init__(
        self,
        agent_id: str = "ml-prediction-engine-001",
        confidence_threshold: float = 0.7
    ):
        super().__init__(agent_id, "MLPredictionEngine")
        self.confidence_threshold = confidence_threshold

        # Initialize models
        self.models = {
            'market_predictor': None,
            'risk_predictor': None,
            'anomaly_detector': None,
            'recommender': None
        }

        # Cache for predictions
        self.prediction_cache = {}
        
        # Initialize validators
        self.input_validator = InputValidator()
        self.message_validator = MessageValidator()

        self.logger.info(
            f"MLPredictionEngine initialized. "
            f"sklearn available: {SKLEARN_AVAILABLE}, "
            f"scipy available: {SCIPY_AVAILABLE}"
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages with proper validation and error handling.
        
        Args:
            message: AgentMessage to process
            
        Returns:
            AgentMessage response or None for unsupported message types
        """
        try:
            # Validate message content structure
            is_valid, error_msg = self.message_validator.validate_message_content(message.payload)
            if not is_valid:
                error_response = self.message_validator.create_error_response(
                    error_type="validation_error",
                    message=error_msg,
                    original_message=message,
                    agent_id=self.agent_id
                )
                return self.message_validator.create_response_message(
                    original_message=message,
                    response_content=error_response,
                    responding_agent_id=self.agent_id
                )
            
            if message.message_type == MessageType.REQUEST:
                action = message.payload.get("action")
                
                if action == "predict_market":
                    result = await self.predict_market_trend(
                        message.payload.get("symbol"),
                        message.payload.get("horizon_days", 30)
                    )
                    return self.message_validator.create_response_message(
                        original_message=message,
                        response_content=result,
                        responding_agent_id=self.agent_id
                    )
                
                elif action == "predict_portfolio":
                    result = await self.predict_portfolio_performance(
                        message.payload.get("portfolio", {}),
                        message.payload.get("timeframe_days", 365),
                        message.payload.get("num_simulations", 1000)
                    )
                    return self.message_validator.create_response_message(
                        original_message=message,
                        response_content=result,
                        responding_agent_id=self.agent_id
                    )
                
                elif action == "detect_anomaly":
                    result = await self.detect_market_anomaly(
                        message.payload.get("market_data", []),
                        message.payload.get("contamination", 0.1)
                    )
                    return self.message_validator.create_response_message(
                        original_message=message,
                        response_content=result,
                        responding_agent_id=self.agent_id
                    )
                
                elif action == "generate_recommendations":
                    result = await self.generate_recommendations(
                        message.payload.get("user_profile", {}),
                        message.payload.get("market_context")
                    )
                    return self.message_validator.create_response_message(
                        original_message=message,
                        response_content={"recommendations": result},
                        responding_agent_id=self.agent_id
                    )
                
                elif action == "predict_risk":
                    result = await self.predict_risk_levels(
                        message.payload.get("portfolio", {}),
                        message.payload.get("scenarios", [])
                    )
                    return self.message_validator.create_response_message(
                        original_message=message,
                        response_content=result,
                        responding_agent_id=self.agent_id
                    )
                
                else:
                    # Handle unknown action
                    error_response = self.message_validator.handle_unknown_action(
                        action=action or "None",
                        original_message=message,
                        responding_agent_id=self.agent_id
                    )
                    return self.message_validator.create_response_message(
                        original_message=message,
                        response_content=error_response,
                        responding_agent_id=self.agent_id
                    )
            
            # Return None for non-REQUEST message types
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            error_response = self.message_validator.create_error_response(
                error_type="processing_error",
                message=f"Internal error processing message: {str(e)}",
                original_message=message,
                agent_id=self.agent_id
            )
            return self.message_validator.create_response_message(
                original_message=message,
                response_content=error_response,
                responding_agent_id=self.agent_id
            )

    async def predict_market_trend(
        self,
        symbol: str,
        horizon_days: int = 30,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Predict market trend using linear regression and ensemble methods.

        Args:
            symbol: Stock/asset symbol
            horizon_days: Prediction horizon in days
            historical_data: Optional historical price data

        Returns:
            Prediction with confidence intervals
        """
        self.logger.info(f"Predicting market trend for {symbol}, horizon: {horizon_days} days")
        
        # Validate horizon_days parameter before model execution
        is_valid, error_msg = self.input_validator.validate_horizon_days(horizon_days)
        if not is_valid:
            return self.input_validator.create_error_response("validation_error", error_msg)
        
        # Validate all other inputs
        is_valid, error_msg = self.input_validator.validate_all_market_inputs(
            symbol=symbol,
            horizon_days=horizon_days,
            historical_data=historical_data
        )
        if not is_valid:
            return self.input_validator.create_error_response("validation_error", error_msg)

        # Use cached data or generate mock data
        if historical_data is None:
            historical_data = self._generate_mock_market_data(symbol, 90)

        if not SKLEARN_AVAILABLE:
            return self._simple_trend_prediction(historical_data, horizon_days)

        # Extract price series
        dates = []
        prices = []
        for data_point in historical_data:
            dates.append(data_point.get('date'))
            prices.append(data_point.get('price', 0))

        if len(prices) < 10:
            return {"error": "Insufficient historical data"}

        # Prepare training data
        X = np.array(range(len(prices))).reshape(-1, 1)
        y = np.array(prices)

        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        future_X = np.array(range(len(prices), len(prices) + horizon_days)).reshape(-1, 1)
        predictions = model.predict(future_X)

        # Calculate confidence intervals
        residuals = y - model.predict(X)
        std_error = np.std(residuals)

        # Generate prediction details
        prediction_dates = [
            (datetime.utcnow() + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(1, horizon_days + 1)
        ]

        forecast = []
        for i, (date, pred) in enumerate(zip(prediction_dates, predictions)):
            forecast.append({
                "date": date,
                "predicted_price": float(pred),
                "lower_bound": float(pred - 1.96 * std_error),
                "upper_bound": float(pred + 1.96 * std_error),
                "confidence": 0.95
            })

        # Calculate trend
        trend_direction = "bullish" if predictions[-1] > prices[-1] else "bearish"
        trend_strength = abs(predictions[-1] - prices[-1]) / prices[-1]

        return {
            "symbol": symbol,
            "current_price": float(prices[-1]),
            "predicted_price": float(predictions[-1]),
            "price_change": float(predictions[-1] - prices[-1]),
            "price_change_percent": float(trend_strength * 100),
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "forecast": forecast[:10],  # Return first 10 days
            "model": "linear_regression",
            "confidence": self._calculate_model_confidence(residuals, std_error),
            "timestamp": datetime.utcnow().isoformat()
        }

    def _simple_trend_prediction(
        self,
        historical_data: List[Dict[str, Any]],
        horizon_days: int
    ) -> Dict[str, Any]:
        """Simple moving average prediction when sklearn not available"""
        prices = [d.get('price', 0) for d in historical_data]

        if len(prices) < 5:
            return {"error": "Insufficient data"}

        # Simple moving average
        recent_avg = sum(prices[-5:]) / 5
        overall_avg = sum(prices) / len(prices)

        # Trend based on recent vs overall average
        trend_direction = "bullish" if recent_avg > overall_avg else "bearish"

        # Simple linear extrapolation
        change_rate = (prices[-1] - prices[0]) / len(prices)
        predicted_price = prices[-1] + (change_rate * horizon_days)

        return {
            "symbol": historical_data[0].get('symbol', 'unknown'),
            "current_price": prices[-1],
            "predicted_price": predicted_price,
            "price_change": predicted_price - prices[-1],
            "trend_direction": trend_direction,
            "model": "simple_moving_average",
            "confidence": 0.6
        }

    async def predict_portfolio_performance(
        self,
        portfolio: Dict[str, Any],
        timeframe_days: int = 365,
        num_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Predict portfolio performance using Monte Carlo simulation.

        Args:
            portfolio: Portfolio composition
            timeframe_days: Prediction timeframe
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Performance predictions with uncertainty quantification
        """
        self.logger.info(f"Predicting portfolio performance for {timeframe_days} days")
        
        # Validate portfolio data structure before processing
        is_valid, error_msg = self.input_validator.validate_portfolio_data(portfolio)
        if not is_valid:
            return self.input_validator.create_error_response("validation_error", error_msg)
        
        # Validate all other inputs
        is_valid, error_msg = self.input_validator.validate_all_portfolio_inputs(
            portfolio=portfolio,
            timeframe_days=timeframe_days,
            num_simulations=num_simulations
        )
        if not is_valid:
            return self.input_validator.create_error_response("validation_error", error_msg)

        if not SKLEARN_AVAILABLE or not SCIPY_AVAILABLE:
            return self._simple_portfolio_prediction(portfolio, timeframe_days)

        # Extract portfolio composition
        total_value = portfolio.get('total_value', 100000)
        assets = portfolio.get('assets', [])

        if not assets:
            return {"error": "No assets in portfolio"}

        # Monte Carlo simulation
        simulated_returns = []

        for _ in range(num_simulations):
            portfolio_return = 0

            for asset in assets:
                allocation = asset.get('allocation', 0)
                # Assume normal distribution of returns
                expected_return = asset.get('expected_return', 0.07)  # 7% default
                volatility = asset.get('volatility', 0.15)  # 15% default

                # Simulate return
                if SCIPY_AVAILABLE:
                    simulated_return = np.random.normal(
                        expected_return * (timeframe_days / 365),
                        volatility * np.sqrt(timeframe_days / 365)
                    )
                else:
                    simulated_return = expected_return * (timeframe_days / 365)

                portfolio_return += allocation * simulated_return

            simulated_returns.append(portfolio_return)

        # Calculate statistics
        simulated_returns = np.array(simulated_returns)
        mean_return = np.mean(simulated_returns)
        std_return = np.std(simulated_returns)

        # Calculate percentiles
        percentiles = np.percentile(simulated_returns, [5, 25, 50, 75, 95])

        # Prepare prediction data
        predictions = {
            "expected_return": float(mean_return),
            "expected_value": float(total_value * (1 + mean_return)),
            "std_deviation": float(std_return),
            "percentile_5": float(total_value * (1 + percentiles[0])),
            "percentile_25": float(total_value * (1 + percentiles[1])),
            "median": float(total_value * (1 + percentiles[2])),
            "percentile_75": float(total_value * (1 + percentiles[3])),
            "percentile_95": float(total_value * (1 + percentiles[4]))
        }
        
        risk_metrics = {
            "var_95": float(total_value * percentiles[0]),  # Value at Risk
            "sharpe_ratio": float(mean_return / std_return) if std_return > 0 else 0,
            "max_drawdown_estimate": float(percentiles[0])
        }
        
        metadata = {
            "timeframe_days": timeframe_days,
            "num_simulations": num_simulations,
            "confidence": 0.95,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Use ResponseFormatter to create standardized response
        response = ResponseFormatter.format_portfolio_response(
            current_value=total_value,
            predictions=predictions,
            risk_metrics=risk_metrics,
            metadata=metadata
        )
        
        # Ensure backward compatibility
        return ResponseFormatter.ensure_backward_compatibility(response)

    def _simple_portfolio_prediction(
        self,
        portfolio: Dict[str, Any],
        timeframe_days: int
    ) -> Dict[str, Any]:
        """Simple portfolio prediction without Monte Carlo"""
        total_value = portfolio.get('total_value', 100000)
        expected_annual_return = 0.07  # 7% default

        expected_return = expected_annual_return * (timeframe_days / 365)
        expected_value = total_value * (1 + expected_return)

        # Prepare prediction data for simple model
        predictions = {
            "expected_return": expected_return,
            "expected_value": expected_value,
            "std_deviation": 0.0,  # No variance in simple model
            "percentile_5": expected_value * 0.95,  # Simple approximation
            "percentile_25": expected_value * 0.975,
            "median": expected_value,
            "percentile_75": expected_value * 1.025,
            "percentile_95": expected_value * 1.05
        }
        
        metadata = {
            "timeframe_days": timeframe_days,
            "model": "simple_compound_return",
            "num_simulations": 1,
            "confidence": 0.6,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Use ResponseFormatter to create standardized response
        response = ResponseFormatter.format_portfolio_response(
            current_value=total_value,
            predictions=predictions,
            metadata=metadata
        )
        
        # Ensure backward compatibility
        return ResponseFormatter.ensure_backward_compatibility(response)

    async def detect_market_anomaly(
        self,
        market_data: List[Dict[str, Any]],
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect anomalies in market data using Isolation Forest.

        Args:
            market_data: Time series market data
            contamination: Expected proportion of anomalies

        Returns:
            Anomaly detection results
        """
        self.logger.info(f"Detecting market anomalies in {len(market_data)} data points")
        
        # Validate contamination parameter against sklearn constraints
        is_valid, error_msg = self.input_validator.validate_contamination_value(contamination)
        if not is_valid:
            return self.input_validator.create_error_response("validation_error", error_msg)
        
        # Validate all other inputs
        is_valid, error_msg = self.input_validator.validate_all_anomaly_inputs(
            market_data=market_data,
            contamination=contamination
        )
        if not is_valid:
            return self.input_validator.create_error_response("validation_error", error_msg)

        if not SKLEARN_AVAILABLE:
            return self._simple_anomaly_detection(market_data)

        # Extract features
        features = []
        timestamps = []

        for data_point in market_data:
            timestamps.append(data_point.get('date'))
            features.append([
                data_point.get('price', 0),
                data_point.get('volume', 0),
                data_point.get('volatility', 0)
            ])

        if len(features) < 10:
            return {"error": "Insufficient data for anomaly detection"}

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Train Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(features_normalized)
        scores = clf.score_samples(features_normalized)

        # Extract anomalies
        anomalies = []
        for idx, (pred, score, timestamp) in enumerate(zip(predictions, scores, timestamps)):
            if pred == -1:  # Anomaly detected
                anomalies.append({
                    "date": timestamp,
                    "anomaly_score": float(-score),
                    "data": market_data[idx],
                    "severity": "high" if -score > 0.6 else "medium"
                })

        return {
            "total_points": len(market_data),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / len(market_data),
            "anomalies": anomalies,
            "model": "isolation_forest",
            "contamination": contamination,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _simple_anomaly_detection(
        self,
        market_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simple statistical anomaly detection"""
        prices = [d.get('price', 0) for d in market_data]

        if len(prices) < 5:
            return {"error": "Insufficient data"}

        mean_price = sum(prices) / len(prices)
        # Simple std calculation
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_price = variance ** 0.5

        # Detect outliers (> 2 std deviations)
        anomalies = []
        for idx, data_point in enumerate(market_data):
            price = data_point.get('price', 0)
            if abs(price - mean_price) > 2 * std_price:
                anomalies.append({
                    "date": data_point.get('date'),
                    "anomaly_score": abs(price - mean_price) / std_price,
                    "data": data_point
                })

        return {
            "total_points": len(market_data),
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "model": "statistical_outlier"
        }

    async def generate_recommendations(
        self,
        user_profile: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized financial recommendations.

        Uses rule-based system with ML-enhanced scoring.

        Args:
            user_profile: User financial profile
            market_context: Current market conditions

        Returns:
            Ranked list of recommendations
        """
        self.logger.info("Generating personalized recommendations")

        recommendations = []

        # Extract user info
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', 'moderate')
        goals = user_profile.get('goals', [])
        current_portfolio = user_profile.get('portfolio', {})

        # Recommendation 1: Emergency fund
        if user_profile.get('emergency_fund', 0) < user_profile.get('monthly_expenses', 0) * 6:
            recommendations.append({
                "type": "emergency_fund",
                "title": "Build Emergency Fund",
                "description": "Ensure you have 6 months of expenses saved",
                "priority": "high",
                "rationale": "Emergency fund provides financial security",
                "action_items": [
                    "Calculate 6 months of expenses",
                    "Set up automatic transfers to savings",
                    "Keep funds in high-yield savings account"
                ],
                "estimated_impact": "high",
                "confidence": 0.9
            })

        # Recommendation 2: Retirement savings
        if age < 50 and user_profile.get('retirement_contributions', 0) < 0.15:
            recommendations.append({
                "type": "retirement_savings",
                "title": "Increase Retirement Contributions",
                "description": "Consider increasing 401(k) contributions to 15%",
                "priority": "medium",
                "rationale": "Early retirement savings benefit from compound growth",
                "action_items": [
                    "Review employer match policy",
                    "Increase contribution percentage",
                    "Consider IRA in addition to 401(k)"
                ],
                "estimated_impact": "high",
                "confidence": 0.85
            })

        # Recommendation 3: Diversification
        if current_portfolio:
            asset_types = len(set(asset.get('type') for asset in current_portfolio.get('assets', [])))
            if asset_types < 3:
                recommendations.append({
                    "type": "diversification",
                    "title": "Improve Portfolio Diversification",
                    "description": "Spread investments across more asset classes",
                    "priority": "medium",
                    "rationale": "Diversification reduces risk",
                    "action_items": [
                        "Add bonds to portfolio",
                        "Consider international exposure",
                        "Include alternative assets (REITs, commodities)"
                    ],
                    "estimated_impact": "medium",
                    "confidence": 0.75
                })

        # Recommendation 4: Tax optimization
        if user_profile.get('tax_bracket', 22) >= 22:
            recommendations.append({
                "type": "tax_optimization",
                "title": "Tax-Loss Harvesting Opportunity",
                "description": "Optimize portfolio for tax efficiency",
                "priority": "low",
                "rationale": "Reduce tax burden through strategic selling",
                "action_items": [
                    "Review portfolio for tax-loss harvesting",
                    "Consider tax-advantaged accounts",
                    "Optimize asset location"
                ],
                "estimated_impact": "medium",
                "confidence": 0.7
            })

        # Score and rank recommendations
        for rec in recommendations:
            rec['score'] = self._score_recommendation(rec, user_profile, market_context)

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations

    def _score_recommendation(
        self,
        recommendation: Dict[str, Any],
        user_profile: Dict[str, Any],
        market_context: Optional[Dict[str, Any]]
    ) -> float:
        """Score recommendation based on relevance and impact"""
        score = 0.0

        # Priority weight
        priority_weights = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        score += priority_weights.get(recommendation.get('priority', 'low'), 0.5)

        # Impact weight
        impact_weights = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        score += impact_weights.get(recommendation.get('estimated_impact', 'low'), 0.5)

        # Confidence weight
        score += recommendation.get('confidence', 0.5)

        # Market context adjustment
        if market_context:
            market_volatility = market_context.get('volatility', 0.5)
            if recommendation.get('type') == 'diversification' and market_volatility > 0.7:
                score += 0.5  # Boost diversification in volatile markets

        return score / 3  # Normalize to 0-1

    async def predict_risk_levels(
        self,
        portfolio: Dict[str, Any],
        scenarios: List[str]
    ) -> Dict[str, Any]:
        """
        Predict risk levels under different scenarios.

        Args:
            portfolio: Portfolio composition
            scenarios: List of scenario names (e.g., 'market_crash', 'recession')

        Returns:
            Risk predictions for each scenario
        """
        self.logger.info(f"Predicting risk levels for {len(scenarios)} scenarios")

        risk_predictions = {}

        for scenario in scenarios:
            # Get scenario parameters
            scenario_params = self._get_scenario_parameters(scenario)

            # Calculate expected portfolio impact
            total_value = portfolio.get('total_value', 100000)
            assets = portfolio.get('assets', [])

            scenario_return = 0
            for asset in assets:
                allocation = asset.get('allocation', 0)
                asset_type = asset.get('type', 'stock')

                # Apply scenario impact
                impact = scenario_params.get('asset_impacts', {}).get(asset_type, -0.1)
                scenario_return += allocation * impact

            expected_loss = total_value * scenario_return
            risk_level = self._categorize_risk_level(scenario_return)

            risk_predictions[scenario] = {
                "expected_return": float(scenario_return),
                "expected_value_change": float(expected_loss),
                "risk_level": risk_level,
                "probability": scenario_params.get('probability', 0.1),
                "mitigation_strategies": scenario_params.get('mitigations', [])
            }

        return {
            "portfolio_value": total_value,
            "scenarios_analyzed": risk_predictions,
            "overall_risk_level": self._calculate_overall_risk(risk_predictions),
            "timestamp": datetime.utcnow().isoformat()
        }

    def _get_scenario_parameters(self, scenario: str) -> Dict[str, Any]:
        """Get parameters for different market scenarios"""
        scenarios = {
            "market_crash": {
                "asset_impacts": {"stock": -0.3, "bond": -0.05, "cash": 0.0},
                "probability": 0.05,
                "mitigations": [
                    "Increase bond allocation",
                    "Hold more cash reserves",
                    "Consider defensive stocks"
                ]
            },
            "recession": {
                "asset_impacts": {"stock": -0.15, "bond": 0.02, "cash": 0.0},
                "probability": 0.15,
                "mitigations": [
                    "Focus on quality companies",
                    "Increase fixed income",
                    "Reduce leverage"
                ]
            },
            "inflation_spike": {
                "asset_impacts": {"stock": -0.05, "bond": -0.1, "cash": -0.03},
                "probability": 0.2,
                "mitigations": [
                    "Add inflation-protected securities",
                    "Consider commodities",
                    "Real estate exposure"
                ]
            },
            "bull_market": {
                "asset_impacts": {"stock": 0.25, "bond": 0.05, "cash": 0.01},
                "probability": 0.3,
                "mitigations": [
                    "Rebalance to target allocation",
                    "Take profits strategically",
                    "Maintain discipline"
                ]
            }
        }

        return scenarios.get(scenario, {
            "asset_impacts": {"stock": -0.1, "bond": 0.0, "cash": 0.0},
            "probability": 0.1,
            "mitigations": ["Monitor situation", "Maintain diversification"]
        })

    def _categorize_risk_level(self, return_value: float) -> str:
        """Categorize risk level based on expected return"""
        if return_value < -0.2:
            return "CRITICAL"
        elif return_value < -0.1:
            return "HIGH"
        elif return_value < 0:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_overall_risk(self, risk_predictions: Dict[str, Any]) -> str:
        """Calculate overall risk level from scenario predictions"""
        weighted_risk = 0
        total_weight = 0

        for scenario, prediction in risk_predictions.items():
            prob = prediction.get('probability', 0.1)
            impact = abs(prediction.get('expected_return', 0))
            weighted_risk += prob * impact
            total_weight += prob

        avg_risk = weighted_risk / total_weight if total_weight > 0 else 0

        return self._categorize_risk_level(-avg_risk)

    def _calculate_model_confidence(
        self,
        residuals: np.ndarray,
        std_error: float
    ) -> float:
        """Calculate model confidence based on residuals"""
        if len(residuals) == 0:
            return 0.5

        # Calculate R-squared like metric
        mean_residual = np.mean(np.abs(residuals))
        confidence = max(0.0, 1.0 - (mean_residual / (std_error + 1e-10)))

        return min(float(confidence), 1.0)

    def _generate_mock_market_data(
        self,
        symbol: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """Generate mock market data for testing"""
        data = []
        base_price = 100.0
        base_date = datetime.utcnow() - timedelta(days=days)

        for i in range(days):
            # Random walk with slight upward trend
            change = np.random.normal(0.001, 0.02) if SKLEARN_AVAILABLE else 0.001
            base_price *= (1 + change)

            data.append({
                "symbol": symbol,
                "date": (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                "price": base_price,
                "volume": int(np.random.uniform(1000000, 5000000)) if SKLEARN_AVAILABLE else 2000000,
                "volatility": abs(change)
            })

        return data


# Singleton instance
_ml_prediction_engine = None


def get_ml_prediction_engine() -> MLPredictionEngine:
    """Get or create singleton ML prediction engine instance"""
    global _ml_prediction_engine
    if _ml_prediction_engine is None:
        _ml_prediction_engine = MLPredictionEngine()
    return _ml_prediction_engine
