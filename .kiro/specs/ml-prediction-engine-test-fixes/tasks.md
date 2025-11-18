# Implementation Plan

- [x] 1. Implement input validation framework





  - Create InputValidator class with comprehensive validation methods
  - Add portfolio data structure validation to prevent 'str' object attribute errors
  - Implement horizon days validation to ensure positive values
  - Add contamination parameter validation for sklearn compatibility
  - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [-] 2. Fix portfolio prediction response structure





  - [x] 2.1 Implement ResponseFormatter class for consistent response formatting








    - Create standardized portfolio response structure with top-level "expected_value" field
    - Ensure backward compatibility by maintaining existing nested structure
    - Add response validation before returning results
    - _Requirements: 1.1, 1.3, 1.4_

  - [-] 2.2 Update portfolio prediction methods to use standardized response format

    - Modify predict_portfolio_performance method to use ResponseFormatter
    - Ensure consistent structure across different timeframes and simulation counts
    - Fix test failures in TestPortfolioPerformancePrediction and TestPerformanceAndStress
    - _Requirements: 1.1, 1.2, 1.3_

- [ ] 3. Enhance message processing validation
  - [ ] 3.1 Implement MessageValidator class for Agent_Message handling
    - Create proper AgentMessage validation with all required Pydantic fields
    - Add default field population for "payload" and "trace_id" fields
    - Implement error response generation for validation failures
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ] 3.2 Update message processing methods to use proper validation
    - Fix TestMessageProcessing test failures by ensuring complete AgentMessage creation
    - Add graceful handling of missing required fields
    - Implement appropriate error responses for unknown actions
    - _Requirements: 2.2, 2.3, 2.4_

- [ ] 4. Integrate input validation into ML prediction methods
  - [ ] 4.1 Add validation calls to predict_portfolio_performance method
    - Validate portfolio data structure before processing
    - Return structured error response for invalid asset data
    - Fix TestErrorHandling.test_invalid_portfolio_data failure
    - _Requirements: 3.1, 3.4_

  - [ ] 4.2 Add validation calls to predict_market_trend method
    - Validate horizon_days parameter before model execution
    - Prevent sklearn errors by checking input parameters
    - Fix TestErrorHandling.test_negative_horizon_days failure
    - _Requirements: 3.2, 3.4, 3.5_

  - [ ] 4.3 Add validation calls to detect_market_anomaly method
    - Validate contamination parameter against sklearn constraints
    - Return appropriate error for out-of-range contamination values
    - Fix TestErrorHandling.test_extreme_contamination_values failure
    - _Requirements: 3.3, 3.4, 3.5_

- [ ] 5. Run comprehensive test validation
  - Execute all ML prediction engine tests to verify fixes resolve the 7 identified failures
  - Perform regression testing to ensure no new issues are introduced
  - Update any test assertions if needed to match new response structures
  - _Requirements: 1.4, 2.4, 3.4_