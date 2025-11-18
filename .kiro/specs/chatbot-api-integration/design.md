# Design Document

## Overview

The chatbot API integration failure is caused by missing conversational endpoints in the main API server and potential dependency issues with the conversational agent. The design will integrate the conversational endpoints into the main FastAPI application while ensuring graceful handling of missing dependencies and proper error responses.

## Architecture

### Current State
- Main API server (`main.py`) serves VP-MAS agent endpoints but excludes conversational endpoints
- Conversational endpoints exist in `api/conversational_endpoints.py` but are only included in standalone servers
- Frontend receives "Failed to fetch" errors when trying to access conversational API endpoints
- Conversational agent may have dependency issues with Ollama that cause import failures

### Target State
- Main API server includes conversational endpoints router
- All conversational endpoints are accessible through the main API server
- Dependency issues are handled gracefully with fallback mechanisms
- Frontend can successfully communicate with chatbot functionality

## Components and Interfaces

### 1. Main API Server Integration
**File**: `main.py`
- Import conversational endpoints router
- Include router in FastAPI application with proper prefix
- Handle potential import errors gracefully
- Ensure CORS configuration supports conversational endpoints

### 2. Conversational Agent Initialization
**File**: `main.py` (lifespan function)
- Initialize conversational agent during application startup
- Handle missing Ollama dependencies gracefully
- Provide fallback initialization if dependencies are unavailable
- Add conversational agent to the global agents dictionary

### 3. Error Handling Enhancement
**Files**: `api/conversational_endpoints.py`, `agents/conversational_agent.py`
- Improve error handling for missing dependencies
- Ensure endpoints return proper HTTP status codes
- Provide meaningful error messages for debugging
- Implement graceful degradation when Ollama is unavailable

### 4. Dependency Management
**File**: `agents/conversational_agent.py`
- Wrap Ollama imports in try-catch blocks
- Provide clear logging when dependencies are missing
- Ensure agent can initialize even without Ollama
- Implement rule-based fallbacks for all functionality

## Data Models

### Request/Response Models
- Existing Pydantic models in `api/conversational_endpoints.py` are sufficient
- No changes needed to request/response schemas
- Error responses will follow existing API response format

### Agent Integration
- Conversational agent will be added to the global `agents` dictionary
- Agent initialization will follow the same pattern as other agents
- Message passing will use existing `AgentMessage` schema

## Error Handling

### Import Error Handling
```python
try:
    from api.conversational_endpoints import router as conversational_router
    CONVERSATIONAL_AVAILABLE = True
except ImportError as e:
    CONVERSATIONAL_AVAILABLE = False
    logger.warning(f"Conversational endpoints unavailable: {e}")
```

### Dependency Error Handling
- Ollama import failures will be caught and logged
- Agent will initialize with `ollama_available = False`
- All endpoints will use rule-based fallbacks when LLM is unavailable
- HTTP 503 responses for service degradation, not 500 errors

### Runtime Error Handling
- Invalid requests return HTTP 400 with clear error messages
- Agent failures return HTTP 500 with correlation IDs for debugging
- Network timeouts handled with appropriate HTTP status codes

## Testing Strategy

### Integration Testing
- Test main API server startup with conversational endpoints included
- Verify all conversational endpoints are accessible through main server
- Test error handling when Ollama dependencies are missing
- Validate CORS configuration for frontend communication

### Endpoint Testing
- Test each conversational endpoint through main API server
- Verify request/response schemas match documentation
- Test error scenarios (invalid input, missing dependencies)
- Validate fallback behavior when LLM is unavailable

### Frontend Integration Testing
- Test frontend can successfully fetch from conversational endpoints
- Verify no "Failed to fetch" errors occur
- Test error handling in frontend when API returns errors
- Validate end-to-end chatbot functionality

## Implementation Approach

### Phase 1: Basic Integration
1. Add conversational router import to main.py with error handling
2. Include conversational router in FastAPI application
3. Initialize conversational agent in lifespan function
4. Test basic endpoint accessibility

### Phase 2: Error Handling
1. Enhance dependency error handling in conversational agent
2. Improve error responses in conversational endpoints
3. Add proper logging for debugging
4. Test graceful degradation scenarios

### Phase 3: Validation
1. Test all endpoints through main API server
2. Verify frontend integration works without fetch errors
3. Validate error handling and fallback mechanisms
4. Update API documentation to reflect changes