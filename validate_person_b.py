#!/usr/bin/env python3
"""
Validation Script for Person B Implementation

This script validates that all Person B tasks (2.6-2.10) have been
completed correctly without requiring external dependencies.

Checks:
1. All required files exist
2. Code structure and organization
3. Syntax validation
4. Documentation completeness
5. Implementation coverage
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Tuple, Dict

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_file_exists(filepath: str) -> bool:
    """Check if file exists"""
    return Path(filepath).exists()

def validate_syntax(filepath: str) -> Tuple[bool, str]:
    """Validate Python file syntax"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, "Valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def count_lines(filepath: str) -> int:
    """Count lines in file"""
    try:
        with open(filepath, 'r') as f:
            return len(f.readlines())
    except:
        return 0

def extract_classes(filepath: str) -> List[str]:
    """Extract class names from Python file"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    except:
        return []

def extract_functions(filepath: str) -> List[str]:
    """Extract function names from Python file"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except:
        return []

def has_docstring(filepath: str) -> bool:
    """Check if file has module docstring"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        return ast.get_docstring(tree) is not None
    except:
        return False

# ============================================================================
# VALIDATION TESTS
# ============================================================================

def validate_file_structure():
    """Task 1.2 - Validate file structure is correct"""
    print_header("PHASE 1: FILE STRUCTURE VALIDATION")

    required_files = {
        "Foundation Files": [
            "data_models/schemas.py",
            "data_models/__init__.py",
            "utils/logger.py",
            "utils/constants.py",
            "utils/__init__.py",
            "requirements.txt"
        ],
        "Person B Core Files": [
            "agents/retriever.py",
        ],
        "Test Files": [
            "tests/test_retriever.py",
        ]
    }

    all_passed = True

    for category, files in required_files.items():
        print(f"\n{Colors.BOLD}{category}:{Colors.END}")
        for filepath in files:
            if check_file_exists(filepath):
                lines = count_lines(filepath)
                print_success(f"{filepath:50} ({lines:4} lines)")
            else:
                print_error(f"{filepath:50} MISSING")
                all_passed = False

    return all_passed

def validate_syntax_all_files():
    """Validate Python syntax for all files"""
    print_header("PHASE 2: SYNTAX VALIDATION")

    python_files = [
        "data_models/schemas.py",
        "data_models/__init__.py",
        "utils/logger.py",
        "utils/constants.py",
        "utils/__init__.py",
        "agents/retriever.py",
        "tests/test_retriever.py"
    ]

    all_passed = True

    for filepath in python_files:
        if not check_file_exists(filepath):
            continue

        valid, message = validate_syntax(filepath)
        if valid:
            print_success(f"{filepath:50} {message}")
        else:
            print_error(f"{filepath:50} {message}")
            all_passed = False

    return all_passed

def validate_documentation():
    """Validate documentation completeness"""
    print_header("PHASE 3: DOCUMENTATION VALIDATION")

    files_to_check = [
        "data_models/schemas.py",
        "utils/logger.py",
        "utils/constants.py",
        "agents/retriever.py",
        "tests/test_retriever.py"
    ]

    all_passed = True

    for filepath in files_to_check:
        if not check_file_exists(filepath):
            continue

        if has_docstring(filepath):
            print_success(f"{filepath:50} Has module docstring")
        else:
            print_warning(f"{filepath:50} Missing module docstring")
            all_passed = False

    return all_passed

def validate_schemas():
    """Task 1.2 - Validate Pydantic schemas are defined"""
    print_header("PHASE 4: SCHEMA VALIDATION")

    required_schemas = [
        "MarketData", "MarketContext", "TriggerEvent",
        "MarketEventType", "SeverityLevel", "LifeEventType",
        "FinancialState", "RiskProfile", "TaxContext",
        "Constraint", "PlanRequest", "PlanStep", "FinancialPlan",
        "VerificationReport", "AgentMessage", "ExecutionLog",
        "ReasoningTrace", "SearchPath", "PerformanceMetrics"
    ]

    classes = extract_classes("data_models/schemas.py")

    all_passed = True

    print(f"Found {len(classes)} classes/enums in schemas.py:\n")

    for schema in required_schemas:
        if schema in classes:
            print_success(f"{schema:40}")
        else:
            print_error(f"{schema:40} MISSING")
            all_passed = False

    return all_passed

def validate_retriever_implementation():
    """Tasks 2.7-2.9 - Validate IRA implementation"""
    print_header("PHASE 5: IRA IMPLEMENTATION VALIDATION")

    required_methods = [
        "get_market_data",          # Task 2.7
        "get_market_context",       # Task 2.7
        "detect_market_triggers",   # Task 2.8
        "monitor_volatility",       # Task 2.9
        "query_financial_knowledge" # Task 2.7 RAG
    ]

    classes = extract_classes("agents/retriever.py")

    all_passed = True

    print(f"Found InformationRetrievalAgent class: ", end="")
    if "InformationRetrievalAgent" in classes:
        print_success("YES")
    else:
        print_error("NO")
        all_passed = False

    print(f"\nRequired methods:\n")

    # Check methods within the file content (more reliable for class methods)
    try:
        with open("agents/retriever.py", 'r') as f:
            content = f.read()

        for method in required_methods:
            # Look for method definition pattern
            if f"async def {method}(" in content or f"def {method}(" in content:
                print_success(f"{method:40}")
            else:
                print_error(f"{method:40} MISSING")
                all_passed = False
    except Exception as e:
        print_error(f"Could not validate methods: {e}")
        all_passed = False

    return all_passed

def validate_test_coverage():
    """Task 2.10 - Validate test coverage"""
    print_header("PHASE 6: TEST COVERAGE VALIDATION")

    required_test_classes = [
        "TestIRAInitialization",
        "TestMarketDataRetrieval",
        "TestMarketContext",
        "TestTriggerDetection",
        "TestVolatilityMonitoring",
        "TestMockDataGeneration",
        "TestPerformanceMetrics"
    ]

    classes = extract_classes("tests/test_retriever.py")

    all_passed = True

    print(f"Found {len(classes)} test classes in test_retriever.py:\n")

    for test_class in required_test_classes:
        if test_class in classes:
            print_success(f"{test_class:40}")
        else:
            print_error(f"{test_class:40} MISSING")
            all_passed = False

    return all_passed

def validate_constants():
    """Task 1.3 - Validate constants are defined"""
    print_header("PHASE 7: CONSTANTS VALIDATION")

    required_constants = [
        "VOLATILITY_THRESHOLDS",
        "MARKET_CHANGE_THRESHOLDS",
        "TRIGGER_THRESHOLDS",
        "MOCK_SCENARIOS",
        "MOCK_INDICES",
        "CACHE_TTL",
        "API_TIMEOUT_SECONDS"
    ]

    # Read constants file and check for definitions
    try:
        with open("utils/constants.py", 'r') as f:
            content = f.read()

        all_passed = True

        for const in required_constants:
            if const in content:
                print_success(f"{const:40}")
            else:
                print_error(f"{const:40} MISSING")
                all_passed = False

        return all_passed
    except:
        print_error("Could not read constants.py")
        return False

def generate_summary_report():
    """Generate final summary report"""
    print_header("VALIDATION SUMMARY REPORT")

    results = {
        "File Structure": validate_file_structure(),
        "Syntax Validation": validate_syntax_all_files(),
        "Documentation": validate_documentation(),
        "Pydantic Schemas": validate_schemas(),
        "IRA Implementation": validate_retriever_implementation(),
        "Test Coverage": validate_test_coverage(),
        "Constants Definition": validate_constants()
    }

    print(f"\n{Colors.BOLD}Overall Results:{Colors.END}\n")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for check, result in results.items():
        status = "PASSED" if result else "FAILED"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{check:30} {status:>10}{Colors.END}")

    print(f"\n{Colors.BOLD}Score: {passed}/{total} checks passed{Colors.END}")

    if all(results.values()):
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL VALIDATIONS PASSED!{Colors.END}")
        print(f"{Colors.GREEN}Person B implementation is complete and ready for testing.{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME VALIDATIONS FAILED{Colors.END}")
        print(f"{Colors.RED}Please review and fix the issues above.{Colors.END}")
        return 1

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{'FinPilot Person B Implementation Validation':^70}{Colors.END}")
    print(f"{Colors.BOLD}{'Information Retrieval Agent (IRA) - Tasks 2.6-2.10':^70}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")

    exit_code = generate_summary_report()

    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}\n")

    sys.exit(exit_code)
