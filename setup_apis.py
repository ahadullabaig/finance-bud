#!/usr/bin/env python3
"""
API Setup Script for FinPilot VP-MAS

This script helps configure external API connections for the FinPilot system.
It handles secure key storage, configuration validation, and testing.

Usage:
    python setup_apis.py --interactive
    python setup_apis.py --barchart-key YOUR_KEY --alpha-vantage-key YOUR_KEY
    python setup_apis.py --test-connections
    python setup_apis.py --mock-mode

Requirements: 5.1, 5.2, 4.4, 12.1
"""

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

from agents.api_config import setup_api_keys, get_api_config
from agents.external_apis import create_api_manager
from utils.logger import LoggingStandards

logger = LoggingStandards.create_system_logger("setup_apis")


def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("FinPilot VP-MAS External API Setup")
    print("=" * 60)
    print()


def interactive_setup():
    """Interactive API key setup"""
    print("Interactive API Key Setup")
    print("-" * 30)
    print()
    
    # Collect API keys
    api_keys = {}
    
    print("Enter your API keys (press Enter to skip):")
    print()
    
    # Barchart API
    barchart_key = getpass.getpass("Barchart API Key: ").strip()
    if barchart_key:
        api_keys['barchart'] = barchart_key
    
    # Alpha Vantage API
    alpha_vantage_key = getpass.getpass("Alpha Vantage API Key: ").strip()
    if alpha_vantage_key:
        api_keys['alpha_vantage'] = alpha_vantage_key
    
    # Massive API
    massive_key = getpass.getpass("Massive API Key: ").strip()
    if massive_key:
        api_keys['massive'] = massive_key
    
    if not api_keys:
        print("No API keys provided. Setting up mock mode only.")
        return setup_mock_mode()
    
    # Setup keys
    print("\nSetting up API configuration...")
    config_manager = setup_api_keys(
        barchart_key=api_keys.get('barchart'),
        alpha_vantage_key=api_keys.get('alpha_vantage'),
        massive_key=api_keys.get('massive'),
        save_config=True
    )
    
    print("✓ API keys configured and saved securely")
    
    # Show configuration summary
    show_configuration_summary(config_manager)
    
    # Ask if user wants to test connections
    test_connections = input("\nTest API connections? (y/N): ").strip().lower()
    if test_connections in ['y', 'yes']:
        asyncio.run(test_api_connections())
    
    return True


def setup_mock_mode():
    """Setup mock mode for offline development"""
    print("Setting up mock mode for offline development...")
    
    config_manager = get_api_config()
    
    # Ensure mock mode is available
    print("✓ Mock mode configured")
    print("✓ You can now use the system offline for development and testing")
    
    return True


def show_configuration_summary(config_manager=None):
    """Show current configuration summary"""
    if config_manager is None:
        config_manager = get_api_config()
    
    summary = config_manager.get_configuration_summary()
    
    print("\nConfiguration Summary:")
    print("-" * 30)
    print(f"Total providers: {summary['total_providers']}")
    print(f"Enabled providers: {summary['enabled_providers']}")
    print()
    
    for provider_name, provider_info in summary['providers'].items():
        status = "✓ Enabled" if provider_info['enabled'] else "✗ Disabled"
        key_status = "✓" if provider_info['has_primary_key'] else "✗"
        backup_status = "✓" if provider_info['has_backup_key'] else "✗"
        
        print(f"{provider_name.upper()}:")
        print(f"  Status: {status}")
        print(f"  Primary Key: {key_status}")
        print(f"  Backup Key: {backup_status}")
        print(f"  Priority: {provider_info['priority']}")
        print(f"  Usage Count: {provider_info['usage_count']}")
        
        if provider_info['rotation_due']:
            print("  ⚠️  Key rotation recommended")
        
        print()


async def test_api_connections():
    """Test API connections"""
    print("Testing API connections...")
    print("-" * 30)
    
    try:
        # Create API manager with real keys
        api_manager = await create_api_manager(mock_mode=False)
        
        async with api_manager.managed_connections():
            # Test health checks
            print("Performing health checks...")
            health_results = await api_manager.health_check_all()
            
            for provider, is_healthy in health_results.items():
                status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
                print(f"  {provider.value}: {status}")
            
            print()
            
            # Test market data retrieval
            print("Testing market data retrieval...")
            response = await api_manager.fetch_market_data(["AAPL", "GOOGL"])
            
            if response.success:
                print(f"✓ Market data retrieved successfully from {response.provider.value}")
                print(f"  Response time: {response.response_time:.3f}s")
                if response.from_cache:
                    print("  Data served from cache")
            else:
                print(f"✗ Market data retrieval failed: {response.error}")
            
            print()
            
            # Test volatility data
            print("Testing volatility data...")
            volatility_response = await api_manager.get_market_volatility()
            
            if volatility_response.success:
                print(f"✓ Volatility data retrieved from {volatility_response.provider.value}")
            else:
                print(f"✗ Volatility data failed: {volatility_response.error}")
            
            print()
            
            # Test economic indicators
            print("Testing economic indicators...")
            econ_response = await api_manager.get_economic_indicators()
            
            if econ_response.success:
                print(f"✓ Economic indicators retrieved from {econ_response.provider.value}")
            else:
                print(f"✗ Economic indicators failed: {econ_response.error}")
    
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False
    
    print("\nConnection tests completed!")
    return True


async def test_mock_mode():
    """Test mock mode functionality"""
    print("Testing mock mode...")
    print("-" * 30)
    
    try:
        # Create API manager in mock mode
        api_manager = await create_api_manager(mock_mode=True)
        
        async with api_manager.managed_connections():
            # Test market data
            print("Testing mock market data...")
            response = await api_manager.fetch_market_data(["AAPL", "GOOGL", "MSFT"])
            
            if response.success:
                print("✓ Mock market data retrieved successfully")
                print(f"  Symbols: {list(response.data['quotes'].keys())}")
                print(f"  Response time: {response.response_time:.3f}s")
            else:
                print(f"✗ Mock market data failed: {response.error}")
            
            # Test volatility
            print("Testing mock volatility data...")
            volatility_response = await api_manager.get_market_volatility()
            
            if volatility_response.success:
                print("✓ Mock volatility data retrieved successfully")
            else:
                print(f"✗ Mock volatility data failed: {volatility_response.error}")
            
            # Test economic indicators
            print("Testing mock economic indicators...")
            econ_response = await api_manager.get_economic_indicators()
            
            if econ_response.success:
                print("✓ Mock economic indicators retrieved successfully")
                indicators = list(econ_response.data.keys())
                print(f"  Indicators: {indicators}")
            else:
                print(f"✗ Mock economic indicators failed: {econ_response.error}")
    
    except Exception as e:
        print(f"✗ Mock mode test failed: {e}")
        return False
    
    print("\nMock mode tests completed!")
    return True


def validate_environment():
    """Validate environment setup"""
    print("Validating environment...")
    print("-" * 30)
    
    # Check required directories
    required_dirs = ['agents', 'data_models', 'utils', 'tests']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"✗ Missing directory: {dir_name}")
            return False
        else:
            print(f"✓ Directory exists: {dir_name}")
    
    # Check required files
    required_files = [
        'agents/external_apis.py',
        'agents/api_config.py',
        'data_models/schemas.py',
        'utils/logger.py'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"✗ Missing file: {file_path}")
            return False
        else:
            print(f"✓ File exists: {file_path}")
    
    print("\n✓ Environment validation passed!")
    return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup external API connections for FinPilot VP-MAS"
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive setup mode'
    )
    
    parser.add_argument(
        '--barchart-key',
        help='Barchart API key'
    )
    
    parser.add_argument(
        '--alpha-vantage-key',
        help='Alpha Vantage API key'
    )
    
    parser.add_argument(
        '--massive-key',
        help='Massive API key'
    )
    
    parser.add_argument(
        '--test-connections',
        action='store_true',
        help='Test API connections'
    )
    
    parser.add_argument(
        '--test-mock',
        action='store_true',
        help='Test mock mode'
    )
    
    parser.add_argument(
        '--mock-mode',
        action='store_true',
        help='Setup mock mode only'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show current configuration'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate environment setup'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Validate environment first
    if not validate_environment():
        print("Environment validation failed. Please check your installation.")
        sys.exit(1)
    
    try:
        if args.interactive:
            interactive_setup()
        
        elif args.mock_mode:
            setup_mock_mode()
        
        elif args.show_config:
            show_configuration_summary()
        
        elif args.test_connections:
            success = asyncio.run(test_api_connections())
            if not success:
                sys.exit(1)
        
        elif args.test_mock:
            success = asyncio.run(test_mock_mode())
            if not success:
                sys.exit(1)
        
        elif args.barchart_key or args.alpha_vantage_key or args.massive_key:
            # Command line key setup
            config_manager = setup_api_keys(
                barchart_key=args.barchart_key,
                alpha_vantage_key=args.alpha_vantage_key,
                massive_key=args.massive_key,
                save_config=True
            )
            
            print("✓ API keys configured successfully")
            show_configuration_summary(config_manager)
        
        elif args.validate:
            print("✓ Environment validation completed")
        
        else:
            # Default: show help and current config
            parser.print_help()
            print()
            show_configuration_summary()
    
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nSetup failed: {e}")
        logger.exception("Setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()