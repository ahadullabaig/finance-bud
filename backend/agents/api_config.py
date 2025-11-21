"""
API Configuration Management for FinPilot VP-MAS

Handles secure API key management, configuration loading, and key rotation
for external market data providers.

Requirements: 5.1, 5.2, 4.4, 12.1
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
import base64

from utils.logger import LoggingStandards

logger = LoggingStandards.create_system_logger("api_config")


@dataclass
class APIKeyConfig:
    """Configuration for API key management"""
    primary_key: str
    backup_key: Optional[str] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    rotation_interval_days: int = 90
    usage_count: int = 0
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ProviderConfig:
    """Configuration for external API provider"""
    name: str
    base_url: str
    api_keys: APIKeyConfig
    rate_limits: Dict[str, int]
    timeout: float = 30.0
    max_retries: int = 3
    health_check_endpoint: str = "/health"
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True


class SecureKeyManager:
    """Secure API key management with encryption and rotation"""
    
    def __init__(self, key_file: str = ".api_keys.enc"):
        self.key_file = Path(key_file)
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for API keys"""
        key_env = os.getenv("FINPILOT_ENCRYPTION_KEY")
        if key_env:
            return base64.urlsafe_b64decode(key_env.encode())
        
        # Generate new key
        key = Fernet.generate_key()
        logger.warning(
            "Generated new encryption key. Set FINPILOT_ENCRYPTION_KEY environment variable:\n"
            f"export FINPILOT_ENCRYPTION_KEY={base64.urlsafe_b64encode(key).decode()}"
        )
        return key
    
    def encrypt_key(self, api_key: str) -> str:
        """Encrypt API key"""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt API key"""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def save_keys(self, keys: Dict[str, str]):
        """Save encrypted API keys to file"""
        encrypted_keys = {
            provider: self.encrypt_key(key)
            for provider, key in keys.items()
        }
        
        with open(self.key_file, 'w') as f:
            json.dump(encrypted_keys, f, indent=2)
        
        # Set restrictive permissions
        os.chmod(self.key_file, 0o600)
        logger.info(f"Saved encrypted API keys to {self.key_file}")
    
    def load_keys(self) -> Dict[str, str]:
        """Load and decrypt API keys from file"""
        if not self.key_file.exists():
            logger.warning(f"API key file {self.key_file} not found")
            return {}
        
        try:
            with open(self.key_file, 'r') as f:
                encrypted_keys = json.load(f)
            
            return {
                provider: self.decrypt_key(encrypted_key)
                for provider, encrypted_key in encrypted_keys.items()
            }
        
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}


class APIConfigManager:
    """Manages API configuration for all external providers"""
    
    def __init__(self, config_file: str = "api_config.json"):
        self.config_file = Path(config_file)
        self.key_manager = SecureKeyManager()
        self.providers: Dict[str, ProviderConfig] = {}
        self.load_configuration()
    
    def load_configuration(self):
        """Load API configuration from file and environment"""
        # Load from config file
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self._parse_config_data(config_data)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        # Load API keys
        api_keys = self.key_manager.load_keys()
        
        # Override with environment variables
        self._load_from_environment(api_keys)
        
        # Set default configurations if not loaded
        self._set_default_configurations()
    
    def _parse_config_data(self, config_data: Dict[str, Any]):
        """Parse configuration data from file"""
        for provider_name, provider_data in config_data.get("providers", {}).items():
            try:
                # Parse API keys
                key_data = provider_data.get("api_keys", {})
                api_keys = APIKeyConfig(
                    primary_key=key_data.get("primary_key", ""),
                    backup_key=key_data.get("backup_key"),
                    created_at=datetime.fromisoformat(key_data.get("created_at", datetime.utcnow().isoformat())),
                    expires_at=datetime.fromisoformat(key_data["expires_at"]) if key_data.get("expires_at") else None,
                    rotation_interval_days=key_data.get("rotation_interval_days", 90),
                    usage_count=key_data.get("usage_count", 0),
                    last_used=datetime.fromisoformat(key_data["last_used"]) if key_data.get("last_used") else None,
                    is_active=key_data.get("is_active", True)
                )
                
                # Create provider config
                provider_config = ProviderConfig(
                    name=provider_name,
                    base_url=provider_data["base_url"],
                    api_keys=api_keys,
                    rate_limits=provider_data.get("rate_limits", {}),
                    timeout=provider_data.get("timeout", 30.0),
                    max_retries=provider_data.get("max_retries", 3),
                    health_check_endpoint=provider_data.get("health_check_endpoint", "/health"),
                    priority=provider_data.get("priority", 1),
                    enabled=provider_data.get("enabled", True)
                )
                
                self.providers[provider_name] = provider_config
                
            except Exception as e:
                logger.error(f"Failed to parse config for provider {provider_name}: {e}")
    
    def _load_from_environment(self, stored_keys: Dict[str, str]):
        """Load configuration from environment variables"""
        env_mappings = {
            "barchart": {
                "key_env": "BARCHART_API_KEY",
                "backup_key_env": "BARCHART_BACKUP_KEY",
                "base_url": "https://api.barchart.com"
            },
            "alpha_vantage": {
                "key_env": "ALPHA_VANTAGE_API_KEY",
                "backup_key_env": "ALPHA_VANTAGE_BACKUP_KEY",
                "base_url": "https://www.alphavantage.co"
            },
            "massive": {
                "key_env": "MASSIVE_API_KEY",
                "backup_key_env": "MASSIVE_BACKUP_KEY",
                "base_url": "https://api.massive.com"
            }
        }
        
        for provider_name, env_config in env_mappings.items():
            primary_key = os.getenv(env_config["key_env"]) or stored_keys.get(f"{provider_name}_primary")
            backup_key = os.getenv(env_config["backup_key_env"]) or stored_keys.get(f"{provider_name}_backup")
            
            if primary_key:
                if provider_name not in self.providers:
                    # Create new provider config
                    api_keys = APIKeyConfig(
                        primary_key=primary_key,
                        backup_key=backup_key
                    )
                    
                    self.providers[provider_name] = ProviderConfig(
                        name=provider_name,
                        base_url=env_config["base_url"],
                        api_keys=api_keys,
                        rate_limits=self._get_default_rate_limits(provider_name),
                        priority=self._get_default_priority(provider_name)
                    )
                else:
                    # Update existing config
                    self.providers[provider_name].api_keys.primary_key = primary_key
                    if backup_key:
                        self.providers[provider_name].api_keys.backup_key = backup_key
    
    def _set_default_configurations(self):
        """Set default configurations for providers"""
        defaults = {
            "barchart": {
                "base_url": "https://api.barchart.com",
                "rate_limits": {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000,
                    "requests_per_day": 10000
                },
                "priority": 1
            },
            "alpha_vantage": {
                "base_url": "https://www.alphavantage.co",
                "rate_limits": {
                    "requests_per_minute": 5,
                    "requests_per_hour": 500,
                    "requests_per_day": 500
                },
                "priority": 2
            },
            "massive": {
                "base_url": "https://api.massive.com",
                "rate_limits": {
                    "requests_per_minute": 100,
                    "requests_per_hour": 2000,
                    "requests_per_day": 20000
                },
                "priority": 3
            }
        }
        
        for provider_name, default_config in defaults.items():
            if provider_name not in self.providers:
                # Create placeholder config (will need API key to be functional)
                api_keys = APIKeyConfig(primary_key="")
                
                self.providers[provider_name] = ProviderConfig(
                    name=provider_name,
                    base_url=default_config["base_url"],
                    api_keys=api_keys,
                    rate_limits=default_config["rate_limits"],
                    priority=default_config["priority"],
                    enabled=False  # Disabled until API key is provided
                )
    
    def _get_default_rate_limits(self, provider_name: str) -> Dict[str, int]:
        """Get default rate limits for provider"""
        defaults = {
            "barchart": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 10000
            },
            "alpha_vantage": {
                "requests_per_minute": 5,
                "requests_per_hour": 500,
                "requests_per_day": 500
            },
            "massive": {
                "requests_per_minute": 100,
                "requests_per_hour": 2000,
                "requests_per_day": 20000
            }
        }
        return defaults.get(provider_name, {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000
        })
    
    def _get_default_priority(self, provider_name: str) -> int:
        """Get default priority for provider"""
        priorities = {
            "barchart": 1,
            "alpha_vantage": 2,
            "massive": 3
        }
        return priorities.get(provider_name, 5)
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for specific provider"""
        return self.providers.get(provider_name)
    
    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        """Get all enabled providers sorted by priority"""
        enabled = {
            name: config for name, config in self.providers.items()
            if config.enabled and config.api_keys.primary_key and config.api_keys.is_active
        }
        
        return dict(sorted(enabled.items(), key=lambda x: x[1].priority))
    
    def update_api_key(self, provider_name: str, new_key: str, is_backup: bool = False):
        """Update API key for provider"""
        if provider_name not in self.providers:
            logger.error(f"Provider {provider_name} not found")
            return False
        
        provider = self.providers[provider_name]
        
        if is_backup:
            provider.api_keys.backup_key = new_key
        else:
            # Rotate keys: current primary becomes backup
            if provider.api_keys.primary_key:
                provider.api_keys.backup_key = provider.api_keys.primary_key
            
            provider.api_keys.primary_key = new_key
            provider.api_keys.created_at = datetime.utcnow()
            provider.api_keys.usage_count = 0
            provider.api_keys.last_used = None
        
        # Save updated keys
        self.save_configuration()
        logger.info(f"Updated {'backup' if is_backup else 'primary'} API key for {provider_name}")
        return True
    
    def rotate_api_key(self, provider_name: str) -> bool:
        """Rotate API key if rotation is due"""
        if provider_name not in self.providers:
            return False
        
        provider = self.providers[provider_name]
        
        # Check if rotation is needed
        if provider.api_keys.created_at:
            days_since_creation = (datetime.utcnow() - provider.api_keys.created_at).days
            if days_since_creation >= provider.api_keys.rotation_interval_days:
                logger.warning(f"API key rotation due for {provider_name} (created {days_since_creation} days ago)")
                return True
        
        return False
    
    def record_api_usage(self, provider_name: str):
        """Record API key usage"""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.api_keys.usage_count += 1
            provider.api_keys.last_used = datetime.utcnow()
    
    def disable_provider(self, provider_name: str, reason: str = ""):
        """Disable provider due to errors or other issues"""
        if provider_name in self.providers:
            self.providers[provider_name].enabled = False
            logger.warning(f"Disabled provider {provider_name}: {reason}")
    
    def enable_provider(self, provider_name: str):
        """Enable provider"""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            if provider.api_keys.primary_key:
                provider.enabled = True
                logger.info(f"Enabled provider {provider_name}")
            else:
                logger.error(f"Cannot enable provider {provider_name}: no API key configured")
    
    def save_configuration(self):
        """Save current configuration to file"""
        config_data = {
            "providers": {},
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Prepare API keys for secure storage
        keys_to_store = {}
        
        for provider_name, provider in self.providers.items():
            # Store configuration (without sensitive keys)
            provider_data = {
                "base_url": provider.base_url,
                "rate_limits": provider.rate_limits,
                "timeout": provider.timeout,
                "max_retries": provider.max_retries,
                "health_check_endpoint": provider.health_check_endpoint,
                "priority": provider.priority,
                "enabled": provider.enabled,
                "api_keys": {
                    "created_at": provider.api_keys.created_at.isoformat() if provider.api_keys.created_at else None,
                    "expires_at": provider.api_keys.expires_at.isoformat() if provider.api_keys.expires_at else None,
                    "rotation_interval_days": provider.api_keys.rotation_interval_days,
                    "usage_count": provider.api_keys.usage_count,
                    "last_used": provider.api_keys.last_used.isoformat() if provider.api_keys.last_used else None,
                    "is_active": provider.api_keys.is_active
                }
            }
            
            config_data["providers"][provider_name] = provider_data
            
            # Store keys separately for encryption
            if provider.api_keys.primary_key:
                keys_to_store[f"{provider_name}_primary"] = provider.api_keys.primary_key
            if provider.api_keys.backup_key:
                keys_to_store[f"{provider_name}_backup"] = provider.api_keys.backup_key
        
        # Save configuration file
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Save encrypted keys
        if keys_to_store:
            self.key_manager.save_keys(keys_to_store)
        
        logger.info(f"Saved API configuration to {self.config_file}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        summary = {
            "total_providers": len(self.providers),
            "enabled_providers": len(self.get_enabled_providers()),
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            summary["providers"][name] = {
                "enabled": provider.enabled,
                "has_primary_key": bool(provider.api_keys.primary_key),
                "has_backup_key": bool(provider.api_keys.backup_key),
                "priority": provider.priority,
                "usage_count": provider.api_keys.usage_count,
                "last_used": provider.api_keys.last_used.isoformat() if provider.api_keys.last_used else None,
                "rotation_due": self.rotate_api_key(name)
            }
        
        return summary


# Global configuration manager instance
config_manager = APIConfigManager()


def get_api_config() -> APIConfigManager:
    """Get the global API configuration manager"""
    return config_manager


def setup_api_keys(
    barchart_key: Optional[str] = None,
    alpha_vantage_key: Optional[str] = None,
    massive_key: Optional[str] = None,
    save_config: bool = True
) -> APIConfigManager:
    """Setup API keys for external providers"""
    
    if barchart_key:
        config_manager.update_api_key("barchart", barchart_key)
        config_manager.enable_provider("barchart")
    
    if alpha_vantage_key:
        config_manager.update_api_key("alpha_vantage", alpha_vantage_key)
        config_manager.enable_provider("alpha_vantage")
    
    if massive_key:
        config_manager.update_api_key("massive", massive_key)
        config_manager.enable_provider("massive")
    
    if save_config:
        config_manager.save_configuration()
    
    return config_manager