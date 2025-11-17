"""
Supabase Configuration and Client Setup for FinPilot VP-MAS

Provides centralized Supabase client configuration, connection management,
and database utilities for the multi-agent financial planning system.
"""

import os
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from datetime import datetime
import asyncio
from dataclasses import dataclass


@dataclass
class SupabaseConfig:
    """Supabase configuration settings"""
    url: str
    anon_key: str
    service_key: str
    timeout: int = 30
    auto_refresh_token: bool = True
    persist_session: bool = True


class SupabaseClient:
    """
    Enhanced Supabase client with connection pooling, error handling,
    and specialized methods for financial data operations.
    """
    
    def __init__(self, config: SupabaseConfig):
        self.config = config
        self._client: Optional[Client] = None
        self._service_client: Optional[Client] = None
        
    @property
    def client(self) -> Client:
        """Get the standard Supabase client"""
        if not self._client:
            self._client = create_client(
                self.config.url,
                self.config.anon_key
            )
        return self._client
    
    @property
    def service_client(self) -> Client:
        """Get the service role client for admin operations"""
        if not self._service_client:
            self._service_client = create_client(
                self.config.url,
                self.config.service_key
            )
        return self._service_client
    
    async def health_check(self) -> bool:
        """Check if Supabase connection is healthy"""
        try:
            result = self.client.table('health_check').select('*').limit(1).execute()
            return True
        except Exception:
            return False
    
    async def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user with Supabase Auth"""
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return {
                "success": True,
                "user": response.user,
                "session": response.session
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_user(self, email: str, password: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create new user account"""
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": metadata or {}
                }
            })
            return {
                "success": True,
                "user": response.user,
                "session": response.session
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def get_supabase_config() -> SupabaseConfig:
    """Get Supabase configuration from environment variables"""
    return SupabaseConfig(
        url=os.getenv("SUPABASE_URL", ""),
        anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
        service_key=os.getenv("SUPABASE_SERVICE_KEY", ""),
        timeout=int(os.getenv("SUPABASE_TIMEOUT", "30")),
        auto_refresh_token=os.getenv("SUPABASE_AUTO_REFRESH", "true").lower() == "true",
        persist_session=os.getenv("SUPABASE_PERSIST_SESSION", "true").lower() == "true"
    )


# Global Supabase client instance
_supabase_client: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get the global Supabase client instance"""
    global _supabase_client
    if not _supabase_client:
        config = get_supabase_config()
        _supabase_client = SupabaseClient(config)
    return _supabase_client