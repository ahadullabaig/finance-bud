"""
Session Persistence Layer

Provides persistent storage for user sessions and agent state to survive system restarts.

Features:
- File-based session storage
- Automatic session expiration
- Session recovery after crashes
- State snapshots
- Session migration support

Requirements: Enhanced robustness for production deployment
"""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import asyncio


@dataclass
class SessionData:
    """Persistent session data"""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    state: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'state': self.state,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create SessionData from dictionary"""
        return cls(
            session_id=data['session_id'],
            user_id=data['user_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            state=data.get('state', {}),
            metadata=data.get('metadata', {})
        )

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at


class SessionPersistenceManager:
    """
    Manages persistent storage of user sessions

    Provides:
    - Automatic session persistence to disk
    - Session recovery after system crashes
    - Expired session cleanup
    - Session state snapshots
    - Multi-format storage (JSON, Pickle)
    """

    def __init__(
        self,
        storage_path: str = "./data/sessions",
        default_ttl_hours: int = 24,
        cleanup_interval_minutes: int = 60,
        storage_format: str = "json"  # "json" or "pickle"
    ):
        self.storage_path = Path(storage_path)
        self.default_ttl_hours = default_ttl_hours
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.storage_format = storage_format

        # In-memory cache
        self.sessions: Dict[str, SessionData] = {}

        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing sessions
        self._load_sessions()

        # Start cleanup task
        asyncio.create_task(self._cleanup_worker())

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session"""
        extension = ".json" if self.storage_format == "json" else ".pkl"
        return self.storage_path / f"session_{session_id}{extension}"

    def _load_sessions(self) -> None:
        """Load all sessions from disk"""
        pattern = "session_*"
        if self.storage_format == "json":
            pattern += ".json"
        else:
            pattern += ".pkl"

        for session_file in self.storage_path.glob(pattern):
            try:
                session_data = self._load_session_file(session_file)
                if session_data and not session_data.is_expired():
                    self.sessions[session_data.session_id] = session_data
                else:
                    # Remove expired session file
                    session_file.unlink()
            except Exception as e:
                print(f"Error loading session from {session_file}: {e}")

    def _load_session_file(self, file_path: Path) -> Optional[SessionData]:
        """Load a single session file"""
        try:
            if self.storage_format == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return SessionData.from_dict(data)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading session file: {e}")
            return None

    def _save_session_file(self, session_data: SessionData) -> None:
        """Save a session to disk"""
        file_path = self._get_session_file(session_data.session_id)

        try:
            if self.storage_format == "json":
                with open(file_path, 'w') as f:
                    json.dump(session_data.to_dict(), f, indent=2)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(session_data, f)
        except Exception as e:
            print(f"Error saving session: {e}")

    async def _cleanup_worker(self) -> None:
        """Background task to cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                self._cleanup_expired_sessions()
            except Exception as e:
                print(f"Error in cleanup worker: {e}")

    def _cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from memory and disk"""
        expired_count = 0
        expired_session_ids = []

        for session_id, session_data in self.sessions.items():
            if session_data.is_expired():
                expired_session_ids.append(session_id)

        for session_id in expired_session_ids:
            del self.sessions[session_id]

            # Remove file
            session_file = self._get_session_file(session_id)
            if session_file.exists():
                try:
                    session_file.unlink()
                    expired_count += 1
                except Exception as e:
                    print(f"Error deleting expired session file: {e}")

        if expired_count > 0:
            print(f"Cleaned up {expired_count} expired sessions")

        return expired_count

    def create_session(
        self,
        session_id: str,
        user_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None
    ) -> SessionData:
        """
        Create a new session

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            initial_state: Initial session state
            metadata: Session metadata
            ttl_hours: Time-to-live in hours (uses default if not specified)

        Returns:
            Created SessionData object
        """
        now = datetime.utcnow()
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours

        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_accessed=now,
            expires_at=now + timedelta(hours=ttl),
            state=initial_state or {},
            metadata=metadata or {}
        )

        self.sessions[session_id] = session_data
        self._save_session_file(session_data)

        return session_data

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get a session by ID

        Args:
            session_id: Session identifier

        Returns:
            SessionData or None if not found/expired
        """
        session_data = self.sessions.get(session_id)

        if session_data:
            if session_data.is_expired():
                self.delete_session(session_id)
                return None

            # Update last accessed
            session_data.last_accessed = datetime.utcnow()
            self._save_session_file(session_data)

        return session_data

    def update_session(
        self,
        session_id: str,
        state_updates: Optional[Dict[str, Any]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        extend_ttl_hours: Optional[int] = None
    ) -> bool:
        """
        Update session data

        Args:
            session_id: Session identifier
            state_updates: Updates to session state (merged with existing)
            metadata_updates: Updates to metadata (merged with existing)
            extend_ttl_hours: Extend expiration by this many hours

        Returns:
            True if updated successfully, False if session not found
        """
        session_data = self.sessions.get(session_id)

        if not session_data or session_data.is_expired():
            return False

        if state_updates:
            session_data.state.update(state_updates)

        if metadata_updates:
            session_data.metadata.update(metadata_updates)

        if extend_ttl_hours:
            session_data.expires_at = datetime.utcnow() + timedelta(hours=extend_ttl_hours)

        session_data.last_accessed = datetime.utcnow()
        self._save_session_file(session_data)

        return True

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        if session_id not in self.sessions:
            return False

        del self.sessions[session_id]

        # Remove file
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            try:
                session_file.unlink()
            except Exception as e:
                print(f"Error deleting session file: {e}")

        return True

    def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """
        Get all sessions for a user

        Args:
            user_id: User identifier

        Returns:
            List of SessionData objects
        """
        return [
            session for session in self.sessions.values()
            if session.user_id == user_id and not session.is_expired()
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """Get session management metrics"""
        total_sessions = len(self.sessions)
        expired_sessions = sum(1 for s in self.sessions.values() if s.is_expired())

        return {
            'total_sessions': total_sessions,
            'active_sessions': total_sessions - expired_sessions,
            'expired_sessions': expired_sessions,
            'storage_path': str(self.storage_path),
            'storage_format': self.storage_format,
            'default_ttl_hours': self.default_ttl_hours
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown and save all sessions"""
        for session_data in self.sessions.values():
            self._save_session_file(session_data)


# Global session manager instance (singleton pattern)
_session_manager: Optional[SessionPersistenceManager] = None


def get_session_manager(**kwargs) -> SessionPersistenceManager:
    """Get or create the global session manager"""
    global _session_manager

    if _session_manager is None:
        _session_manager = SessionPersistenceManager(**kwargs)

    return _session_manager
