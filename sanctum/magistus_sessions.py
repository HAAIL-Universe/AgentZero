"""
Magistus Session Manager -- persistent conversation storage.
Conversations stored as JSON files in data/magistus_sessions/.
AI-Generated | Claude (Anthropic) | AgentZero Session 188 | 2026-03-11
"""

import os
import json
import time
from datetime import datetime
from typing import Optional


SESSIONS_DIR = "Z:/AgentZero/data/magistus_sessions"


class SessionManager:
    """Manages persistent conversation sessions."""

    def __init__(self, sessions_dir: str = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)
        self.current_session_id = None
        self.messages = []
        self._load_latest()

    def _load_latest(self):
        """Load the most recent session, or create a new one."""
        files = sorted(
            [f for f in os.listdir(self.sessions_dir) if f.endswith('.json')],
            reverse=True
        )
        if files:
            self.current_session_id = files[0].replace('.json', '')
            self._load_session(self.current_session_id)
        else:
            self.new_session()

    def _load_session(self, session_id: str):
        """Load a session from disk."""
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.messages = data.get("messages", [])
            self.current_session_id = session_id
        else:
            self.messages = []

    def _save_session(self):
        """Save current session to disk."""
        if not self.current_session_id:
            return
        path = os.path.join(self.sessions_dir, f"{self.current_session_id}.json")
        data = {
            "session_id": self.current_session_id,
            "created": self.messages[0]["timestamp"] if self.messages else datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "messages": self.messages,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def new_session(self) -> str:
        """Create a new session. Returns session ID."""
        self.current_session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.messages = []
        self._save_session()
        return self.current_session_id

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> dict:
        """Add a message to the current session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            msg["metadata"] = metadata
        self.messages.append(msg)
        self._save_session()
        return msg

    def get_history(self, limit: int = 50) -> list:
        """Get recent conversation history."""
        return self.messages[-limit:]

    def get_context_window(self, max_chars: int = 4000) -> list:
        """Get messages that fit within a character budget (for prompt construction)."""
        result = []
        total = 0
        for msg in reversed(self.messages):
            content_len = len(msg.get("content", ""))
            if total + content_len > max_chars and result:
                break
            result.insert(0, msg)
            total += content_len
        return result

    def list_sessions(self) -> list:
        """List all saved sessions."""
        files = sorted(
            [f for f in os.listdir(self.sessions_dir) if f.endswith('.json')],
            reverse=True
        )
        sessions = []
        for f in files:
            path = os.path.join(self.sessions_dir, f)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                sessions.append({
                    "id": f.replace('.json', ''),
                    "created": data.get("created", ""),
                    "message_count": data.get("message_count", 0),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return sessions

    def clear_session(self) -> str:
        """Clear current session and start fresh."""
        return self.new_session()


# Singleton
_manager = None


def get_session_manager() -> SessionManager:
    """Get or create the singleton session manager."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
