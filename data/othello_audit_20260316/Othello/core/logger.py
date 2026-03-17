"""
core/logger.py

Central logging module for FELLO. Handles behavioral event logging, system messages, and future extensible logging types.
All logs saved as JSON. Keeps everything traceable and auditable.

Author: Julius
"""

import json
import datetime
import os
from core.shadow_manager import ShadowManager

# --- DEFAULT PATHS (can be overridden in use) ---
DEFAULT_EVENT_LOG = "data/behavior_events.json"
DEFAULT_SYSTEM_LOG = "data/system_log.json"

class FelloLogger:
    def __init__(self, event_log_path=DEFAULT_EVENT_LOG, system_log_path=DEFAULT_SYSTEM_LOG):
        self.event_log_path = event_log_path
        self.system_log_path = system_log_path
        self.shadow = ShadowManager()

        # Make sure storage paths exist
        os.makedirs(os.path.dirname(event_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(system_log_path), exist_ok=True)

    def _load_json(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def log_event(self, event_type, detail, context=None):
        """
        Log a behavioral event: nudge, task, result, impatience_trigger, etc.
        """
        if event_type == "nudge_sent":
            try:
                from core.self_reflection import SelfReflectionEngine
                sref = SelfReflectionEngine()
                sref.log_intervention(
                    nudge_type=detail.get("msg", "unknown"),
                    reason=detail.get("reason", "unspecified"),
                    context=context or {}
                )
            except Exception as e:
                self.log_system("WARN", f"Self-reflection logging failed: {e}")

        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "detail": detail,
            "context": context or {},
        }
        events = self._load_json(self.event_log_path)
        events.append(event)
        self._save_json(self.event_log_path, events)

        # Update shadow
        self.shadow.safe_update({"decisions": {"logged_event": event}}, consent_level="auto")

    def log_system(self, level, message):
        """
        Log a system event (startup, errors, warnings, impatience notes, etc.)
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
        }
        logs = self._load_json(self.system_log_path)
        logs.append(entry)
        self._save_json(self.system_log_path, logs)

        # Update shadow
        self.shadow.safe_update({"decisions": {"system_log": entry}}, consent_level="auto")

    def get_recent_events(self, limit=50):
        events = self._load_json(self.event_log_path)
        return events[-limit:]

    def get_recent_system_logs(self, limit=50):
        logs = self._load_json(self.system_log_path)
        return logs[-limit:]

    def log_impatience_trigger(self, level, input_text):
        """
        Specific helper for logging impatience detection triggers.
        """
        self.log_event("impatience_triggered", {"level": level, "input": input_text})
        self.log_system("INFO", f"Impatience trigger logged: {level}")

if __name__ == "__main__":
    logger = FelloLogger()
    logger.log_event("nudge_sent", {"msg": "Take a walk"}, {"mood": "neutral"})
    logger.log_system("INFO", "Logger initialized.")
    logger.log_impatience_trigger("high", "Do it now!")
    print(logger.get_recent_events())
    print(logger.get_recent_system_logs())
