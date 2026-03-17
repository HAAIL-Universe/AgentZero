"""
core/energy_feed.py

Centralizes energy/mood/activity data pulls for FELLO/Othello.
Handles user logs, app activity, and plug-in for external APIs/devices (WOOP, Oura, etc).
Standardizes all input into a common “energy event” format.

Author: Julius
"""

import datetime
import json
import os

ENERGY_LOG_PATH = "data/energy_triggers.json"

# --- Standard Event Format ---
def make_energy_event(source, level, confidence=1.0, detail=None):
    """
    Standardizes an energy event.
    :param source: str ('user', 'woop', 'oura', 'behavior', etc)
    :param level: str/int (e.g. 'high', 'medium', 'low', or 0-100)
    :param confidence: float 0-1
    :param detail: dict, optional extra info (raw payload, HRV, etc)
    :return: dict
    """
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": source,
        "level": level,
        "confidence": confidence,
        "detail": detail or {}
    }

# --- Data Pulls / API Parsers ---

def import_user_log(energy_str, confidence=1.0):
    """
    Direct user input ("I feel wiped", "Crushing it today").
    :param energy_str: str
    :return: event dict
    """
    # Simple NLP mapping (expandable)
    mapping = {
        "tired": "low",
        "wiped": "low",
        "exhausted": "low",
        "burnt": "low",
        "okay": "medium",
        "average": "medium",
        "alright": "medium",
        "good": "high",
        "crushing": "high",
        "fired up": "high"
    }
    for k, v in mapping.items():
        if k in energy_str.lower():
            return make_energy_event("user", v, confidence, {"raw": energy_str})
    return make_energy_event("user", "unknown", confidence, {"raw": energy_str})

def import_woop_data(woop_json):
    """
    Stub for WOOP integration.
    Takes WOOP's exported payload, maps to standard event.
    """
    # TODO: Fill out mapping based on WOOP's API when ready.
    # Example:
    # energy = woop_json.get("recovery_score")
    # level = "high" if energy > 80 else "medium" if energy > 50 else "low"
    # confidence = woop_json.get("score_confidence", 0.7)
    # return make_energy_event("woop", level, confidence, {"raw": woop_json})
    return make_energy_event("woop", "unknown", 0.5, {"raw": woop_json})

def import_behavioral_data(events):
    """
    Maps usage patterns to energy.
    :param events: list of FELLO events (tasks completed, nudges ignored, etc)
    :return: event dict
    """
    completed = sum(1 for e in events if e.get("event_type") in ["task_completed", "nudge_accepted"])
    ignored = sum(1 for e in events if e.get("event_type") in ["nudge_ignored", "task_skipped", "denied_goal"])
    if completed + ignored == 0:
        level = "unknown"
        confidence = 0.3
    elif completed > ignored * 2:
        level = "high"
        confidence = 0.8
    elif ignored > completed * 2:
        level = "low"
        confidence = 0.8
    else:
        level = "medium"
        confidence = 0.7
    return make_energy_event("behavior", level, confidence, {"completed": completed, "ignored": ignored})

# --- Event Logging ---

def log_energy_event(event):
    """
    Appends energy event to history in data/energy_triggers.json.
    """
    if not os.path.exists(ENERGY_LOG_PATH):
        events = []
    else:
        with open(ENERGY_LOG_PATH, "r") as f:
            try:
                events = json.load(f)
            except Exception:
                events = []
    events.append(event)
    with open(ENERGY_LOG_PATH, "w") as f:
        json.dump(events, f, indent=2)

# --- Example Usage / Testing ---

if __name__ == "__main__":
    # User log
    e1 = import_user_log("I'm absolutely crushed, barely slept.")
    log_energy_event(e1)
    # WOOP log (stubbed)
    e2 = import_woop_data({"recovery_score": 30})
    log_energy_event(e2)
    # Behavioral (simulate)
    fake_events = [
        {"event_type": "task_completed"},
        {"event_type": "nudge_ignored"},
        {"event_type": "task_completed"},
        {"event_type": "task_completed"},
    ]
    e3 = import_behavioral_data(fake_events)
    log_energy_event(e3)
    print("Sample energy events written to data/energy_triggers.json")
