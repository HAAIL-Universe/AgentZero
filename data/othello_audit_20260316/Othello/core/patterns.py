"""
core/patterns.py

Behavioral pattern recognition engine for FELLO.
Scans event logs to identify recurring user patterns, avoidance triggers, peak performance windows, and “emerging” behaviors.
Designed to be called from behavioralist.py and other modules.

Author: Julius
"""

import datetime
from core.shadow_linker import ShadowLinker

def find_avoidance_patterns(events, min_repeats=3):
    patterns = []
    ignore_counts = {}
    for event in events:
        if event.get("event_type") == "nudge_ignored":
            key = event["detail"].get("target_habit", "unknown")
            ignore_counts[key] = ignore_counts.get(key, 0) + 1
            if ignore_counts[key] == min_repeats:
                patterns.append({
                    "pattern": "avoidance",
                    "target": key,
                    "count": min_repeats,
                    "example": event
                })
    return patterns

def find_peak_windows(events, min_count=5):
    hour_counts = {}
    for event in events:
        if event.get("event_type") == "task_completed":
            timestamp = event.get("timestamp")
            if timestamp:
                hour = datetime.datetime.fromisoformat(timestamp).hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
    peaks = []
    for hour, count in hour_counts.items():
        if count >= min_count:
            peaks.append({
                "pattern": "peak_window",
                "hour": hour,
                "count": count
            })
    return peaks

def find_trigger_chains(events, trigger="denied_goal", result="mood_drop", window_minutes=30):
    chains = []
    for i, event in enumerate(events):
        if event.get("event_type") == trigger:
            trigger_time = datetime.datetime.fromisoformat(event["timestamp"])
            for next_event in events[i+1:]:
                next_time = datetime.datetime.fromisoformat(next_event["timestamp"])
                if (next_time - trigger_time).total_seconds() / 60 > window_minutes:
                    break
                if next_event.get("event_type") == result:
                    chains.append({
                        "pattern": "trigger_chain",
                        "trigger": trigger,
                        "result": result,
                        "window_minutes": window_minutes,
                        "trigger_event": event,
                        "result_event": next_event
                    })
    return chains

def find_emerging_patterns(events, min_novelty=2):
    return []

def summarize_patterns(events):
    summary = {
        "avoidance": find_avoidance_patterns(events),
        "peaks": find_peak_windows(events),
        "trigger_chains": find_trigger_chains(events),
        "emerging": find_emerging_patterns(events)
    }

    linker = ShadowLinker()
    linker.safe_update({"micro_patterns": {"pattern_summary": summary}}, consent_level="auto")

    return summary

if __name__ == "__main__":
    import json
    with open("data/behavior_events.json", "r") as f:
        events = json.load(f)
    summary = summarize_patterns(events)
    print(summary)
