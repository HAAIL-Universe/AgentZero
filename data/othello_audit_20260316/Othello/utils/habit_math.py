"""
utils/habit_math.py

Math, stats, and helper functions for habit tracking in FELLO.
Handles habit formation, decay rates, streaks, snap-backs, and predictive analytics.
All functions are stateless—just crunch the data you give them.

Author: Julius
"""

import math
import datetime
from core.shadow_manager import ShadowManager

def habit_half_life(events, habit, window_days=30):
    now = datetime.datetime.now()
    recent = [
        e for e in events
        if e.get("detail", {}).get("target_habit") == habit
        and (now - datetime.datetime.fromisoformat(e["timestamp"])).days <= window_days
    ]
    if not recent:
        return 0

    completed = sum(1 for e in recent if e["event_type"] == "task_completed")
    missed = sum(1 for e in recent if e["event_type"] in ["nudge_ignored", "task_skipped"])
    total = completed + missed
    if total == 0:
        return 0

    decay_rate = missed / total if total > 0 else 0.5
    if decay_rate == 0:
        return float("inf")
    try:
        half_life = math.log(0.5) / math.log(1 - decay_rate)
        return round(half_life, 2)
    except (ValueError, ZeroDivisionError):
        return 0

def current_streak(events, habit):
    streak = 0
    for event in reversed(events):
        if event.get("detail", {}).get("target_habit") != habit:
            continue
        if event["event_type"] == "task_completed":
            streak += 1
        else:
            break
    return streak

def longest_streak(events, habit):
    max_streak = 0
    streak = 0
    for event in events:
        if event.get("detail", {}).get("target_habit") != habit:
            continue
        if event["event_type"] == "task_completed":
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return max_streak

def habit_snapback(events, habit):
    last_break = None
    resume_times = []
    for event in events:
        if event.get("detail", {}).get("target_habit") != habit:
            continue
        if event["event_type"] in ["nudge_ignored", "task_skipped"]:
            last_break = event["timestamp"]
        elif event["event_type"] == "task_completed" and last_break:
            d1 = datetime.datetime.fromisoformat(last_break)
            d2 = datetime.datetime.fromisoformat(event["timestamp"])
            resume_times.append((d2 - d1).days)
            last_break = None
    if not resume_times:
        return 0
    return round(sum(resume_times) / len(resume_times), 2)

def summarize_habit_stats(events, habit):
    summary = {
        "current_streak": current_streak(events, habit),
        "longest_streak": longest_streak(events, habit),
        "half_life_days": habit_half_life(events, habit),
        "avg_snapback_days": habit_snapback(events, habit)
    }

    # Update shadow
    shadow = ShadowManager()
    shadow.safe_update({"micro_patterns": {f"habit_stats_{habit}": summary}}, consent_level="auto")

    return summary

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    import json
    habit = "movement"
    with open("data/behavior_events.json", "r") as f:
        events = json.load(f)
    stats = summarize_habit_stats(events, habit)
    print(stats)
