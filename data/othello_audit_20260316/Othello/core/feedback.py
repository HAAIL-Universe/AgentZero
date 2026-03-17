"""
core/feedback.py

Behavioralist feedback loop for FELLO.
Processes user responses to nudges/actions, updates performance stats, and adapts nudge strategies in real time.

Author: Julius
"""

import datetime

def process_feedback(event_type, result, context=None):
    """
    Handle the result of a nudge/action.
    :param event_type: str, what happened (e.g., "nudge_sent", "task_completed", "nudge_ignored")
    :param result: dict, includes outcome ("success", "fail", "ignored", "delayed"), plus raw info
    :param context: dict, optional extra (mood, overlays, time)
    :return: dict, summary of updates
    """
    import datetime
    feedback_event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "result": result,
        "context": context or {}
    }

    # Self-Reflection hook for outcome tracking
    try:
        from core.self_reflection import SelfReflectionEngine
        sref = SelfReflectionEngine()
        nudge_id = result.get("nudge_id")
        outcome = result.get("outcome")
        if nudge_id and outcome:
            sref.log_outcome(nudge_id, outcome)
    except Exception as e:
        # Optional: add real logger call here
        pass  # Silently skip if reflection logging fails

    return feedback_event

def update_success_rate(events, nudge_type=None, window=30):
    """
    Calculate user’s recent success rate for a type of nudge/task.
    :param events: list of event dicts (newest last)
    :param nudge_type: filter by nudge/task type, or None for all
    :param window: int, number of recent events to check
    :return: float, percent success (0-100)
    """
    relevant = [e for e in events[-window:]
                if not nudge_type or e.get("detail", {}).get("nudge_type") == nudge_type]
    if not relevant:
        return 0.0
    successes = sum(1 for e in relevant if e.get("event_type") in ["task_completed", "nudge_accepted"])
    return round(100.0 * successes / len(relevant), 2)

def update_failure_rate(events, nudge_type=None, window=30):
    """
    Calculate user’s recent failure/ignore rate for a type of nudge/task.
    :return: float, percent failed/ignored (0-100)
    """
    relevant = [e for e in events[-window:]
                if not nudge_type or e.get("detail", {}).get("nudge_type") == nudge_type]
    if not relevant:
        return 0.0
    failures = sum(1 for e in relevant if e.get("event_type") in ["nudge_ignored", "task_skipped", "denied_goal"])
    return round(100.0 * failures / len(relevant), 2)

def adaptive_strategy_update(events, state, nudge_type=None):
    """
    Suggests tweak to nudge strategy based on recent outcomes and current state.
    :param events: list of event dicts
    :param state: str, hot/cold/neutral (from state_detect)
    :param nudge_type: optional filter
    :return: dict, suggestion for next strategy tweak
    """
    success = update_success_rate(events, nudge_type)
    fail = update_failure_rate(events, nudge_type)
    suggestion = "standard"

    if state == "cold" or fail > 50:
        suggestion = "softer"  # back off, less frequent or gentler nudges
    elif state == "hot" and success > 70:
        suggestion = "harder"  # ramp up, offer stretch goals
    # else: neutral/standard approach

    return {
        "current_state": state,
        "success_rate": success,
        "failure_rate": fail,
        "suggestion": suggestion
    }

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    import json
    with open("data/behavior_events.json", "r") as f:
        events = json.load(f)
    state = "neutral"
    print(adaptive_strategy_update(events, state))
