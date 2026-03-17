from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from modules.conversation_parser import ConversationParser

# Single shared parser instance for the postprocessor
parser = ConversationParser()


def _fallback_goal_from_text(text: str) -> Dict[str, Any]:
    """
    Fallback heuristic when the parser didn't detect any goals but the
    message clearly mentions 'goal' / 'goals'.

    We keep the schema that the rest of the system expects:
    {text, deadline, soft, confidence, source_phrase, timestamp}
    """
    lower = text.lower()

    # Try to extract the clause around the word 'goal'
    # e.g. "I'm planning to build an AI assistant. That is the goal."
    snippet = text.strip()
    idx = lower.find("goal")
    if idx != -1:
        # Take the sentence containing 'goal'
        # (very rough: split on '.', keep the one with 'goal')
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for s in sentences:
            if "goal" in s.lower():
                snippet = s
                break

    return {
        "text": snippet,
        "deadline": None,
        "soft": False,
        "confidence": 0.8,
        "source_phrase": "goal",
        "timestamp": datetime.utcnow().isoformat(),
    }


def postprocess_and_save(user_input: str) -> Dict[str, Any]:
    """
    Lightweight post-processor:

    - Uses ConversationParser to extract goals / traits / routines.
    - If no goals are detected but the text clearly mentions 'goal',
      synthesises a goal entry.
    - DOES NOT persist to any managers (Architect is the single source
      of truth for long-term persona state).
    - Returns a summary dict for logging/UI.
    """
    text = (user_input or "").strip()
    if not text:
        summary = {"goals": [], "traits": [], "routines": [], "mood": "neutral"}
        return summary

    # Primary parses
    goals: List[Dict[str, Any]] = parser.extract_goals(text) or []
    traits = parser.extract_traits(text) or []
    routines = parser.extract_routines(text) or []

    lower = text.lower()

    # --- Fallback: message mentions 'goal' but no goal objects detected ----
    if not goals and "goal" in lower:
        fallback = _fallback_goal_from_text(text)
        goals.append(fallback)

    # No persistence here — Architect handles GoalManager / TraitManager /
    # RoutineTracker updates during plan_and_execute.

    summary: Dict[str, Any] = {
        "goals": goals,
        "traits": traits,
        "routines": routines,
        "mood": "neutral",
    }
    return summary
