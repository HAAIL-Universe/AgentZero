# modules/routine_tracker.py

from typing import List, Dict, Any, Optional

class RoutineTracker:
    """
    Minimal stub RoutineTracker for Othello v0.1.

    - Accepts routines from Architect/postprocessor.
    - Stores them in memory.
    - Prints them to console for debugging.
    """

    def __init__(self) -> None:
        self.routines: List[Dict[str, Any]] = []

    def log_routine_detected(self, routine: Any, context: Optional[str] = None) -> None:
        entry = {
            "routine": routine,
            "context": context,
        }
        self.routines.append(entry)
        print(f"[RoutineTracker] Logged routine: {entry}")
