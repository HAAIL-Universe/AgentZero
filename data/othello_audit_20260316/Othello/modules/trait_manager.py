# modules/trait_manager.py

from typing import List, Any, Dict, Optional

class TraitManager:
    """
    Minimal stub TraitManager for Othello v0.1.

    - Accepts trait lists from Architect/postprocessor.
    - Stores them in memory.
    - Prints them to console for debugging.
    """

    def __init__(self) -> None:
        self.traits: List[Dict[str, Any]] = []

    def record_traits(self, traits: List[Any], context: Optional[str] = None) -> None:
        entry = {
            "traits": traits,
            "context": context,
        }
        self.traits.append(entry)
        print(f"[TraitManager] Recorded traits: {entry}")
