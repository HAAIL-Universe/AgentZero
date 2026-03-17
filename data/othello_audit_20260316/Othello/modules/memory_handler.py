import json
import os
from typing import List, Dict

MEMORY_FILE = "data/memory.json"  # Path where memory will be stored

class MemoryHandler:
    """
    Handles agent memory storage and retrieval for conversational and agentic continuity.
    Uses persistent storage for agent's conversation/context recall.
    """
    def __init__(self):
        # Load full history on boot
        self.history = self.load_memory()

    def load_memory(self) -> List[Dict]:
        """
        Loads memory from the persistent storage (JSON file).
        """
        if not os.path.exists(MEMORY_FILE):
            return []  # Return an empty list if no memory file exists
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)

    def get_recent_messages(self, limit=5) -> List[Dict]:
        """
        Returns the latest N agent/user message pairs for context injection in the prompt.
        """
        return self.history[-limit:] if self.history else []

    def add_interaction(self, user_input: str, response: str):
        """
        Records each agent-user exchange in persistent memory.
        """
        entry = {
            "user": user_input,
            "assistant": response
        }
        self.history.append(entry)
        self.save_memory()  # Write-through to disk

    def save_memory(self):
        """
        Saves the current memory state to a JSON file.
        """
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=2)

    def clear_memory(self):
        """
        Clears the entire memory.
        """
        self.history = []
        self.save_memory()
        print("[MemoryHandler] Memory cleared.")
