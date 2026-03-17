# utils/memory.py

import json
import os
from datetime import datetime

MEMORY_FILE = "fello_memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def add_memory_entry(entry):
    memory = load_memory()
    memory.append(entry)
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def get_latest_entry():
    memory = load_memory()
    if memory:
        return memory[-1]
    return None

class MemoryHandler:
    """🧠 Manages conversation history for contextual memory."""
    def __init__(self):
        self.history = load_memory()

    def get_recent_messages(self, limit=10):
        return self.history[-limit:] if self.history else []

    def add_interaction(self, user_input, ai_output):
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "user": user_input,
            "fello": ai_output
        }
        self.history.append(entry)
        add_memory_entry(entry)
