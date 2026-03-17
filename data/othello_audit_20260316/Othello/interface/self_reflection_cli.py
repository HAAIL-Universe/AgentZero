"""
interface/self_reflection_cli.py

Minimal CLI for querying FELLO’s self-reflection summaries and intervention reasons.

Author: Julius
"""

import json
import os
from core.self_reflection import SelfReflectionEngine

META_LOG_PATH = "data/self_reflection_log.json"

def load_meta_log():
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH, "r") as f:
        try:
            return json.load(f)
        except:
            return []

def show_last_summary():
    """
    Display the most recent reflection summary.
    """
    meta_log = load_meta_log()
    if not meta_log:
        print("No reflection summaries found.")
        return
    last = meta_log[-1]
    print("--- Last Reflection Summary ---")
    print(json.dumps(last, indent=2))

def explain_intervention():
    """
    Ask for an intervention ID and show its reason/context.
    """
    engine = SelfReflectionEngine()
    nudge_id = input("Enter intervention ID: ").strip()
    info = engine.explain_intervention(nudge_id)
    if not info:
        print("Intervention ID not found.")
    else:
        print("--- Intervention Explanation ---")
        print(json.dumps(info, indent=2))

def main():
    print("FELLO Self-Reflection CLI")
    print("(1) Show last reflection summary")
    print("(2) Explain intervention by ID")
    choice = input("Select option: ").strip()
    if choice == "1":
        show_last_summary()
    elif choice == "2":
        explain_intervention()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
