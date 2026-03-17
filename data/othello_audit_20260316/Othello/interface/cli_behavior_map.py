"""
interface/cli_behavior_map.py

CLI for viewing and editing FELLO’s detected behavioral patterns, overlays, and assumptions.
Lets user review, approve, edit, or wipe FELLO’s “understanding” of their current behavioral map.

Author: Julius
"""

import json
import os
from core.shadow_linker import ShadowLinker

DATA_PATH = "data/behavior_map.json"

def load_behavior_map():
    if not os.path.exists(DATA_PATH):
        return {}
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def save_behavior_map(map_data):
    with open(DATA_PATH, "w") as f:
        json.dump(map_data, f, indent=2)
    linker = ShadowLinker()
    linker.safe_update({"micro_patterns": {"behavior_map": map_data}}, consent_level="auto")

def display_behavior_map(map_data):
    print("\n--- FELLO Behavior Map ---")
    if not map_data:
        print("No patterns or overlays detected yet.")
        return
    for k, v in map_data.items():
        print(f"{k}: {v}")
    print("-" * 30)

def edit_overlays(map_data):
    overlays = map_data.get("overlays", [])
    print(f"\nCurrent overlays: {overlays}")
    action = input("Add or remove overlay? (add/remove/none): ").strip().lower()
    if action == "add":
        overlay = input("Overlay to add: ").strip()
        overlays.append(overlay)
    elif action == "remove":
        overlay = input("Overlay to remove: ").strip()
        overlays = [o for o in overlays if o != overlay]
    map_data["overlays"] = list(set(overlays))
    return map_data

def approve_behavior_map(map_data):
    print("\nApprove FELLO’s current behavioral assumptions? (yes/no/wipe)")
    choice = input("Choice: ").strip().lower()
    linker = ShadowLinker()
    if choice == "wipe":
        map_data.clear()
        print("Behavior map wiped.")
        linker.safe_update({"micro_patterns": {"behavior_map": "wiped"}}, consent_level="auto")
    elif choice == "no":
        print("No changes made.")
    elif choice == "yes":
        print("Behavior map approved.")
        linker.safe_update({"micro_patterns": {"behavior_map": map_data, "status": "approved"}}, consent_level="auto")
    return map_data

def main():
    map_data = load_behavior_map()
    while True:
        display_behavior_map(map_data)
        print("Options: (1) Edit overlays (2) Approve/Wipe (3) Save & Exit")
        choice = input("Pick option: ").strip()
        if choice == "1":
            map_data = edit_overlays(map_data)
        elif choice == "2":
            map_data = approve_behavior_map(map_data)
        elif choice == "3":
            save_behavior_map(map_data)
            print("Map saved. Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
