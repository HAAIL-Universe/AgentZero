import json
import os
import random
from typing import List, Dict
from datetime import datetime

COACH_LOG_PATH = "data/coach_action_log.json"

class AspirationalCoachAgent:
    """
    AspirationalCoachAgent: Analyzes user’s goals, motivation, and behavior to autonomously suggest actions and provide motivational nudges.
    Tracks progress and adjusts strategies based on feedback.
    """

    def __init__(self, hub=None):
        self.hub = hub
        self.activation_events = []
        self.aspirations = []
        self.session_log = []
        self.last_checkin = None
        self.reward_system = {"goal_progress": 10, "trait_achievement": 5, "habit_formation": 7}
        self.rl_feedback_threshold = 5

        # Safe calls via hub if present
        if self.hub:
            self.hub.update_shadow({"init": "AspirationalCoachAgent initialized"})
            self.decision_data = self.hub.get_decision_vault_data({})
            self.psyche_analysis = self.hub.run_psyche_analysis({})
        else:
            self.decision_data = None
            self.psyche_analysis = None

    def activate(self, summary_str, traits=None, goals=None):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "activation",
            "summary": summary_str,
            "traits": traits or [],
            "goals": goals or []
        }
        self.activation_events.append(event)

        if traits:
            for t in traits:
                self.add_aspirational_trait(t)
        if goals:
            for g in goals:
                self.add_aspirational_goal(g)

        if self.hub:
            self.hub.update_shadow({"decisions": {"activation_event": event}})
        else:
            print("[AspirationalCoachAgent] No hub assigned — shadow update skipped.")

    def add_aspirational_trait(self, trait):
        for asp in self.aspirations:
            if asp["trait"] == trait:
                return
        new_asp = {
            "trait": trait,
            "goal": None,
            "date_set": datetime.now().isoformat(),
            "progress": []
        }
        self.aspirations.append(new_asp)

        if self.hub:
            self.hub.update_shadow({"decisions": {"aspirational_trait": new_asp}})

    def add_aspirational_goal(self, goal):
        for asp in self.aspirations:
            if asp.get("goal") == goal:
                return
        new_asp = {
            "trait": None,
            "goal": goal,
            "date_set": datetime.now().isoformat(),
            "progress": []
        }
        self.aspirations.append(new_asp)

        if self.hub:
            self.hub.update_shadow({"decisions": {"aspirational_goal": new_asp}})

    def log_checkin(self, prompt, user_reply):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "user_reply": user_reply
        }
        self.session_log.append(log_entry)
        self.last_checkin = datetime.now().isoformat()

        if self.hub:
            self.hub.update_shadow({"decisions": {"checkin": log_entry}})

    def progress_update(self, trait_or_goal, progress_note):
        for asp in self.aspirations:
            if asp["trait"] == trait_or_goal or asp["goal"] == trait_or_goal:
                prog = {
                    "timestamp": datetime.now().isoformat(),
                    "note": progress_note
                }
                asp.setdefault("progress", []).append(prog)

                if self.hub:
                    self.hub.update_shadow({"decisions": {"progress_update": prog}})

    INTERVENTIONS = {
        "push": [
            "You’re on a roll—want to stretch further?",
            "Energy is high, let’s tackle something big.",
            "You’re flying. Ready for a challenge?"
        ],
        "recharge": [
            "Looks like you’re running low. Time for a quick recharge?",
            "Energy’s dipping—maybe take a break or move around.",
            "No shame in a breather. Micro-rest?"
        ],
        "checkin": [
            "How are you actually feeling? Anything I can tweak?",
            "On the fence? Want to shift gears or keep coasting?",
            "Not sure? Let’s check in and plan next move."
        ]
    }

    def pick_intervention(self, state, trend):
        if state == "high" and trend in ["rising", "stable"]:
            return "push"
        elif state == "low" or trend == "falling":
            return "recharge"
        else:
            return "checkin"

    def coach_action(self, energy_summary):
        state = energy_summary.get("current_state", "unknown")
        trend = energy_summary.get("trend", "unknown")
        intervention_type = self.pick_intervention(state, trend)
        msg = random.choice(self.INTERVENTIONS.get(intervention_type, ["No suitable intervention."]))

        action_log = {
            "timestamp": datetime.now().isoformat(),
            "energy_state": state,
            "trend": trend,
            "intervention": intervention_type,
            "message": msg,
            "result": None
        }

        self.append_coach_log(action_log)

        if self.hub:
            self.hub.update_shadow({"decisions": {"coach_action": action_log}})

        return msg

    def append_coach_log(self, action):
        if not action or not isinstance(action, dict):
            return
        try:
            if not os.path.exists(COACH_LOG_PATH):
                actions = []
            else:
                with open(COACH_LOG_PATH, "r") as f:
                    actions = json.load(f)
        except Exception:
            actions = []
        actions.append(action)
        with open(COACH_LOG_PATH, "w") as f:
            json.dump(actions, f, indent=2)


    def log_action_result(self, index, result):
        if not os.path.exists(COACH_LOG_PATH):
            return False
        with open(COACH_LOG_PATH, "r") as f:
            actions = json.load(f)
        if index < 0 or index >= len(actions):
            return False
        actions[index]["result"] = result
        with open(COACH_LOG_PATH, "w") as f:
            json.dump(actions, f, indent=2)

        if self.hub:
            self.hub.update_shadow({"decisions": {"coach_action_result": {"index": index, "result": result}}})

        return True

    def parse_and_update_aspirations(self, conversation_input: str):
        parsed_aspirations = self.extract_aspirations_from_text(conversation_input)
        for aspiration in parsed_aspirations:
            if aspiration.get("goal"):
                self.add_aspirational_goal(aspiration["goal"])
            elif aspiration.get("trait"):
                self.add_aspirational_trait(aspiration["trait"])

    def extract_aspirations_from_text(self, text: str) -> List[Dict]:
        return [{"goal": text, "trait": None}]  # Stub for now
    
    def _save_state(self):
        state = {
            "aspirations": self.aspirations,
            "activation_events": self.activation_events,
            "session_log": self.session_log,
        }
        os.makedirs("data", exist_ok=True)
        with open("data/coach_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        try:
            with open("data/coach_state.json", "r") as f:
                state = json.load(f)
            self.aspirations = state.get("aspirations", [])
            self.activation_events = state.get("activation_events", [])
            self.session_log = state.get("session_log", [])
        except FileNotFoundError:
            pass

