import json
import os
from typing import List, Dict
from datetime import datetime

ROUTINE_LOG = "data/routines.json"
ROUTINE_STATE = "data/routine_state.json"

class RoutineTrackerAgent:
    """
    RoutineTrackerAgent: Tracks and analyzes user routines—wake/sleep, meals, work, activity—
    suggesting improvements for optimizing habits and increasing consistency.
    """

    def __init__(self, hub=None):
        self.hub = hub
        self.routine_data = self._load_log()
        self.question_queue = self._init_question_queue()
        self.streaks = {}
        self.max_streaks = {}
        self.last_break = {}
        self.reward_system = {"routine_formed": 10, "routine_improvement": 5}
        self.rl_feedback_threshold = 5
        self._load_state()  # <---- load streaks/max_streaks/last_break on boot

    def _load_log(self):
        if not os.path.exists(ROUTINE_LOG):
            return []
        with open(ROUTINE_LOG, "r") as f:
            return json.load(f)

    def _save_log(self):
        os.makedirs(os.path.dirname(ROUTINE_LOG), exist_ok=True)
        with open(ROUTINE_LOG, "w") as f:
            json.dump(self.routine_data, f, indent=4)

    def _save_state(self):
        state = {
            "streaks": self.streaks,
            "max_streaks": self.max_streaks,
            "last_break": self.last_break,
        }
        os.makedirs(os.path.dirname(ROUTINE_STATE), exist_ok=True)
        with open(ROUTINE_STATE, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        try:
            with open(ROUTINE_STATE, "r") as f:
                state = json.load(f)
            self.streaks = state.get("streaks", {})
            self.max_streaks = state.get("max_streaks", {})
            self.last_break = state.get("last_break", {})
        except FileNotFoundError:
            pass

    def _init_question_queue(self):
        return [
            "What time did you wake up today?",
            "Did you have coffee or breakfast around a usual time?",
            "When did you start focused work or your main task?",
            "Did you move your body today (gym, walk, stretch)?",
            "What time do you usually wind down for the night?"
        ]

    def get_next_question(self):
        if not self.question_queue:
            self.question_queue = self._init_question_queue()
        return self.question_queue.pop(0)

    def log_answer(self, question, answer, mood=None, trait_link=None, goal_link=None):
        routine_type = self._extract_routine_type(question)
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "routine_type": routine_type,
            "question": question,
            "answer": answer,
            "mood": mood,
            "linked_traits": [trait_link] if trait_link else [],
            "linked_goals": [goal_link] if goal_link else [],
        }

        if not isinstance(self.routine_data, list):
            print(f"⚠ [RoutineTrackerAgent] routine_data was not a list. Resetting to empty list.")
            self.routine_data = []

        self.routine_data.append(entry)
        self._update_streak(routine_type, entry)
        self._save_log()

        if self.hub:
            self.hub.update_shadow({"routine": {routine_type: entry}})
        else:
            print(f"[RoutineTrackerAgent] No hub assigned — shadow update skipped.")

    def _update_streak(self, routine_type, entry):
        today = entry["date"]
        prev_entries = [e for e in self.routine_data if e.get("routine_type") == routine_type]
        prev_dates = sorted(set(e["date"] for e in prev_entries if "date" in e))

        if len(prev_dates) >= 2 and (
            (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(prev_dates[-2], "%Y-%m-%d")).days == 1
        ):
            self.streaks[routine_type] = self.streaks.get(routine_type, 0) + 1
        else:
            self.streaks[routine_type] = 1

        if self.streaks[routine_type] > self.max_streaks.get(routine_type, 0):
            self.max_streaks[routine_type] = self.streaks[routine_type]
        if len(prev_dates) >= 2 and (
            (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(prev_dates[-2], "%Y-%m-%d")).days > 1
        ):
            self.last_break[routine_type] = prev_dates[-2]

        self._save_state()  # <--- persist state on streak change

    def analyze_routines(self):
        routine_analysis = {
            "suggestions": [],
            "overall_status": "stable"
        }

        if "wake_up" not in [entry["routine_type"] for entry in self.routine_data]:
            routine_analysis["suggestions"].append("Add a wake-up routine to improve consistency.")
        else:
            routine_analysis["overall_status"] = "routine established"

        return routine_analysis

    def _extract_routine_type(self, question):
        q = question.lower()
        if "wake" in q:
            return "wake_up"
        if "coffee" in q or "breakfast" in q:
            return "morning_intake"
        if "work" in q or "focus" in q:
            return "focus_block"
        if "move your body" in q or "gym" in q:
            return "activity"
        if "wind down" in q or "sleep" in q:
            return "wind_down"
        return "other"

    def reward_user_for_routine(self, routine_type: str):
        if routine_type in self.streaks and self.streaks[routine_type] > 0:
            reward_points = self.reward_system["routine_formed"]
            if self.hub:
                self.hub.update_shadow({"reward_points": reward_points})
            print(f"[RoutineTrackerAgent] User rewarded with {reward_points} points for maintaining {routine_type} routine.")
        else:
            print(f"[RoutineTrackerAgent] No streak found for {routine_type} — no reward applied.")

    def set_routines(self, routines: List[Dict]):
        """
        Accepts a list of parsed routines and logs each one.
        """
        for routine in routines:
            self.log_answer(
                question=routine.get("question", "Routine update"),
                answer=routine.get("answer", "No answer provided"),
                mood=routine.get("mood"),
                trait_link=routine.get("trait_link"),
                goal_link=routine.get("goal_link")
            )
        print(f"[RoutineTrackerAgent] Set and logged {len(routines)} routines.")

    def get_routines(self):
        """
        Returns a summary of tracked routines (for CentralHub access).
        """
        if not hasattr(self, 'routine_data'):
            self.routine_data = {}
        return self.routine_data

    def build_snapshot(self):
        """
        Build a snapshot of the user's current routine state.
        """
        return {
            "routines": self.get_routines(),
            "streaks": self.streaks,
            "max_streaks": self.max_streaks,
            "last_break": self.last_break
        }

