import json
import os
from typing import List, Dict, Any, Optional, cast
from datetime import datetime
import logging

# File paths
BEHAVIOR_FILE = "data/behavior.json"
HABIT_FILE = "data/habit.json"
EVENTS_FILE = "data/events.json"

logging.basicConfig(level=logging.DEBUG)

class BehavioralAgent:
    """
    BehavioralAgent: Tracks and analyzes user behavior, habits, and events over time.
    Flags anomalies, builds trait/goal links, and prepares high-fidelity data for RL/ML integration.
    Everything is persisted; no state lost on restart.
    """

    def __init__(self, hub=None):
        self.hub = hub
        self.behavior_data = self._load_json(BEHAVIOR_FILE, data_type='dict')
        self.habit_data = self._load_json(HABIT_FILE, data_type='dict')
        self.event_data = self._load_json(EVENTS_FILE, data_type='list')

        self.reward_system = {
            "base": 1,
            "streak_bonus": 5,
            "recovery_bonus": 10,
            "trait_goal_bonus": 3
        }
        self.rl_feedback_threshold = 5
        self.behavior_threshold = 3  # for anomaly detection

    # --- JSON IO ---
    def _load_json(self, path: str, data_type: str = 'dict') -> Any:
        default_data = {} if data_type == 'dict' else []
        try:
            if not os.path.exists(path):
                self._save_json(path, default_data)
                return default_data
            with open(path, 'r') as f:
                data = json.load(f)
                if data_type == 'dict' and isinstance(data, dict):
                    return data
                if data_type == 'list' and isinstance(data, list):
                    return data
                raise ValueError(f"Incorrect data type for {path}")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            self._save_json(path, default_data)
            return default_data

    def _save_json(self, path: str, data: Any):
        full_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.debug(f"[BehavioralAgent] JSON saved to: {full_path}")

    # --- Context ---
    def _get_context(self) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "timestamp": now.isoformat(),
            "time_of_day": "morning" if now.hour < 12 else "afternoon" if now.hour < 18 else "evening",
            "weekday": now.strftime("%A")
        }

    # --- Behavior Tracking ---
    def track_behavior(
        self,
        behavior_name: str,
        score_change: int = 1,
        mood: Optional[str] = None,
        emotional_state: Optional[str] = None,
        trait_links: Optional[List[str]] = None,
        goal_links: Optional[List[str]] = None
    ):
        logging.debug(f"[BehavioralAgent] TRACK_BEHAVIOR CALLED: {behavior_name}, Score change: {score_change}")
        """
        Track/update a behavior, record its history, persist change.
        """
        now = datetime.now().isoformat()
        context = self._get_context()
        trait_links = trait_links or []
        goal_links = goal_links or []

        behavior = self.behavior_data.get(behavior_name, {
            "habit_score": 0,
            "streak": 0,
            "regression_flags": 0,
            "type": None,
            "linked_traits": [],
            "linked_goals": [],
            "history": []
        })

        anomaly = abs(score_change) > self.behavior_threshold

        # Score update
        behavior["habit_score"] += score_change
        behavior["last_checked"] = now
        behavior["linked_traits"] = list(set(behavior["linked_traits"] + trait_links))
        behavior["linked_goals"] = list(set(behavior["linked_goals"] + goal_links))

        # History log
        behavior["history"].append({
            "timestamp": now,
            "score_change": score_change,
            "context": context,
            "mood": mood,
            "emotional_state": emotional_state,
            "anomaly_flag": anomaly
        })

        # Streak/regression
        if score_change > 0:
            behavior["streak"] = behavior.get("streak", 0) + 1
        else:
            behavior["regression_flags"] = behavior.get("regression_flags", 0) + 1
            behavior["streak"] = 0

        # Reward
        reward_points = self.calculate_reward(behavior)
        behavior["last_reward"] = reward_points

        self.behavior_data[behavior_name] = behavior
        self._save_json(BEHAVIOR_FILE, self.behavior_data)

        if self.hub:
            logging.debug(f"[BehavioralAgent] Passing behavior data to hub: {behavior_name}")
            self.hub.update_shadow({"behavior": {behavior_name: behavior}, "reward_points": reward_points})
        else:
            logging.debug(f"[BehavioralAgent] Behavior tracked: {behavior_name} | Reward: {reward_points}")

    def calculate_reward(self, behavior: dict) -> int:
        reward = self.reward_system["base"]
        if behavior["streak"] and behavior["streak"] % 3 == 0:
            reward += self.reward_system["streak_bonus"]
        if behavior["regression_flags"] == 0 and behavior["streak"] >= 3:
            reward += self.reward_system["recovery_bonus"]
        if behavior["linked_traits"] or behavior["linked_goals"]:
            reward += self.reward_system["trait_goal_bonus"]
        logging.debug(f"[BehavioralAgent] Reward calculated for {behavior}: {reward}")
        return reward

    # --- Habit Tracking ---
    def track_habit(
        self,
        habit_name: str,
        score_change: int = 1,
        context: Optional[str] = None
    ):
        logging.debug(f"[BehavioralAgent] TRACK_HABIT CALLED: {habit_name}, Score change: {score_change}")
        """
        Track or update a habit, record its history, and persist.
        """
        now = datetime.now().isoformat()
        habit = self.habit_data.get(habit_name, {
            "score": 0,
            "history": []
        })
        habit["score"] += score_change
        habit["history"].append({
            "timestamp": now,
            "score_change": score_change,
            "context": context or self._get_context()
        })
        self.habit_data[habit_name] = habit
        self._save_json(HABIT_FILE, self.habit_data)
        logging.debug(f"[BehavioralAgent] Habit tracked: {habit_name}")

    # --- Event Tracking ---
    def track_event(
        self,
        event_desc: str,
        details: Optional[Dict[str, Any]] = None
    ):
        logging.debug(f"[BehavioralAgent] TRACK_EVENT CALLED: {event_desc}")
        """
        Log an event (discrete, not part of an ongoing behavior/habit).
        """
        now = datetime.now().isoformat()
        event = {
            "timestamp": now,
            "event": event_desc,
            "details": details or {},
            "context": self._get_context()
        }
        self.event_data.append(event)
        self._save_json(EVENTS_FILE, self.event_data)
        logging.debug(f"[BehavioralAgent] Event tracked: {event_desc}")

    # --- Summaries/Analysis ---
    def get_behavioral_summary(self) -> dict:
        logging.debug("[BehavioralAgent] Generating behavioral summary")
        summary = {
            "tracked_behaviors": list(self.behavior_data.keys()),
            "active_streaks": {k: v.get("streak", 0) for k, v in self.behavior_data.items()},
            "recent_anomalies": [k for k, v in self.behavior_data.items() if any(h.get("anomaly_flag") for h in v.get("history", [])[-3:])],
            "reward_points": sum(v.get("last_reward", 0) for v in self.behavior_data.values())
        }
        logging.debug(f"[BehavioralAgent] Behavioral summary: {summary}")
        return summary

    def analyze_behavior(self, shadow_snapshot: Optional[dict] = None, context_notes: Optional[str] = None) -> dict:
        logging.debug("[BehavioralAgent] Analyzing behavior data")
        """
        Analyze stored behaviors for patterns, anomalies, progression, and ML/RL hooks.
        """
        summary = {
            "anomaly_flags": [],
            "positive_streaks": [],
            "regressions": [],
            "linked_traits": {},
            "linked_goals": {},
            "total_rewards": 0,
            "summary_notes": [],
            "delta_change": {},
            "momentum": {},
            "emotional_state": None,
            "ml_rl_ready_vector": {},
            "shadow_alignment": {},
            "persona_alignment": {},
        }

        for name, data in self.behavior_data.items():
            history = data.get("history", [])
            last_entries = history[-3:] if len(history) >= 3 else history
            reward = data.get("last_reward", 0)

            # Anomaly detection
            if any(h.get("anomaly_flag") for h in last_entries):
                summary["anomaly_flags"].append(name)

            # Positive streaks
            if data.get("streak", 0) >= 3:
                summary["positive_streaks"].append(name)
                summary["summary_notes"].append(f"{name} shows growing consistency.")

            # Regression
            if data.get("regression_flags", 0) >= 2:
                summary["regressions"].append(name)
                summary["summary_notes"].append(f"{name} may need intervention — frequent dips detected.")

            # Trait/Goal Links
            for trait in data.get("linked_traits", []):
                summary["linked_traits"].setdefault(trait, 0)
                summary["linked_traits"][trait] += 1

            for goal in data.get("linked_goals", []):
                summary["linked_goals"].setdefault(goal, 0)
                summary["linked_goals"][goal] += 1

            # Reward accumulation
            summary["total_rewards"] += reward

            # RL/ML prep: delta/momentum
            if len(history) >= 2:
                delta = history[-1]["score_change"] - history[-2]["score_change"]
                summary["delta_change"][name] = delta
                if delta > 0:
                    summary["momentum"][name] = "upward"
                elif delta < 0:
                    summary["momentum"][name] = "downward"
                else:
                    summary["momentum"][name] = "flat"

            # Shadow/persona alignment hooks
            if shadow_snapshot:
                shadow_behaviors = shadow_snapshot.get("behavior", {})
                shadow_entry = shadow_behaviors.get(name)
                if shadow_entry:
                    summary["shadow_alignment"][name] = {
                        "habit_score": shadow_entry.get("habit_score"),
                        "last_checked": shadow_entry.get("last_checked")
                    }
            summary["persona_alignment"][name] = {
                "habit_score": data.get("habit_score"),
                "last_checked": data.get("last_checked")
            }

            summary["ml_rl_ready_vector"][name] = {
                "score": data.get("habit_score"),
                "streak": data.get("streak", 0),
                "regression_flags": data.get("regression_flags", 0)
            }

        # Emotional state estimate
        if summary["regressions"]:
            summary["emotional_state"] = "stressed"
        elif summary["positive_streaks"]:
            summary["emotional_state"] = "motivated"
        else:
            summary["emotional_state"] = "neutral"

        if context_notes:
            summary["summary_notes"].append(f"Context: {context_notes}")

        # Optional: Sync back to Shadow or Hub if context provided
        if self.hub:
            self.hub.update_shadow({"behavioral_insights": summary})

        logging.debug(f"[BehavioralAgent] Behavioral summary generated (v2).")
        return summary
