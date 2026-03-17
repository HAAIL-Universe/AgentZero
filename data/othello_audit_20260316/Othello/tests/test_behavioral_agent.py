import unittest
import os
import json
from modules.agents.behavioral_agent import BehavioralAgent  # adjust import if needed

class TestPureBehavioralAgent(unittest.TestCase):

    def setUp(self):
        self.behavior_file = "data/behavior.json"
        self.habit_file = "data/habit.json"
        self.events_file = "data/events.json"
        self.agent = BehavioralAgent(hub=None)

    def test_load_json_creates_file_and_defaults(self):
        test_file = "data/temp_test.json"
        if os.path.exists(test_file):
            os.remove(test_file)
        result = self.agent.load_json(test_file, data_type="dict")
        self.assertEqual(result, {})
        self.assertTrue(os.path.exists(test_file))

    def test_save_json_writes_data_correctly(self):
        test_file = "data/temp_save.json"
        data = {"hello": "world"}
        self.agent.save_json(test_file, data)
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r") as f:
            loaded = json.load(f)
            self.assertEqual(loaded, data)

    def test_get_context_contains_expected_keys(self):
        context = self.agent._get_context()
        self.assertIn("timestamp", context)
        self.assertIn("time_of_day", context)
        self.assertIn("weekday", context)

    def test_track_behavior_creates_entry_and_saves(self):
        self.agent.track_behavior("meditate", score_change=2, mood="calm", emotional_state="neutral",
                                  trait_links=["mindfulness"], goal_links=["consistency"])
        with open(self.behavior_file, "r") as f:
            data = json.load(f)
            self.assertIn("meditate", data)
            entry = data["meditate"]
            self.assertEqual(entry["habit_score"], 2)
            self.assertEqual(entry["linked_traits"], ["mindfulness"])
            self.assertEqual(entry["linked_goals"], ["consistency"])
            self.assertEqual(entry["streak"], 1)
            self.assertIn("history", entry)
            self.assertEqual(len(entry["history"]), 1)

    def test_anomaly_flag_triggers_on_high_score(self):
        self.agent.track_behavior("explode", score_change=20)
        with open(self.behavior_file, "r") as f:
            data = json.load(f)
            entry = data["explode"]
            last = entry["history"][-1]
            self.assertTrue(last["anomaly_flag"])

    def test_calculate_reward_all_conditions(self):
        behavior = {
            "streak": 3,
            "regression_flags": 0,
            "linked_traits": ["focus"],
            "linked_goals": ["discipline"]
        }
        reward = self.agent.calculate_reward(behavior)
        expected_min = sum([
            self.agent.reward_system["base"],
            self.agent.reward_system["streak_bonus"],
            self.agent.reward_system["recovery_bonus"],
            self.agent.reward_system["trait_goal_bonus"]
        ])
        self.assertGreaterEqual(reward, expected_min)

    def test_get_behavioral_summary_reports_correctly(self):
        self.agent.track_behavior("write", score_change=1)
        self.agent.track_behavior("write", score_change=1)
        summary = self.agent.get_behavioral_summary()
        self.assertIn("tracked_behaviors", summary)
        self.assertIn("active_streaks", summary)
        self.assertIn("recent_anomalies", summary)
        self.assertIn("reward_points", summary)
        self.assertIn("write", summary["tracked_behaviors"])
        self.assertGreaterEqual(summary["reward_points"], 1)

if __name__ == "__main__":
    unittest.main()
