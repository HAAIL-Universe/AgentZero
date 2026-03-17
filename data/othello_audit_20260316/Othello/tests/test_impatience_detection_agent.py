import unittest
import os
import json
import time
from modules.agents.impatience_detection_agent import ImpatienceDetectionAgent  # Adjust if needed

class TestUpgradedImpatienceDetectionAgent(unittest.TestCase):

    def setUp(self):
        self.impatience_file = "data/impatience_data.json"
        self.reward_file = "data/rewards/impatience_rewards.json"
        self.cleanup_files([self.impatience_file, self.reward_file])
        self.agent = ImpatienceDetectionAgent(hub=None)

    def cleanup_files(self, paths):
        for path in paths:
            if os.path.exists(path):
                os.remove(path)

    def test_analyze_scoring(self):
        score = self.agent.analyze("Oh fuck this!")
        self.assertGreaterEqual(score, 2)

    def test_update_impatience_adds_to_score(self):
        self.agent.update_impatience("Shit!")
        self.assertGreater(self.agent.impatience_score, 0)

    def test_get_impatience_level_logic(self):
        self.agent.impatience_score = 4
        self.assertEqual(self.agent.get_impatience_level(), "low")
        self.agent.impatience_score = 9
        self.assertEqual(self.agent.get_impatience_level(), "medium")
        self.agent.impatience_score = 20
        self.assertEqual(self.agent.get_impatience_level(), "high")

    def test_normalized_score_bounds(self):
        self.agent.impatience_score = 5
        self.assertLessEqual(self.agent.get_normalized_score(), 1.0)
        self.agent.impatience_score = 100
        self.assertEqual(self.agent.get_normalized_score(), 1.0)

    def test_detect_impatience_structure(self):
        result = self.agent.detect_impatience("You're useless!", "agitated", source="test_case")
        self.assertIn("impatience_level", result)
        self.assertIn("normalized_score", result)
        self.assertIn("suggestion", result)
        self.assertIn("source", result)
        self.assertIn("timestamp", result)

    def test_save_impatience_data_creates_file(self):
        self.agent.impatience_score = 12
        self.agent.save_impatience_data("medium")
        self.assertTrue(os.path.exists(self.impatience_file))
        with open(self.impatience_file, "r") as f:
            data = json.load(f)
            self.assertIn("compiled_summary", data)
            self.assertIn("data", data)

    def test_reward_user_logging(self):
        self.agent.reward_user("impatience_reduction")
        self.assertTrue(os.path.exists(self.reward_file))
        with open(self.reward_file, "r") as f:
            data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertGreaterEqual(len(data), 1)

    def test_summary_snapshot_contains_recent_data(self):
        self.agent.detect_impatience("Not this again!", "annoyed", source="UX")
        snapshot = self.agent.get_summary_snapshot()
        self.assertIn("impatience_score", snapshot)
        self.assertIn("normalized_score", snapshot)
        self.assertIn("level", snapshot)
        self.assertIn("last_inputs", snapshot)
        self.assertIn("emotion_summary", snapshot)

    def test_predict_next_state_stub(self):
        prediction = self.agent.predict_next_state("This sucks")
        self.assertIn("predicted_score_change", prediction)

if __name__ == "__main__":
    unittest.main()
