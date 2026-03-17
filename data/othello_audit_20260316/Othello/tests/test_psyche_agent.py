import unittest
import os
import json
from unittest.mock import MagicMock
from modules.agentic_agents.psyche_agent import PsycheAgent

class TestPsycheAgent(unittest.TestCase):
    def setUp(self):
        """Set up a PsycheAgent instance for testing."""
        self.mock_central_hub = MagicMock()
        self.mock_agentic_hub = MagicMock()
        self.agent = PsycheAgent(
            central_hub=self.mock_central_hub,
            agentic_hub=self.mock_agentic_hub
        )

        # Remove any existing psyche.json to start fresh
        if os.path.exists("data/psyche.json"):
            os.remove("data/psyche.json")

    def test_log_activation_creates_entry_and_file(self):
        """Test that logging an activation creates a log entry and writes to file."""
        self.agent.log_activation(2, "Test activation")
        
        self.assertEqual(self.agent.activation_log[-1]["level"], 2)
        self.assertTrue(os.path.exists("data/psyche.json"))
        
        with open("data/psyche.json", "r") as f:
            data = json.load(f)
        self.assertGreater(len(data), 0)

    def test_log_trigger_creates_entry_and_file(self):
        """Test that logging a trigger creates a trigger entry and writes to file."""
        self.agent.log_trigger("test_event", "success", feedback="All good")
        
        self.assertEqual(self.agent.trigger_history[-1]["event"], "test_event")
        self.assertTrue(os.path.exists("data/psyche.json"))
        
        with open("data/psyche.json", "r") as f:
            data = json.load(f)
        self.assertGreater(len(data), 0)

    def test_analyze_psyche_returns_expected_keys(self):
        """Test that analyze_psyche returns a proper insights dict."""
        shadow_data = {"mood": "positive"}
        user_data = {"goal_streak": 6}
        
        insights = self.agent.analyze_psyche(shadow_data, user_data)
        self.assertIn("emotional_state", insights)
        self.assertIn("motivation_level", insights)
        self.assertIn("suggestion", insights)
        self.assertEqual(insights["emotional_state"], "positive")
        self.assertEqual(insights["motivation_level"], 6)

    def test_reward_user_outputs_message(self):
        """Test reward_user outputs the expected print message."""
        from io import StringIO
        import sys
        
        captured_output = StringIO()
        sys.stdout = captured_output
        self.agent.reward_user("motivation_boost")
        sys.stdout = sys.__stdout__
        
        self.assertIn("User rewarded with 5 points for motivation_boost", captured_output.getvalue())

if __name__ == "__main__":
    unittest.main()
