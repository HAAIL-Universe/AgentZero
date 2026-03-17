import unittest
from unittest.mock import MagicMock
from modules.agentic_agents.prism_agent import PrismAgent

class TestPrismAgent(unittest.TestCase):

    def setUp(self):
        self.mock_agentic_hub = MagicMock()
        self.mock_central_hub = MagicMock()
        self.agent = PrismAgent(
            central_hub=self.mock_central_hub,
            agentic_hub=self.mock_agentic_hub
        )

        self.mock_shadow = {
            "mood": "positive",
            "traits": {"kind": 1},
            "goals": [{"description": "Test goal"}]
        }

    def test_update_user_data_sets_profile_and_insights(self):
        self.agent.update_user_data(self.mock_shadow)
        self.assertEqual(self.agent.mood_state, "positive")
        self.assertIn("traits", self.agent.profile_summary)
        self.assertIn("goals", self.agent.profile_summary)
        self.assertIsInstance(self.agent.systematic_insights, list)

    def test_analyze_behavior_returns_expected_structure(self):
        result = self.agent.analyze_behavior(self.mock_shadow)
        self.assertIn("emotional_state", result)
        self.assertIn("behavior_suggestion", result)

    def test_reward_user_updates_shadow(self):
        self.agent.reward_user("goal_progress")
        self.mock_agentic_hub.update_shadow.assert_called_once()
        call_args = self.mock_agentic_hub.update_shadow.call_args[0][0]
        self.assertEqual(call_args.get("reward_points"), 10)

    def test_run_daily_update_returns_expected_structure(self):
        result = self.agent.run_daily_update(self.mock_shadow)
        self.assertIn("daily_prompt", result)
        self.assertIn("profile_summary", result)
        self.assertIn("systematic_insights", result)

    def test_retrieve_user_data_returns_expected_structure(self):
        self.agent.update_user_data(self.mock_shadow)
        data = self.agent.retrieve_user_data()
        self.assertIn("profile_summary", data)
        self.assertIn("systematic_insights", data)
        self.assertIn("mood_state", data)
        self.assertIn("behavioral_insights", data)

if __name__ == "__main__":
    unittest.main()
