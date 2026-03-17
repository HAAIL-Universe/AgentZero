import unittest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
import json
import os

from modules.agents.aspirational_coach_agent import AspirationalCoachAgent

class TestAspirationalCoachAgent(unittest.TestCase):

    def setUp(self):
        self.mock_hub = MagicMock()
        self.agent = AspirationalCoachAgent(hub=self.mock_hub)

    def test_activate(self):
        self.agent.activate("Daily motivation check", traits=["resilient"], goals=["finish project"])
        self.assertEqual(len(self.agent.activation_events), 1)
        self.assertIn("resilient", [a["trait"] for a in self.agent.aspirations if a["trait"]])
        self.assertIn("finish project", [a["goal"] for a in self.agent.aspirations if a["goal"]])

    def test_add_aspirational_trait_and_goal(self):
        self.agent.add_aspirational_trait("creative")
        self.agent.add_aspirational_goal("learn piano")
        self.assertEqual(len(self.agent.aspirations), 2)
        self.assertTrue(any(asp["trait"] == "creative" for asp in self.agent.aspirations))
        self.assertTrue(any(asp["goal"] == "learn piano" for asp in self.agent.aspirations))

    def test_log_checkin(self):
        self.agent.log_checkin("How are you?", "Feeling great")
        self.assertEqual(len(self.agent.session_log), 1)
        self.assertEqual(self.agent.session_log[0]["user_reply"], "Feeling great")

    def test_progress_update(self):
        self.agent.add_aspirational_trait("focused")
        self.agent.progress_update("focused", "Practiced deep work")
        asp = next(asp for asp in self.agent.aspirations if asp["trait"] == "focused")
        self.assertEqual(len(asp["progress"]), 1)
        self.assertIn("Practiced deep work", asp["progress"][0]["note"])

    def test_pick_intervention(self):
        self.assertEqual(self.agent.pick_intervention("high", "rising"), "push")
        self.assertEqual(self.agent.pick_intervention("low", "falling"), "recharge")
        self.assertEqual(self.agent.pick_intervention("medium", "unknown"), "checkin")

    @patch("builtins.open", new_callable=mock_open, read_data="[]")
    @patch("os.path.exists", return_value=True)
    def test_coach_action_and_log(self, mock_exists, mock_file):
        msg = self.agent.coach_action({"current_state": "high", "trend": "stable"})
        self.assertIsInstance(msg, str)
        self.assertTrue(self.mock_hub.update_shadow.called)
        mock_file.assert_called_with("data/coach_action_log.json", "w")

    @patch("builtins.open", new_callable=mock_open, read_data='[{"result": null}]')
    @patch("os.path.exists", return_value=True)
    def test_log_action_result_success(self, mock_exists, mock_file):
        self.assertTrue(self.agent.log_action_result(0, "Success"))

    @patch("builtins.open", new_callable=mock_open, read_data="[]")
    @patch("os.path.exists", return_value=True)
    def test_log_action_result_out_of_bounds(self, mock_exists, mock_file):
        self.assertFalse(self.agent.log_action_result(5, "Too Far"))

    def test_parse_and_update_aspirations(self):
        self.agent.parse_and_update_aspirations("I want to run a marathon")
        self.assertTrue(any(asp["goal"] == "I want to run a marathon" for asp in self.agent.aspirations))

    def test_extract_aspirations_from_text(self):
        result = self.agent.extract_aspirations_from_text("Become more confident")
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["goal"], "Become more confident")


if __name__ == "__main__":
    unittest.main()
