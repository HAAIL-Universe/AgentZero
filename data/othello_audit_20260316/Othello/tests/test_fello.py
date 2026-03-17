import unittest
from unittest.mock import MagicMock, patch
from fello import Fello

class TestFello(unittest.TestCase):

    def setUp(self):
        # Patch all agent imports used inside Fello
        patchers = {
            'ReflectiveAgent': patch('fello.ReflectiveAgent'),
            'ArchitectAgent': patch('fello.ArchitectAgent'),
            'ShadowAgent': patch('fello.ShadowAgent'),
            'GoalManagementAgent': patch('fello.GoalManagementAgent'),
            'BehavioralAgent': patch('fello.BehavioralAgent'),
            'PsycheAgent': patch('fello.PsycheAgent'),
            'RoutineTrackerAgent': patch('fello.RoutineTrackerAgent'),
            'ImpatienceDetectionAgent': patch('fello.ImpatienceDetectionAgent'),
            'TraitAgent': patch('fello.TraitAgent'),
            'DecisionVaultAgent': patch('fello.DecisionVaultAgent'),
            'MemoryHandler': patch('fello.MemoryHandler'),
            'generate_daily_prompt': patch('fello.generate_daily_prompt', return_value="Mock prompt")
        }

        self.mocks = {key: p.start() for key, p in patchers.items()}
        for p in patchers.values():
            self.addCleanup(p.stop)

        # Set up mock return values
        self.architect_mock = self.mocks["ArchitectAgent"].return_value
        self.architect_mock.get_goals.return_value = [{"goal": "test goal"}]

        self.reflective_result = {
            "summary_text": "You're doing well.",
            "mood": 7,
            "reflection": "Great progress",
            "goal_update": "Update applied",
            "delta": {},
            "anomalies": []
        }

        self.mocks["ReflectiveAgent"].return_value.run_full_reflection.return_value = self.reflective_result
        self.mocks["BehavioralAgent"].return_value.analyze_behavior.return_value = {"emotional_state": "calm"}
        self.mocks["PsycheAgent"].return_value.analyze_psyche.return_value = {"mental_state": "stable"}
        self.mocks["RoutineTrackerAgent"].return_value.build_snapshot.return_value = {"snapshot": "routine"}
        self.mocks["ImpatienceDetectionAgent"].return_value.detect_impatience.return_value = "low"
        self.mocks["DecisionVaultAgent"].return_value.analyze_decisions.return_value = {"decisions": []}

        self.fello = Fello()

    def test_set_autonomy_level_valid(self):
        level = self.fello.set_autonomy_level("suggestive", consent_tier=2)
        self.assertEqual(level, 1)

    def test_set_autonomy_level_invalid(self):
        with self.assertRaises(ValueError):
            self.fello.set_autonomy_level("insane")

    def test_run_daily_check_in_returns_all_keys(self):
        result = self.fello.run_daily_check_in()
        self.assertIn("prompt", result)
        self.assertIn("mood", result)
        self.assertIn("psyche_insights", result)
        self.assertIn("routine_snapshot", result)
        self.assertIn("impatience_result", result)

    def test_add_goal_and_edit_goal_call_save_goals(self):
        self.fello._save_goals = MagicMock()
        self.fello.add_goal({"goal": "new one"})
        self.fello.edit_goal(0, {"goal": "edited one"})
        self.assertEqual(self.fello._save_goals.call_count, 2)

    def test_view_goals_returns_data(self):
        goals = self.fello.view_goals()
        self.assertEqual(goals, [{"goal": "test goal"}])

if __name__ == '__main__':
    unittest.main()
