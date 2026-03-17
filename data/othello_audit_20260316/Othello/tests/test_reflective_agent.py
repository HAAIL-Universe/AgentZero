import unittest
from datetime import datetime, timedelta
from modules.agents.reflective_agent import ReflectiveAgent  # Adjust path if needed

class TestReflectiveAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ReflectiveAgent()

    def test_run_full_reflection_structure(self):
        shadow_data = {
            "mood": "neutral",
            "goal_streak": 3,
            "traits": {"confidence": 0.8},
            "persona_traits": {"confidence": 0.8},
            "shadow_traits": {"confidence": 0.8}
        }
        result = self.agent.run_full_reflection(shadow_data)
        self.assertIn("summary_text", result)
        self.assertIn("delta", result)
        self.assertIn("anomalies", result)
        self.assertIn("reward_points", result)

    def test_goal_reward_scaling(self):
        shadow_data = {
            "mood": "positive",
            "goal_streak": 6,
            "traits": {"resilience": 1.0},
            "persona_traits": {"resilience": 1.0},
            "shadow_traits": {"resilience": 1.0}
        }
        result = self.agent.run_full_reflection(shadow_data)
        self.assertGreaterEqual(result["reward_points"], 1)

    def test_detect_trait_conflicts_none(self):
        shadow_data = {
            "mood": "neutral",
            "goal_streak": 1,
            "traits": {"calm": 0.6},
            "persona_traits": {"calm": 0.6},
            "shadow_traits": {"calm": 0.6}
        }
        result = self.agent.run_full_reflection(shadow_data)
        self.assertIsInstance(result["anomalies"], list)

    def test_apply_rl_for_goal_no_goals(self):
        goals = self.agent.apply_rl_for_goal()
        self.assertEqual(goals, [])

    def test_add_goal_and_prioritize(self):
        self.agent.add_goal("Finish project", "in_progress", ["urgent"])
        self.agent.goals[0]["reward_points"] = 15
        prioritized = self.agent.apply_rl_for_goal()
        self.assertEqual(len(prioritized), 1)
        self.assertEqual(prioritized[0]["description"], "Finish project")

    def test_parse_and_add_goals(self):
        self.agent.parse_and_update_goals([
            {"description": "Write summary", "status": "planned", "tags": ["writing"]}
        ])
        self.assertEqual(len(self.agent.goals), 1)
        self.assertEqual(self.agent.goals[0]["description"], "Write summary")

    def test_extract_goals_from_text(self):
        goals = self.agent.extract_goals_from_text("Plan weekend trip")
        self.assertEqual(goals[0]["description"], "Plan weekend trip")
        self.assertIn("auto", goals[0]["tags"])

if __name__ == "__main__":
    unittest.main()
