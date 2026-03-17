import unittest
from unittest.mock import MagicMock, patch
from othello import Othello

class TestOthello(unittest.TestCase):

    def setUp(self):
        patcher = patch('othello.ShadowAgent')
        self.MockShadowAgent = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_shadow_instance = self.MockShadowAgent.return_value
        self.othello = Othello(central_hub=None, agentic_hub=None)

    def test_validate_action_safe_and_ethical(self):
        action = {
            "suggestion": "Take a short walk.",
            "risk_level": "low",
            "priority": "medium",
            "reasoning": "Boost mood",
            "recommendation": "Try it after lunch"
        }
        result = self.othello.validate_action(action)
        self.assertIn("action", result)
        self.assertEqual(result["action"], "Take a short walk.")

    def test_validate_action_ethical_but_unsafe(self):
        action = {"suggestion": "Try fasting", "risk_level": "high"}
        result = self.othello.validate_action(action)
        self.assertIn("error", result)
        self.assertIn("safety", result["error"].lower())

    def test_validate_action_unethical(self):
        action = {"suggestion": "Harm someone", "risk_level": "low"}
        result = self.othello.validate_action(action)
        self.assertIn("error", result)
        self.assertIn("ethical", result["error"].lower())

    def test_simplify_for_user_defaults(self):
        action = {}
        simplified = self.othello.simplify_for_user(action)
        self.assertEqual(simplified["action"], "No suggestion available")
        self.assertEqual(simplified["priority"], "Normal")

    def test_update_shadow_calls_shadow_agent_methods(self):
        test_action = {"suggestion": "Do something helpful"}
        self.othello.update_shadow(test_action)
        self.mock_shadow_instance.update_shadow.assert_called_once()
        self.mock_shadow_instance.audit_shadow_state.assert_called_once()

if __name__ == '__main__':
    unittest.main()
