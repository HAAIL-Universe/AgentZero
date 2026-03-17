import unittest
from unittest.mock import MagicMock
from modules.agentic_agents.decision_vault_agent import DecisionVaultAgent

class TestDecisionVaultAgent(unittest.TestCase):

    def setUp(self):
        self.mock_central_hub = MagicMock()
        self.mock_agentic_hub = MagicMock()
        self.agent = DecisionVaultAgent(
            central_hub=self.mock_central_hub,
            agentic_hub=self.mock_agentic_hub
        )
        self.agent._save_decisions = MagicMock()  # Disable actual file writes
        self.agent.decision_data = []

    def test_log_decision_stores_and_updates_shadow(self):
        self.agent.log_decision("choice", {"type": "test"}, outcome="success")
        self.assertEqual(len(self.agent.decision_data), 1)
        self.mock_agentic_hub.update_shadow.assert_called_once()

    def test_analyze_decisions_flags_avoidance_and_risk(self):
        self.agent.decision_data = [
            {"decision_type": "test", "details": {"type": "low"}, "outcome": None},
            {"decision_type": "test", "details": {"type": "high risk"}, "outcome": "fail"},
            {"decision_type": "test", "details": {"type": "low"}, "outcome": None},
            {"decision_type": "test", "details": {"type": "low"}, "outcome": None},
            {"decision_type": "test", "details": {"type": "low"}, "outcome": None}
        ]
        analysis = self.agent.analyze_decisions()
        self.assertGreaterEqual(len(analysis["decision_avoidance"]), 3)
        self.assertEqual(len(analysis["risky_decisions"]), 1)
        self.assertIn("Consider breaking down large decisions", analysis["suggestions"][0])

    def test_simulate_decision_outcome_high_risk(self):
        decision = {"details": {"type": "high_risk"}}
        result = self.agent.simulate_decision_outcome(decision)
        self.assertEqual(result["outcome"], "high_risk_failure")

    def test_simulate_decision_outcome_low_risk(self):
        decision = {"details": {"type": "low"}}
        result = self.agent.simulate_decision_outcome(decision)
        self.assertEqual(result["outcome"], "low_risk_success")

    def test_remind_review_pending_with_pending(self):
        self.agent.decision_data = [{"resolved": False}, {"resolved": False}]
        msg = self.agent.remind_review_pending()
        if msg is not None:
            self.assertIn("pending decisions", msg)
        else:
            self.fail("Expected a reminder message but got None.")

    def test_remind_review_pending_none(self):
        self.agent.decision_data = [{"resolved": True}]
        msg = self.agent.remind_review_pending()
        self.assertIsNone(msg)

    def test_summary_counts_correctly(self):
        self.agent.decision_data = [
            {"outcome": None},
            {"outcome": "done"}
        ]
        summary = self.agent.summary()
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["resolved"], 1)

    def test_list_all_returns_decisions(self):
        self.agent.decision_data = [{"test": "decision"}]
        all_decisions = self.agent.list_all()
        self.assertEqual(all_decisions, [{"test": "decision"}])

if __name__ == '__main__':
    unittest.main()
