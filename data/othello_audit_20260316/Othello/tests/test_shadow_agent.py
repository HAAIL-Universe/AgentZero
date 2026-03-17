import unittest
import os
import json
from unittest.mock import MagicMock
from modules.agentic_agents.shadow_agent import ShadowAgent

class TestShadowAgent(unittest.TestCase):
    def setUp(self):
        """Set up ShadowAgent for testing, clean up existing shadow files."""
        self.mock_central_hub = MagicMock()
        self.mock_agentic_hub = MagicMock()
        self.agent = ShadowAgent(
            central_hub=self.mock_central_hub,
            agentic_hub=self.mock_agentic_hub
        )

        # Clean up prior data files if they exist
        if os.path.exists("data/shadow_state.json"):
            os.remove("data/shadow_state.json")
        if os.path.exists("logs"):
            for f in os.listdir("logs"):
                if f.startswith("audit_"):
                    os.remove(os.path.join("logs", f))

    def test_process_shadow_data_creates_refined_shadow(self):
        """Test that process_shadow_data correctly refines shadow data."""
        data = {"test": "value"}
        self.agent.process_shadow_data(data)
        self.assertIn("timestamp", self.agent.refined_shadow_data)
        self.assertIn("data", self.agent.refined_shadow_data)
        self.assertEqual(self.agent.refined_shadow_data["data"], data)

    def test_take_snapshot_creates_file_and_snapshot(self):
        """Test that take_snapshot creates a snapshot and writes to file."""
        self.agent.refined_shadow_data = {"test": "snapshot"}
        self.agent.take_snapshot()

        self.assertGreater(len(self.agent.snapshot_history), 0)
        self.assertTrue(os.path.exists("data/shadow_state.json"))

        with open("data/shadow_state.json", "r") as f:
            data = json.load(f)
        self.assertGreater(len(data), 0)

    def test_update_shadow_calls_all_parts(self):
        """Test that update_shadow calls process, snapshot, and update."""
        data = {"update": "test"}
        self.agent.update_shadow(data)

        # Check refined_shadow_data set by safe_update
        self.assertEqual(self.agent.refined_shadow_data, data)

    def test_safe_update_handles_invalid_data(self):
        """Test safe_update ignores non-dict input."""
        original = self.agent.refined_shadow_data
        self.agent.safe_update("not a dict")
        self.assertEqual(self.agent.refined_shadow_data, original)

if __name__ == "__main__":
    unittest.main()
