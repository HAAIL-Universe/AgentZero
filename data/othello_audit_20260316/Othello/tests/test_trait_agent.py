import unittest
from datetime import datetime, timedelta
from modules.agents.trait_agent import TraitAgent  # Adjust this path if your file lives elsewhere

class TestTraitAgent(unittest.TestCase):

    def setUp(self):
        self.agent = TraitAgent()

    def test_add_new_trait(self):
        """Test that a new trait is added correctly."""
        self.agent.update_trait("resilience", 1.5, context="initial", source="test")
        self.assertIn("resilience", self.agent.traits)
        self.assertEqual(self.agent.traits["resilience"]["score"], 1.5)

    def test_update_existing_trait(self):
        """Test that updating a trait increases its score and updates source."""
        self.agent.update_trait("curiosity", 2.0, context="initial", source="test")
        self.agent.update_trait("curiosity", 3.0, context="follow-up", source="test2")
        self.assertGreaterEqual(self.agent.traits["curiosity"]["score"], 5.0)

    def test_decay_trait_score_over_time(self):
        """Test decayed trait scoring logic (future ML extension stub)."""
        self.agent.update_trait("confidence", 2.0, context="initial", source="test")
        self.agent.traits["confidence"]["last_updated"] = (datetime.now() - timedelta(days=7)).isoformat()
        decayed_score = self.agent.traits["confidence"]["score"]  # No decay logic active in current version
        self.assertEqual(decayed_score, 2.0)  # Placeholder until decay is implemented in update logic

    def test_get_all_traits_returns_dict(self):
        """Check that traits summary returns a dict."""
        self.agent.update_trait("focus", 1.0, context="check", source="test")
        traits = self.agent.get_all_traits()
        self.assertIsInstance(traits, dict)
        self.assertIn("focus", traits)

    def test_trait_context_structure(self):
        """Validate trait context and fields structure."""
        self.agent.update_trait("adaptability", 2.0, context="initial", source="test")
        context = self.agent.get_trait_context()
        self.assertIn("adaptability", context)
        self.assertIn("score", context["adaptability"])
        self.assertIn("sources", context["adaptability"])

    def test_snapshot_output(self):
        """Verify snapshot includes expected traits."""
        self.agent.update_trait("integrity", 2.5, context="initial", source="test")
        snapshot = self.agent.build_snapshot()
        self.assertIn("traits", snapshot)
        self.assertIn("integrity", snapshot["traits"])

if __name__ == "__main__":
    unittest.main()
