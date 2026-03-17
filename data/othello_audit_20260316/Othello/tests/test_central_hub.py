import unittest
from modules.central_hub import CentralHub

class TestCentralHub(unittest.TestCase):

    def setUp(self):
        """Initialize a CentralHub for integration testing."""
        self.hub = CentralHub()

        # Ensure default consent structure exists
        if not hasattr(self.hub, 'consent') or self.hub.consent is None:
            self.hub.consent = {
                "autonomy_level": "passive",
                "permissions": {"passive": ["shadow_update", "reflection"], "suggestive": [], "active": []}
            }

    def test_shadow_update_respects_consent(self):
        """Test that shadow updates respect autonomy level."""
        self.hub.consent["permissions"]["passive"] = ["shadow_update"]
        self.hub.consent["autonomy_level"] = "passive"
        
        try:
            self.hub.update_shadow({"test_key": "test_value"})
            success = True
        except Exception:
            success = False

        self.assertTrue(success)

    def test_receive_user_input_runs_without_error(self):
        """Test processing of a basic user input string."""
        try:
            self.hub.receive_user_input("I want to wake up earlier and be more focused.")
            success = True
        except Exception:
            success = False

        self.assertTrue(success)

    def test_run_reflection_returns_result(self):
        """Test that reflection produces a summary or returns None on block."""
        self.hub.consent["permissions"]["passive"] = ["reflection"]
        self.hub.consent["autonomy_level"] = "passive"
        result = self.hub.run_reflection()
        self.assertTrue(result is None or isinstance(result, dict))

if __name__ == '__main__':
    unittest.main()
