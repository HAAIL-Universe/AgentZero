import unittest
from unittest.mock import MagicMock, AsyncMock
import asyncio
from modules.agentic_agents.architect_agent import ArchitectAgent

class TestArchitectAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Mock model with async chat method
        self.mock_model = MagicMock()
        self.mock_model.chat = AsyncMock(return_value="Mock response")

        # Mock hubs
        self.mock_central_hub = MagicMock()
        self.mock_agentic_hub = MagicMock()
        self.mock_agentic_hub.memory = MagicMock()
        self.mock_agentic_hub.update_shadow = MagicMock()

        self.agent = ArchitectAgent(model=self.mock_model, central_hub=self.mock_central_hub, agentic_hub=self.mock_agentic_hub)

    async def test_design_agent(self):
        result = self.agent.design_agent()
        self.assertIn("Designing agent based on", result)

    async def test_set_autonomy_level_valid(self):
        self.agent.set_autonomy_level('active')
        self.assertEqual(self.agent.autonomy_level, 'active')

    async def test_set_autonomy_level_invalid(self):
        with self.assertRaises(ValueError):
            self.agent.set_autonomy_level('invalid')

    async def test_add_and_get_goals(self):
        self.agent.add_goal("Test goal")
        goals = self.agent.get_goals()
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["description"], "Test goal")

    async def test_edit_goal(self):
        self.agent.add_goal("Test goal")
        success = self.agent.edit_goal(0, new_description="Updated goal", new_status="completed")
        self.assertTrue(success)
        goal = self.agent.get_goals()[0]
        self.assertEqual(goal["description"], "Updated goal")
        self.assertEqual(goal["status"], "completed")

    async def test_edit_goal_out_of_bounds(self):
        success = self.agent.edit_goal(99, new_description="Doesn't exist")
        self.assertFalse(success)

    async def test_plan_and_execute_passive(self):
        self.agent.set_autonomy_level('passive')
        result = await self.agent.plan_and_execute("What is my plan?")
        self.assertEqual(result, "Mock response")
        self.mock_agentic_hub.update_shadow.assert_not_called()

    async def test_plan_and_execute_active(self):
        self.agent.set_autonomy_level('active')
        result = await self.agent.plan_and_execute("What is my plan?")
        self.assertEqual(result, "Mock response")
        self.mock_agentic_hub.update_shadow.assert_called_once()

    async def test_set_memory_state_toggle(self):
        # Disable memory
        self.agent.set_memory_state(False)
        self.assertIsNone(self.agent.memory)

        # Enable memory (should reattach to agentic hub memory)
        self.agent.set_memory_state(True)
        self.assertIs(self.agent.memory, self.mock_agentic_hub.memory)

if __name__ == '__main__':
    unittest.main()
