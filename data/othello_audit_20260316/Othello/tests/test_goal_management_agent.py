import unittest
import os
import json
from unittest.mock import MagicMock
from modules.agentic_agents.goal_management_agent import GoalManagementAgent

class TestGoalManagementAgent(unittest.TestCase):

    def setUp(self):
        # Remove existing test files to start fresh
        if os.path.exists("data/goals.json"):
            os.remove("data/goals.json")
        if os.path.exists("data/archived_goals.json"):
            os.remove("data/archived_goals.json")

        self.mock_central_hub = MagicMock()
        self.mock_agentic_hub = MagicMock()

        self.agent = GoalManagementAgent(
            central_hub=self.mock_central_hub,
            agentic_hub=self.mock_agentic_hub
        )

    def tearDown(self):
        if os.path.exists("data/goals.json"):
            os.remove("data/goals.json")
        if os.path.exists("data/archived_goals.json"):
            os.remove("data/archived_goals.json")

    def test_add_goal_creates_and_saves_goal(self):
        self.agent.add_goal("Test Goal 1")
        self.assertEqual(len(self.agent.goals), 1)
        self.assertEqual(self.agent.goals[0]["description"], "Test Goal 1")

        with open("data/goals.json", "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["description"], "Test Goal 1")

    def test_add_or_edit_goal_adds_new_if_not_exists(self):
        self.agent.add_or_edit_goal({"description": "Unique Goal"})
        self.assertEqual(len(self.agent.goals), 1)
        self.assertEqual(self.agent.goals[0]["description"], "Unique Goal")

    def test_add_or_edit_goal_edits_existing(self):
        self.agent.add_goal("Editable Goal", tags=["start"])
        self.agent.add_or_edit_goal({
            "description": "Editable Goal",
            "tags": ["updated"],
            "deadline": "2025-12-31"
        })

        goal = self.agent.goals[0]
        self.assertIn("updated", goal["tags"])
        self.assertEqual(goal["deadline"], "2025-12-31")

    def test_load_and_save_goals_work_correctly(self):
        self.agent.add_goal("Persistent Goal")
        new_agent = GoalManagementAgent(
            central_hub=self.mock_central_hub,
            agentic_hub=self.mock_agentic_hub
        )
        self.assertEqual(len(new_agent.goals), 1)
        self.assertEqual(new_agent.goals[0]["description"], "Persistent Goal")

    def test_load_archived_goals_empty_on_first_run(self):
        self.assertEqual(len(self.agent.archived_goals), 0)

if __name__ == "__main__":
    unittest.main()
