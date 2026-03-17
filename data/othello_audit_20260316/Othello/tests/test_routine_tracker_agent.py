import os
import json
import unittest
from datetime import datetime, timedelta
from modules.agents.routine_tracker_agent import RoutineTrackerAgent  # Adjust path if needed

class TestRoutineTrackerAgent(unittest.TestCase):

    def setUp(self):
        self.test_file = "data/routines.json"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        self.agent = RoutineTrackerAgent()

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_question_queue_initialization(self):
        self.assertIsInstance(self.agent.question_queue, list)
        self.assertGreater(len(self.agent.question_queue), 0)

    def test_get_next_question_recycles_queue(self):
        questions = [self.agent.get_next_question() for _ in range(6)]
        self.assertIn(questions[-1], self.agent._init_question_queue())

    def test_log_answer_appends_correctly(self):
        question = "What time did you wake up today?"
        self.agent.log_answer(question, "7:00 AM", mood="refreshed", trait_link="discipline", goal_link="consistency")
        data = self.agent.get_routines()
        self.assertIsInstance(data, list)
        self.assertTrue(any(entry["question"] == question for entry in data))

    def test_extract_routine_type_logic(self):
        self.assertEqual(self.agent._extract_routine_type("When did you wake up?"), "wake_up")
        self.assertEqual(self.agent._extract_routine_type("Did you eat breakfast?"), "morning_intake")
        self.assertEqual(self.agent._extract_routine_type("Time for focus block?"), "focus_block")
        self.assertEqual(self.agent._extract_routine_type("Did you go to the gym?"), "activity")
        self.assertEqual(self.agent._extract_routine_type("Do you wind down now?"), "wind_down")

    def test_analyze_routines_empty(self):
        result = self.agent.analyze_routines()
        self.assertIn("suggestions", result)
        self.assertEqual(result["overall_status"], "stable")

    def test_set_routines_bulk_logging(self):
        routines = [
            {"question": "Did you go to the gym?", "answer": "Yes"},
            {"question": "What time did you wake up today?", "answer": "6:30 AM"}
        ]
        self.agent.set_routines(routines)
        data = self.agent.get_routines()
        self.assertEqual(len(data), 2)

    def test_streak_tracking_logic(self):
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Make sure routine_data is a list
        if not isinstance(self.agent.routine_data, list):
            self.agent.routine_data = []

        # Add yesterday's entry
        entry_yesterday = {
            "routine_type": "wake_up",
            "date": yesterday,
            "question": "What time did you wake up today?",
            "answer": "7:00 AM"
        }
        self.agent.routine_data.append(entry_yesterday)
        self.agent._update_streak("wake_up", entry_yesterday)

        # Add today's entry
        entry_today = {
            "routine_type": "wake_up",
            "date": today,
            "question": "What time did you wake up today?",
            "answer": "7:30 AM"
        }
        self.agent.routine_data.append(entry_today)
        self.agent._update_streak("wake_up", entry_today)

        self.assertEqual(self.agent.streaks.get("wake_up"), 2)

if __name__ == "__main__":
    unittest.main()
