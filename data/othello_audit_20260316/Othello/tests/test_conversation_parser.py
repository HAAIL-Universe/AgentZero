import unittest
from core.conversation_parser import ConversationParser

class TestConversationParser(unittest.TestCase):
    def setUp(self):
        self.parser = ConversationParser()

    def test_extract_goals_detects_strong_goal(self):
        text = "I want to finish my project by Friday."
        goals = self.parser.extract_goals(text)
        self.assertTrue(any("want to" in g["source_phrase"] for g in goals))
        self.assertTrue(any("by friday" in (g["deadline"] or "").lower() for g in goals))

    def test_extract_goals_detects_progress_pattern(self):
        text = "I made progress on my goal of learning Python."
        goals = self.parser.extract_goals(text)
        self.assertTrue(any("progress-pattern" == g["source_phrase"] for g in goals))

    def test_extract_routines_detects_wake_up(self):
        text = "I woke up at 7 today."
        routines = self.parser.extract_routines(text)
        self.assertTrue(any("wake up" in r["question"].lower() for r in routines))

    def test_extract_traits_detects_trait(self):
        text = "I am compassionate and kind."
        traits = self.parser.extract_traits(text)
        self.assertIn("compassionate", traits)

    def test_extract_behaviors_detects_behavior(self):
        text = "I tend to avoid difficult tasks."
        behaviors = self.parser.extract_behaviors(text)
        self.assertIn("avoidant", behaviors)

    def test_detect_mood_positive(self):
        text = "I feel so happy and excited!"
        mood = self.parser.detect_mood(text)
        self.assertEqual(mood, "positive")

    def test_detect_mood_negative(self):
        text = "I am feeling tired and frustrated."
        mood = self.parser.detect_mood(text)
        self.assertEqual(mood, "negative")

    def test_extract_decisions_combines_data(self):
        text = "I want to exercise more. I am organized."
        decisions = self.parser.extract_decisions(text)
        self.assertTrue(any(d["type"] == "goal" for d in decisions))
        self.assertTrue(any(d["type"] == "trait" for d in decisions))

if __name__ == "__main__":
    unittest.main()
