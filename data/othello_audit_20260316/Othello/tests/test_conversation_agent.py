import unittest
from unittest.mock import MagicMock, patch

from modules.agents.conversation_agent import ConversationAgent

class TestConversationAgent(unittest.TestCase):

    @patch("modules.agents.conversation_agent.ConversationParser")
    def setUp(self, mock_parser_class):
        self.mock_parser = MagicMock()
        mock_parser_class.return_value = self.mock_parser
        self.agent = ConversationAgent()

    def test_parse_delegates_to_parser(self):
        user_input = "I want to run daily and be more focused."
        self.mock_parser.extract_behaviors.return_value = ["motivated"]
        self.mock_parser.extract_goals.return_value = ["run daily"]
        self.mock_parser.extract_traits.return_value = ["focused"]
        self.mock_parser.extract_routines.return_value = ["exercise"]

        result = self.agent.parse(user_input)

        self.assertEqual(result["behaviors"], ["motivated"])
        self.assertEqual(result["goals"], ["run daily"])
        self.assertEqual(result["traits"], {"focused": 1})
        self.assertEqual(result["routines"], ["exercise"])

        self.mock_parser.extract_behaviors.assert_called_once_with(user_input)
        self.mock_parser.extract_goals.assert_called_once_with(user_input)
        self.mock_parser.extract_traits.assert_called_once_with(user_input)
        self.mock_parser.extract_routines.assert_called_once_with(user_input)

if __name__ == "__main__":
    unittest.main()
