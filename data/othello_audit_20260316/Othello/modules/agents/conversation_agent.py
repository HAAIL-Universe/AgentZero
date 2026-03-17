from core.conversation_parser import ConversationParser
import logging

logging.basicConfig(level=logging.DEBUG)

class ConversationAgent:
    """
    Central interface for parsing user input and providing structured data.
    """

    def __init__(self, hub=None):
        if hub is None:
            raise ValueError("CentralHub must be provided to ConversationAgent.")
        self.hub = hub
        self.parser = ConversationParser()

    def parse(self, user_input: str) -> dict:
        logging.debug(f"ConversationAgent: Received user input: {user_input}")
        parsed_data = self.parser.parse(user_input)
        logging.debug(f"ConversationAgent: Parsed data: {parsed_data}")
        self.hub.receive_parsed_data(parsed_data)
        logging.debug(f"ConversationAgent: Released data to CentralHub: {parsed_data}")
        return parsed_data
