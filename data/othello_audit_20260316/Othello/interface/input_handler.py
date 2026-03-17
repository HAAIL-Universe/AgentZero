# interface/input_handler.py

import logging

class InputHandler:
    def __init__(self):
        self.logger = logging.getLogger("InputHandler")

    def get_input(self) -> str:
        print("You (type your message, end with a blank line):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            except Exception as e:
                self.logger.error(f"Input error: {e}")
                break
        user_input = " ".join(lines).strip()
        self.logger.info(f"User said: {user_input}")
        return user_input

