import logging
from core.architect_brain import Architect


class FelloAgent:
    def __init__(self, model):
        self.logger = logging.getLogger("FELLO")
        self.architect = Architect(model=model)  # Only pass model!

    async def respond(self, user_input: str) -> str:
        """🎯 Primary method to get a response from FELLO."""
        self.logger.info(f"📥 Input received: {user_input}")
        reply = await self.architect.plan_and_execute(user_input)
        self.logger.info(f"📤 Response sent: {reply}")
        return reply

    def set_memory_window(self, window_size: int):
        """Set RAM-only memory window for agent context."""
        self.architect.set_memory_window(window_size)

    def clear_short_term_memory(self):
        """Clear short-term in-RAM memory window (not long-term data)."""
        self.architect.clear_short_term_memory()

