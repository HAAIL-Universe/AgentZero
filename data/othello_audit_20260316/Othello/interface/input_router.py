from interface.training_mode_controller import TrainingModeController
from modules.user_profile_builder import UserProfileBuilder
from core.agent import FelloAgent
from modules.conversation_parser import ConversationParser

class InputRouter:
    def __init__(self, fello_agent: FelloAgent, training_controller: TrainingModeController):
        self.fello = fello_agent
        self.training_controller = training_controller
        self.parser = ConversationParser()
        self.profile_builder = UserProfileBuilder()

    async def handle_input(self, user_input: str) -> str:
        if self.training_controller.is_active():
            return await self._handle_training_mode(user_input)
        else:
            return await self.fello.respond(user_input)

    async def _handle_training_mode(self, user_input: str) -> str:
        # Enrich user profile and look for routine data
        traits = self.profile_builder.analyze_text(user_input)
        routines = self.parser.extract_routines(user_input)

        if routines:
            print("📅 Detected Routine Patterns:")
            for r in routines:
                print(f" - {r}")

        summary = self.profile_builder.summarize_profile()

        print("\n🧠 Updated Profile Traits:")
        for trait, score in summary["top_traits"]:
            print(f" - {trait}: {score}")

        print("\n🔖 Identity Phrases:")
        for line in summary["identity_lines"]:
            print(f" - {line.strip()}")

        return await self.fello.respond(
            "Training mode conversation.\n"
            f"User said: {user_input}\n"
            "Extract: personality traits, routines, suggestions for habit scaffolding."
        )
