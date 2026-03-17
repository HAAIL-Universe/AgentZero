import logging
from interface.suggestion_display import SuggestionDisplay
from modules.conversation_parser import ConversationParser
from modules.user_profile_builder import UserProfileBuilder
from modules.routine_tracker import RoutineTracker

class OpenChatInterface:
    def __init__(self, fello_agent):
        self.logger = logging.getLogger("ChatInterface")
        self.agent = fello_agent
        self.training_mode = False
        self.suggestion_display = SuggestionDisplay()
        self.parser = ConversationParser()
        self.profile_builder = UserProfileBuilder()
        self.routine_tracker = RoutineTracker()

    def toggle_training_mode(self):
        self.training_mode = not self.training_mode
        status = "ON" if self.training_mode else "OFF"
        print(f"\n🧠 Routine Training Mode: {status}")

    async def launch(self):
        print("\n💬 Welcome to Open Chat Mode. Type 'train' to toggle routine learning.\n")

        while True:
            self.suggestion_display.display()

            user_input = input("🗣️ You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Ending open chat session.")
                break
            if user_input.lower() == "train":
                self.toggle_training_mode()
                continue

            if not user_input:
                continue

            # Phase 1: Parse and extract
            mood = self.parser.detect_mood(user_input)
            goals = self.parser.extract_goals(user_input)
            self.profile_builder.analyze_text(user_input)

            # Phase 2: Optional training questions
            if self.training_mode:
                routine_qs = [self.routine_tracker.get_next_question()]
                for q in routine_qs:
                    print(f"🧠 {q}")
                    ans = input("🗣️ You: ")
                    self.routine_tracker.log_answer(q, ans)

            # Phase 3: Reflect and respond
            reflection_prompt = (
                f"Open chat message:\n"
                f"{user_input}\n"
                "Extract insight and respond naturally with advice or guidance."
            )
            response = await self.agent.respond(reflection_prompt)

            print(f"\n🤖 FELLO: {response}\n")
