from modules.central_hub import CentralHub
from othello import Othello
from fello import Fello

class MainAgent:
    def __init__(self):
        self.hub = CentralHub()
        self.othello = Othello()
        self.fello = Fello()

    def run_daily_check_in(self):
        # Example flow using CentralHub
        print("[MainAgent] Running daily check-in...")

        # Update shadow with some example data
        example_shadow_data = {"mood": "neutral", "energy": "steady"}
        self.hub.update_shadow(example_shadow_data)

        # Run reflection
        reflection_result = self.hub.run_reflection()
        if reflection_result:
            print(f"[MainAgent] Reflection result: {reflection_result}")

        # Update goals (example goal data)
        example_goal = {"goal": "Stay focused", "priority": "high"}
        self.hub.update_goals(example_goal)

        # Pass data up to Fellow for creative suggestions
        self.fello.process_reflection(reflection_result)

        # Final pass through Othello gatekeeper
        self.othello.evaluate_and_safeguard(reflection_result)

        print("[MainAgent] Daily check-in complete.")

if __name__ == "__main__":
    agent = MainAgent()
    agent.run_daily_check_in()
