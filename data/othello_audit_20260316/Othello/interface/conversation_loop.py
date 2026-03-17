# interface/conversation_loop.py

from interface.input_handler import InputHandler
from interface.console_display import display_response, display_nudge_bar
from interface.response_router import route_input

input_handler = InputHandler()

def run_conversation():
    """Main conversation loop for FELLO."""
    display_nudge_bar()  # 🔄 Show dynamic insight before conversation starts
    print("FELLO is ready. Type 'exit' to quit.\n")

    while True:
        user_input = input_handler.get_input()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("FELLO: Goodbye for now.")
            break

        response = route_input(user_input)
        display_response(response)
