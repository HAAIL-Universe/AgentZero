"""
interface/console_display.py

Handles console output formatting for FELLO.
Provides display helpers for responses, nudges, and other agent messages.
"""

def display_response(response_text: str):
    """
    Nicely formats and prints the agent's response to the user.
    """
    print(f"\n🤖 FELLO: {response_text}\n")


def display_nudge_bar(nudge_text: str):
    """
    Prints a visually distinct nudge bar to draw attention to nudges.
    """
    print("=" * 50)
    print(f"⚡ NUDGE: {nudge_text}")
    print("=" * 50)
