import time
import random
import logging
from core.llm_wrapper import LLMWrapper

import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")

logging.debug("This is a debug message.")
logging.info("This is an info message.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")
logging.critical("This is a critical message.")

# Simulated user phrases, as requested
sample_inputs = [
    {"text": "I went for a run this morning.", "mood": "energized", "context": "morning"},
    {"text": "Skipped breakfast again.", "mood": "tired", "context": "morning"},
    {"text": "I’ve been avoiding my tasks all week.", "mood": "avoidant", "context": "work"},
    {"text": "Finally finished my project!", "mood": "accomplished", "context": "work"},
    {"text": "Slept in until noon.", "mood": "lazy", "context": "home"},
    {"text": "Had my usual coffee at 7am.", "mood": "neutral", "context": "morning"},
    {"text": "Missed my workout today.", "mood": "disappointed", "context": "evening"},
    {"text": "Going to bed early tonight.", "mood": "relaxed", "context": "night"},
    {"text": "Worked straight through lunch.", "mood": "drained", "context": "work"},
    {"text": "I’m feeling resilient after a tough week.", "mood": "resilient", "context": "reflection"},
    {"text": "I’m really curious about new projects.", "mood": "curious", "context": "work"},
    {"text": "My patience is running out.", "mood": "impatient", "context": "frustrated"},
    {"text": "I’m calm despite the stress.", "mood": "calm", "context": "stress"},
    {"text": "Fuck this, I’m sick of it!", "mood": "frustrated", "context": "work"},
    {"text": "Can you hurry up already?", "mood": "impatient", "context": "waiting"},
    {"text": "All good, take your time.", "mood": "calm", "context": "waiting"},
    {"text": "Whatever, just do it!", "mood": "impatient", "context": "command"},
    {"text": "I want to run a marathon next year.", "mood": "motivated", "context": "planning"},
    {"text": "Maybe I should quit and move to Bali.", "mood": "restless", "context": "life"},
    {"text": "Let’s check in on my weekly goals.", "mood": "reflective", "context": "planning"},
    {"text": "Didn’t hit my target, but I tried.", "mood": "accepting", "context": "reflection"},
    {"text": "I’m so happy and yet I feel lost.", "mood": "confused", "context": "emotion"},
    {"text": "Super productive, but nothing got done.", "mood": "mixed", "context": "work"},
    {"text": "Calm and angry at the same time.", "mood": "conflicted", "context": "emotion"},
    {"text": "Avoiding work but really want to succeed.", "mood": "ambivalent", "context": "work"},
    {"text": "Why do I keep making the same mistakes?", "mood": "frustrated", "context": "reflection"},
    {"text": "Today was a win!", "mood": "victorious", "context": "evening"},
    {"text": "Zero motivation right now.", "mood": "unmotivated", "context": "morning"},
    {"text": "I feel like giving up, but I know I won’t.", "mood": "determined", "context": "night"},
]

def main():
    llm = LLMWrapper(model="gpt-4o")  # or whatever you want to use

    for i, input_data in enumerate(sample_inputs):
        user_text = input_data["text"]
        print(f"\n[User]: {user_text}")
        # Send directly to the LLM as if user typed it
        logging.debug("Sending request to LLM...")
        response = llm.generate(user_text)
        logging.debug(f"Received response: {response}")
        print(f"[LLM]: {response}")
        time.sleep(2)

if __name__ == "__main__":
    main()
