import random
import datetime

class ConversationStarter:
    """
    Handles context-aware opening and follow-up prompts for FELLO.
    Designed for modular use in daily check-ins, journaling, or nudges.
    """

    def __init__(self):
        """
        Initializes a list of morning prompts for user engagement at start of day.
        """
        self.morning_prompts = [
            "Good morning! What’s the vibe today?",
            "How are you feeling as you start the day?",
            "What’s on your mind this morning?",
            "Got any goals or plans for the day ahead?",
            "Tell me about today — anything important happening?",
            "What’s your energy like this morning?"
        ]
    
    def get_morning_prompt(self):
        """
        Returns a morning prompt based on the current time.
        If called before noon, picks a random 'morning' message.
        If called after noon, prompts user for midday reflection.
        """
        now = datetime.datetime.now()
        hour = now.hour

        # If before noon, give a morning check-in. Otherwise, offer a midday nudge.
        if hour < 12:
            return random.choice(self.morning_prompts)
        else:
            return "Want to reflect a bit on how today’s going so far?"

    def get_followup_prompt(self, user_response):
        """
        Suggests a follow-up message based on the user's initial response length.
        - Short replies get gentle encouragement.
        - Longer replies get supportive feedback.
        Extend this method with NLP for tone/emotion-based prompts.
        
        Args:
            user_response (str): The user's previous reply.

        Returns:
            str: A follow-up prompt for conversation flow.
        """
        if len(user_response.strip()) < 10:
            return "Feel free to give me as much or as little as you like — even a brain dump helps."
        return "Thanks for sharing. Let’s unpack that together if you’d like."
