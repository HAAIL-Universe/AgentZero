import logging
import random
from datetime import datetime


class NudgeDisplayManager:
    def __init__(self):
        self.logger = logging.getLogger("NudgeManager")
        self.default_nudges = [
            "💡 Small steps beat no steps.",
            "🧘 Breathe. Refocus. Try again.",
            "🚀 Build momentum, not pressure.",
            "🌱 What you repeat, you become.",
        ]

    def get_time_of_day(self):
        hour = datetime.now().hour
        if 5 <= hour < 11:
            return "morning"
        elif 11 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "late"

    def prioritize_nudges(self, mood=None, goals=None, spiral_risk=False):
        nudges = []

        # Mood-based nudges
        if mood:
            if mood in ["anxious", "stressed", "overwhelmed"]:
                nudges.append("🧘 It's okay to pause. Your peace is productive.")
            elif mood in ["low", "tired", "unmotivated"]:
                nudges.append("🌞 Energy is built, not found. One small win to start.")

        # Goal-related nudge
        if goals:
            nudges.append("🎯 Your goals remember you, even when you forget them.")

        # Spiral risk flag
        if spiral_risk:
            nudges.append("🌀 You’re entering a dip. Try a reset ritual today.")

        # Add a time-of-day message
        time_label = self.get_time_of_day()
        if time_label == "morning":
            nudges.append("🌅 New day. Clean slate. Anchor your intention.")
        elif time_label == "afternoon":
            nudges.append("☀️ Regroup. You still have time to move forward.")
        elif time_label == "evening":
            nudges.append("🌇 Reflect, then rest. You did better than you think.")
        else:
            nudges.append("🌙 Still awake? Honor your body’s rhythm when you can.")

        # Ensure at least one default
        if not nudges:
            nudges.append(random.choice(self.default_nudges))

        return nudges

    def display(self, mood=None, goals=None, spiral_risk=False):
        nudges = self.prioritize_nudges(mood, goals, spiral_risk)
        print("\n🔔 Suggested Nudges:")
        for n in nudges:
            print(f" - {n}")
