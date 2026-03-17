# prompts.py

import random

def load_prompt(name):
    """🧾 Load a prompt string by name."""
    prompts = {
        "life_architect": "You are FELLO™, a helpful agentic AI for life guidance. Respond clearly and kindly."
    }
    return prompts.get(name, "Prompt not found.")

def generate_daily_prompt(mood, reflection, goal_update):
    try:
        mood = int(mood)
    except ValueError:
        mood = 5  # Default to neutral if user input isn't numeric

    if mood >= 8:
        tone = "🔥 High energy detected. Let's channel that!"
        prompts = [
            "What's one bold move you could make today?",
            "You're on fire — how can you use that energy to help someone else?",
            "Momentum's rare. Lock it in with a micro-win right now."
        ]
    elif 5 <= mood < 8:
        tone = "🔄 You're steady. Let's nudge things forward."
        prompts = [
            "What's one small task you’ve been avoiding that could unlock progress?",
            "You’re balanced. Where can you apply that to create traction today?",
            "Can you refine your plan — or simplify it — to make movement easier?"
        ]
    elif 3 <= mood < 5:
        tone = "🌥️ Something's weighing you down. Time to lighten the load."
        prompts = [
            "What emotion or thought is looping today — and what’s beneath it?",
            "If you could offload one mental weight, what would it be?",
            "Small wins matter most on heavy days. What’s one low-effort, high-reward task?"
        ]
    else:
        tone = "🌧️ Tough day. Let’s move gently."
        prompts = [
            "What do you need more than anything today — rest, space, or kindness?",
            "If today had a color, what would it be? Why?",
            "What’s one way you can show yourself grace today?"
        ]

    prompt = random.choice(prompts)

    return f"{tone}\n\n🧠 Reflective Prompt: {prompt}"
