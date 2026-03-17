import time
import random
import os
import json
import asyncio
import yaml
from datetime import datetime, timezone
from fello import Fello
from modules.central_hub import CentralHub
from modules.agentic_hub import AgenticHub

def load_consent_config(config_path="config/consent.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tier = config.get("tier", 1)
    autonomy = "active" if tier == 3 else "suggestive" if tier == 2 else "passive"
    features = config.get("feature_toggles", {})
    return tier, autonomy, features

# Sample Inputs: Simulated user entries.
sample_inputs = [
    # Behavior & Habits
    {"text": "I went for a run this morning.", "mood": "energized", "context": "morning"},
    {"text": "Skipped breakfast again.", "mood": "tired", "context": "morning"},
    {"text": "I’ve been avoiding my tasks all week.", "mood": "avoidant", "context": "work"},
    {"text": "Finally finished my project!", "mood": "accomplished", "context": "work"},
    {"text": "Slept in until noon.", "mood": "lazy", "context": "home"},

    # Routines
    {"text": "Had my usual coffee at 7am.", "mood": "neutral", "context": "morning"},
    {"text": "Missed my workout today.", "mood": "disappointed", "context": "evening"},
    {"text": "Going to bed early tonight.", "mood": "relaxed", "context": "night"},
    {"text": "Worked straight through lunch.", "mood": "drained", "context": "work"},

    # Traits
    {"text": "I’m feeling resilient after a tough week.", "mood": "resilient", "context": "reflection"},
    {"text": "I’m really curious about new projects.", "mood": "curious", "context": "work"},
    {"text": "My patience is running out.", "mood": "impatient", "context": "frustrated"},
    {"text": "I’m calm despite the stress.", "mood": "calm", "context": "stress"},

    # Impatience/Emotion
    {"text": "Fuck this, I’m sick of it!", "mood": "frustrated", "context": "work"},
    {"text": "Can you hurry up already?", "mood": "impatient", "context": "waiting"},
    {"text": "All good, take your time.", "mood": "calm", "context": "waiting"},
    {"text": "Whatever, just do it!", "mood": "impatient", "context": "command"},

    # Aspirational/Goals
    {"text": "I want to run a marathon next year.", "mood": "motivated", "context": "planning"},
    {"text": "Maybe I should quit and move to Bali.", "mood": "restless", "context": "life"},
    {"text": "Let’s check in on my weekly goals.", "mood": "reflective", "context": "planning"},
    {"text": "Didn’t hit my target, but I tried.", "mood": "accepting", "context": "reflection"},

    # Contradictory/Mixed
    {"text": "I’m so happy and yet I feel lost.", "mood": "confused", "context": "emotion"},
    {"text": "Super productive, but nothing got done.", "mood": "mixed", "context": "work"},
    {"text": "Calm and angry at the same time.", "mood": "conflicted", "context": "emotion"},
    {"text": "Avoiding work but really want to succeed.", "mood": "ambivalent", "context": "work"},

    # Misc/Edge
    {"text": "Why do I keep making the same mistakes?", "mood": "frustrated", "context": "reflection"},
    {"text": "Today was a win!", "mood": "victorious", "context": "evening"},
    {"text": "Zero motivation right now.", "mood": "unmotivated", "context": "morning"},
    {"text": "I feel like giving up, but I know I won’t.", "mood": "determined", "context": "night"},
]

# Log directories (these are for logging only, do NOT inject or patch any agent state).
SIM_LOG_DIR = "sim_logs"
DEBUG_LOG_DIR = "debug_logs"
SUMMARY_LOG_DIR = "summary_logs"
os.makedirs(SIM_LOG_DIR, exist_ok=True)
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
os.makedirs(SUMMARY_LOG_DIR, exist_ok=True)

async def main():
    

    print("[Simulation] Starting 20-minute emergence loop...")
    end_time = time.time() + 20 * 60

    while time.time() < end_time:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        input_data = random.choice(sample_inputs)

        # Consent & features for this cycle
        consent_tier, autonomy_level, feature_toggles = load_consent_config()
        print(f"Consent Level {consent_tier}, Autonomy Set {autonomy_level}. Features: {feature_toggles}")

        # Inject feature_toggles into user_data (if you want agents to see it)
        user_input = {**input_data, "feature_toggles": feature_toggles}
        # Init hubs and main entry agent.
        central_hub = CentralHub()
        agentic_hub = AgenticHub()
        fello = Fello(central_hub=central_hub, agentic_hub=agentic_hub, model="gpt-4")
        # Pass only allowed args to run_daily_check_in
        result = await fello.run_daily_check_in(
            consent_tier=consent_tier,
            autonomy_level=autonomy_level,
            user_data=user_input
        )
        # --- Output, no agent state hacks ---
        sim_log_path = os.path.join(SIM_LOG_DIR, f"emergence_{timestamp}.json")
        debug_log_path = os.path.join(DEBUG_LOG_DIR, f"emergence_{timestamp}_debug.txt")
        summary_log_path = os.path.join(SUMMARY_LOG_DIR, f"emergence_{timestamp}_summary.txt")

        with open(sim_log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2))

        with open(summary_log_path, "w", encoding="utf-8") as f:
            f.write(f"--- SUMMARY {timestamp} ---\n")
            f.write(f"Mood: {input_data['mood']}\n")
            f.write(f"Text: {input_data['text']}\n")
            f.write(f"Reflection: {result.get('reflection', 'N/A')}\n")
            f.write(f"Behavioral Insight: {result.get('behavioral_insights', {}).get('summary_notes', 'N/A')}\n")
            f.write(f"Psyche: {result.get('psyche_insights', {})}\n")
            f.write(f"Routine: {result.get('routine_snapshot', {})}\n")
            f.write(f"Impatience: {result.get('impatience_result', {}).get('impatience_level', 'N/A')}\n")
            f.write(f"Delta: {result.get('delta', 'N/A')}\n")
            f.write(f"Anomalies: {result.get('anomalies', 'N/A')}\n")
            f.write(f"Decision Vault: {result.get('decision_analysis', 'N/A')}\n")

        print(f"[Cycle Complete] Logged @ {timestamp}")
        await asyncio.sleep(5)

    agentic_hub.memory["last_run"] = datetime.now().isoformat()
    agentic_hub.save_memory()
    print("[AgenticHub] Memory saved after simulation.")
    print("[Simulation] Complete. Logs saved.")

if __name__ == "__main__":
    asyncio.run(main())
