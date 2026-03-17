from modules.conversation_parser import ConversationParser
from modules.user_profile_builder import UserProfileBuilder
from modules.reflective_response import ReflectiveResponse
from prism.prism_engine import PRISMEngine
from core.llm_wrapper import LLMWrapper
from modules.trait_manager import TraitManager
from modules.routine_tracker import RoutineTracker
from modules.decision_vault import DecisionVault

import json
from pathlib import Path
from datetime import datetime
import uuid  # For unique goal IDs

# === Passive Pending Goals Setup ===
PENDING_GOALS_FILE = Path("data/pending_goals.json")
pending_goals = []
PENDING_GOALS_FILE.parent.mkdir(exist_ok=True)
if PENDING_GOALS_FILE.exists():
    with open(PENDING_GOALS_FILE, "r", encoding="utf-8") as f:
        pending_goals = json.load(f)

# === Persistent Facet Setup ===
trait_manager = TraitManager()
routine_tracker = RoutineTracker()
decision_vault = DecisionVault()
parser = ConversationParser()
profile_builder = UserProfileBuilder()
reflector = ReflectiveResponse()
prism = PRISMEngine()
llm = LLMWrapper()

def should_interject(goals, traits, user_input):
    """
    If a goal is detected, generate an agentic nudge to offer tracking/support.
    """
    if goals:
        return (
            "\n\nFELLO Insight: I noticed you mentioned a goal or a deadline. "
            "Would you like to track this, set a reminder, or build a roadmap? "
            "I can nudge you when you want help."
        )
    return ""

def add_or_update_pending_goal(goal_text, user_input, mood, pending_goals):
    """
    Add a new pending goal, or update an existing one.
    Keeps full context history, affirmation count, denied/edit tracking, etc.
    (This should later move to its own PendingGoalManager module.)
    """
    now = datetime.now().isoformat()
    # Try to find a match (case-insensitive, exact match for now)
    match = None
    for g in pending_goals:
        if goal_text.lower().strip() == g["description"].lower().strip():
            match = g
            break

    if match:
        # Already present: stack affirmations, add to context history, update timestamps, mood
        match["affirmation_count"] = match.get("affirmation_count", 1) + 1
        match["last_mentioned"] = now
        # Add a context history object for every mention
        match.setdefault("contexts", []).append({"text": user_input, "timestamp": now, "mood": mood})
        match["mood_at_mention"] = mood
        print(f"[Passive Goal Capture] Pending goal re-affirmed: {goal_text} (count: {match['affirmation_count']})")
    else:
        # Brand new goal: create and initialize all fields
        new_id = str(uuid.uuid4())
        pending_goals.append({
            "id": new_id,
            "description": goal_text,
            "affirmation_count": 1,
            "contexts": [{"text": user_input, "timestamp": now, "mood": mood}],
            "first_mentioned": now,
            "last_mentioned": now,
            "denied_count": 0,  # Upgrade-ready
            "edit_history": [],  # Upgrade-ready
            "related_to": [],    # Upgrade-ready (for future trees/graphs)
            "status": "pending",
            "mood_at_mention": mood
        })
        print(f"[Passive Goal Capture] Pending goal added: {goal_text}")

    # Save updated pending goals to disk (always after any add/update)
    with open(PENDING_GOALS_FILE, "w", encoding="utf-8") as f:
        json.dump(pending_goals, f, indent=4)

def route_input(user_input: str) -> str:
    """
    Main conversational pipeline:
    - User input goes to LLM for friendly reply.
    - Agentic pipeline parses for mood, goals, traits, routines, and logs all to respective state engines.
    - Any detected goals are stored instantly as pending goals (with context, affirmation counts, and upgrade hooks).
    - Interjects if past decision patterns or soft cues are detected.
    """
    # === Platinum+ Passive Decision Logging ===
    decision_vault.passive_log_from_input(user_input)

    # 1. LLM reply first (natural, uninterrupted chat)
    llm_reply = llm.generate(
        prompt=user_input,
        system_prompt="You are FELLO, a loyal, psychology-powered life assistant. Reply helpfully and supportively."
    )

    # 2. Background agentic processing
    mood = parser.detect_mood(user_input)
    goals = parser.extract_goals(user_input)
    traits = parser.extract_traits(user_input)
    routines = parser.extract_routines(user_input)
    profile_builder.analyze_text(user_input)
    _ = prism.run_daily_update(mood=mood, user_text=user_input, goal_streak=0)

    # --- Passive pending goals logic (platinum-level, modular, upgrade-ready) ---
    for goal_text in goals:
        add_or_update_pending_goal(goal_text, user_input, mood, pending_goals)

    # --- Persistent trait logging ---
    if traits:
        trait_manager.record_traits(traits, context=user_input)

    # --- Persistent routine logging ---
    if routines:
        for routine_type in routines:
            routine_tracker.log_routine_detected(routine_type, user_input)
        # Print handled in RoutineTracker

    # --- DecisionVault: Interject if repeated decision detected ---
    decision_matches = decision_vault.match_past_pattern(user_input)
    if decision_matches:
        insight_lines = ["\n\n🗂️ FELLO Insight: You've faced a similar decision before:"]
        for match in decision_matches:
            insight_lines.append(
                f"- [{match['timestamp'][:10]}] '{match['situation']}' | Option chosen: {match['chosen_option']} | Outcome: {match.get('outcome', 'N/A')}"
            )
            if match.get("reflection"):
                insight_lines.append(f"    Reflection: {match['reflection']}")
        insight_lines.append("Want to run a scenario or review other possible outcomes? Just ask!")
        # Show as nudge under LLM reply (always appears if a match)
        return llm_reply + "\n" + "\n".join(insight_lines)

    # --- Regular agentic interjection (goal/trait etc) ---
    interjection = should_interject(goals, traits, user_input)
    return llm_reply + interjection

