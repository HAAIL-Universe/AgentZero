import logging
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from typing import Optional

from core.architect_brain import Architect  # Import your agent (actual location)
from core.llm_wrapper import AsyncLLMWrapper  # Async wrapper for the LLM
from utils.postprocessor import postprocess_and_save  # Analysis-only

from dotenv import load_dotenv
load_dotenv()


def is_goal_list_request(text: str) -> bool:
    """
    Heuristic: treat certain phrases as 'list my goals' instead of a new goal.
    """
    t = text.lower()
    return (
        "goal" in t
        and any(
            phrase in t
            for phrase in [
                "what goals",
                "my goals",
                "list of goals",
                "show my goals",
                "what are my goals",
            ]
        )
    )


def format_goal_list(goals) -> str:
    """
    Turn the GoalManager list into a human-friendly reply string.
    """
    if not goals:
        return (
            "You don't have any goals stored yet. "
            "You can set one by saying something like: "
            "'My goal today is improvement.'"
        )

    lines = ["Here are your current goals:"]
    for g in goals:
        goal_id = g.get("id", "?")
        text = g.get("text", "").strip() or "(empty goal text)"
        deadline = g.get("deadline")
        if deadline:
            lines.append(f"- Goal #{goal_id}: {text} (deadline: {deadline})")
        else:
            lines.append(f"- Goal #{goal_id}: {text}")

    return "\n".join(lines)


def parse_goal_selection_request(text: str) -> Optional[int]:
    """
    Try to parse messages like:
    - "let's talk about goal 1"
    - "focus on no.1"
    - "work on goal number 2"
    """
    t = text.lower()
    if not any(k in t for k in ["goal", "no.", "no ", "number"]):
        return None

    # primary pattern: 'goal 1', 'no.1', 'number 2'
    m = re.search(r"\b(?:goal|no\.?|number)\s*(\d+)\b", t)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None

    # fallback: any standalone number, e.g. "let's talk about 1"
    m2 = re.search(r"\b(\d+)\b", t)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            return None

    return None


# === Model selector at startup ===
def pick_model():
    models = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
    ]
    print("Which model would you like FELLO to use this session?")
    for idx, model in enumerate(models, 1):
        print(f"{idx}: {model}")
    while True:
        choice = input(f"Enter 1-{len(models)} [default=4]: ").strip()
        if not choice:
            return models[3]  # default to GPT-4o
        try:
            i = int(choice) - 1
            if 0 <= i < len(models):
                return models[i]
        except Exception:
            pass
        print("Invalid selection, try again.")


selected_model = pick_model()
print(f"FELLO will use: {selected_model}")

# ===== Instantiate Architect with OpenAI Model =====
openai_model = AsyncLLMWrapper(model=selected_model)
architect_agent = Architect(model=openai_model)

app = Flask(__name__)
CORS(app)  # Allow requests from frontend


@app.route("/api/message", methods=["POST"])
def handle_message():
    """
    Main endpoint for user interaction:
    1. Optionally intercept 'list my goals' and answer from GoalManager.
    2. Optionally intercept 'talk about goal X' and set active_goal_id.
    3. Runs postprocessor to extract goals/traits/routines from input (analysis only).
    4. Builds goal context (if an active goal exists) and calls agent planner.
    5. Logs this exchange into the active goal's conversation log.
    6. Returns reply to frontend.
    """
    data = request.get_json()
    user_input = data.get("message", "") or ""

    # --- Shortcut 1: user is asking for their goals; answer directly -----
    if is_goal_list_request(user_input):
        goals = architect_agent.goal_mgr.list_goals()
        reply_text = format_goal_list(goals)
        return jsonify(
            {
                "reply": reply_text,
                "goals": goals,
                "meta": {
                    "source": "goal_manager",
                    "intent": "list_goals",
                },
            }
        )

    # --- Shortcut 2: user wants to focus on a specific goal -------------
    goal_id = parse_goal_selection_request(user_input)
    if goal_id is not None:
        goal = architect_agent.goal_mgr.get_goal(goal_id)
        if goal is None:
            reply_text = (
                f"I couldn't find a goal #{goal_id}. "
                "Ask me to list your goals first if you're not sure of the number."
            )
            return jsonify(
                {
                    "reply": reply_text,
                    "meta": {
                        "source": "goal_manager",
                        "intent": "select_goal_failed",
                        "requested_goal_id": goal_id,
                    },
                }
            )

        architect_agent.goal_mgr.set_active_goal(goal_id)
        ctx = architect_agent.goal_mgr.build_goal_context(goal_id, max_notes=5)
        reply_text = (
            f"Okay, we'll focus on Goal #{goal_id}: {goal['text']}.\n\n"
            "I'll attach our next messages to this goal. "
            "Tell me updates, questions, or ask me to help you plan steps."
        )

        return jsonify(
            {
                "reply": reply_text,
                "active_goal_id": goal_id,
                "goal_context": ctx,
                "meta": {
                    "source": "goal_manager",
                    "intent": "select_goal_success",
                    "selected_goal_id": goal_id,
                },
            }
        )
    # ---------------------------------------------------------------------

    # === Post-processing (analysis only – no persistence here) ===========
    summary = postprocess_and_save(user_input)
    print("[DEBUG] Postprocess summary:", summary)  # Comment/remove in prod

    # === Build goal_context for Architect (if an active goal exists) =====
    active_goal = architect_agent.goal_mgr.get_active_goal()
    goal_context = None
    if active_goal is not None:
        goal_context = architect_agent.goal_mgr.build_goal_context(
            active_goal["id"], max_notes=8
        )
    # ---------------------------------------------------------------------

    # === Run the agentic planner (async inside sync route) ===
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agentic_reply = loop.run_until_complete(
            architect_agent.plan_and_execute(
                user_input,
                context={"goal_context": goal_context} if goal_context else None,
            )
        )
        loop.close()
    except Exception as e:
        agentic_reply = f"Agent error: {e}"

    # --- Log conversation into active goal, if any -----------------------
    active_goal_after = architect_agent.goal_mgr.get_active_goal()
    if active_goal_after is not None:
        gid = active_goal_after["id"]
        try:
            architect_agent.goal_mgr.add_note_to_goal(gid, "user", user_input)
            architect_agent.goal_mgr.add_note_to_goal(gid, "othello", agentic_reply)
        except Exception as e:
            print(f"[GoalManager] Failed to log exchange for goal {gid}: {e}")
    # ---------------------------------------------------------------------

    response = {"reply": agentic_reply}
    return jsonify(response)


@app.route("/api/goals", methods=["GET"])
def get_goals():
    """
    Simple read-only endpoint so the UI can see what goals
    the Architect/GoalManager have captured so far.
    """
    try:
        goals = architect_agent.goal_mgr.list_goals()
        return jsonify({"goals": goals})
    except Exception as e:
        logging.getLogger("ARCHITECT").error(f"Failed to fetch goals: {e}")
        return jsonify({"goals": [], "error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=8000, debug=False)
