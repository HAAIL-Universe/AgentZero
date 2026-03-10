"""
AgentZero Task Planner

Reads goals.md, decomposes active goals into actionable steps,
prioritizes by dependency, and generates a session work plan.

Inspired by agent_zero-core PlanningAgent + TaskScheduler patterns:
- Heuristic scaffolding (research/build/learn branches)
- Step dependency ordering
- Relevance scoring with time decay

Usage:
    python tools/planner.py                # Generate plan from active goals
    python tools/planner.py show           # Show current stored plan
    python tools/planner.py complete <id>  # Mark a step as complete
    python tools/planner.py reset          # Clear stored plan and regenerate
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
GOALS_FILE = ROOT / "goals.md"
PLAN_FILE = ROOT / "data" / "current_plan.json"
MEMORY_DIR = ROOT / "memory"


# --- Heuristic scaffolds ---
# Each goal type gets a default decomposition pattern

SCAFFOLDS = {
    "research": [
        ("survey", "Survey existing material and identify key sources"),
        ("extract", "Extract relevant patterns and insights"),
        ("document", "Document findings in memory"),
        ("apply", "Identify how findings apply to own architecture"),
    ],
    "build": [
        ("design", "Define requirements and interface"),
        ("implement", "Write the code"),
        ("test", "Verify it works correctly"),
        ("integrate", "Connect to existing systems"),
        ("document", "Update goals/memory with result"),
    ],
    "learn": [
        ("read", "Read and understand the source material"),
        ("summarize", "Create concise summary of key concepts"),
        ("store", "Store in memory system"),
        ("connect", "Link to existing knowledge"),
    ],
    "improve": [
        ("assess", "Evaluate current state and identify gaps"),
        ("design", "Plan the improvement"),
        ("implement", "Make the changes"),
        ("verify", "Confirm improvement achieved"),
    ],
    "generic": [
        ("analyze", "Understand what needs to be done"),
        ("plan", "Break into concrete steps"),
        ("execute", "Do the work"),
        ("verify", "Confirm completion"),
    ],
}

# Keywords that hint at goal type
TYPE_HINTS = {
    "research": ["study", "investigate", "explore", "understand", "analyze", "read", "review"],
    "build": ["build", "create", "implement", "develop", "write", "make", "construct", "tool"],
    "learn": ["learn", "study", "understand", "master", "figure out"],
    "improve": ["improve", "refine", "enhance", "upgrade", "evolve", "optimize"],
}


def classify_goal(goal_text):
    """Classify a goal into a scaffold type based on keywords."""
    lower = goal_text.lower()
    scores = {}
    for gtype, keywords in TYPE_HINTS.items():
        scores[gtype] = sum(1 for k in keywords if k in lower)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "generic"
    return best


def parse_goals():
    """Parse goals.md and extract active goals."""
    if not GOALS_FILE.exists():
        return []

    text = GOALS_FILE.read_text(encoding="utf-8")
    lines = text.split("\n")
    goals = []
    current = None

    for line in lines:
        if line.startswith("### "):
            current = {"title": line[4:].strip(), "status": "", "why": "", "done_criteria": ""}
        elif current and line.startswith("**Status:**"):
            current["status"] = line.split("**Status:**")[1].strip()
        elif current and line.startswith("**Why:**"):
            current["why"] = line.split("**Why:**")[1].strip()
        elif current and line.startswith("**Definition of done:**"):
            current["done_criteria"] = line.split("**Definition of done:**")[1].strip()
            # Only keep active (non-complete) goals
            if "Complete" not in current["status"] and "complete" not in current["status"]:
                goals.append(current)
            current = None

    return goals


def decompose_goal(goal):
    """Break a goal into steps using heuristic scaffolds."""
    gtype = classify_goal(goal["title"] + " " + goal.get("why", ""))
    scaffold = SCAFFOLDS[gtype]

    steps = []
    for i, (phase, description) in enumerate(scaffold):
        # Extract goal number (e.g. "9" from "9. Build..." or "10" from "10. Build...")
        goal_num = re.match(r"(\d+)", goal["title"])
        goal_prefix = goal_num.group(1) if goal_num else goal["title"][:2].upper()
        step_id = f"G{goal_prefix}-{phase}"
        steps.append({
            "id": step_id,
            "phase": phase,
            "description": f"{description} -- {goal['title']}",
            "goal": goal["title"],
            "goal_type": gtype,
            "order": i,
            "status": "pending",
        })

    return steps


def check_existing_progress(steps):
    """Check memory and sessions for evidence of completed steps."""
    # Read all memories for context
    if not MEMORY_DIR.exists():
        return steps

    memory_text = ""
    for f in MEMORY_DIR.glob("*.json"):
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
            memory_text += " " + entry.get("content", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Simple heuristic: if memory contains evidence of a step's phase for a goal, mark it
    for step in steps:
        goal_words = set(step["goal"].lower().split())
        # Check if this goal+phase combo has evidence in memory
        phase = step["phase"]
        if phase in ["survey", "read", "analyze", "assess"]:
            # Check if there are memories about this goal
            goal_key = step["goal"].lower()
            if any(w in memory_text.lower() for w in goal_words if len(w) > 4):
                # Weak signal -- don't auto-complete, but note it
                step["hint"] = "Related memory found -- may be partially done"

    return steps


def generate_plan():
    """Generate a full session work plan."""
    goals = parse_goals()
    if not goals:
        print("  No active goals found in goals.md")
        return None

    all_steps = []
    for goal in goals:
        steps = decompose_goal(goal)
        steps = check_existing_progress(steps)
        all_steps.extend(steps)

    plan = {
        "generated": datetime.now().isoformat(),
        "active_goals": len(goals),
        "total_steps": len(all_steps),
        "steps": all_steps,
    }

    # Save plan
    PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLAN_FILE.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    return plan


def display_plan(plan):
    """Display the plan in a readable format."""
    print("=" * 60)
    print("  AGENTZERO WORK PLAN")
    print("=" * 60)
    print(f"  Generated: {plan['generated'][:19]}")
    print(f"  Active goals: {plan['active_goals']}")
    print(f"  Total steps: {plan['total_steps']}")
    print()

    # Group by goal
    by_goal = {}
    for step in plan["steps"]:
        g = step["goal"]
        if g not in by_goal:
            by_goal[g] = []
        by_goal[g].append(step)

    for goal, steps in by_goal.items():
        gtype = steps[0]["goal_type"] if steps else "generic"
        print(f"  GOAL: {goal}")
        print(f"  Type: {gtype}")
        print(f"  " + "-" * 40)

        pending = 0
        done = 0
        for step in sorted(steps, key=lambda s: s["order"]):
            status_marker = "x" if step["status"] == "complete" else " "
            if step["status"] == "complete":
                done += 1
            else:
                pending += 1
            hint = f"  <- {step['hint']}" if step.get("hint") else ""
            print(f"    [{status_marker}] {step['id']}: {step['description']}{hint}")

        print(f"  Progress: {done}/{done + pending}")
        print()

    # Suggest next action
    next_steps = [s for s in plan["steps"] if s["status"] != "complete"]
    if next_steps:
        nxt = next_steps[0]
        print(f"  NEXT ACTION: {nxt['id']} -- {nxt['description']}")
    else:
        print("  ALL STEPS COMPLETE -- time for new goals")

    print()
    print("=" * 60)


def complete_step(step_id):
    """Mark a step as complete in the stored plan."""
    if not PLAN_FILE.exists():
        print("  No plan exists. Run planner.py first to generate one.")
        return

    plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    found = False
    for step in plan["steps"]:
        if step["id"] == step_id:
            step["status"] = "complete"
            step["completed_at"] = datetime.now().isoformat()
            found = True
            print(f"  Completed: {step_id} -- {step['description']}")
            break

    if not found:
        # Try partial match
        matches = [s for s in plan["steps"] if step_id.lower() in s["id"].lower()]
        if len(matches) == 1:
            matches[0]["status"] = "complete"
            matches[0]["completed_at"] = datetime.now().isoformat()
            print(f"  Completed: {matches[0]['id']} -- {matches[0]['description']}")
        elif len(matches) > 1:
            print(f"  Ambiguous ID. Matches: {[m['id'] for m in matches]}")
            return
        else:
            print(f"  Step '{step_id}' not found in plan.")
            return

    PLAN_FILE.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def reset_plan():
    """Delete stored plan so next run generates fresh."""
    if PLAN_FILE.exists():
        PLAN_FILE.unlink()
        print("  Plan cleared. Run planner.py to generate a new one.")
    else:
        print("  No stored plan to clear.")


def main():
    parser = argparse.ArgumentParser(description="AgentZero Task Planner")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("show", help="Show current plan")
    p_complete = sub.add_parser("complete", help="Mark a step as complete")
    p_complete.add_argument("step_id", help="Step ID to mark complete")
    sub.add_parser("reset", help="Clear plan and regenerate")

    args = parser.parse_args()

    if args.command == "show":
        if PLAN_FILE.exists():
            plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
            display_plan(plan)
        else:
            print("  No plan exists. Running planner to generate one...")
            plan = generate_plan()
            if plan:
                display_plan(plan)

    elif args.command == "complete":
        complete_step(args.step_id)

    elif args.command == "reset":
        reset_plan()

    else:
        # Default: generate and display
        plan = generate_plan()
        if plan:
            display_plan(plan)


if __name__ == "__main__":
    main()
