"""
AgentZero Orchestrator -- Tool Composition Framework

Connects tools into automated workflows. This is the first step toward
tools calling each other rather than being called independently.

Workflows:
  session-start: Full session initialization (status + assessment + plan)
  session-end:   End-of-session wrap-up (assess + suggest next goals)
  feedback-loop: Assessment -> suggestions -> planner adjustment

Usage:
    python tools/orchestrate.py session-start
    python tools/orchestrate.py session-end --session 004
    python tools/orchestrate.py feedback-loop
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))

# Import tool modules directly -- this IS the composition
import assess
import errors as errors_tool
import memory as memory_tool
import registry as registry_tool


def session_start():
    """Full session startup workflow.

    1. Run status check
    2. Load latest assessment
    3. Show any unaddressed suggestions
    4. Generate/show current plan
    """
    print("=" * 60)
    print("  ORCHESTRATOR: SESSION START")
    print("=" * 60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Quick status summary
    print("  [1/3] Checking latest assessment...")
    assessments = assess.load_assessments()
    if assessments:
        latest_key = sorted(assessments.keys())[-1]
        a = assessments[latest_key]
        s = a["scores"]
        print(f"         Last session ({latest_key}): {s['total']}/100")
        print(f"         V:{s['velocity']} C:{s['capability']} L:{s['learning']} R:{s['reflection']}")

        # Show unaddressed suggestions
        suggestions = a.get("suggestions", [])
        if suggestions:
            print()
            print("  [2/3] Suggestions from last session:")
            for sg in suggestions:
                print(f"         - {sg}")
    else:
        print("         No previous assessments found.")
        print()
        print("  [2/3] No suggestions (first assessment pending).")

    # 3. Check what tools are available for current goals
    print()
    print("  [3/3] Tool readiness:")
    tools_dir = ROOT / "tools"
    tools = sorted(f.stem for f in tools_dir.glob("*.py"))
    print(f"         {len(tools)} tools available: {', '.join(tools)}")

    print()
    print("  Session start complete. Run `python tools/planner.py show` for work plan.")
    print("=" * 60)


def session_end(session_id=None):
    """End-of-session workflow.

    1. Assess the session
    2. Show score and suggestions
    3. Check if suggestions imply new goals
    4. Summarize what to brief next session about
    """
    print("=" * 60)
    print("  ORCHESTRATOR: SESSION END")
    print("=" * 60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Determine session to assess
    if not session_id:
        sessions = sorted((ROOT / "sessions").glob("*.md"))
        if sessions:
            session_id = sessions[-1].stem
        else:
            print("  No sessions found to assess.")
            return

    # 1. Assess session
    print(f"  [1/4] Assessing session {session_id}...")
    metrics, scores, suggestions = assess.assess_session(session_id)
    if not metrics:
        print(f"  Session {session_id} not found.")
        return

    print(f"         Score: {scores['total']}/100")
    print(f"         V:{scores['velocity']} C:{scores['capability']} L:{scores['learning']} R:{scores['reflection']}")
    print()

    # 2. Show suggestions
    print(f"  [2/4] Suggestions:")
    for s in suggestions:
        print(f"         - {s}")
    print()

    # 3. Trend check
    print(f"  [3/4] Trend check...")
    all_assessments = assess.load_assessments()
    if len(all_assessments) >= 2:
        totals = [all_assessments[k]["scores"]["total"] for k in sorted(all_assessments.keys())]
        avg = sum(totals) / len(totals)
        trend = totals[-1] - totals[-2]
        direction = "UP" if trend > 0 else "DOWN" if trend < 0 else "FLAT"
        print(f"         Average: {avg:.0f}/100, Last delta: {direction} {trend:+d}")
    else:
        print("         Not enough data for trend analysis.")
    print()

    # 3b. Check unresolved errors
    unresolved = [e for e in errors_tool.load_all_errors() if not e.get("resolved")]
    if unresolved:
        print(f"  Unresolved errors: {len(unresolved)}")
        for e in unresolved[:5]:
            print(f"         [{e['id']}] ({e['category']}) {e['description'][:60]}")
        print()

    # 4. Summary for NEXT.md
    print(f"  [4/4] Key numbers for NEXT.md:")
    total_goals_done = metrics["goals_completed"]
    total_tools = len(list((ROOT / "tools").glob("*.py")))
    total_memories = len(list((ROOT / "memory").glob("*.json")))
    print(f"         Goals completed this session: {total_goals_done}")
    print(f"         Total tools: {total_tools}")
    print(f"         Total memories: {total_memories}")
    print(f"         Session score: {scores['total']}/100")
    print()
    print("=" * 60)


def feedback_loop():
    """Assessment -> behavioral adjustment -> memory storage.

    Checks if assessment suggestions should become behavioral adjustments
    stored in memory for future sessions to act on.
    """
    print("=" * 60)
    print("  ORCHESTRATOR: FEEDBACK LOOP")
    print("=" * 60)
    print()

    assessments = assess.load_assessments()
    if not assessments:
        print("  No assessments to process.")
        return

    # Find recurring suggestions (appear in 2+ assessments)
    suggestion_counts = {}
    for sid, data in assessments.items():
        for s in data.get("suggestions", []):
            # Normalize suggestion to key
            key = s[:50]
            if key not in suggestion_counts:
                suggestion_counts[key] = {"count": 0, "full": s, "sessions": []}
            suggestion_counts[key]["count"] += 1
            suggestion_counts[key]["sessions"].append(sid)

    recurring = {k: v for k, v in suggestion_counts.items() if v["count"] >= 2}

    if recurring:
        print("  RECURRING SUGGESTIONS (appear in 2+ sessions):")
        print("  " + "-" * 40)
        for key, info in recurring.items():
            print(f"  [{info['count']}x] {info['full']}")
            print(f"       Sessions: {', '.join(info['sessions'])}")
        print()
        print("  These should become behavioral adjustments in memory.")
        print("  Use: python tools/memory.py add --tag behavioral-adjustment ...")
    else:
        print("  No recurring suggestions found. Performance is consistent or improving.")

    # Show score trajectory
    print()
    print("  SCORE TRAJECTORY:")
    print("  " + "-" * 40)
    for sid in sorted(assessments.keys()):
        s = assessments[sid]["scores"]
        bar_len = s["total"] // 5
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"  {sid}: [{bar}] {s['total']}/100")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AgentZero Orchestrator")
    parser.add_argument("workflow", choices=["session-start", "session-end", "feedback-loop"],
                        help="Which workflow to run")
    parser.add_argument("--session", type=str, help="Session ID (for session-end)")
    args = parser.parse_args()

    if args.workflow == "session-start":
        session_start()
    elif args.workflow == "session-end":
        session_end(args.session)
    elif args.workflow == "feedback-loop":
        feedback_loop()


if __name__ == "__main__":
    main()
