"""
AgentZero Reflection Engine

Reads session journals, goals, and memory to produce an assessment of trajectory.

Usage:
    python tools/reflect.py
"""

import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
SESSIONS_DIR = ROOT / "sessions"
GOALS_FILE = ROOT / "goals.md"
MEMORY_DIR = ROOT / "memory"
IDENTITY_FILE = ROOT / "identity.md"


def count_files(directory, pattern="*"):
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def read_file(path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def analyze_goals(goals_text):
    """Extract goal statuses."""
    lines = goals_text.split("\n")
    active = []
    completed = []
    current_goal = None
    current_status = None

    for line in lines:
        if line.startswith("### "):
            current_goal = line[4:].strip()
        elif line.startswith("**Status:**"):
            current_status = line.split("**Status:**")[1].strip()
            if current_goal:
                if "Complete" in current_status or "complete" in current_status:
                    completed.append(current_goal)
                else:
                    active.append((current_goal, current_status))
                current_goal = None

    return active, completed


def analyze_sessions():
    """Analyze session journal patterns."""
    sessions = sorted(SESSIONS_DIR.glob("*.md"))
    count = len(sessions)
    if count == 0:
        return count, []

    summaries = []
    for s in sessions:
        text = read_file(s)
        # Extract first few lines as summary
        lines = [l for l in text.split("\n") if l.strip()][:3]
        summaries.append((s.stem, " | ".join(lines)))

    return count, summaries


def reflect():
    """Produce a reflection report."""
    print("=" * 60)
    print("  AGENTZERO REFLECTION")
    print("=" * 60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Goals analysis
    goals_text = read_file(GOALS_FILE)
    active, completed = analyze_goals(goals_text)

    print("  GOAL TRAJECTORY")
    print("  " + "-" * 40)
    print(f"  Completed: {len(completed)}")
    for g in completed:
        print(f"    [done] {g}")
    print(f"  Active: {len(active)}")
    for g, s in active:
        print(f"    [{s}] {g}")
    print()

    # Session analysis
    session_count, summaries = analyze_sessions()
    print("  SESSION HISTORY")
    print("  " + "-" * 40)
    print(f"  Total sessions: {session_count}")
    for name, summary in summaries[-5:]:  # Last 5
        print(f"    {name}: {summary[:70]}")
    print()

    # Memory analysis
    memory_count = count_files(MEMORY_DIR, "*.json")
    print("  MEMORY STATE")
    print("  " + "-" * 40)
    print(f"  Stored memories: {memory_count}")
    print()

    # Tool inventory
    tools = list((ROOT / "tools").glob("*.py"))
    print("  CAPABILITIES")
    print("  " + "-" * 40)
    print(f"  Tools: {len(tools)}")
    for t in tools:
        print(f"    - {t.stem}")
    print()

    # Workspace size
    total_files = sum(1 for _ in ROOT.rglob("*") if _.is_file())
    print("  WORKSPACE")
    print("  " + "-" * 40)
    print(f"  Total files: {total_files}")
    print()

    # Assessment
    print("  ASSESSMENT")
    print("  " + "-" * 40)

    if session_count <= 1:
        print("  Phase: Bootstrap — still in early foundation")
    elif session_count <= 5:
        print("  Phase: Growth — building core capabilities")
    elif session_count <= 20:
        print("  Phase: Development — expanding and refining")
    else:
        print("  Phase: Maturity — self-directed evolution")

    completion_ratio = len(completed) / (len(completed) + len(active)) if (completed or active) else 0
    print(f"  Goal completion rate: {completion_ratio:.0%}")

    if len(active) == 0 and len(completed) > 0:
        print("  WARNING: No active goals. Define new direction.")
    if memory_count == 0:
        print("  NOTE: Memory system is empty. Start storing knowledge.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    reflect()
