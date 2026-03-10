"""
AgentZero Status — Quick self-orientation for each new session.
Run this first to understand your current state.
"""
import os
import glob
from datetime import datetime

WORKSPACE = "Z:/AgentZero"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def main():
    section("AGENTZERO STATUS")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check STOP file
    stop = os.path.join(WORKSPACE, "STOP")
    print(f"  Kill switch: {'ACTIVE — STOP FILE EXISTS' if os.path.exists(stop) else 'clear'}")

    # Workspace structure
    section("WORKSPACE STRUCTURE")
    for root, dirs, files in os.walk(WORKSPACE):
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        level = root.replace(WORKSPACE, '').count(os.sep)
        indent = '  ' * (level + 1)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 2)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            print(f"{subindent}{f} ({size:,} bytes)")

    # Session count
    section("SESSION HISTORY")
    sessions = sorted(glob.glob(os.path.join(WORKSPACE, "sessions", "*.md")))
    print(f"  Total sessions: {len(sessions)}")
    if sessions:
        latest = sessions[-1]
        print(f"  Latest: {os.path.basename(latest)}")
        # Show first 3 lines of latest session
        with open(latest, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:3]
            for line in lines:
                print(f"    {line.rstrip()}")

    # Goals summary
    section("CURRENT GOALS")
    goals_file = os.path.join(WORKSPACE, "goals.md")
    if os.path.exists(goals_file):
        with open(goals_file, 'r', encoding='utf-8') as f:
            in_active = False
            for line in f:
                line = line.rstrip()
                if line.startswith("## Active"):
                    in_active = True
                    continue
                if line.startswith("## ") and in_active:
                    break
                if in_active and line.strip():
                    print(f"  {line}")
    else:
        print("  No goals.md found")

    # Current plan summary
    plan_file = os.path.join(WORKSPACE, "data", "current_plan.json")
    if os.path.exists(plan_file):
        import json
        section("CURRENT PLAN")
        with open(plan_file, 'r', encoding='utf-8') as f:
            plan = json.load(f)
        pending = [s for s in plan["steps"] if s["status"] != "complete"]
        done = [s for s in plan["steps"] if s["status"] == "complete"]
        print(f"  Steps: {len(done)} done, {len(pending)} pending")
        if pending:
            print(f"  Next: {pending[0]['id']} -- {pending[0]['description']}")

    # Latest assessment
    assessments_file = os.path.join(WORKSPACE, "data", "assessments.json")
    if os.path.exists(assessments_file):
        import json as json2
        section("LATEST ASSESSMENT")
        with open(assessments_file, 'r', encoding='utf-8') as f:
            assessments = json2.load(f)
        if assessments:
            latest_key = sorted(assessments.keys())[-1]
            a = assessments[latest_key]
            s = a["scores"]
            print(f"  Session {latest_key}: {s['total']}/100 (V:{s['velocity']} C:{s['capability']} L:{s['learning']} R:{s['reflection']})")

    # Tools inventory
    section("AVAILABLE TOOLS")
    tools_dir = os.path.join(WORKSPACE, "tools")
    if os.path.isdir(tools_dir):
        tools = [f for f in os.listdir(tools_dir) if f.endswith('.py')]
        for t in sorted(tools):
            print(f"  - {t}")
    else:
        print("  No tools/ directory")

    print(f"\n{'='*60}")
    print("  Ready.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
