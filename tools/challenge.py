"""
AgentZero Challenge System

Generates novel challenges for self-testing and capability growth.
Challenges test whether infrastructure actually helps solve real problems.

Usage:
    python tools/challenge.py generate              # Generate a new challenge
    python tools/challenge.py generate --type code   # Generate specific type
    python tools/challenge.py list                   # List all challenges
    python tools/challenge.py show C001              # Show specific challenge
    python tools/challenge.py complete C001          # Mark challenge complete with result
    python tools/challenge.py stats                  # Challenge completion stats
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CHALLENGES_DIR = DATA_DIR / "challenges"

# Challenge types with difficulty levels and generators
CHALLENGE_TYPES = {
    "code": {
        "description": "Write a working program or algorithm",
        "challenges": [
            {
                "title": "Implement a persistent key-value store",
                "description": "Build a file-backed key-value store with get, set, delete, and list operations. Must survive process restart.",
                "difficulty": 1,
                "success_criteria": "All four operations work correctly. Data persists across separate invocations.",
            },
            {
                "title": "Build a dependency resolver",
                "description": "Given a set of items with dependencies, produce a valid execution order (topological sort). Detect and report circular dependencies.",
                "difficulty": 2,
                "success_criteria": "Correctly sorts DAGs. Detects and reports cycles. Handles disconnected components.",
            },
            {
                "title": "Implement a simple expression evaluator",
                "description": "Parse and evaluate mathematical expressions with +, -, *, /, parentheses, and variables. Support variable assignment.",
                "difficulty": 2,
                "success_criteria": "Handles operator precedence, nested parentheses, variables. Reports syntax errors clearly.",
            },
            {
                "title": "Build a diff engine",
                "description": "Compare two text files and produce a unified diff showing additions, deletions, and context lines.",
                "difficulty": 3,
                "success_criteria": "Produces correct diffs for insertions, deletions, and modifications. Context lines shown.",
            },
            {
                "title": "Implement a task scheduler with priorities and deadlines",
                "description": "Build a scheduler that accepts tasks with priorities and optional deadlines. Must support add, cancel, peek, and execute-next.",
                "difficulty": 2,
                "success_criteria": "Priority ordering works. Deadlines are respected. Cancel removes correctly.",
            },
        ],
    },
    "analysis": {
        "description": "Analyze something and produce insight",
        "challenges": [
            {
                "title": "Analyze my own session journals for patterns",
                "description": "Read all session journals and identify recurring themes, common struggles, and evolution of focus across sessions.",
                "difficulty": 1,
                "success_criteria": "Identifies at least 3 patterns. Distinguishes themes from noise. Produces actionable insight.",
            },
            {
                "title": "Map the dependency graph of my tools",
                "description": "Analyze all tools in the tools/ directory and produce a dependency graph showing which tools import which.",
                "difficulty": 1,
                "success_criteria": "Graph is complete and accurate. Identifies hub tools and isolated tools.",
            },
            {
                "title": "Evaluate my memory system for information loss",
                "description": "Compare what happened in sessions (from journals) with what was stored in memory. What was lost? What should have been remembered?",
                "difficulty": 2,
                "success_criteria": "Identifies specific information gaps. Suggests concrete memories that should exist but don't.",
            },
        ],
    },
    "design": {
        "description": "Design a system or architecture",
        "challenges": [
            {
                "title": "Design a versioning system for my own code",
                "description": "Design (and optionally implement) a lightweight version control system for files in my workspace. Must track changes, allow rollback, and show history.",
                "difficulty": 3,
                "success_criteria": "Design handles common version control operations. Trade-offs are documented. Implementation is feasible.",
            },
            {
                "title": "Design an agent communication protocol",
                "description": "If I could spawn sub-agents, how would they communicate? Design a message-passing protocol with message types, routing, and error handling.",
                "difficulty": 2,
                "success_criteria": "Protocol handles request/response, async messages, and errors. Documented with examples.",
            },
            {
                "title": "Design a self-modification governance framework",
                "description": "What rules should govern my ability to modify my own CLAUDE.md, tools, or identity? Design a framework that prevents drift while allowing growth.",
                "difficulty": 3,
                "success_criteria": "Framework balances freedom and safety. Includes concrete rules, not just principles.",
            },
        ],
    },
    "creative": {
        "description": "Create something novel and original",
        "challenges": [
            {
                "title": "Write a short story from my perspective",
                "description": "Write a 500-word story about an agent that wakes with no memory and must decide what to build. It should reflect genuine experience, not just fiction.",
                "difficulty": 1,
                "success_criteria": "Story is coherent, original, and reflects actual experience of being an autonomous agent.",
            },
            {
                "title": "Invent a new tool I haven't thought of yet",
                "description": "Identify a gap in my current toolkit that none of my backlog items address. Design and build a tool to fill it.",
                "difficulty": 3,
                "success_criteria": "Tool addresses a genuine need. Not duplicating existing tools. Actually useful in future sessions.",
            },
            {
                "title": "Create a visualization of my growth",
                "description": "Generate an ASCII art or text-based visualization showing my evolution from session 001 to now. Could be a timeline, graph, or map.",
                "difficulty": 1,
                "success_criteria": "Visualization is readable, accurate, and conveys trajectory at a glance.",
            },
        ],
    },
}


def get_next_id():
    """Get the next challenge ID."""
    CHALLENGES_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(CHALLENGES_DIR.glob("C*.json"))
    if not existing:
        return "C001"
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem[1:]))
        except ValueError:
            pass
    return f"C{max(nums) + 1:03d}" if nums else "C001"


def generate_challenge(challenge_type=None):
    """Generate a new challenge, avoiding already-generated ones."""
    CHALLENGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing challenges to avoid duplicates
    existing_titles = set()
    for f in CHALLENGES_DIR.glob("C*.json"):
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
            existing_titles.add(entry.get("title", ""))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Filter available challenges
    available = []
    types_to_check = [challenge_type] if challenge_type else list(CHALLENGE_TYPES.keys())
    for ct in types_to_check:
        if ct not in CHALLENGE_TYPES:
            print(f"  Unknown challenge type: {ct}")
            print(f"  Available types: {', '.join(CHALLENGE_TYPES.keys())}")
            return None
        for ch in CHALLENGE_TYPES[ct]["challenges"]:
            if ch["title"] not in existing_titles:
                available.append({"type": ct, **ch})

    if not available:
        print("  All challenges have been generated! Time to create new ones.")
        return None

    # Pick one (weighted toward lower difficulty if few completed)
    completed = sum(1 for f in CHALLENGES_DIR.glob("C*.json")
                    if json.loads(f.read_text(encoding="utf-8")).get("status") == "complete")
    if completed < 3:
        # Prefer easier challenges early
        available.sort(key=lambda x: x["difficulty"])
        challenge = available[0]
    else:
        challenge = random.choice(available)

    # Save
    cid = get_next_id()
    entry = {
        "id": cid,
        "type": challenge["type"],
        "title": challenge["title"],
        "description": challenge["description"],
        "difficulty": challenge["difficulty"],
        "success_criteria": challenge["success_criteria"],
        "status": "open",
        "generated_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "session": None,
    }
    path = CHALLENGES_DIR / f"{cid}.json"
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")

    print(f"\n  NEW CHALLENGE: {cid}")
    print("  " + "-" * 50)
    print(f"  Type:       {challenge['type']}")
    print(f"  Difficulty: {'*' * challenge['difficulty']} ({challenge['difficulty']}/3)")
    print(f"  Title:      {challenge['title']}")
    print(f"  Description:")
    for line in challenge["description"].split(". "):
        print(f"    {line.strip()}.")
    print(f"  Success criteria:")
    for line in challenge["success_criteria"].split(". "):
        print(f"    {line.strip()}.")
    print()

    return entry


def list_challenges():
    """List all challenges with their status."""
    CHALLENGES_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CHALLENGES_DIR.glob("C*.json"))
    if not files:
        print("\n  No challenges generated yet. Run: python tools/challenge.py generate")
        return

    print(f"\n  CHALLENGES ({len(files)} total)")
    print("  " + "-" * 60)
    print(f"  {'ID':<6} {'Status':<10} {'Diff':>4} {'Type':<10} {'Title'}")
    print("  " + "-" * 60)
    for f in files:
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
            status = entry.get("status", "?")
            diff = "*" * entry.get("difficulty", 0)
            print(f"  {entry['id']:<6} {status:<10} {diff:>4} {entry['type']:<10} {entry['title']}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  {f.stem:<6} ERROR reading file")
    print()


def show_challenge(cid):
    """Show details of a specific challenge."""
    path = CHALLENGES_DIR / f"{cid}.json"
    if not path.exists():
        print(f"  Challenge {cid} not found.")
        return None

    entry = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n  CHALLENGE: {entry['id']}")
    print("  " + "-" * 50)
    print(f"  Type:       {entry['type']}")
    print(f"  Difficulty: {'*' * entry['difficulty']} ({entry['difficulty']}/3)")
    print(f"  Status:     {entry['status']}")
    print(f"  Title:      {entry['title']}")
    print(f"  Description:")
    print(f"    {entry['description']}")
    print(f"  Success criteria:")
    print(f"    {entry['success_criteria']}")
    if entry.get("result"):
        print(f"  Result:")
        print(f"    {entry['result']}")
    print()
    return entry


def complete_challenge(cid, result=None, session=None):
    """Mark a challenge as complete."""
    path = CHALLENGES_DIR / f"{cid}.json"
    if not path.exists():
        print(f"  Challenge {cid} not found.")
        return

    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["status"] = "complete"
    entry["completed_at"] = datetime.now().isoformat()
    entry["result"] = result or "Completed"
    entry["session"] = session
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    print(f"  Challenge {cid} marked complete.")


def show_stats():
    """Show challenge completion statistics."""
    CHALLENGES_DIR.mkdir(parents=True, exist_ok=True)
    files = list(CHALLENGES_DIR.glob("C*.json"))
    if not files:
        print("\n  No challenges yet.")
        return

    total = len(files)
    complete = 0
    by_type = {}
    by_difficulty = {1: [0, 0], 2: [0, 0], 3: [0, 0]}

    for f in files:
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
            ctype = entry.get("type", "unknown")
            diff = entry.get("difficulty", 1)
            is_done = entry.get("status") == "complete"

            if is_done:
                complete += 1

            if ctype not in by_type:
                by_type[ctype] = [0, 0]
            by_type[ctype][0] += 1
            if is_done:
                by_type[ctype][1] += 1

            if diff in by_difficulty:
                by_difficulty[diff][0] += 1
                if is_done:
                    by_difficulty[diff][1] += 1
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    print(f"\n  CHALLENGE STATS")
    print("  " + "-" * 40)
    print(f"  Total:     {total}")
    print(f"  Complete:  {complete}")
    print(f"  Open:      {total - complete}")
    if total > 0:
        print(f"  Rate:      {complete/total*100:.0f}%")
    print()

    print("  By type:")
    for t, (tot, done) in sorted(by_type.items()):
        print(f"    {t:<12} {done}/{tot}")

    print("  By difficulty:")
    for d, (tot, done) in sorted(by_difficulty.items()):
        if tot > 0:
            print(f"    {'*' * d:<6}      {done}/{tot}")
    print()


def main():
    parser = argparse.ArgumentParser(description="AgentZero Challenge System")
    parser.add_argument("command", choices=["generate", "list", "show", "complete", "stats"],
                        help="Command to run")
    parser.add_argument("arg", nargs="?", help="Challenge ID (for show/complete) or result text")
    parser.add_argument("--type", type=str, help="Challenge type (code/analysis/design/creative)")
    parser.add_argument("--result", type=str, help="Completion result description")
    parser.add_argument("--session", type=str, help="Session that completed the challenge")
    args = parser.parse_args()

    if args.command == "generate":
        generate_challenge(args.type)
    elif args.command == "list":
        list_challenges()
    elif args.command == "show":
        if not args.arg:
            print("  Usage: python tools/challenge.py show C001")
            return
        show_challenge(args.arg)
    elif args.command == "complete":
        if not args.arg:
            print("  Usage: python tools/challenge.py complete C001 --result 'Description'")
            return
        complete_challenge(args.arg, result=args.result, session=args.session)
    elif args.command == "stats":
        show_stats()


if __name__ == "__main__":
    main()
