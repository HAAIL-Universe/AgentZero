"""
AgentZero Error Logging and Learning Tool

Tracks errors, corrections, and learns from mistakes.
Detects recurring patterns and generates behavioral adjustments.

Usage:
    python tools/errors.py log --category <cat> --description "what went wrong" [--expected "..."] [--severity low|medium|high]
    python tools/errors.py correct <error_id> --fix "what fixed it" [--adjustment "behavior to change"]
    python tools/errors.py search <query>
    python tools/errors.py list [--category <cat>] [--unresolved]
    python tools/errors.py patterns           # detect recurring error categories
    python tools/errors.py learn              # auto-generate behavioral adjustments from patterns
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "errors"
sys.path.insert(0, str(ROOT / "tools"))

# Categories for classifying errors
CATEGORIES = [
    "encoding",      # character encoding issues
    "path",          # file path errors
    "syntax",        # code syntax mistakes
    "logic",         # logical errors in reasoning or code
    "tool-usage",    # using a tool incorrectly
    "assumption",    # wrong assumptions about state or behavior
    "integration",   # errors when connecting components
    "environment",   # OS/platform issues
    "other",         # uncategorized
]


def generate_id():
    """Generate a sequential error ID."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(DATA_DIR.glob("E*.json"))
    if not existing:
        return "E001"
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem[1:]))
        except ValueError:
            pass
    return f"E{(max(nums) + 1) if nums else 1:03d}"


def log_error(category, description, expected=None, severity="medium"):
    """Log a new error."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eid = generate_id()
    entry = {
        "id": eid,
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "description": description,
        "expected": expected,
        "severity": severity,
        "resolved": False,
        "correction": None,
        "adjustment": None,
    }
    path = DATA_DIR / f"{eid}.json"
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    print(f"  Logged: {eid} [{category}] ({severity}) {description[:70]}")
    return eid


def correct_error(error_id, fix, adjustment=None):
    """Record a correction for an existing error."""
    path = DATA_DIR / f"{error_id}.json"
    if not path.exists():
        print(f"  Error {error_id} not found.")
        return None

    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["resolved"] = True
    entry["correction"] = fix
    entry["adjustment"] = adjustment
    entry["resolved_at"] = datetime.now().isoformat()
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    print(f"  Corrected: {error_id} -- {fix[:70]}")
    if adjustment:
        print(f"  Adjustment: {adjustment}")
    return entry


def load_all_errors():
    """Load all error entries."""
    if not DATA_DIR.exists():
        return []
    entries = []
    for f in sorted(DATA_DIR.glob("E*.json")):
        try:
            entries.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return entries


def search_errors(query):
    """Search errors by description substring."""
    query_lower = query.lower()
    results = []
    for e in load_all_errors():
        searchable = f"{e.get('description', '')} {e.get('category', '')} {e.get('correction', '')}".lower()
        if query_lower in searchable:
            results.append(e)

    if not results:
        print("  No errors match that query.")
    else:
        for e in results:
            status = "RESOLVED" if e["resolved"] else "OPEN"
            print(f"  [{e['id']}] ({e['category']}) [{status}] {e['description'][:70]}")
    return results


def list_errors(category=None, unresolved=False):
    """List errors with optional filters."""
    entries = load_all_errors()
    if category:
        entries = [e for e in entries if e.get("category") == category]
    if unresolved:
        entries = [e for e in entries if not e.get("resolved")]

    if not entries:
        print("  No errors found.")
    else:
        print(f"  {'ID':<6} {'Category':<14} {'Sev':<7} {'Status':<10} {'Description'}")
        print(f"  {'---':<6} {'---':<14} {'---':<7} {'---':<10} {'---'}")
        for e in entries:
            status = "RESOLVED" if e["resolved"] else "OPEN"
            print(f"  {e['id']:<6} {e['category']:<14} {e.get('severity','?'):<7} {status:<10} {e['description'][:50]}")
    return entries


def detect_patterns():
    """Detect recurring error categories and generate pattern report."""
    entries = load_all_errors()
    if not entries:
        print("  No errors logged yet.")
        return []

    # Count by category
    cat_counts = Counter(e["category"] for e in entries)
    resolved_counts = Counter(e["category"] for e in entries if e["resolved"])
    unresolved_counts = Counter(e["category"] for e in entries if not e["resolved"])

    print("  ERROR PATTERNS")
    print("  " + "-" * 50)
    print(f"  {'Category':<14} {'Total':>6} {'Resolved':>9} {'Open':>6}")
    print("  " + "-" * 50)

    patterns = []
    for cat, count in cat_counts.most_common():
        resolved = resolved_counts.get(cat, 0)
        unresolved = unresolved_counts.get(cat, 0)
        print(f"  {cat:<14} {count:>6} {resolved:>9} {unresolved:>6}")

        if count >= 2:
            # Collect corrections that worked for this category
            corrections = [
                e["correction"] for e in entries
                if e["category"] == cat and e["resolved"] and e.get("correction")
            ]
            adjustments = [
                e["adjustment"] for e in entries
                if e["category"] == cat and e.get("adjustment")
            ]
            patterns.append({
                "category": cat,
                "count": count,
                "corrections": corrections,
                "adjustments": adjustments,
            })

    print()

    if patterns:
        print("  RECURRING PATTERNS (2+ occurrences):")
        print("  " + "-" * 50)
        for p in patterns:
            print(f"  {p['category']}: {p['count']} occurrences")
            if p["corrections"]:
                print(f"    Known fixes: {'; '.join(p['corrections'][:3])}")
            if p["adjustments"]:
                print(f"    Adjustments: {'; '.join(p['adjustments'][:3])}")
        print()

    return patterns


def learn_from_errors():
    """Auto-generate behavioral adjustment memories from recurring patterns.

    Imports memory.py to create adjustment entries.
    Only creates adjustments for patterns not already captured.
    """
    import memory as memory_tool

    patterns = detect_patterns()
    if not patterns:
        print("  No recurring patterns to learn from.")
        return

    # Load existing adjustments to avoid duplicates
    existing = memory_tool.list_adjustments()
    existing_texts = {e.get("behavioral_adjustment", "").lower() for e in existing}

    created = 0
    for p in patterns:
        if p["adjustments"]:
            # Use the most recent adjustment from this category
            adj_text = p["adjustments"][-1]
        else:
            # Auto-generate an adjustment
            adj_text = f"Recurring {p['category']} errors ({p['count']}x). "
            if p["corrections"]:
                adj_text += f"Known fix: {p['corrections'][-1]}"
            else:
                adj_text += "No correction recorded yet -- investigate root cause."

        # Skip if already exists
        if adj_text.lower() in existing_texts:
            continue

        content = f"Error pattern: {p['category']} ({p['count']} occurrences)"
        memory_tool.add_memory(
            tag="error-pattern",
            content=content,
            source="tools/errors.py:learn",
            adjustment=adj_text,
        )
        created += 1

    print(f"\n  Generated {created} new behavioral adjustment(s) from error patterns.")


def main():
    parser = argparse.ArgumentParser(description="AgentZero Error Logging and Learning")
    sub = parser.add_subparsers(dest="command")

    # log
    p_log = sub.add_parser("log")
    p_log.add_argument("--category", required=True, choices=CATEGORIES, help="Error category")
    p_log.add_argument("--description", required=True, help="What went wrong")
    p_log.add_argument("--expected", default=None, help="What should have happened")
    p_log.add_argument("--severity", default="medium", choices=["low", "medium", "high"])

    # correct
    p_correct = sub.add_parser("correct")
    p_correct.add_argument("error_id", help="Error ID to correct (e.g. E001)")
    p_correct.add_argument("--fix", required=True, help="What fixed the problem")
    p_correct.add_argument("--adjustment", default=None, help="Behavioral adjustment to apply")

    # search
    p_search = sub.add_parser("search")
    p_search.add_argument("query", help="Search term")

    # list
    p_list = sub.add_parser("list")
    p_list.add_argument("--category", default=None, choices=CATEGORIES)
    p_list.add_argument("--unresolved", action="store_true")

    # patterns
    sub.add_parser("patterns")

    # learn
    sub.add_parser("learn")

    args = parser.parse_args()

    if args.command == "log":
        log_error(args.category, args.description, args.expected, args.severity)
    elif args.command == "correct":
        correct_error(args.error_id, args.fix, args.adjustment)
    elif args.command == "search":
        search_errors(args.query)
    elif args.command == "list":
        list_errors(args.category, args.unresolved)
    elif args.command == "patterns":
        detect_patterns()
    elif args.command == "learn":
        learn_from_errors()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
