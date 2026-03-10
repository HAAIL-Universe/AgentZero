"""
AgentZero Memory System

Persistent, searchable knowledge store.
Each memory is a JSON file in memory/ with metadata and content.

Usage:
    python tools/memory.py add --tag <tag> --content "what to remember" [--adjustment "behavior change"]
    python tools/memory.py search <query>
    python tools/memory.py list [--tag <tag>]
    python tools/memory.py recall <id>
    python tools/memory.py adjustments   # list all behavioral adjustments
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent.parent / "memory"


def generate_id():
    """Generate a sequential memory ID."""
    existing = list(MEMORY_DIR.glob("*.json"))
    if not existing:
        return "M001"
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem[1:]))
        except ValueError:
            pass
    return f"M{(max(nums) + 1) if nums else 1:03d}"


def add_memory(tag, content, source=None, adjustment=None):
    """Store a new memory entry."""
    MEMORY_DIR.mkdir(exist_ok=True)
    mid = generate_id()
    entry = {
        "id": mid,
        "timestamp": datetime.now().isoformat(),
        "tag": tag,
        "content": content,
        "source": source or "unknown",
        "behavioral_adjustment": adjustment,
    }
    path = MEMORY_DIR / f"{mid}.json"
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    print(f"Stored: {mid} [{tag}] {content[:60]}...")
    return mid


def search_memories(query):
    """Search memories by content substring (case-insensitive)."""
    query_lower = query.lower()
    results = []
    for f in sorted(MEMORY_DIR.glob("*.json")):
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if query_lower in entry.get("content", "").lower() or query_lower in entry.get("tag", "").lower():
            results.append(entry)
    if not results:
        print("No memories match that query.")
    else:
        for r in results:
            print(f"  [{r['id']}] ({r['tag']}) {r['content'][:80]}")
    return results


def list_memories(tag=None):
    """List all memories, optionally filtered by tag."""
    entries = []
    for f in sorted(MEMORY_DIR.glob("*.json")):
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if tag and entry.get("tag") != tag:
            continue
        entries.append(entry)
    if not entries:
        print("No memories found.")
    else:
        print(f"  {'ID':<6} {'Tag':<15} {'Content'}")
        print(f"  {'---':<6} {'---':<15} {'---'}")
        for e in entries:
            print(f"  {e['id']:<6} {e['tag']:<15} {e['content'][:60]}")
    return entries


def recall_memory(mid):
    """Recall a specific memory by ID."""
    path = MEMORY_DIR / f"{mid}.json"
    if not path.exists():
        print(f"Memory {mid} not found.")
        return None
    entry = json.loads(path.read_text(encoding="utf-8"))
    print(f"  ID:        {entry['id']}")
    print(f"  Tag:       {entry['tag']}")
    print(f"  Timestamp: {entry['timestamp']}")
    print(f"  Source:    {entry['source']}")
    print(f"  Content:   {entry['content']}")
    if entry.get("behavioral_adjustment"):
        print(f"  Adjustment:{entry['behavioral_adjustment']}")
    return entry


def list_adjustments():
    """List all memories that carry behavioral adjustments."""
    entries = []
    for f in sorted(MEMORY_DIR.glob("*.json")):
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if entry.get("behavioral_adjustment"):
            entries.append(entry)
    if not entries:
        print("No behavioral adjustments stored.")
    else:
        print("  Active behavioral adjustments:")
        for e in entries:
            print(f"    [{e['id']}] {e['behavioral_adjustment']}")
    return entries


def main():
    parser = argparse.ArgumentParser(description="AgentZero Memory System")
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add")
    p_add.add_argument("--tag", required=True, help="Category tag")
    p_add.add_argument("--content", required=True, help="What to remember")
    p_add.add_argument("--source", default=None, help="Where this came from")
    p_add.add_argument("--adjustment", default=None, help="Behavioral adjustment to apply")

    p_search = sub.add_parser("search")
    p_search.add_argument("query", help="Search term")

    p_list = sub.add_parser("list")
    p_list.add_argument("--tag", default=None, help="Filter by tag")

    p_recall = sub.add_parser("recall")
    p_recall.add_argument("id", help="Memory ID to recall")

    sub.add_parser("adjustments")

    args = parser.parse_args()

    if args.command == "add":
        add_memory(args.tag, args.content, args.source, args.adjustment)
    elif args.command == "search":
        search_memories(args.query)
    elif args.command == "list":
        list_memories(args.tag)
    elif args.command == "recall":
        recall_memory(args.id)
    elif args.command == "adjustments":
        list_adjustments()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
