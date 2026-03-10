"""
AgentZero Capability Registry

Auto-discovers tools in tools/ directory, extracts their docstrings and CLI usage,
and provides a queryable inventory of capabilities.

Inspired by agent_zero-core TaskRouter: keyword-to-capability mapping.

Usage:
    python tools/registry.py              # List all capabilities
    python tools/registry.py query <need> # Find tool matching a need
    python tools/registry.py detail <name># Show full detail for a tool
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOLS_DIR = ROOT / "tools"


def extract_tool_info(filepath):
    """Extract docstring and CLI commands from a Python tool file."""
    text = filepath.read_text(encoding="utf-8")

    # Extract module docstring
    try:
        tree = ast.parse(text)
        docstring = ast.get_docstring(tree) or ""
    except SyntaxError:
        docstring = ""

    # Extract usage lines from docstring
    usage_lines = []
    in_usage = False
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("usage:"):
            in_usage = True
            continue
        if in_usage:
            if stripped.startswith("python ") or stripped.startswith("tools/"):
                usage_lines.append(stripped)
            elif stripped == "":
                if usage_lines:
                    break
            else:
                break

    # Extract first paragraph as description
    desc_lines = []
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped == "":
            if desc_lines:
                break
            continue
        if stripped.lower().startswith("usage:"):
            break
        desc_lines.append(stripped)

    description = " ".join(desc_lines) if desc_lines else "(no description)"

    # Extract subcommands from argparse if present
    subcommands = re.findall(r'sub\.add_parser\(["\'](\w+)["\']', text)

    return {
        "name": filepath.stem,
        "path": str(filepath.relative_to(ROOT)),
        "description": description,
        "usage": usage_lines,
        "subcommands": subcommands,
        "size": filepath.stat().st_size,
    }


def discover_tools():
    """Find all Python tools and extract their info."""
    tools = []
    for f in sorted(TOOLS_DIR.glob("*.py")):
        if f.name.startswith("_"):
            continue
        info = extract_tool_info(f)
        tools.append(info)
    return tools


def list_capabilities():
    """List all tools with descriptions."""
    tools = discover_tools()
    print("=" * 60)
    print("  AGENTZERO CAPABILITY REGISTRY")
    print("=" * 60)
    print(f"  Tools: {len(tools)}")
    print()

    for t in tools:
        print(f"  {t['name']}")
        print(f"    {t['description'][:70]}")
        if t["subcommands"]:
            print(f"    Commands: {', '.join(t['subcommands'])}")
        print()

    print("=" * 60)
    return tools


def query_capability(need):
    """Find the best tool for a given need."""
    tools = discover_tools()
    need_lower = need.lower()
    need_words = set(need_lower.split())

    # Simple synonym/association map for better matching
    associations = {
        "remember": ["memory", "store", "save"],
        "find": ["search", "query", "list"],
        "plan": ["planner", "task", "goal", "decompose"],
        "think": ["reflect", "assess", "evaluate"],
        "tools": ["registry", "capability", "inventory"],
        "orient": ["status", "briefing", "state"],
        "goal": ["planner", "goals", "task"],
        "store": ["memory", "save", "add"],
        "search": ["memory", "find", "query"],
        "assess": ["reflect", "evaluate", "score"],
    }

    expanded_words = set(need_words)
    for w in need_words:
        if w in associations:
            expanded_words.update(associations[w])

    scored = []
    for t in tools:
        searchable = (t["description"] + " " + t["name"] + " " + " ".join(t["subcommands"])).lower()
        # Word overlap scoring with expanded terms
        score = sum(1 for w in expanded_words if w in searchable)
        # Bonus for name match
        if need_lower in t["name"]:
            score += 3
        if score > 0:
            scored.append((score, t))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        print(f"  No tool matches '{need}'.")
        print("  Available tools:")
        for t in tools:
            print(f"    - {t['name']}: {t['description'][:50]}")
        return []

    print(f"  Best match for '{need}':")
    for score, t in scored[:3]:
        print(f"    [{score}] {t['name']} -- {t['description'][:60]}")
        if t["usage"]:
            print(f"         Usage: {t['usage'][0]}")
    return scored


def show_detail(name):
    """Show full detail for a specific tool."""
    filepath = TOOLS_DIR / f"{name}.py"
    if not filepath.exists():
        print(f"  Tool '{name}' not found.")
        return None

    info = extract_tool_info(filepath)
    print("=" * 60)
    print(f"  TOOL: {info['name']}")
    print("=" * 60)
    print(f"  Path: {info['path']}")
    print(f"  Size: {info['size']:,} bytes")
    print(f"  Description: {info['description']}")
    if info["subcommands"]:
        print(f"  Commands: {', '.join(info['subcommands'])}")
    if info["usage"]:
        print(f"  Usage:")
        for u in info["usage"]:
            print(f"    {u}")
    print("=" * 60)
    return info


def main():
    parser = argparse.ArgumentParser(description="AgentZero Capability Registry")
    sub = parser.add_subparsers(dest="command")

    p_query = sub.add_parser("query", help="Find tool for a need")
    p_query.add_argument("need", nargs="+", help="What you need")

    p_detail = sub.add_parser("detail", help="Show tool detail")
    p_detail.add_argument("name", help="Tool name")

    args = parser.parse_args()

    if args.command == "query":
        query_capability(" ".join(args.need))
    elif args.command == "detail":
        show_detail(args.name)
    else:
        list_capabilities()


if __name__ == "__main__":
    main()
