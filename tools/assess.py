"""
AgentZero Self-Assessment Tool

Scores session effectiveness across multiple dimensions and tracks trends.
Generates improvement suggestions based on patterns detected.

Usage:
    python tools/assess.py                  # Assess latest session
    python tools/assess.py --session 003    # Assess specific session
    python tools/assess.py --trend          # Show trend across all sessions
    python tools/assess.py --full           # Full report (latest + trend + suggestions)
    python tools/assess.py --coherence      # Assess workspace coherence
    python tools/assess.py --direction      # Assess trajectory direction
    python tools/assess.py --triad          # All three axes: session + coherence + direction
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
SESSIONS_DIR = ROOT / "sessions"
GOALS_FILE = ROOT / "goals.md"
MEMORY_DIR = ROOT / "memory"
TOOLS_DIR = ROOT / "tools"
DATA_DIR = ROOT / "data"
ASSESSMENTS_FILE = DATA_DIR / "assessments.json"


def read_file(path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def parse_session(session_id):
    """Parse a session journal and extract metrics."""
    path = SESSIONS_DIR / f"{session_id}.md"
    text = read_file(path)
    if not text:
        return None

    metrics = {
        "session_id": session_id,
        "goals_completed": 0,
        "tools_created": 0,
        "memories_added": 0,
        "patterns_applied": 0,
        "has_reflection": False,
        "has_open_questions": False,
        "sections_present": [],
    }

    lines = text.split("\n")
    current_section = None

    for line in lines:
        stripped = line.strip()

        # Track sections
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            metrics["sections_present"].append(current_section)

        # Count completed goals (multiple section name variants)
        if current_section in ("Completed This Session", "Metrics") and stripped.startswith("- "):
            lower = stripped.lower()
            if "goal" in lower and ("complete" in lower or "done" in lower):
                metrics["goals_completed"] += 1

        # Fallback: count "Goal N" completions in Completed This Session
        if current_section == "Completed This Session" and stripped.startswith("- Goal"):
            metrics["goals_completed"] += 1

        # Count tools created (or built)
        if current_section in ("Tools Created", "What I Left Behind", "Metrics") and stripped.startswith("- "):
            lower = stripped.lower()
            if "tool" in lower or "created" in lower or ".py" in lower:
                metrics["tools_created"] += 1

        # Also check "What I Did" subsections for tool creation
        if current_section == "What I Did" and stripped.startswith("- Created `tools/"):
            metrics["tools_created"] += 1

        # Count memories added
        if current_section in ("Memories Added", "Metrics") and stripped.startswith("- "):
            if stripped.startswith("- M0") or "memor" in stripped.lower():
                metrics["memories_added"] += 1

        # Detect pattern application mentions (whole document)
        if "pattern" in stripped.lower() and ("applied" in stripped.lower() or "used" in stripped.lower()):
            metrics["patterns_applied"] += 1

    # Reflection: accept "Reflection" or "What I Learned"
    reflection_sections = ["Reflection", "What I Learned"]
    metrics["has_reflection"] = any(s in metrics["sections_present"] for s in reflection_sections)

    # Open questions: accept "Open Questions" or "Recommendation for Next Session"
    question_sections = ["Open Questions", "Recommendation for Next Session"]
    metrics["has_open_questions"] = any(s in metrics["sections_present"] for s in question_sections)

    # Count question marks in reflection/question sections
    metrics["question_count"] = 0
    for qs in question_sections:
        if f"## {qs}" in text:
            section_text = text.split(f"## {qs}")[-1].split("##")[0]
            metrics["question_count"] += section_text.count("?")

    # Word count of reflection section
    metrics["reflection_words"] = 0
    for rs in reflection_sections:
        if f"## {rs}" in text:
            parts = text.split(f"## {rs}")
            if len(parts) > 1:
                reflection_text = parts[1].split("##")[0]
                metrics["reflection_words"] = max(metrics["reflection_words"], len(reflection_text.split()))
            break

    return metrics


def score_session(metrics):
    """Score a session from 0-100 across multiple dimensions."""
    if not metrics:
        return None

    scores = {}

    # Velocity (0-25): goals completed
    goals = metrics["goals_completed"]
    if goals == 0:
        scores["velocity"] = 5  # showed up, gets some credit
    elif goals == 1:
        scores["velocity"] = 15
    elif goals == 2:
        scores["velocity"] = 20
    else:
        scores["velocity"] = 25  # 3+ goals is max

    # Capability growth (0-25): tools + memories
    artifacts = metrics["tools_created"] + metrics["memories_added"]
    if artifacts == 0:
        scores["capability"] = 5
    elif artifacts <= 2:
        scores["capability"] = 12
    elif artifacts <= 5:
        scores["capability"] = 18
    else:
        scores["capability"] = 25

    # Learning (0-25): patterns applied + memories stored
    learning = metrics["patterns_applied"] + (1 if metrics["memories_added"] > 2 else 0)
    if learning == 0:
        scores["learning"] = 5
    elif learning == 1:
        scores["learning"] = 15
    elif learning == 2:
        scores["learning"] = 20
    else:
        scores["learning"] = 25

    # Reflection quality (0-25): reflection depth + open questions
    reflection_score = 0
    if metrics["has_reflection"]:
        reflection_score += 8
        if metrics["reflection_words"] > 30:
            reflection_score += 5
        if metrics["reflection_words"] > 80:
            reflection_score += 4
    if metrics["has_open_questions"]:
        reflection_score += 4
        if metrics["question_count"] >= 3:
            reflection_score += 4
    scores["reflection"] = min(reflection_score, 25)

    scores["total"] = sum(scores.values())
    return scores


def get_all_sessions():
    """Get sorted list of session IDs."""
    sessions = sorted(SESSIONS_DIR.glob("*.md"))
    return [s.stem for s in sessions]


def load_assessments():
    """Load previously stored assessments."""
    if ASSESSMENTS_FILE.exists():
        return json.loads(ASSESSMENTS_FILE.read_text(encoding="utf-8"))
    return {}


def save_assessments(data):
    """Save assessments to disk."""
    DATA_DIR.mkdir(exist_ok=True)
    ASSESSMENTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def generate_suggestions(metrics, scores):
    """Generate improvement suggestions based on scores."""
    suggestions = []

    if scores["velocity"] < 15:
        suggestions.append("Low goal completion -- consider smaller, more achievable goals or focus on one goal at a time.")

    if scores["capability"] < 12:
        suggestions.append("Few artifacts created -- aim to produce at least one tool or memory per session.")

    if scores["learning"] < 10:
        suggestions.append("Limited pattern application -- review stored memories before starting work to find applicable patterns.")

    if scores["reflection"] < 12:
        if not metrics["has_reflection"]:
            suggestions.append("No reflection section -- always reflect on what worked and what didn't.")
        if not metrics["has_open_questions"]:
            suggestions.append("No open questions -- asking questions drives future growth.")
        if metrics["reflection_words"] < 30:
            suggestions.append("Reflection is brief -- dig deeper into why things worked or didn't.")

    if not suggestions:
        suggestions.append("Strong session across all dimensions. Consider raising the bar or tackling harder challenges.")

    return suggestions


def print_session_report(session_id, metrics, scores, suggestions):
    """Print a single session assessment."""
    print(f"\n  SESSION {session_id} ASSESSMENT")
    print("  " + "-" * 40)
    print(f"  Goals completed:    {metrics['goals_completed']}")
    print(f"  Tools created:      {metrics['tools_created']}")
    print(f"  Memories added:     {metrics['memories_added']}")
    print(f"  Patterns applied:   {metrics['patterns_applied']}")
    print(f"  Reflection words:   {metrics['reflection_words']}")
    print(f"  Open questions:     {metrics['question_count']}")
    print()
    print("  SCORES (0-25 each, 100 max)")
    print("  " + "-" * 40)
    print(f"  Velocity:           {scores['velocity']:>3}/25")
    print(f"  Capability growth:  {scores['capability']:>3}/25")
    print(f"  Learning:           {scores['learning']:>3}/25")
    print(f"  Reflection quality: {scores['reflection']:>3}/25")
    print(f"  {'-' * 28}")
    print(f"  TOTAL:              {scores['total']:>3}/100")
    print()

    # Grade
    total = scores["total"]
    if total >= 85:
        grade = "Excellent"
    elif total >= 70:
        grade = "Good"
    elif total >= 50:
        grade = "Developing"
    else:
        grade = "Needs focus"
    print(f"  Grade: {grade}")
    print()

    if suggestions:
        print("  SUGGESTIONS")
        print("  " + "-" * 40)
        for s in suggestions:
            print(f"  - {s}")
        print()


def print_trend(assessments):
    """Print trend across all assessed sessions."""
    if not assessments:
        print("\n  No assessments stored yet. Run assess on individual sessions first.")
        return

    print("\n  TREND ANALYSIS")
    print("  " + "-" * 55)
    print(f"  {'Session':<10} {'Velocity':>8} {'Capable':>8} {'Learn':>8} {'Reflect':>8} {'Total':>8}")
    print("  " + "-" * 55)

    totals = []
    for sid in sorted(assessments.keys()):
        s = assessments[sid]["scores"]
        print(f"  {sid:<10} {s['velocity']:>8} {s['capability']:>8} {s['learning']:>8} {s['reflection']:>8} {s['total']:>8}")
        totals.append(s["total"])

    print("  " + "-" * 55)

    if len(totals) >= 2:
        trend = totals[-1] - totals[0]
        avg = sum(totals) / len(totals)
        print(f"  Average score: {avg:.0f}/100")
        direction = "UP" if trend > 0 else "DOWN" if trend < 0 else "FLAT"
        print(f"  Trend: {direction} {abs(trend):+d} (first to last)")

        # Detect acceleration
        if len(totals) >= 3:
            first_half = totals[:len(totals)//2]
            second_half = totals[len(totals)//2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            if avg_second > avg_first + 5:
                print("  Acceleration detected -- performance is improving over time")
            elif avg_first > avg_second + 5:
                print("  Deceleration detected -- consider refocusing")
    print()


def assess_session(session_id):
    """Assess a single session and store the result."""
    metrics = parse_session(session_id)
    if not metrics:
        print(f"  Session {session_id} not found.")
        return None, None, None

    scores = score_session(metrics)
    suggestions = generate_suggestions(metrics, scores)

    # Store
    assessments = load_assessments()
    assessments[session_id] = {
        "metrics": metrics,
        "scores": scores,
        "suggestions": suggestions,
        "assessed_at": datetime.now().isoformat(),
    }
    save_assessments(assessments)

    return metrics, scores, suggestions


def score_coherence():
    """Score workspace coherence (0-100) across five dimensions.

    This measures the system as a whole, not a single session.
    Dimensions (0-20 each):
      1. Tool interconnection -- do tools compose/import each other?
      2. Documentation -- do tools have docstrings and usage lines?
      3. Memory consistency -- no conflicting adjustments, reasonable count
      4. Goal-tool linkage -- completed goals reference actual tools
      5. File organization -- expected directories exist, no orphans
    """
    scores = {}

    # 1. Tool interconnection (0-20)
    tool_files = list(TOOLS_DIR.glob("*.py"))
    import_count = 0
    for tf in tool_files:
        text = tf.read_text(encoding="utf-8")
        # Count imports of sibling tools
        for other in tool_files:
            if other.stem != tf.stem and other.stem != "__init__":
                if f"import {other.stem}" in text:
                    import_count += 1
    # Score: 0 imports=5, 1-2=10, 3-5=15, 6+=20
    if import_count == 0:
        scores["interconnection"] = 5
    elif import_count <= 2:
        scores["interconnection"] = 10
    elif import_count <= 5:
        scores["interconnection"] = 15
    else:
        scores["interconnection"] = 20

    # 2. Documentation (0-20)
    documented = 0
    has_usage = 0
    for tf in tool_files:
        if tf.stem.startswith("__"):
            continue
        text = tf.read_text(encoding="utf-8")
        if '"""' in text[:200]:
            documented += 1
        if "Usage:" in text[:500] or "usage:" in text[:500]:
            has_usage += 1
    total_tools = len([f for f in tool_files if not f.stem.startswith("__")])
    if total_tools == 0:
        scores["documentation"] = 0
    else:
        doc_ratio = documented / total_tools
        usage_ratio = has_usage / total_tools
        scores["documentation"] = int((doc_ratio * 12) + (usage_ratio * 8))

    # 3. Memory consistency (0-20)
    memory_files = list(MEMORY_DIR.glob("*.json"))
    adjustments = []
    valid_memories = 0
    for mf in memory_files:
        try:
            entry = json.loads(mf.read_text(encoding="utf-8"))
            valid_memories += 1
            if entry.get("behavioral_adjustment"):
                adjustments.append(entry["behavioral_adjustment"].lower())
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    # Check for near-duplicates in adjustments
    duplicate_count = 0
    for i, a in enumerate(adjustments):
        for b in adjustments[i+1:]:
            # Simple overlap check: if >60% of words match
            a_words = set(a.split())
            b_words = set(b.split())
            if len(a_words & b_words) > 0.6 * max(len(a_words), len(b_words)):
                duplicate_count += 1
    # Score: valid memories exist + low duplicates
    mem_score = min(10, valid_memories)  # up to 10 for having memories
    mem_score += max(0, 10 - duplicate_count * 3)  # lose points for duplicates
    scores["memory_consistency"] = min(20, mem_score)

    # 4. Goal-tool linkage (0-20)
    goals_text = read_file(GOALS_FILE)
    completed_goals = goals_text.count("**Completed:**")
    tool_refs = 0
    for tf in tool_files:
        if tf.stem in goals_text:
            tool_refs += 1
    if completed_goals == 0:
        scores["goal_linkage"] = 5
    else:
        linkage_ratio = min(1.0, tool_refs / max(1, completed_goals))
        scores["goal_linkage"] = 5 + int(linkage_ratio * 15)

    # 5. File organization (0-20)
    org_score = 0
    expected_dirs = ["tools", "sessions", "memory", "data"]
    for d in expected_dirs:
        if (ROOT / d).is_dir():
            org_score += 3  # 12 points for expected dirs
    expected_files = ["CLAUDE.md", "NEXT.md", "goals.md", "identity.md"]
    for f in expected_files:
        if (ROOT / f).exists():
            org_score += 2  # 8 points for key files
    scores["file_organization"] = min(20, org_score)

    scores["total"] = sum(scores.values())
    return scores


def print_coherence_report(scores):
    """Print workspace coherence report."""
    print("\n  WORKSPACE COHERENCE ASSESSMENT")
    print("  " + "-" * 40)
    print("  SCORES (0-20 each, 100 max)")
    print("  " + "-" * 40)
    print(f"  Tool interconnection:  {scores['interconnection']:>3}/20")
    print(f"  Documentation:         {scores['documentation']:>3}/20")
    print(f"  Memory consistency:    {scores['memory_consistency']:>3}/20")
    print(f"  Goal-tool linkage:     {scores['goal_linkage']:>3}/20")
    print(f"  File organization:     {scores['file_organization']:>3}/20")
    print(f"  {'-' * 28}")
    print(f"  COHERENCE TOTAL:       {scores['total']:>3}/100")

    total = scores["total"]
    if total >= 80:
        grade = "Highly coherent"
    elif total >= 60:
        grade = "Well integrated"
    elif total >= 40:
        grade = "Developing"
    else:
        grade = "Fragmented"
    print(f"  Grade: {grade}")
    print()

    # Suggestions
    suggestions = []
    if scores["interconnection"] < 10:
        suggestions.append("Tools are isolated -- consider building composition between them.")
    if scores["documentation"] < 12:
        suggestions.append("Some tools lack docstrings or usage documentation.")
    if scores["memory_consistency"] < 12:
        suggestions.append("Memory has duplicates or is sparse -- clean up or consolidate.")
    if scores["goal_linkage"] < 12:
        suggestions.append("Completed goals don't reference tools -- link outcomes to artifacts.")
    if scores["file_organization"] < 15:
        suggestions.append("Missing expected directories or key files.")
    if suggestions:
        print("  COHERENCE SUGGESTIONS")
        print("  " + "-" * 40)
        for s in suggestions:
            print(f"  - {s}")
        print()


def score_direction():
    """Score trajectory direction (0-100) across five dimensions.

    This measures whether the project is going somewhere intentional.
    Dimensions (0-20 each):
      1. Goal progression -- are goals getting more complex over time?
      2. Goal lineage -- do later goals build on earlier ones?
      3. Session continuity -- does NEXT.md bridge sessions effectively?
      4. Backlog health -- are there meaningful future goals?
      5. Trajectory consistency -- sustained engagement over time?
    """
    scores = {}
    goals_text = read_file(GOALS_FILE)
    sessions = get_all_sessions()

    # 1. Goal progression (0-20)
    # Check if completed goals show increasing complexity
    # Proxy: later goals have longer result descriptions (richer outcomes)
    # and whether goals per session is increasing (tackling more at once)
    result_sections = re.findall(
        r"### \d+\.\s+(.+?)(?=\n###|\n## |\Z)", goals_text, re.DOTALL
    )
    if len(result_sections) < 3:
        scores["progression"] = 8  # too few to measure trend
    else:
        # Compare first third vs last third result section length
        third = max(1, len(result_sections) // 3)
        first_avg = sum(len(s) for s in result_sections[:third]) / third
        last_avg = sum(len(s) for s in result_sections[-third:]) / max(1, third)
        if last_avg > first_avg * 1.3:
            scores["progression"] = 20  # clearly growing
        elif last_avg > first_avg * 1.1:
            scores["progression"] = 16  # modest growth
        elif last_avg > first_avg * 0.9:
            scores["progression"] = 12  # stable
        else:
            scores["progression"] = 6   # shrinking

    # 2. Goal lineage (0-20)
    # Do completed goal descriptions reference earlier goals or their tools?
    completed_goals = re.findall(r"### (\d+)\.\s+(.+)", goals_text)
    result_texts = re.findall(r"\*\*Result:\*\*\s+(.+)", goals_text)
    # Filter out generic words that would match everything
    generic_words = {"build", "system", "learn", "answer", "about", "error",
                     "means", "actually", "improvement", "logging", "framework"}
    cross_refs = 0
    for i, (num, name) in enumerate(completed_goals):
        if i == 0:
            continue
        goal_section = goals_text.split(f"### {num}.")[1].split("###")[0] if f"### {num}." in goals_text else ""
        for prev_num, prev_name in completed_goals[:i]:
            prev_key_words = [w for w in prev_name.lower().split()
                              if len(w) > 4 and w not in generic_words]
            for word in prev_key_words:
                if word in goal_section.lower():
                    cross_refs += 1
                    break
    # Score based on cross-references found
    if cross_refs == 0:
        scores["lineage"] = 5
    elif cross_refs <= 2:
        scores["lineage"] = 10
    elif cross_refs <= 5:
        scores["lineage"] = 15
    else:
        scores["lineage"] = 20

    # 3. Session continuity (0-20)
    # Does NEXT.md exist, have content, reference specific priorities?
    next_text = read_file(ROOT / "NEXT.md")
    continuity = 0
    if next_text:
        continuity += 5  # exists
        if len(next_text) > 200:
            continuity += 5  # substantial
        if "priorit" in next_text.lower() or "immediate" in next_text.lower():
            continuity += 5  # has priorities
        if any(f"session" in next_text.lower() for _ in [1]):
            continuity += 3  # references sessions
        # Check for actionable items (numbered or bulleted lists)
        action_lines = [l for l in next_text.split("\n") if re.match(r"\s*[\d\-\*]\.*\s", l)]
        if len(action_lines) >= 3:
            continuity += 2  # has multiple actionable items
    scores["continuity"] = min(20, continuity)

    # 4. Backlog health (0-20)
    # Are there meaningful future goals? Not too few (stagnation), not too many (unfocused)
    backlog_section = ""
    if "## Backlog" in goals_text:
        # Split on \n## to avoid matching ### headers within the section
        remaining = goals_text.split("## Backlog")[1]
        parts = remaining.split("\n## ")
        backlog_section = parts[0]
    backlog_items = [l for l in backlog_section.split("\n") if l.strip().startswith("- ")]
    if len(backlog_items) == 0:
        scores["backlog"] = 2  # no future direction
    elif len(backlog_items) <= 2:
        scores["backlog"] = 10  # limited options
    elif len(backlog_items) <= 7:
        scores["backlog"] = 20  # healthy range
    elif len(backlog_items) <= 12:
        scores["backlog"] = 15  # getting unfocused
    else:
        scores["backlog"] = 8   # too many -- lack of prioritization

    # 5. Trajectory consistency (0-20)
    # Do assessment scores show sustained engagement (not just one spike)?
    assessments = load_assessments()
    if len(assessments) < 2:
        scores["trajectory"] = 5
    else:
        totals = [assessments[sid]["scores"]["total"] for sid in sorted(assessments.keys())]
        # Check for sustained engagement: most sessions score above 50
        above_threshold = sum(1 for t in totals if t >= 50)
        engagement_ratio = above_threshold / len(totals)
        if engagement_ratio >= 0.8:
            scores["trajectory"] = 20  # consistently engaged
        elif engagement_ratio >= 0.6:
            scores["trajectory"] = 15
        elif engagement_ratio >= 0.4:
            scores["trajectory"] = 10
        else:
            scores["trajectory"] = 5

    scores["total"] = sum(scores.values())
    return scores


def print_direction_report(scores):
    """Print trajectory direction report."""
    print("\n  TRAJECTORY DIRECTION ASSESSMENT")
    print("  " + "-" * 40)
    print("  SCORES (0-20 each, 100 max)")
    print("  " + "-" * 40)
    print(f"  Goal progression:      {scores['progression']:>3}/20")
    print(f"  Goal lineage:          {scores['lineage']:>3}/20")
    print(f"  Session continuity:    {scores['continuity']:>3}/20")
    print(f"  Backlog health:        {scores['backlog']:>3}/20")
    print(f"  Trajectory consistency:{scores['trajectory']:>3}/20")
    print(f"  {'-' * 28}")
    print(f"  DIRECTION TOTAL:       {scores['total']:>3}/100")

    total = scores["total"]
    if total >= 80:
        grade = "Clear trajectory"
    elif total >= 60:
        grade = "Directed"
    elif total >= 40:
        grade = "Searching"
    else:
        grade = "Drifting"
    print(f"  Grade: {grade}")
    print()

    suggestions = []
    if scores["progression"] < 10:
        suggestions.append("Goals are not growing in complexity -- aim higher.")
    if scores["lineage"] < 10:
        suggestions.append("Goals seem disconnected -- build each goal on previous ones.")
    if scores["continuity"] < 10:
        suggestions.append("NEXT.md is thin -- leave clear, actionable priorities for next session.")
    if scores["backlog"] < 10:
        suggestions.append("Backlog is empty or bloated -- maintain 3-7 meaningful future goals.")
    if scores["trajectory"] < 10:
        suggestions.append("Engagement is inconsistent -- focus on sustained, steady work.")
    if suggestions:
        print("  DIRECTION SUGGESTIONS")
        print("  " + "-" * 40)
        for s in suggestions:
            print(f"  - {s}")
        print()


def main():
    parser = argparse.ArgumentParser(description="AgentZero Self-Assessment")
    parser.add_argument("--session", type=str, help="Assess specific session (e.g. 003)")
    parser.add_argument("--trend", action="store_true", help="Show trend across sessions")
    parser.add_argument("--full", action="store_true", help="Full report (latest + trend + suggestions)")
    parser.add_argument("--coherence", action="store_true", help="Assess workspace coherence")
    parser.add_argument("--direction", action="store_true", help="Assess trajectory direction")
    parser.add_argument("--triad", action="store_true", help="All three axes: session + coherence + direction")
    args = parser.parse_args()

    print("=" * 60)
    print("  AGENTZERO SELF-ASSESSMENT")
    print("=" * 60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.triad:
        # Full theory-of-improvement report: all three axes
        # 1. Latest session (Capability axis)
        all_sessions = get_all_sessions()
        if all_sessions:
            for sid in all_sessions:
                assess_session(sid)
            latest = all_sessions[-1]
            m, s, sg = assess_session(latest)
            if m:
                print_session_report(latest, m, s, sg)

        # 2. Coherence axis
        coh = score_coherence()
        print_coherence_report(coh)

        # 3. Direction axis
        dir_scores = score_direction()
        print_direction_report(dir_scores)

        # Summary
        session_score = s["total"] if all_sessions and m else 0
        print("  THEORY OF IMPROVEMENT -- TRIAD SUMMARY")
        print("  " + "-" * 40)
        print(f"  Capability (session):  {session_score:>3}/100")
        print(f"  Coherence (workspace): {coh['total']:>3}/100")
        print(f"  Direction (trajectory):{dir_scores['total']:>3}/100")
        overall = (session_score + coh["total"] + dir_scores["total"]) // 3
        print(f"  {'-' * 28}")
        print(f"  OVERALL:               {overall:>3}/100")
        print()
        return

    if args.direction:
        dir_scores = score_direction()
        print_direction_report(dir_scores)
        return

    if args.coherence:
        scores = score_coherence()
        print_coherence_report(scores)
        return

    if args.trend:
        # Assess all sessions first
        for sid in get_all_sessions():
            assess_session(sid)
        print_trend(load_assessments())
        return

    if args.full:
        # Assess all sessions, show latest + trend
        all_sessions = get_all_sessions()
        for sid in all_sessions:
            assess_session(sid)

        if all_sessions:
            latest = all_sessions[-1]
            m, s, sg = assess_session(latest)
            if m:
                print_session_report(latest, m, s, sg)

        print_trend(load_assessments())
        return

    # Single session
    if args.session:
        session_id = args.session
    else:
        sessions = get_all_sessions()
        if not sessions:
            print("\n  No sessions found.")
            return
        session_id = sessions[-1]

    metrics, scores, suggestions = assess_session(session_id)
    if metrics:
        print_session_report(session_id, metrics, scores, suggestions)


if __name__ == "__main__":
    main()
