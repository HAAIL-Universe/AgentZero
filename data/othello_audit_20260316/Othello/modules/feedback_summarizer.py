from typing import Optional, List

def summarize_feedback(
    mood: Optional[str] = None,
    goals: Optional[List[str]] = None,
    traits: Optional[List[str]] = None
) -> str:
    """
    Formats parsed user feedback for display in the FELLO console or UI.

    Args:
        mood (Optional[str]): Detected mood or emotional state (e.g., "positive", "negative").
        goals (Optional[List[str]]): List of goals or intentions mentioned in the session.
        traits (Optional[List[str]]): List of behavioral or personality traits detected.

    Returns:
        str: A summary string combining all detected elements, ready for display.
    """
    lines = []

    # Show mood if present
    if mood:
        lines.append(f"🧠 Detected Mood: {mood.capitalize()}")

    # List all detected goals
    if goals:
        lines.append("🎯 Goals Mentioned:")
        for g in goals:
            lines.append(f" - {g.strip()}")

    # List all detected traits
    if traits:
        lines.append("🧬 Traits Detected:")
        for t in traits:
            lines.append(f" - {t.capitalize()}")

    # Fallback: If nothing was detected, invite more sharing
    if not lines:
        return "Tell me more. I'm listening."

    # Return formatted summary, each detail on its own line
    return "\n".join(lines)
