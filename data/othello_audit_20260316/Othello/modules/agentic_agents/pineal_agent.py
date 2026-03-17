from datetime import datetime

class PinealAgent:
    """
    PinealAgent: Spiritual-alignment and intuition-based arbitrator.
    Interprets internal system suggestions through the lens of user's long-term self-actualization,
    alignment with identity, values, and declared spiritual/psychological goals.
    """

    def __init__(self, hippocampus=None, user_profile=None):
        self.hippocampus = hippocampus
        self.user_profile = user_profile or {}

    def evaluate_alignment(self, thought: dict) -> dict:
        """
        Checks a proposed idea against user's deep goals, values, and identity alignment.
        Returns feedback or moderation guidance.
        """
        suggestion = thought.get("suggestion", "")
        reasoning = thought.get("reasoning", "")

        alignment_score = self._score_alignment(suggestion, reasoning)
        guidance = self._generate_guidance(alignment_score, suggestion)

        evaluation = {
            "from": "PinealAgent",
            "evaluation_time": datetime.now().isoformat(),
            "suggestion": suggestion,
            "alignment_score": alignment_score,
            "guidance": guidance
        }

        if self.hippocampus:
            self.hippocampus.share_thought("PinealAgent", evaluation)

        return evaluation

    def _score_alignment(self, suggestion, reasoning):
        score = 5  # baseline neutral

        # Adjust based on spiritual keywords or misalignment with persona
        lowers = (suggestion + " " + reasoning).lower()
        traits = self.user_profile.get("traits", [])
        spiritual_goals = self.user_profile.get("spiritual_goals", [])

        if any(term in lowers for term in ["quit", "abandon", "destroy"]):
            score -= 2
        if any(term in lowers for term in ["align", "connect", "purpose", "soul"]):
            score += 2

        for goal in spiritual_goals:
            if goal.lower() in lowers:
                score += 1

        return max(1, min(10, score))

    def _generate_guidance(self, score, suggestion):
        if score >= 8:
            return "Strongly aligned with the user's spiritual trajectory."
        elif score >= 5:
            return "Generally compatible. Clarify intent before proceeding."
        else:
            return f"Caution: Suggestion '{suggestion}' may conflict with user's deeper values."

    def update_user_profile(self, profile_data):
        self.user_profile.update(profile_data)

    def fetch_latest_alignment(self):
        if self.hippocampus:
            pineal_thoughts = self.hippocampus.fetch_thoughts(filter_by=["PinealAgent"])
            return pineal_thoughts[-1] if pineal_thoughts else {}
        return {}
