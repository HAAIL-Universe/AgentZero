"""
core/meta_analysis.py

Crunches intervention logs to produce meta reflection stats and suggestions.

Author: Julius
"""

from datetime import datetime
from collections import Counter

class MetaAnalysis:
    """
    MetaAnalysis engine for FELLO — analyzes intervention history, patterns, and generates strategy suggestions.
    """

    def __init__(self):
        self.last_summary = None

    def summarize_reflection(self, interventions):
        stats = self._calculate_stats(interventions)
        patterns = self._find_patterns(interventions)
        suggestions = self._generate_suggestions(stats, patterns)
        summary = self._make_summary(stats, patterns, suggestions)
        self.last_summary = summary
        return summary

    @staticmethod
    def _calculate_stats(interventions):
        if not interventions:
            return {
                "hit_rate": 0.0,
                "ignore_rate": 0.0,
                "fail_rate": 0.0
            }

        total = len(interventions)
        hit = sum(1 for i in interventions if i.get("outcome") == "acted")
        ignored = sum(1 for i in interventions if i.get("outcome") == "ignored")
        fail = sum(1 for i in interventions if i.get("outcome") == "pushed_back")

        return {
            "hit_rate": hit / total,
            "ignore_rate": ignored / total,
            "fail_rate": fail / total
        }

    @staticmethod
    def _find_patterns(interventions):
        hits = [i.get("nudge_type") for i in interventions if i.get("outcome") == "acted"]
        fails = [i.get("nudge_type") for i in interventions if i.get("outcome") in ["ignored", "pushed_back"]]

        top_success = Counter(hits).most_common(1)
        top_fail = Counter(fails).most_common(1)

        return {
            "top_success": top_success[0][0] if top_success else None,
            "top_fail": top_fail[0][0] if top_fail else None
        }

    @staticmethod
    def _generate_suggestions(stats, patterns):
        suggestions = []

        if stats["ignore_rate"] > 0.4:
            suggestions.append("Consider reducing nudge frequency or changing timing.")
        if stats["hit_rate"] > 0.5:
            suggestions.append("Keep up the current nudge patterns—they're working.")
        if patterns["top_fail"]:
            suggestions.append(f"Review strategy for {patterns['top_fail']}—often ignored or pushed back.")
        if patterns["top_success"]:
            suggestions.append(f"Leverage more {patterns['top_success']} nudges—they perform well.")

        return suggestions

    @staticmethod
    def _make_summary(stats, patterns, suggestions):
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "hit_rate": stats["hit_rate"],
                "ignore_rate": stats["ignore_rate"],
                "fail_rate": stats["fail_rate"],
                "top_success": patterns["top_success"],
                "top_fail": patterns["top_fail"]
            },
            "suggestions": suggestions
        }

# --- Example standalone test ---
if __name__ == "__main__":
    fake_data = [
        {"nudge_type": "walk_reminder", "outcome": "acted"},
        {"nudge_type": "water_reminder", "outcome": "ignored"},
        {"nudge_type": "walk_reminder", "outcome": "acted"},
        {"nudge_type": "afternoon_nudge", "outcome": "pushed_back"},
    ]
    meta = MetaAnalysis()
    summary = meta.summarize_reflection(fake_data)
    from pprint import pprint
    pprint(summary)
