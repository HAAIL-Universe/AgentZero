class Hippocampus:
    def __init__(self):
        self.thought_arena = []  # shared working memory
        self.decision_log = []   # historical suggestions
        self.outcome_log = []    # real-world consequences

    def share_thought(self, source: str, thought: dict):
        self.thought_arena.append({"from": source, **thought})

    def fetch_thoughts(self, filter_by=None):
        if not filter_by:
            return self.thought_arena
        return [t for t in self.thought_arena if t.get("from") in filter_by]

    def archive_decision(self, decision: dict):
        self.decision_log.append(decision)

    def log_outcome(self, outcome: dict):
        self.outcome_log.append(outcome)

    def suggest_modification(self, agent: str, old_thought: dict, revised: dict):
        self.thought_arena.append({
            "from": agent,
            "modification_of": old_thought,
            "revision": revised
        })

    def clear_thought_arena(self):
        self.thought_arena = []
